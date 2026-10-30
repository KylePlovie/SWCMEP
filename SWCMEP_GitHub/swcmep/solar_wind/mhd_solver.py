from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
from swcmep.grid.spherical_grid import SphericalGrid
from swcmep.solar_wind.background_2d import I_P, I_RHO, I_VPHI, I_VR, N_VARS
from swcmep.utils.constants import M_PROTON
RHO = I_RHO
VR = I_VR
VPHI = I_VPHI
P = I_P

@dataclass
class MHDSolver:
    grid: SphericalGrid
    U: np.ndarray
    gamma: float = 1.05
    cfl_factor: float = 0.5
    inner_boundary: str = 'fixed'
    outer_boundary: str = 'outflow'
    t: float = field(default=0.0, init=False)
    step: int = field(default=0, init=False)
    dt: float = field(default=0.0, init=False)
    _U_inner_fixed: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._U_inner_fixed = self.U[:, :2, :].copy()
        assert self.U.shape == (N_VARS, self.grid.nr, self.grid.nphi), f'State shape mismatch: {self.U.shape} != ({N_VARS}, {self.grid.nr}, {self.grid.nphi})'

    def sound_speed(self, U: np.ndarray) -> np.ndarray:
        rho = np.maximum(U[RHO], 1e-20)
        p = np.maximum(U[P], 1e-20)
        return np.sqrt(self.gamma * p / rho)

    def max_wave_speed(self, U: np.ndarray) -> float:
        cs = self.sound_speed(U)
        s_r = np.abs(U[VR]) + cs
        s_phi = np.abs(U[VPHI]) + cs
        return float(max(s_r.max(), s_phi.max()))

    def compute_dt(self) -> float:
        cs = self.sound_speed(self.U)
        v_r = np.abs(self.U[VR])
        v_phi = np.abs(self.U[VPHI])
        dr_2d = self.grid.dr[:, np.newaxis]
        dt_r = dr_2d / (v_r + cs + 1e-20)
        r_2d = self.grid.R
        dt_phi = r_2d * self.grid.dphi / (v_phi + cs + 1e-20)
        dt = self.cfl_factor * min(dt_r.min(), dt_phi.min())
        return float(dt)

    def _apply_boundaries(self, U: np.ndarray):
        if self.inner_boundary == 'fixed':
            U[:, 0, :] = self._U_inner_fixed[:, 0, :]
            U[:, 1, :] = self._U_inner_fixed[:, 1, :]
        else:
            U[:, 0, :] = U[:, 1, :]
        U[:, -1, :] = U[:, -2, :]
        U[:, -2, :] = U[:, -3, :]
        U[RHO] = np.maximum(U[RHO], 1e-25)
        U[P] = np.maximum(U[P], 1e-20)

    def _lax_friedrichs_step(self, U: np.ndarray, dt: float) -> np.ndarray:
        return U.copy()

    def run(self, duration_seconds: float, output_every: float=3600.0, callback: Optional[Callable]=None, verbose: bool=True) -> list[dict]:
        snapshots = []
        t_end = self.t + duration_seconds
        t_next_out = self.t + output_every
        wall_start = time.time()
        if verbose:
            print(f'Running MHD solver for {duration_seconds / 3600:.1f} hours...')
            print(f'  Grid: {self.grid.nr} × {self.grid.nphi} cells')
            print(f'  Output every: {output_every / 60:.0f} min')
        while self.t < t_end:
            self.dt = self.compute_dt()
            if self.t + self.dt > t_end:
                self.dt = t_end - self.t
            U_new = self._lax_friedrichs_step(self.U, self.dt)
            self._apply_boundaries(U_new)
            self.U = U_new
            self.t += self.dt
            self.step += 1
            if self.t >= t_next_out or self.t >= t_end:
                snap = {'t': self.t, 'step': self.step, 'U': self.U.copy(), 'dt': self.dt}
                snapshots.append(snap)
                if callback:
                    callback(self, snap)
                if verbose:
                    wall_elapsed = time.time() - wall_start
                    v_earth = self._v_at_earth()
                    n_earth = self._n_at_earth()
                    print(f'  t={self.t / 3600:.2f}h  step={self.step:6d}  dt={self.dt:.1f}s  v(Earth)={v_earth:.0f}km/s  n(Earth)={n_earth:.2f}cm³  wall={wall_elapsed:.1f}s')
                t_next_out += output_every
        if verbose:
            total_wall = time.time() - wall_start
            print(f'Done. {self.step} steps in {total_wall:.1f}s ({self.step / total_wall:.0f} steps/s)')
        return snapshots

    def _v_at_earth(self) -> float:
        i, j = self.grid.earth_index()
        return float(self.U[VR, i, j]) / 1000.0

    def _n_at_earth(self) -> float:
        i, j = self.grid.earth_index()
        return float(self.U[RHO, i, j]) / M_PROTON / 1000000.0

    def total_mass(self) -> float:
        area = self.grid.cell_area()
        return float((self.U[RHO] * area).sum())

    def is_valid(self) -> bool:
        return np.all(np.isfinite(self.U)) and np.all(self.U[RHO] > 0) and np.all(self.U[P] > 0)
