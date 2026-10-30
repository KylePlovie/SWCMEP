from __future__ import annotations
import math
import time
from typing import Callable, Optional
import numpy as np
import torch
from swcmep.grid.spherical_grid import SphericalGrid, SphericalGrid3D
from swcmep.solar_wind.background_2d import I_BPHI, I_BR, I_P, I_PSI, I_RHO, I_VPHI, I_VR
from swcmep.solar_wind.hlle_muscl import hlle_muscl_step, hlle_muscl_step_3d
from swcmep.solar_wind.mhd_solver import RHO, VPHI, VR, MHDSolver, P
from swcmep.utils.constants import AU, I_BTHETA, I_EINT, I_VTHETA, M_PROTON, M_SUN, MU_0, G
from swcmep.utils.validation import finite_float, integer_value, positive_float

class SimulationError(FloatingPointError):
    pass

class MHDSolverGPU(MHDSolver):

    def __init__(self, grid: SphericalGrid, U: np.ndarray, gamma: float=1.05, cfl_factor: float=0.5, inner_boundary: str='fixed', outer_boundary: str='outflow', device: Optional[str]=None, dtype: torch.dtype=torch.float32, use_glm: bool=True):
        gamma = finite_float(gamma, 'gamma')
        cfl_factor = finite_float(cfl_factor, 'cfl_factor')
        if gamma <= 1.0:
            raise ValueError('gamma must be greater than 1')
        if not 0.0 < cfl_factor <= 1.0:
            raise ValueError('cfl_factor must be in (0, 1]')
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('dtype must be torch.float32 or torch.float64')
        if inner_boundary not in ('fixed', 'outflow') or outer_boundary != 'outflow':
            raise ValueError('supported boundaries: inner fixed/outflow, outer outflow')
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.dtype = dtype
        self.use_glm = use_glm
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1000000000.0
            print(f'GPU solver initialized: {gpu_name} ({vram_gb:.1f} GB VRAM)')
        else:
            print('GPU not available — running on CPU with PyTorch')
        super().__init__(grid, U, gamma, cfl_factor, inner_boundary, outer_boundary)
        self._U_gpu = torch.tensor(self.U, dtype=self.dtype, device=self.device)
        self._U_inner_gpu = torch.tensor(self._U_inner_fixed, dtype=self.dtype, device=self.device)
        self._dr_gpu = torch.tensor(self.grid.dr[:, np.newaxis], dtype=self.dtype, device=self.device)
        self._r_gpu = torch.tensor(self.grid.R, dtype=self.dtype, device=self.device)
        self._dphi_gpu = float(self.grid.dphi)
        self._grav_gpu = torch.tensor(-G * M_SUN / self.grid.R ** 2, dtype=self.dtype, device=self.device)
        self._U_eq_gpu = torch.tensor(self.U, dtype=self.dtype, device=self.device)
        self._i_earth, self._j_earth = self.grid.earth_index()
        self.use_well_balanced = True
        self.use_boundary_sponge = True
        self.sponge_width = 6
        self.sponge_alpha0 = 0.35

    def set_equilibrium_state(self, U_eq):
        self._U_eq_gpu = torch.tensor(U_eq, dtype=self.dtype, device=self.device)
        self._U_inner_gpu = self._U_eq_gpu[:, :2, :].clone()

    def _sound_speed_gpu(self, U):
        rho = torch.clamp(U[RHO], min=1e-25)
        p = torch.clamp(U[P], min=1e-20)
        return torch.sqrt(self.gamma * p / rho)

    def compute_dt(self):
        cs = self._sound_speed_gpu(self._U_gpu)
        v_r = torch.abs(self._U_gpu[VR])
        v_phi = torch.abs(self._U_gpu[VPHI])
        rho = torch.clamp(self._U_gpu[RHO], min=1e-30)
        B2 = self._U_gpu[I_BR] ** 2 + self._U_gpu[I_BPHI] ** 2
        v_A = torch.sqrt(B2 / (MU_0 * rho))
        cf = torch.sqrt(cs ** 2 + v_A ** 2)
        lambda_r = v_r + cf
        lambda_phi = v_phi + cf
        if self.use_glm:
            c_h = torch.maximum(lambda_r, lambda_phi).max()
            lambda_r = torch.maximum(lambda_r, c_h)
            lambda_phi = torch.maximum(lambda_phi, c_h)
        rate = lambda_r / self._dr_gpu + lambda_phi / (self._r_gpu * self._dphi_gpu)
        if not bool(torch.isfinite(rate).all()) or float(rate.max()) <= 0.0:
            raise SimulationError('invalid 2D CFL rate')
        return self.cfl_factor / float(rate.max())

    def _lax_friedrichs_step_gpu(self, U, dt):
        return hlle_muscl_step(U=U, dr_gpu=self._dr_gpu, r_gpu=self._r_gpu, dphi=self._dphi_gpu, grav_gpu=self._grav_gpu, gamma=self.gamma, dt=dt, G_MSun=G * M_SUN, U_eq=self._U_eq_gpu if self.use_well_balanced else None, use_glm=self.use_glm)

    def _apply_boundaries_gpu(self, U, injector=None, t_seconds=None):
        boundary_time = self.t if t_seconds is None else t_seconds
        if injector is not None and injector.is_injecting(boundary_time):
            pass
        elif self.inner_boundary == 'fixed':
            U[:, 0, :] = self._U_eq_gpu[:, 0, :]
            U[:, 1, :] = self._U_eq_gpu[:, 1, :]
        else:
            U[:, 0, :] = U[:, 1, :]
        U[:, -1, :] = self._U_eq_gpu[:, -1, :]
        U[:, -2, :] = self._U_eq_gpu[:, -2, :]
        apply_sponge = self.use_boundary_sponge
        if apply_sponge and (injector is None or not injector.is_injecting(boundary_time)):
            n_sponge = self.sponge_width
            for ii in range(2, min(2 + n_sponge, U.shape[1] - 2)):
                alpha = self.sponge_alpha0 * (1.0 - (ii - 2) / max(n_sponge, 1))
                U[:, ii, :] = (1.0 - alpha) * U[:, ii, :] + alpha * self._U_eq_gpu[:, ii, :]
        U[I_RHO] = torch.clamp(U[I_RHO], min=1e-25)
        U[I_P] = torch.clamp(U[I_P], min=1e-20)
        if U.shape[0] > I_EINT:
            U[I_EINT] = torch.maximum(torch.clamp(U[I_EINT], min=1e-20 / (self.gamma - 1.0 + 1e-30)), U[I_P] / (self.gamma - 1.0 + 1e-30))
        U[I_PSI, 0, :] = 0.0
        U[I_PSI, 1, :] = 0.0
        U[I_PSI, -1, :] = 0.0
        U[I_PSI, -2, :] = 0.0

    def compute_divB(self):
        U = self._U_gpu
        r = self._r_gpu
        dr = self._dr_gpu
        dphi = self._dphi_gpu
        B_r = U[I_BR]
        B_phi = U[I_BPHI]
        r2Br = r ** 2 * B_r
        divB_r = (r2Br[2:, :] - r2Br[:-2, :]) / (2.0 * r[1:-1, :] ** 2 * dr[1:-1, :])
        divB_phi = (torch.roll(B_phi, -1, dims=1) - torch.roll(B_phi, 1, dims=1)) / (2.0 * r * dphi)
        divB = divB_r + divB_phi[1:-1, :]
        B_mag = torch.sqrt(B_r[1:-1, :] ** 2 + B_phi[1:-1, :] ** 2) + 1e-30
        dx = torch.minimum(dr[1:-1, :], r[1:-1, :] * dphi)
        divB_norm = torch.abs(divB) * dx / B_mag
        return (float(divB_norm.max()), float(divB_norm.mean()))

    def diagnostics(self):
        U = self._U_gpu
        area = torch.tensor(self.grid.cell_area(), dtype=self.dtype, device=self.device)
        rho = U[I_RHO]
        v_r = U[I_VR]
        p = U[I_P]
        g1 = self.gamma - 1.0 + 1e-20
        E = p / g1 + 0.5 * rho * (v_r ** 2 + U[I_VPHI] ** 2)
        B_r = U[I_BR]
        B_phi = U[I_BPHI]
        E_mag = (B_r ** 2 + B_phi ** 2) / (2.0 * MU_0)
        divB_max, divB_mean = self.compute_divB()
        return {'total_mass': float((rho * area).sum()), 'total_energy': float(((E + E_mag) * area).sum()), 'min_pressure': float(p.min()), 'max_velocity': float(v_r.max()) / 1000.0, 'min_density': float(rho.min()), 'max_B_nT': float(torch.sqrt(B_r ** 2 + B_phi ** 2).max()) * 1000000000.0, 'max_psi': float(torch.abs(U[I_PSI]).max()), 'divB_max': divB_max, 'divB_mean': divB_mean, 'is_valid': self.is_valid()}

    def run(self, duration_seconds: float, output_every: float=3600.0, callback: Optional[Callable]=None, verbose: bool=True, injector: object=None, retain_snapshots: bool=True) -> list[dict]:
        duration_seconds = positive_float(duration_seconds, 'duration_seconds', allow_zero=True)
        output_every = positive_float(output_every, 'output_every')
        t_start = finite_float(self.t, 'solver time')
        t_end = finite_float(t_start + duration_seconds, 'end time')
        if duration_seconds and t_end <= t_start:
            raise ValueError('duration_seconds is too small to advance the current solver time')
        if t_start + output_every <= t_start:
            raise ValueError('output_every is too small for the current solver time')
        snapshots = []
        output_count = 0
        t_next_out = t_start + output_every
        wall_start = time.perf_counter()
        self.termination_reason = None
        self.last_error = None
        apply_inflow = None
        apply_drag = None
        if injector is not None:
            for name in ('apply_boundary_gpu_3d', 'apply_boundary_gpu', 'apply_gpu'):
                apply_inflow = getattr(injector, name, None)
                if apply_inflow is not None:
                    break
            if apply_inflow is None:
                raise TypeError('injector must provide a GPU boundary method')
            apply_drag = getattr(injector, 'apply_drag_gpu', None)

        def validate_state(state, stage):
            bad = ~torch.isfinite(state)
            if bool(bad.any()):
                index = tuple((int(x) for x in bad.nonzero()[0]))
                self.termination_reason = 'nonfinite_state'
                raise SimulationError(f'non-finite state {stage} at index {index}')
            if bool((state[I_RHO] <= 0).any()) or bool((state[I_P] <= 0).any()):
                self.termination_reason = 'invalid_state'
                raise SimulationError(f'non-positive density or pressure {stage}')
        if verbose:
            print(f'Running {self._U_gpu.ndim - 1}D MHD for {duration_seconds / 3600:.3f} h on {self.device} ({self.dtype}); grid={tuple(self._U_gpu.shape[1:])}')
        try:
            validate_state(self._U_gpu, 'at run start')
            while self.t < t_end:
                if apply_inflow is not None:
                    apply_inflow(self._U_gpu, self.t)
                    validate_state(self._U_gpu, 'after injection')
                self.dt_cfl = float(self.compute_dt())
                if not math.isfinite(self.dt_cfl) or self.dt_cfl < 1e-06:
                    self.termination_reason = 'dt_collapse'
                    raise SimulationError(f'CFL timestep collapsed to {self.dt_cfl:.3e} s at step {self.step}, t={self.t:.6g} s')
                dt = min(self.dt_cfl, t_end - self.t, t_next_out - self.t)
                if injector is not None and hasattr(injector, 'timestep_limit'):
                    limit = float(injector.timestep_limit(self.t))
                    if math.isnan(limit) or limit <= 0:
                        raise ValueError('injector timestep_limit must be positive')
                    dt = min(dt, limit)
                next_time = self.t + dt
                if next_time <= self.t:
                    self.termination_reason = 'dt_collapse'
                    raise SimulationError('timestep is too small to advance solver time')
                previous_faces = None
                if getattr(self, 'use_ct', False):
                    previous_faces = (self._Br_face, self._Bth_face, self._Bphi_face)
                try:
                    U_new = self._lax_friedrichs_step_gpu(self._U_gpu, dt)
                    validate_state(U_new, 'after MHD update')
                    if apply_drag is not None:
                        apply_drag(U_new, dt)
                    self._apply_boundaries_gpu(U_new, injector=injector, t_seconds=next_time)
                    if apply_inflow is not None:
                        apply_inflow(U_new, next_time)
                    validate_state(U_new, 'after boundary/drag update')
                except Exception:
                    if previous_faces is not None:
                        self._Br_face, self._Bth_face, self._Bphi_face = previous_faces
                    raise
                self._U_gpu = U_new
                self.t = next_time
                self.dt = dt
                self.step += 1
                tolerance = 8.0 * math.ulp(max(abs(self.t), 1.0))
                if self.t >= t_next_out - tolerance or self.t >= t_end - tolerance:
                    output_count += 1
                    if retain_snapshots or callback is not None:
                        U_cpu = self._U_gpu.detach().cpu().numpy().copy()
                        snap = {'t': self.t, 'step': self.step, 'U': U_cpu, 'dt': self.dt}
                        self.U = U_cpu.copy()
                        if retain_snapshots:
                            snapshots.append(snap)
                        if callback is not None:
                            callback(self, snap)
                    if verbose:
                        earth_index = self.grid.earth_index_3d() if self._U_gpu.ndim == 4 else self.grid.earth_index()
                        v = float(self._U_gpu[I_VR, *earth_index]) / 1000.0
                        n = float(self._U_gpu[I_RHO, *earth_index]) / M_PROTON / 1000000.0
                        print(f'  t={self.t / 3600:.3f}h step={self.step:6d} dt={dt:.2f}s v(Earth)={v:.0f}km/s n(Earth)={n:.2f}cm^-3 wall={time.perf_counter() - wall_start:.1f}s')
                    t_next_out = t_start + (output_count + 1) * output_every
            self.termination_reason = 'completed'
        except Exception as exc:
            self.termination_reason = self.termination_reason or 'error'
            self.last_error = str(exc)
            raise
        finally:
            self.U = self._U_gpu.detach().cpu().numpy().copy()
        if verbose:
            print(f'Completed {self.t - t_start:.3f} simulated seconds in {time.perf_counter() - wall_start:.2f}s')
        return snapshots

    def is_valid(self):
        return bool(torch.all(torch.isfinite(self._U_gpu))) and bool(torch.all(self._U_gpu[I_RHO] > 0)) and bool(torch.all(self._U_gpu[I_P] > 0))

class MHDSolverGPU3D(MHDSolverGPU):

    def __init__(self, grid: SphericalGrid3D, U: np.ndarray, gamma: float=1.05, cfl_factor: float=0.4, inner_boundary: str='fixed', outer_boundary: str='outflow', device: Optional[str]=None, dtype: torch.dtype=torch.float32, use_glm: bool=True, use_ct: bool=False, face_B: Optional[tuple]=None, use_equilibrium_residual: bool=True, inner_guard_cells: int=3, outer_guard_cells: int=3, pole_guard_cells: int=9, use_boundary_sponge: bool=True, sponge_width: int=8, sponge_alpha0: float=0.5, spatial_order: int=2, shock_flattening: bool=False, shock_threshold: float=2.0):
        expected_shape = (grid.nr, grid.ntheta, grid.nphi)
        if U.ndim != 4 or U.shape[0] not in (9, 10):
            raise ValueError('3D state must have shape (9 or 10, nr, ntheta, nphi)')
        if U.shape[1:] != expected_shape:
            raise ValueError(f'state/grid shape mismatch: {U.shape[1:]} != {expected_shape}')
        if not np.isfinite(U).all():
            bad = np.argwhere(~np.isfinite(U))[0]
            raise ValueError(f'initial state contains a non-finite value at var={bad[0]}, r={bad[1]}, theta={bad[2]}, phi={bad[3]}')
        for var, name in ((I_RHO, 'density'), (I_P, 'pressure')):
            if np.any(U[var] <= 0.0):
                bad = np.argwhere(U[var] <= 0.0)[0]
                raise ValueError(f'initial {name} must be positive at r={bad[0]}, theta={bad[1]}, phi={bad[2]}')
        if gamma <= 1.0:
            raise ValueError('gamma must be greater than 1')
        if not 0.0 < cfl_factor <= 1.0:
            raise ValueError('cfl_factor must be in (0, 1]')
        if spatial_order not in (1, 2):
            raise ValueError('spatial_order must be 1 or 2')
        shock_threshold = finite_float(shock_threshold, 'shock_threshold')
        if shock_threshold <= 1.0:
            raise ValueError('shock_threshold must be greater than 1')
        if use_ct and (not use_glm):
            raise ValueError('the current CT path requires use_glm=True')
        if face_B is not None:
            if len(face_B) != 3:
                raise ValueError('face_B must be (Br_face, Btheta_face, Bphi_face)')
            expected_face_shapes = ((grid.nr + 1, grid.ntheta, grid.nphi), (grid.nr, grid.ntheta + 1, grid.nphi), (grid.nr, grid.ntheta, grid.nphi))
            for name, field, shape in zip(('Br_face', 'Btheta_face', 'Bphi_face'), face_B, expected_face_shapes):
                if field.shape != shape:
                    raise ValueError(f'{name} shape {field.shape} != {shape}')
                if not np.isfinite(field).all():
                    raise ValueError(f'{name} contains non-finite values')
        gamma = finite_float(gamma, 'gamma')
        cfl_factor = finite_float(cfl_factor, 'cfl_factor')
        if gamma <= 1.0:
            raise ValueError('gamma must be greater than 1')
        if not 0.0 < cfl_factor <= 1.0:
            raise ValueError('cfl_factor must be in (0, 1]')
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('dtype must be torch.float32 or torch.float64')
        if inner_boundary not in ('fixed', 'outflow') or outer_boundary != 'outflow':
            raise ValueError('supported boundaries: inner fixed/outflow, outer outflow')
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.dtype = dtype
        self.use_glm = use_glm
        self.use_ct = use_ct
        self.gamma = gamma
        self.cfl_factor = cfl_factor
        self.spatial_order = int(spatial_order)
        self.shock_flattening = bool(shock_flattening)
        self.shock_threshold = float(shock_threshold)
        self.inner_boundary = inner_boundary
        self.outer_boundary = outer_boundary
        self.grid = grid
        if U.shape[0] == 9:
            U_de = np.zeros((10,) + U.shape[1:], dtype=U.dtype)
            U_de[:9] = U
            U_de[I_EINT] = np.maximum(U[I_P], 1e-30) / (gamma - 1.0 + 1e-30)
            U = U_de
        self.U = U.copy()
        self.t = 0.0
        self.step = 0
        self.dt = 0.0
        self.use_well_balanced = bool(use_equilibrium_residual and (not use_ct))
        self.use_boundary_sponge = bool(use_boundary_sponge)
        self.sponge_width = integer_value(sponge_width, 'sponge_width', minimum=0)
        self.sponge_alpha0 = float(sponge_alpha0)
        if not 0.0 <= self.sponge_alpha0 <= 1.0:
            raise ValueError('sponge_alpha0 must be between 0 and 1')
        self.inner_guard_cells = integer_value(inner_guard_cells, 'inner_guard_cells')
        self.outer_guard_cells = integer_value(outer_guard_cells, 'outer_guard_cells')
        self.pole_guard_cells = integer_value(pole_guard_cells, 'pole_guard_cells')
        nr, ntheta, nphi = (grid.nr, grid.ntheta, grid.nphi)
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1000000000.0
            print(f'GPU 3D solver initialized: {gpu_name} ({vram_gb:.1f} GB VRAM)')
            bytes_per_value = torch.empty((), dtype=self.dtype).element_size()
            vram_state = U.size * bytes_per_value / 1000000000.0
            print(f"  State array: {vram_state:.2f} GB  ({nr}r × {ntheta}theta × {nphi}phi × {U.shape[0]}vars × {('float32' if dtype == torch.float32 else 'float64')})")
        else:
            print('GPU not available — running on CPU with PyTorch')
        self._U_gpu = torch.tensor(U, dtype=self.dtype, device=self.device)
        self._U_eq_gpu = torch.tensor(U, dtype=self.dtype, device=self.device)
        self._dr_gpu = torch.tensor(grid.dr[:, np.newaxis, np.newaxis], dtype=self.dtype, device=self.device)
        self._r_gpu = torch.tensor(grid.r_centers[:, np.newaxis, np.newaxis], dtype=self.dtype, device=self.device)
        self._dtheta = float(grid.dtheta)
        self._dphi = float(grid.dphi)
        self._sin_th_edges_gpu = torch.tensor(grid.sin_theta_edges, dtype=self.dtype, device=self.device)
        self._sin_th_centers_gpu = torch.tensor(grid.theta_centers, dtype=self.dtype, device=self.device).sin().reshape(1, ntheta, 1)
        self._cot_gpu = torch.tensor(grid.COTT[:, :, 0], dtype=self.dtype, device=self.device).mean(dim=0).reshape(1, ntheta, 1)
        self._grav_gpu = torch.tensor((-G * M_SUN / grid.r_centers ** 2)[:, np.newaxis, np.newaxis], dtype=self.dtype, device=self.device)
        self._i_earth, self._j_earth, self._k_earth = grid.earth_index_3d()
        self._last_c_h = 0.0
        self._c_h_cfl = 0.0
        if face_B is not None:
            Br_np, Bth_np, Bphi_np = face_B
            self._Br_face = torch.tensor(Br_np, dtype=torch.float64, device=self.device)
            self._Bth_face = torch.tensor(Bth_np, dtype=torch.float64, device=self.device)
            self._Bphi_face = torch.tensor(Bphi_np, dtype=torch.float64, device=self.device)
        else:
            re64 = torch.tensor(grid.r_edges, dtype=torch.float64, device=self.device)
            rc64 = torch.tensor(grid.r_centers, dtype=torch.float64, device=self.device)
            Br_c64 = torch.tensor(U[I_BR, :, :, :], dtype=torch.float64, device=self.device)
            Bth_c64 = torch.tensor(U[I_BTHETA, :, :, :], dtype=torch.float64, device=self.device)
            Bphi_c64 = torch.tensor(U[I_BPHI, :, :, :], dtype=torch.float64, device=self.device)
            Br_inner = Br_c64[:1] * (rc64[0] / re64[0]) ** 2
            r2Br = (rc64 ** 2).reshape(-1, 1, 1) * Br_c64
            Br_interior = 0.5 * (r2Br[:-1] + r2Br[1:]) / (re64[1:-1] ** 2).reshape(-1, 1, 1)
            Br_outer = Br_c64[-1:] * (rc64[-1] / re64[-1]) ** 2
            Br_face_t = torch.cat([Br_inner, Br_interior, Br_outer], dim=0)
            Bth_face_t = torch.cat([torch.zeros_like(Bth_c64[:, :1, :]), 0.5 * (Bth_c64[:, :-1, :] + Bth_c64[:, 1:, :]), torch.zeros_like(Bth_c64[:, -1:, :])], dim=1)
            self._Br_face = Br_face_t
            self._Bth_face = Bth_face_t
            self._Bphi_face = Bphi_c64.clone()
        self._Br_face_eq = self._Br_face.clone()
        self._Bth_face_eq = self._Bth_face.clone()
        self._Bphi_face_eq = self._Bphi_face.clone()
        self._r_edges_gpu = torch.tensor(grid.r_edges[:, np.newaxis, np.newaxis], dtype=torch.float64, device=self.device)
        self._theta_edges_gpu = torch.tensor(grid.theta_edges, dtype=torch.float64, device=self.device)
        from swcmep.solar_wind.hlle_muscl import apply_pole_bcs_3d
        self._U_gpu = apply_pole_bcs_3d(self._U_gpu)
        self._U_eq_gpu = apply_pole_bcs_3d(self._U_eq_gpu)
        self.U = self._U_gpu.detach().cpu().numpy().copy()

    def set_equilibrium_state(self, U_eq: np.ndarray):
        from swcmep.solar_wind.hlle_muscl import apply_pole_bcs_3d
        if U_eq.shape[0] == 9:
            U_de = np.zeros((10,) + U_eq.shape[1:], dtype=U_eq.dtype)
            U_de[:9] = U_eq
            U_de[I_EINT] = np.maximum(U_eq[I_P], 1e-30) / (self.gamma - 1.0 + 1e-30)
            U_eq = U_de
        self._U_eq_gpu = torch.tensor(U_eq, dtype=self.dtype, device=self.device)
        self._U_eq_gpu = apply_pole_bcs_3d(self._U_eq_gpu)
        if hasattr(self, '_Br_face'):
            self._Br_face_eq = self._Br_face.clone()
            self._Bth_face_eq = self._Bth_face.clone()
            self._Bphi_face_eq = self._Bphi_face.clone()

    def _resolved_guard_counts(self) -> tuple[int, int, int]:
        nr, ntheta = (self._U_gpu.shape[1], self._U_gpu.shape[2])
        inner = min(self.inner_guard_cells, max(nr - 4, 1))
        earth_room = max(nr - self._i_earth - 1, 1)
        outer = min(self.outer_guard_cells, max(nr - inner - 2, 1), earth_room)
        pole = min(self.pole_guard_cells, max((ntheta - 2) // 2, 1))
        return (inner, outer, pole)

    def compute_dt(self) -> float:
        U = self._U_gpu
        nr, ntheta = (U.shape[1], U.shape[2])
        inner_guard, outer_guard, pole_guard = self._resolved_guard_counts()
        r0_active = inner_guard
        r1_active = max(r0_active + 1, nr - outer_guard)
        th0_active = pole_guard
        th1_active = max(th0_active + 1, ntheta - th0_active)
        r0 = max(r0_active - 1, 0)
        r1 = min(r1_active + 1, nr)
        th0 = max(th0_active - 1, 0)
        th1 = min(th1_active + 1, ntheta)
        U_cfl = U[:, r0:r1, th0:th1, :]
        rho = torch.clamp(U_cfl[I_RHO], min=1e-25)
        p = torch.clamp(U_cfl[I_P], min=1e-20)
        cs2 = self.gamma * p / rho
        B2 = U_cfl[I_BR] ** 2 + U_cfl[I_BPHI] ** 2 + U_cfl[I_BTHETA] ** 2
        va2 = B2 / (MU_0 * rho)
        cf = torch.sqrt(torch.clamp(cs2 + va2, min=0.0))
        v_r = torch.abs(U_cfl[I_VR])
        v_th = torch.abs(U_cfl[I_VTHETA])
        v_phi = torch.abs(U_cfl[I_VPHI])
        if self.use_glm:
            c_h = torch.max(torch.maximum(torch.maximum(v_r, v_th), v_phi) + cf)
            self._c_h_cfl = float(c_h)
            lambda_r = torch.maximum(v_r + cf, c_h)
            lambda_th = torch.maximum(v_th + cf, c_h)
            lambda_phi = torch.maximum(v_phi + cf, c_h)
        else:
            self._c_h_cfl = 0.0
            lambda_r = v_r + cf
            lambda_th = v_th + cf
            lambda_phi = v_phi + cf
        r_cfl = self._r_gpu[r0:r1]
        dr_cfl = self._dr_gpu[r0:r1]
        sint_cfl = self._sin_th_centers_gpu[:, th0:th1, :]
        rate = lambda_r / dr_cfl + lambda_th / (r_cfl * self._dtheta) + lambda_phi / (r_cfl * sint_cfl * self._dphi)
        if not bool(torch.isfinite(rate).all()):
            idx = (~torch.isfinite(rate)).nonzero()[0]
            raise RuntimeError(f'non-finite CFL rate at r={int(idx[0]) + r0}, th={int(idx[1]) + th0}, ph={int(idx[2])}')
        max_rate = rate.max()
        if float(max_rate) <= 0.0:
            raise RuntimeError('non-positive maximum CFL rate')
        return self.cfl_factor / float(max_rate)

    def cfl_limiter_diagnostics(self) -> dict:
        inner_guard, outer_guard, pole_guard = self._resolved_guard_counts()
        r0 = max(inner_guard - 1, 0)
        r1 = min(self.grid.nr - outer_guard + 1, self.grid.nr)
        th0 = max(pole_guard - 1, 0)
        th1 = min(self.grid.ntheta - pole_guard + 1, self.grid.ntheta)
        U = self._U_gpu[:, r0:r1, th0:th1, :]
        rho = torch.clamp(U[I_RHO], min=1e-25)
        p = torch.clamp(U[I_P], min=1e-20)
        br, bphi, bth = (U[I_BR], U[I_BPHI], U[I_BTHETA])
        bmag = torch.sqrt(br ** 2 + bphi ** 2 + bth ** 2)
        cf = torch.sqrt(torch.clamp(self.gamma * p / rho + bmag ** 2 / (MU_0 * rho), min=0.0))
        vr, vphi, vth = (U[I_VR], U[I_VPHI], U[I_VTHETA])
        flow_speed = torch.maximum(torch.maximum(torch.abs(vr), torch.abs(vphi)), torch.abs(vth))
        signal = flow_speed + cf
        idx = (signal == signal.max()).nonzero()[0]
        ri = int(idx[0]) + r0
        ti = int(idx[1]) + th0
        pi = int(idx[2])
        li, lj, lk = (int(idx[0]), int(idx[1]), int(idx[2]))
        return {'r': ri, 'theta': ti, 'phi': pi, 'r_au': float(self.grid.r_centers[ri] / AU), 'theta_deg': float(np.degrees(self.grid.theta_centers[ti])), 'phi_deg': float(np.degrees(self.grid.phi_centers[pi])), 'rho': float(rho[li, lj, lk]), 'p': float(p[li, lj, lk]), 'vr_kms': float(vr[li, lj, lk] / 1000.0), 'vtheta_kms': float(vth[li, lj, lk] / 1000.0), 'vphi_kms': float(vphi[li, lj, lk] / 1000.0), 'b_nt': float(bmag[li, lj, lk] * 1000000000.0), 'cf_kms': float(cf[li, lj, lk] / 1000.0), 'signal_kms': float(signal[li, lj, lk] / 1000.0)}

    def _lax_friedrichs_step_gpu(self, U: torch.Tensor, dt: float) -> torch.Tensor:
        inner_guard, outer_guard, pole_guard = self._resolved_guard_counts()
        result = hlle_muscl_step_3d(U=U, dr_gpu=self._dr_gpu, r_gpu=self._r_gpu, dtheta=self._dtheta, dphi=self._dphi, sin_th_edges_gpu=self._sin_th_edges_gpu, cot_gpu=self._cot_gpu, grav_gpu=self._grav_gpu, gamma=self.gamma, dt=dt, c_h=self._c_h_cfl if self.use_glm else 0.0, c_p_ratio=0.5, U_eq=self._U_eq_gpu if self.use_well_balanced else None, use_glm=self.use_glm, Br_face=self._Br_face if self.use_ct else None, Bth_face=self._Bth_face if self.use_ct else None, Bphi_face=self._Bphi_face if self.use_ct else None, r_edges_gpu=self._r_edges_gpu, theta_edges_gpu=self._theta_edges_gpu, inner_guard_cells=inner_guard, outer_guard_cells=outer_guard, pole_guard_cells=pole_guard, spatial_order=self.spatial_order, shock_flattening=self.shock_flattening, shock_threshold=self.shock_threshold)
        if len(result) == 5:
            U_out, c_h, Br_new, Bth_new, Bphi_new = result
            if self.use_ct:
                self._Br_face = Br_new
                self._Bth_face = Bth_new
                self._Bphi_face = Bphi_new
        else:
            U_out, c_h = result
        self._last_c_h = c_h
        U_out[I_RHO] = torch.clamp(U_out[I_RHO], min=1e-25)
        U_out[I_P] = torch.clamp(U_out[I_P], min=1e-20)
        return U_out

    def _apply_boundaries_gpu(self, U: torch.Tensor, injector=None, t_seconds=None):
        boundary_time = self.t if t_seconds is None else t_seconds
        from swcmep.solar_wind.hlle_muscl import apply_pole_bcs_3d
        inner_guard, outer_guard, pole_zone = self._resolved_guard_counts()
        nr = U.shape[1]
        sponge_start = inner_guard
        sponge_stop = min(sponge_start + self.sponge_width, nr - outer_guard)
        if injector is not None and injector.is_injecting(boundary_time):
            pass
        elif self.inner_boundary == 'fixed':
            U[:, :inner_guard, :, :] = self._U_eq_gpu[:, :inner_guard, :, :]
        else:
            U[:, 0, :, :] = U[:, 1, :, :]
        outer_source = U[:, -outer_guard - 1, :, :].clone()
        if self.use_well_balanced:
            outer_source -= self._U_eq_gpu[:, -outer_guard - 1, :, :]
        for offset in range(outer_guard):
            idx = U.shape[1] - outer_guard + offset
            U[:, idx, :, :] = self._U_eq_gpu[:, idx, :, :] + outer_source if self.use_well_balanced else outer_source
        U[I_BR, -outer_guard:, :, :] = self._U_eq_gpu[I_BR, -outer_guard:, :, :]
        U[I_BPHI, -outer_guard:, :, :] = self._U_eq_gpu[I_BPHI, -outer_guard:, :, :]
        U[I_BTHETA, -outer_guard:, :, :] = self._U_eq_gpu[I_BTHETA, -outer_guard:, :, :]
        cme_cooldown_s = 12.0 * 3600.0
        if injector is not None and hasattr(injector, 'injection_end_seconds'):
            injection_end_s = float(injector.injection_end_seconds)
        elif injector is not None and hasattr(injector, 'cme'):
            injection_end_s = injector.cme.duration_hours * 3600.0
        else:
            injection_end_s = 0.0
        apply_radial_sponge = self.use_boundary_sponge and (injector is None or boundary_time > injection_end_s + cme_cooldown_s)
        if apply_radial_sponge and sponge_stop > sponge_start:
            for ii in range(sponge_start, sponge_stop):
                alpha = self.sponge_alpha0 * (1.0 - (ii - sponge_start) / max(self.sponge_width, 1))
                U[:, ii, :, :] = (1.0 - alpha) * U[:, ii, :, :] + alpha * self._U_eq_gpu[:, ii, :, :]
            corner_stop = min(sponge_start + 3, sponge_stop)
            for ii in range(sponge_start, corner_stop):
                for jj in range(min(2, pole_zone), min(4, pole_zone)):
                    frac_r = 1.0 - (ii - sponge_start) / 3.0
                    frac_t = 1.0 - max(jj - 2, 0) / 2.0
                    alpha_c = 0.5 * frac_r * frac_t
                    if alpha_c > 0.01:
                        U[:, ii, jj, :] = (1.0 - alpha_c) * U[:, ii, jj, :] + alpha_c * self._U_eq_gpu[:, ii, jj, :]
                        U[:, ii, -jj - 1, :] = (1.0 - alpha_c) * U[:, ii, -jj - 1, :] + alpha_c * self._U_eq_gpu[:, ii, -jj - 1, :]
        N_POLE_SPONGE = pole_zone
        pole_alpha = 0.8
        for jj in range(N_POLE_SPONGE):
            alpha_j = pole_alpha * (1.0 - jj / N_POLE_SPONGE)
            if alpha_j > 0.01:
                U[:, :, jj, :] = (1 - alpha_j) * U[:, :, jj, :] + alpha_j * self._U_eq_gpu[:, :, jj, :]
                U[:, :, -(jj + 1), :] = (1 - alpha_j) * U[:, :, -(jj + 1), :] + alpha_j * self._U_eq_gpu[:, :, -(jj + 1), :]
        if apply_radial_sponge and sponge_stop > sponge_start:
            sp_r = slice(sponge_start, sponge_stop)
            rho_eq_sp = self._U_eq_gpu[I_RHO, sp_r, :, :]
            p_eq_sp = self._U_eq_gpu[I_P, sp_r, :, :]
            _rho_clamp_count = int((U[I_RHO, sp_r, :, :] < 0.1 * rho_eq_sp).sum().item())
            _p_clamp_count = int((U[I_P, sp_r, :, :] < 0.1 * p_eq_sp).sum().item())
            U[I_RHO, sp_r, :, :] = torch.maximum(U[I_RHO, sp_r, :, :], 0.1 * rho_eq_sp)
            U[I_P, sp_r, :, :] = torch.maximum(U[I_P, sp_r, :, :], 0.1 * p_eq_sp)
            if _rho_clamp_count > 0 or _p_clamp_count > 0:
                print(f'  [sponge-floor] step={self.step} rho_clamped={_rho_clamp_count}  p_clamped={_p_clamp_count}')
        N_INNER_GUARD = inner_guard
        buf_r = slice(0, N_INNER_GUARD)
        rho_eq_buf = self._U_eq_gpu[I_RHO, buf_r, :, :]
        p_eq_buf = self._U_eq_gpu[I_P, buf_r, :, :]
        U[I_RHO, buf_r, :, :] = torch.maximum(U[I_RHO, buf_r, :, :], 0.05 * rho_eq_buf)
        U[I_P, buf_r, :, :] = torch.maximum(U[I_P, buf_r, :, :], 0.05 * p_eq_buf)
        U[I_PSI, :N_INNER_GUARD, :, :] = 0.0
        U[I_RHO] = torch.clamp(U[I_RHO], min=1e-25)
        U[I_P] = torch.clamp(U[I_P], min=1e-20)
        if U.shape[0] > I_EINT:
            U[I_EINT] = torch.maximum(U[I_EINT], U[I_P] / (self.gamma - 1.0))
        psi_inner_stop = sponge_stop if apply_radial_sponge else inner_guard
        U[I_PSI, :psi_inner_stop, :, :] = 0.0
        U[I_PSI, -outer_guard:, :, :] = 0.0
        U[I_PSI, :, :pole_zone, :] = 0.0
        U[I_PSI, :, -pole_zone:, :] = 0.0
        U[I_VTHETA, :, :pole_zone, :] = 0.0
        U[I_VTHETA, :, -pole_zone:, :] = 0.0
        U = apply_pole_bcs_3d(U)
        if self.use_ct and hasattr(self, '_Br_face_eq'):
            N_INNER = inner_guard
            N_OUTER = outer_guard + 1
            re64 = self._r_edges_gpu.reshape(-1).to(torch.float64)
            rc64 = 0.5 * (re64[:-1] + re64[1:])
            Br_c64 = U[I_BR].to(torch.float64)
            r2Br = (rc64 ** 2).reshape(-1, 1, 1) * Br_c64
            Br_int = 0.5 * (r2Br[:-1] + r2Br[1:]) / (re64[1:-1] ** 2).reshape(-1, 1, 1)
            Br_face_new = torch.cat([Br_c64[:1] * (rc64[0] / re64[0]) ** 2, Br_int, Br_c64[-1:] * (rc64[-1] / re64[-1]) ** 2], dim=0)
            Bphi_face_new = U[I_BPHI].to(torch.float64)
            self._Br_face[:N_INNER, :, :] = Br_face_new[:N_INNER, :, :]
            self._Br_face[-N_OUTER:, :, :] = self._Br_face_eq[-N_OUTER:, :, :]
            self._Bphi_face[:N_INNER, :, :] = Bphi_face_new[:N_INNER, :, :]
            self._Bphi_face[-N_OUTER:, :, :] = self._Bphi_face_eq[-N_OUTER:, :, :]
            cos_te64 = torch.cos(self._theta_edges_gpu.to(torch.float64))
            sin_te64 = torch.sin(self._theta_edges_gpu.to(torch.float64))
            dphi_f = float(self._dphi)
            dtheta_f = float(self._dtheta)
            cos_dff = cos_te64[:-1] - cos_te64[1:]
            rsqdiff = 0.5 * (re64[1:] ** 2 - re64[:-1] ** 2)
            Phi_r = (re64 ** 2).reshape(-1, 1, 1) * cos_dff.reshape(1, -1, 1) * dphi_f * Br_face_new
            Phi_phi = rsqdiff.reshape(-1, 1, 1) * dtheta_f * Bphi_face_new
            dPhi_r = Phi_r[1:] - Phi_r[:-1]
            dPhi_phi = torch.roll(Phi_phi, -1, dims=2) - Phi_phi
            residual = dPhi_r + dPhi_phi
            col_sum = residual.sum(dim=1, keepdim=True)
            res_star = residual - col_sum / residual.shape[1]
            A_th = rsqdiff.reshape(-1, 1, 1) * sin_te64.clamp(min=1e-10).reshape(1, -1, 1) * dphi_f
            Phi_th = torch.zeros(N_INNER, residual.shape[1] + 1, residual.shape[2], dtype=torch.float64, device=U.device)
            for j in range(residual.shape[1]):
                Phi_th[:, j + 1, :] = Phi_th[:, j, :] - res_star[:N_INNER, j, :]
            south_err = Phi_th[:, -1:, :].clone()
            frac = torch.linspace(0, 1, Phi_th.shape[1], device=U.device, dtype=torch.float64).reshape(1, -1, 1)
            Phi_th -= frac * south_err
            Phi_th[:, 0, :] = 0.0
            Phi_th[:, -1, :] = 0.0
            Bth_new_ghost = Phi_th / A_th[:N_INNER].clamp(min=1e-60)
            Bth_new_ghost[:, 0, :] = 0.0
            Bth_new_ghost[:, -1, :] = 0.0
            self._Bth_face[:N_INNER, :, :] = Bth_new_ghost
            self._Bth_face[-N_OUTER:, :, :] = self._Bth_face_eq[-N_OUTER:, :, :]
        return U

    def _divergence_fields(self):
        from swcmep.solar_wind.magnetic_diagnostics import centered_divergence_fields
        return centered_divergence_fields(self._U_gpu[I_BR], self._U_gpu[I_BTHETA], self._U_gpu[I_BPHI], self.grid.r_centers, self.grid.theta_centers, self.grid.dr, self.grid.dtheta, self.grid.dphi)

    def _active_divergence_region(self):
        inner, outer, pole = self._resolved_guard_counts()
        return (slice(max(inner, 1) - 1, min(self.grid.nr - outer, self.grid.nr - 1) - 1), slice(max(pole, 1) - 1, min(self.grid.ntheta - pole, self.grid.ntheta - 1) - 1), slice(None))

    def compute_divB(self):
        normalized = self._divergence_fields()['normalized'][self._active_divergence_region()]
        return (float(normalized.max()), float(normalized.mean()))

    def divergence_report(self):
        from swcmep.solar_wind.magnetic_diagnostics import summarize_divergence
        fields = self._divergence_fields()
        return {'time_seconds': self.t, 'step': self.step, 'operator': 'cell_centered', 'normalization': 'abs(divB) * min(dr, r*dtheta, r*sin(theta)*dphi) / abs(B)', 'active': summarize_divergence(fields, self.grid, region=self._active_divergence_region(), offset=(1, 1, 0)), 'interior_including_guards': summarize_divergence(fields, self.grid, offset=(1, 1, 0))}

    def diagnostics(self) -> dict:
        U = self._U_gpu
        vol = torch.tensor(self.grid.cell_volume(), dtype=self.dtype, device=self.device)
        rho = U[I_RHO]
        v_r = U[I_VR]
        v_th = U[I_VTHETA]
        v_phi = U[I_VPHI]
        p = U[I_P]
        B_r = U[I_BR]
        B_th = U[I_BTHETA]
        B_phi = U[I_BPHI]
        g1 = self.gamma - 1.0 + 1e-20
        E_kin = 0.5 * rho * (v_r ** 2 + v_th ** 2 + v_phi ** 2)
        E_mag = (B_r ** 2 + B_th ** 2 + B_phi ** 2) / (2.0 * MU_0)
        E_tot = p / g1 + E_kin + E_mag
        divB_max, divB_mean = self.compute_divB()
        return {'total_mass': float((rho * vol).sum()), 'total_energy': float((E_tot * vol).sum()), 'min_pressure': float(p.min()), 'max_velocity': float(v_r.max()) / 1000.0, 'min_density': float(rho.min()), 'max_B_nT': float(torch.sqrt(B_r ** 2 + B_th ** 2 + B_phi ** 2).max()) * 1000000000.0, 'max_psi': float(torch.abs(U[I_PSI]).max()), 'max_v_theta': float(torch.abs(v_th).max()) / 1000.0, 'c_h_kms': self._last_c_h / 1000.0, 'divB_max': divB_max, 'divB_mean': divB_mean, 'is_valid': self.is_valid()}

    def _v_at_earth(self) -> float:
        i, j, k = (self._i_earth, self._j_earth, self._k_earth)
        return float(self._U_gpu[I_VR, i, j, k]) / 1000.0

    def _n_at_earth(self) -> float:
        i, j, k = (self._i_earth, self._j_earth, self._k_earth)
        return float(self._U_gpu[I_RHO, i, j, k]) / M_PROTON / 1000000.0

    def is_valid(self) -> bool:
        return bool(torch.all(torch.isfinite(self._U_gpu))) and bool(torch.all(self._U_gpu[I_RHO] > 0)) and bool(torch.all(self._U_gpu[I_P] > 0))
