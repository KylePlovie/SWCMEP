from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from swcmep.grid.spherical_grid import SphericalGrid
from swcmep.solar_wind.parker import ParkerSolution
from swcmep.utils.constants import AU, K_B, M_PROTON, R_SUN, m3_to_cm3
I_RHO = 0
I_VR = 1
I_VPHI = 2
I_P = 3
I_BR = 4
I_BPHI = 5
I_PSI = 6
N_VARS = 7

@dataclass
class Background2D:
    grid: SphericalGrid
    parker: ParkerSolution
    gamma: float = 1.05
    U: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.U = np.zeros((N_VARS, self.grid.nr, self.grid.nphi))
        self._fill_from_parker()
        self._compute_parker_spiral()

    def _compute_parker_spiral(self):
        from swcmep.utils.constants import B_REF, OMEGA_SUN, R_REF
        r = self.grid.R
        v_r = self.U[I_VR, :, :]
        B_r = B_REF * (R_REF / r) ** 2
        B_phi = -B_REF * (R_REF / r) * (OMEGA_SUN * R_REF / v_r)
        self.U[I_BR, :, :] = B_r
        self.U[I_BPHI, :, :] = B_phi
        self.U[I_PSI, :, :] = 0.0

    def _fill_from_parker(self):
        r = self.grid.r_centers
        v_r_1d = np.interp(r, self.parker.r, self.parker.v)
        n_1d = np.interp(r, self.parker.r, self.parker.n)
        rho_1d = n_1d * M_PROTON
        p_1d = n_1d * K_B * self.parker.T_corona
        self.U[I_RHO] = rho_1d[:, np.newaxis] * np.ones((1, self.grid.nphi))
        self.U[I_VR] = v_r_1d[:, np.newaxis] * np.ones((1, self.grid.nphi))
        self.U[I_VPHI] = np.zeros((self.grid.nr, self.grid.nphi))
        self.U[I_P] = p_1d[:, np.newaxis] * np.ones((1, self.grid.nphi))

    @property
    def rho(self) -> np.ndarray:
        return self.U[I_RHO]

    @property
    def v_r(self) -> np.ndarray:
        return self.U[I_VR]

    @property
    def v_phi(self) -> np.ndarray:
        return self.U[I_VPHI]

    @property
    def pressure(self) -> np.ndarray:
        return self.U[I_P]

    def number_density(self) -> np.ndarray:
        return self.U[I_RHO] / M_PROTON

    def sound_speed(self) -> np.ndarray:
        return np.sqrt(self.gamma * self.U[I_P] / self.U[I_RHO])

    def mach_number(self) -> np.ndarray:
        return self.U[I_VR] / self.sound_speed()

    def at_earth(self):
        i, j = self.grid.earth_index()
        return {'r_AU': self.grid.r_centers[i] / AU, 'v_r_kms': self.U[I_VR, i, j] / 1000.0, 'n_cm3': m3_to_cm3(self.number_density()[i, j]), 'p_Pa': self.U[I_P, i, j], 'mach': self.mach_number()[i, j]}

    def summary(self) -> str:
        earth = self.at_earth()
        sonic_idx = np.argmin(np.abs(self.mach_number()[:, 0] - 1.0))
        r_sonic = self.grid.r_centers[sonic_idx] / R_SUN
        return f"Background2D({self.grid.nr}r × {self.grid.nphi}phi)\n  Sonic point      : {r_sonic:.2f} R☉\n  At Earth (1 AU)  :\n    v_r   = {earth['v_r_kms']:.1f} km/s\n    n     = {earth['n_cm3']:.2f} cm⁻³\n    Mach  = {earth['mach']:.2f}\n  Memory           : {self.U.nbytes / 1000000.0:.1f} MB\n"

    def plot(self, save_path: Optional[str]=None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'SWCMEP — 2D Background State  |  T={self.parker.T_corona / 1000000.0:.1f} MK  |  {self.grid.nr}×{self.grid.nphi} grid', fontsize=13)
        ax = axes[0]
        v_kms = self.v_r / 1000.0
        X, Y = self.grid.to_cartesian()
        X_AU = X / AU
        Y_AU = Y / AU
        im = ax.pcolormesh(X_AU, Y_AU, v_kms, cmap='plasma', shading='auto', vmin=v_kms.min(), vmax=v_kms.max())
        plt.colorbar(im, ax=ax, label='v_r [km/s]')
        ax.plot(1.0, 0.0, 'o', color='#3B8BD4', ms=8, zorder=5, label='Earth')
        ax.plot(0.0, 0.0, 'o', color='#F9CB42', ms=12, zorder=5, label='Sun')
        ax.set_xlabel('X [AU]')
        ax.set_ylabel('Y [AU]')
        ax.set_title('Solar wind speed — ecliptic plane')
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        ax = axes[1]
        r_au = self.grid.r_centers / AU
        v_prof = self.v_r[:, 0] / 1000.0
        n_prof = m3_to_cm3(self.number_density()[:, 0])
        mach = self.mach_number()[:, 0]
        ax2 = ax.twinx()
        l1, = ax.plot(r_au, v_prof, color='#E85D24', lw=2, label='v_r [km/s]')
        l2, = ax.plot(r_au, mach, color='#7F77DD', lw=2, ls='--', label='Mach number')
        l3, = ax2.plot(r_au, n_prof, color='#1D9E75', lw=2, label='n [cm⁻³]')
        ax.axvline(1.0, color='gray', ls='--', lw=1, alpha=0.5)
        ax.axhline(1.0, color='#7F77DD', ls=':', lw=1, alpha=0.5)
        ax.set_xlabel('Distance [AU]')
        ax.set_ylabel('Speed [km/s] / Mach number')
        ax2.set_ylabel('Number density [cm⁻³]')
        ax2.set_yscale('log')
        ax.set_title('Radial profiles')
        lines = [l1, l2, l3]
        ax.legend(lines, [line.get_label() for line in lines], fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'Plot saved: {save_path}')
        plt.show()

def _cli():
    import argparse
    parser = argparse.ArgumentParser(description='SWCMEP Phase 2 — 2D background state')
    parser.add_argument('--temperature', type=float, default=1500000.0)
    parser.add_argument('--nr', type=int, default=512)
    parser.add_argument('--nphi', type=int, default=360)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()
    from swcmep.grid.spherical_grid import SphericalGrid
    print('Building grid...')
    grid = SphericalGrid(nr=args.nr, nphi=args.nphi)
    print(grid.summary())
    print('Computing Parker solution...')
    parker = ParkerSolution(T_corona=args.temperature)
    print('Filling 2D background state...')
    bg = Background2D(grid, parker)
    print(bg.summary())
    if args.plot:
        bg.plot(save_path=args.save or 'swcmep_background_2d.png')
if __name__ == '__main__':
    _cli()
