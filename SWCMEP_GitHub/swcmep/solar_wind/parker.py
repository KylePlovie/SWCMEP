from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.integrate import solve_ivp
from swcmep.utils.constants import AU, K_B, M_PROTON, M_SUN, N_SW_1AU, R_INNER, R_OUTER, V_SW_FAST, V_SW_SLOW, G, m3_to_cm3, m_to_Rsun, ms_to_kms

@dataclass
class ParkerSolution:
    T_corona: float = 1500000.0
    gamma: float = 1.0
    n_inner: float = 20000000000.0
    r_inner: float = R_INNER
    r_outer: float = R_OUTER
    n_points: int = 5000
    c_s: float = field(init=False)
    r_c: float = field(init=False)
    r: np.ndarray = field(init=False, repr=False)
    v: np.ndarray = field(init=False, repr=False)
    n: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.c_s = self._sound_speed()
        self.r_c = self._critical_radius()
        self._integrate()
        self._compute_density()

    def _sound_speed(self) -> float:
        return np.sqrt(K_B * self.T_corona / M_PROTON)

    def _critical_radius(self) -> float:
        return G * M_SUN / (2.0 * self.c_s ** 2)

    def _parker_rhs(self, r: float, v_arr) -> list:
        v = v_arr[0]
        numer = 2.0 * self.c_s ** 2 / r - G * M_SUN / r ** 2
        denom = v - self.c_s ** 2 / v
        return [numer / denom]

    def _critical_slope(self) -> float:
        return self.c_s / self.r_c

    def _integrate(self):
        alpha = self._critical_slope()
        eps = self.r_c * 0.0001
        r_start = self.r_c + eps
        v_start = self.c_s + alpha * eps
        r_out_eval = np.linspace(r_start, self.r_outer, self.n_points)
        sol_out = solve_ivp(self._parker_rhs, [r_start, self.r_outer], [v_start], t_eval=r_out_eval, method='RK45', rtol=1e-09, atol=1e-09)
        self.r = sol_out.t
        self.v = sol_out.y[0]

    def _compute_density(self):
        r_ref = self.r[0]
        v_ref = self.v[0]
        self.n = self.n_inner * (r_ref ** 2 * v_ref) / (self.r ** 2 * self.v)

    def v_at(self, r_m: float) -> float:
        return float(np.interp(r_m, self.r, self.v))

    def n_at(self, r_m: float) -> float:
        return float(np.interp(r_m, self.r, self.n))

    def summary(self) -> str:
        v_1au = self.v_at(AU)
        n_1au = self.n_at(AU)
        return f'ParkerSolution(T={self.T_corona / 1000000.0:.1f} MK)\n  Sound speed        : {ms_to_kms(self.c_s):.1f} km/s\n  Critical radius    : {m_to_Rsun(self.r_c):.2f} R☉\n  Speed at 1 AU      : {ms_to_kms(v_1au):.1f} km/s  (observed: {ms_to_kms(V_SW_SLOW):.0f}–{ms_to_kms(V_SW_FAST):.0f} km/s)\n  Density at 1 AU    : {m3_to_cm3(n_1au):.2f} cm⁻³  (observed: ~{m3_to_cm3(N_SW_1AU):.0f} cm⁻³)\n'

    def plot(self, save_path: Optional[str]=None):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'SWCMEP — Parker Solar Wind  |  T = {self.T_corona / 1000000.0:.1f} MK', fontsize=13)
        r_rsun = m_to_Rsun(self.r)
        v_kms = ms_to_kms(self.v)
        n_cm3 = m3_to_cm3(self.n)
        ax = axes[0]
        ax.plot(r_rsun, v_kms, color='#E85D24', lw=2)
        ax.axvline(m_to_Rsun(self.r_c), color='gray', ls='--', lw=1, label=f'r_c = {m_to_Rsun(self.r_c):.1f} R☉')
        ax.axhline(ms_to_kms(self.c_s), color='gray', ls=':', lw=1, label=f'c_s = {ms_to_kms(self.c_s):.0f} km/s')
        ax.axvline(m_to_Rsun(AU), color='steelblue', ls='--', lw=1, alpha=0.7, label='1 AU')
        ax.set_xlabel('Distance [R☉]')
        ax.set_ylabel('Speed [km/s]')
        ax.set_title('Solar wind velocity')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax = axes[1]
        ax.plot(r_rsun, v_kms, color='#185FA5', lw=2)
        ax.axvline(m_to_Rsun(self.r_c), color='gray', ls='--', lw=1)
        ax.set_xlabel('Distance [R☉]')
        ax.set_ylabel('Speed [km/s]')
        ax.set_title('Velocity (log r)')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax = axes[2]
        ax.plot(r_rsun, n_cm3, color='#1D9E75', lw=2)
        ax.axvline(m_to_Rsun(AU), color='steelblue', ls='--', lw=1, alpha=0.7, label='1 AU')
        ax.set_xlabel('Distance [R☉]')
        ax.set_ylabel('Number density [cm⁻³]')
        ax.set_title('Number density')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'Plot saved: {save_path}')
        plt.show()

def _cli():
    parser = argparse.ArgumentParser(description='SWCMEP Phase 1 — Parker solar wind model')
    parser.add_argument('--temperature', type=float, default=1500000.0, help='Coronal temperature [K] (default: 1.5e6)')
    parser.add_argument('--plot', action='store_true', help='Show and save velocity/density plots')
    parser.add_argument('--save', type=str, default=None, help='Path to save plot PNG')
    args = parser.parse_args()
    sol = ParkerSolution(T_corona=args.temperature)
    print(sol.summary())
    if args.plot:
        sol.plot(save_path=args.save or 'swcmep_parker.png')
if __name__ == '__main__':
    _cli()
