from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from swcmep.solar_wind.parker import ParkerSolution
from swcmep.utils.constants import AU, B_SW_1AU, OMEGA_SUN, T_to_nT, m_to_AU

@dataclass
class ParkerSpiral:
    parker: ParkerSolution
    B_ref: float = B_SW_1AU
    r_ref: float = AU
    omega: float = OMEGA_SUN
    r: np.ndarray = field(init=False, repr=False)
    B_r: np.ndarray = field(init=False, repr=False)
    B_phi: np.ndarray = field(init=False, repr=False)
    B_mag: np.ndarray = field(init=False, repr=False)
    psi: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.r = self.parker.r
        self._compute_field()

    def _compute_field(self):
        v = self.parker.v
        r = self.r
        self.B_r = self.B_ref * (self.r_ref / r) ** 2
        self.B_phi = -self.B_r * (self.omega * r / v)
        self.B_mag = np.sqrt(self.B_r ** 2 + self.B_phi ** 2)
        self.psi = np.degrees(np.arctan2(np.abs(self.B_phi), np.abs(self.B_r)))

    def B_r_at(self, r_m: float) -> float:
        return float(np.interp(r_m, self.r, self.B_r))

    def B_phi_at(self, r_m: float) -> float:
        return float(np.interp(r_m, self.r, self.B_phi))

    def B_mag_at(self, r_m: float) -> float:
        return float(np.interp(r_m, self.r, self.B_mag))

    def spiral_angle_at(self, r_m: float) -> float:
        return float(np.interp(r_m, self.r, self.psi))

    def summary(self) -> str:
        psi_1au = self.spiral_angle_at(AU)
        Br_1au = T_to_nT(self.B_r_at(AU))
        Bphi_1au = T_to_nT(self.B_phi_at(AU))
        Bmag_1au = T_to_nT(self.B_mag_at(AU))
        return f'ParkerSpiral(T={self.parker.T_corona / 1000000.0:.1f} MK)\n  Spiral angle at 1 AU : {psi_1au:.1f}°  (slow wind ~45°, fast wind ~25°)\n  B_r  at 1 AU         : {Br_1au:.2f} nT\n  B_phi at 1 AU        : {Bphi_1au:.2f} nT\n  |B|  at 1 AU         : {Bmag_1au:.2f} nT  (observed ~5 nT)\n'

    def plot(self, save_path: Optional[str]=None):
        fig = plt.figure(figsize=(16, 5))
        fig.suptitle(f'SWCMEP — Parker Spiral IMF  |  T = {self.parker.T_corona / 1000000.0:.1f} MK', fontsize=13)
        r_au = m_to_AU(self.r)
        Br_nT = T_to_nT(self.B_r)
        Bph_nT = T_to_nT(self.B_phi)
        Bm_nT = T_to_nT(self.B_mag)
        ax = fig.add_subplot(1, 3, 1)
        ax.plot(r_au, np.abs(Br_nT), color='#E85D24', lw=2, label='|B_r|')
        ax.plot(r_au, np.abs(Bph_nT), color='#185FA5', lw=2, label='|B_φ|')
        ax.plot(r_au, Bm_nT, color='#1D9E75', lw=2, label='|B|', ls='--')
        ax.axvline(1.0, color='gray', ls='--', lw=1, alpha=0.6, label='1 AU')
        ax.set_xlabel('Distance [AU]')
        ax.set_ylabel('Field strength [nT]')
        ax.set_title('IMF components')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        ax = fig.add_subplot(1, 3, 2)
        ax.plot(r_au, self.psi, color='#7F77DD', lw=2)
        ax.axvline(1.0, color='gray', ls='--', lw=1, alpha=0.6, label='1 AU')
        ax.axhline(45, color='#EF9F27', ls=':', lw=1, alpha=0.8, label='45° reference')
        ax.set_xlabel('Distance [AU]')
        ax.set_ylabel('Spiral angle ψ [degrees]')
        ax.set_title('Parker spiral angle')
        ax.set_ylim(0, 90)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax = fig.add_subplot(1, 3, 3, projection='polar')
        r_spiral = self.r[self.r >= self.r_ref * 0.1]
        v_spiral = self.parker.v[self.r >= self.r_ref * 0.1]
        phi_spiral = -self.omega * (r_spiral - self.r_ref) / v_spiral
        ax.plot(phi_spiral, m_to_AU(r_spiral), color='#378ADD', lw=1.5, label='Field line')
        ax.plot(phi_spiral + np.pi, m_to_AU(r_spiral), color='#378ADD', lw=1.5, alpha=0.5)
        ax.plot(0, 0, 'o', color='#F9CB42', ms=12, zorder=5)
        ax.plot(0, 1, 'o', color='#3B8BD4', ms=8, zorder=5, label='Earth (1 AU)')
        ax.set_title('Top-down view\n(ecliptic plane)', pad=15)
        ax.set_rlabel_position(45)
        ax.set_rlim(0, 1.15)
        ax.set_rticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1 AU'], fontsize=7)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'Plot saved: {save_path}')
        plt.show()

def _cli():
    parser = argparse.ArgumentParser(description='SWCMEP — Parker spiral IMF')
    parser.add_argument('--temperature', type=float, default=1500000.0)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()
    sol = ParkerSolution(T_corona=args.temperature)
    spiral = ParkerSpiral(sol)
    print(spiral.summary())
    if args.plot:
        spiral.plot(save_path=args.save or 'swcmep_spiral.png')
if __name__ == '__main__':
    _cli()
