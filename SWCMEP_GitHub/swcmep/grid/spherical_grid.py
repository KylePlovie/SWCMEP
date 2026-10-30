from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from swcmep.utils.constants import AU, R_INNER, R_OUTER, R_SUN
from swcmep.utils.validation import finite_float, integer_value

@dataclass
class SphericalGrid:
    nr: int = 512
    nphi: int = 360
    r_inner: float = R_INNER
    r_outer: float = R_OUTER
    n_ghost: int = 2
    log_radial: bool = True
    r_edges: np.ndarray = field(init=False, repr=False)
    phi_edges: np.ndarray = field(init=False, repr=False)
    r_centers: np.ndarray = field(init=False, repr=False)
    phi_centers: np.ndarray = field(init=False, repr=False)
    R: np.ndarray = field(init=False, repr=False)
    PHI: np.ndarray = field(init=False, repr=False)
    dr: np.ndarray = field(init=False, repr=False)
    dphi: float = field(init=False, repr=False)

    def __post_init__(self):
        self.nr = integer_value(self.nr, 'nr', minimum=4)
        self.nphi = integer_value(self.nphi, 'nphi', minimum=3)
        self.r_inner = finite_float(self.r_inner, 'r_inner')
        self.r_outer = finite_float(self.r_outer, 'r_outer')
        if not 0 < self.r_inner < self.r_outer:
            raise ValueError('radii must satisfy 0 < r_inner < r_outer')
        self._build_radial_grid()
        self._build_azimuthal_grid()
        self._build_meshgrid()

    def _build_radial_grid(self):
        if self.log_radial:
            self.r_edges = np.geomspace(self.r_inner, self.r_outer, self.nr + 1)
        else:
            self.r_edges = np.linspace(self.r_inner, self.r_outer, self.nr + 1)
        self.r_centers = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])
        self.dr = np.diff(self.r_edges)

    def _build_azimuthal_grid(self):
        self.dphi = 2.0 * np.pi / self.nphi
        self.phi_centers = np.arange(self.nphi, dtype=float) * self.dphi
        self.phi_edges = (np.arange(self.nphi + 1, dtype=float) - 0.5) * self.dphi

    def _build_meshgrid(self):
        self.R, self.PHI = np.meshgrid(self.r_centers, self.phi_centers, indexing='ij')

    def cell_area(self) -> np.ndarray:
        dr_2d = self.dr[:, np.newaxis] * np.ones((1, self.nphi))
        return self.R * dr_2d * self.dphi

    def r_index(self, r_m: float) -> int:
        r_m = finite_float(r_m, 'r_m')
        if not self.r_inner <= r_m <= self.r_outer:
            raise ValueError('radius lies outside the grid')
        return min(int(np.searchsorted(self.r_edges, r_m, side='right') - 1), self.nr - 1)

    def phi_index(self, phi_rad: float) -> int:
        phi_wrapped = phi_rad % (2.0 * np.pi)
        return int(np.floor((phi_wrapped + 0.5 * self.dphi) / self.dphi)) % self.nphi

    def earth_index(self):
        return (self.r_index(AU), self.phi_index(0.0))

    def to_cartesian(self):
        X = self.R * np.cos(self.PHI)
        Y = self.R * np.sin(self.PHI)
        return (X, Y)

    def min_cell_size(self) -> float:
        dr_min = self.dr.min()
        dphi_min = self.r_centers.min() * self.dphi
        return min(dr_min, dphi_min)

    def aspect_ratio_at(self, r_m: float) -> float:
        i = self.r_index(r_m)
        return self.dr[i] / (r_m * self.dphi)

    def summary(self) -> str:
        i_earth, j_earth = self.earth_index()
        dr_inner = self.dr[0]
        dr_outer = self.dr[-1]
        dphi_km = self.r_centers[0] * self.dphi / 1000.0
        return f"SphericalGrid({self.nr} r × {self.nphi} phi)\n  Radial range    : {self.r_inner / R_SUN:.1f} – {self.r_outer / AU:.2f} AU\n  Radial spacing  : {('logarithmic' if self.log_radial else 'linear')}\n  dr at inner bdy : {dr_inner / R_SUN:.4f} R☉\n  dr at outer bdy : {dr_outer / AU:.4f} AU\n  dphi            : {np.degrees(self.dphi):.2f}°\n  Arc length at inner bdy: {dphi_km:.1f} km/cell\n  Total cells     : {self.nr * self.nphi:,}\n  Earth cell      : r[{i_earth}], phi[{j_earth}]\n  Min cell size   : {self.min_cell_size() / R_SUN:.4f} R☉\n"

@dataclass
class SphericalGrid3D(SphericalGrid):
    ntheta: int = 90
    theta_inner: float = 0.0
    theta_outer: float = np.pi
    theta_edges: np.ndarray = field(init=False, repr=False)
    theta_centers: np.ndarray = field(init=False, repr=False)
    dtheta: float = field(init=False, repr=False)
    R3: np.ndarray = field(init=False, repr=False)
    THETA: np.ndarray = field(init=False, repr=False)
    PHI3: np.ndarray = field(init=False, repr=False)
    SINT: np.ndarray = field(init=False, repr=False)
    COST: np.ndarray = field(init=False, repr=False)
    COTT: np.ndarray = field(init=False, repr=False)
    sin_theta_edges: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.ntheta = integer_value(self.ntheta, 'ntheta', minimum=3)
        self.theta_inner = finite_float(self.theta_inner, 'theta_inner')
        self.theta_outer = finite_float(self.theta_outer, 'theta_outer')
        if not 0 <= self.theta_inner < self.theta_outer <= np.pi:
            raise ValueError('theta bounds must satisfy 0 <= theta_inner < theta_outer <= pi')
        super().__post_init__()
        self._build_theta_grid()
        self._build_meshgrid_3d()

    def _build_theta_grid(self):
        self.theta_edges = np.linspace(self.theta_inner, self.theta_outer, self.ntheta + 1)
        self.theta_centers = 0.5 * (self.theta_edges[:-1] + self.theta_edges[1:])
        self.dtheta = self.theta_edges[1] - self.theta_edges[0]
        self.sin_theta_edges = np.sin(self.theta_edges)

    def _build_meshgrid_3d(self):
        self.R3, self.THETA, self.PHI3 = np.meshgrid(self.r_centers, self.theta_centers, self.phi_centers, indexing='ij')
        self.SINT = np.sin(self.THETA)
        self.COST = np.cos(self.THETA)
        cot_clip = np.cos(self.dtheta / 2.0) / np.sin(self.dtheta / 2.0)
        sin_safe = np.where(np.abs(self.SINT) < 1e-10, 1e-10, self.SINT)
        self.COTT = np.clip(self.COST / sin_safe, -cot_clip, cot_clip)

    def cell_volume(self) -> np.ndarray:
        radial = np.diff(self.r_edges ** 3) / 3.0
        polar = np.cos(self.theta_edges[:-1]) - np.cos(self.theta_edges[1:])
        return np.broadcast_to(radial[:, None, None] * polar[None, :, None] * self.dphi, (self.nr, self.ntheta, self.nphi)).copy()

    def theta_index(self, theta_rad: float) -> int:
        idx = int(np.searchsorted(self.theta_edges, theta_rad) - 1)
        return max(0, min(idx, self.ntheta - 1))

    def earth_index_3d(self):
        return (self.r_index(AU), self.theta_index(np.pi / 2.0), self.phi_index(0.0))

    def equatorial_slice(self) -> int:
        return self.theta_index(np.pi / 2.0)

    def to_cartesian_3d(self):
        X = self.R3 * self.SINT * np.cos(self.PHI3)
        Y = self.R3 * self.SINT * np.sin(self.PHI3)
        Z = self.R3 * self.COST
        return (X, Y, Z)

    def min_cell_size(self) -> float:
        dr_min = self.dr.min()
        dphi_min = self.r_centers.min() * self.dphi
        dtheta_min = self.r_centers.min() * self.dtheta
        sint_min = np.sin(self.theta_centers[1])
        dphi_pole = self.r_centers.min() * sint_min * self.dphi
        return min(dr_min, dphi_min, dtheta_min, dphi_pole)

    def summary(self) -> str:
        i_e, j_e, k_e = self.earth_index_3d()
        j_eq = self.equatorial_slice()
        total_cells = self.nr * self.ntheta * self.nphi
        vram_float64_gb = total_cells * 10 * 8 / 1000000000.0
        vram_float32_gb = total_cells * 10 * 4 / 1000000000.0
        return f"SphericalGrid3D({self.nr} r × {self.ntheta} theta × {self.nphi} phi)\n  Radial range    : {self.r_inner / R_SUN:.1f} R☉ – {self.r_outer / AU:.2f} AU\n  Radial spacing  : {('logarithmic' if self.log_radial else 'linear')}\n  Theta range     : {np.degrees(self.theta_inner):.1f}° – {np.degrees(self.theta_outer):.1f}°\n  dtheta          : {np.degrees(self.dtheta):.2f}°\n  dphi            : {np.degrees(self.dphi):.2f}°\n  Equatorial slice: theta[{j_eq}] = {np.degrees(self.theta_centers[j_eq]):.2f}°\n  Total cells     : {total_cells:,}\n  Earth cell      : r[{i_e}], theta[{j_e}], phi[{k_e}]\n  VRAM (float64)  : {vram_float64_gb:.2f} GB\n  VRAM (float32)  : {vram_float32_gb:.2f} GB\n  Min cell size   : {self.min_cell_size() / R_SUN:.4f} R☉\n"
