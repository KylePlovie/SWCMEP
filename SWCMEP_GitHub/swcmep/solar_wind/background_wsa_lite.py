from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from swcmep.grid.spherical_grid import SphericalGrid, SphericalGrid3D
from swcmep.solar_wind.background_2d import I_BPHI, I_BR, I_P, I_PSI, I_RHO, I_VPHI, I_VR
from swcmep.solar_wind.background_2d import N_VARS as N_VARS_2D
from swcmep.utils.constants import AU, I_BTHETA, I_VTHETA, K_B, M_PROTON, N_VARS, OMEGA_SUN, R_INNER
B_1AU_T = 5e-09

@dataclass
class WSALiteBoundaryMap:
    phi_centers: np.ndarray
    vr_ms: Optional[np.ndarray] = None
    br_sign: Optional[np.ndarray] = None
    slow_speed_ms: float = 350000.0
    fast_speed_ms: float = 650000.0
    base_temperature_K: float = 1500000.0
    slow_density_cm3_1au: float = 7.0
    fast_density_cm3_1au: float = 3.5

    def __post_init__(self):
        self.phi_centers = np.asarray(self.phi_centers, dtype=float)
        if self.vr_ms is None:
            self.vr_ms = self._synthetic_speed_map()
        else:
            self.vr_ms = np.asarray(self.vr_ms, dtype=float)
        if self.br_sign is None:
            self.br_sign = self._synthetic_polarity_map()
        else:
            self.br_sign = np.asarray(self.br_sign, dtype=float)
        if self.vr_ms.shape != self.phi_centers.shape:
            raise ValueError('vr_ms must have same shape as phi_centers')
        if self.br_sign.shape != self.phi_centers.shape:
            raise ValueError('br_sign must have same shape as phi_centers')
        self.br_sign = np.where(self.br_sign >= 0.0, 1.0, -1.0)
        self.vr_ms = np.clip(self.vr_ms, 250000.0, 900000.0)

    def _angular_distance(self, center_rad: float) -> np.ndarray:
        d = np.abs(self.phi_centers - center_rad)
        return np.where(d > np.pi, 2.0 * np.pi - d, d)

    def _synthetic_speed_map(self) -> np.ndarray:
        phi = self.phi_centers
        vr = np.full_like(phi, self.slow_speed_ms, dtype=float)
        streams = [(np.radians(50.0), np.radians(30.0), self.fast_speed_ms), (np.radians(230.0), np.radians(45.0), self.fast_speed_ms * 0.92)]
        for center, half_width, peak in streams:
            d = self._angular_distance(center)
            x = np.clip((half_width - d) / max(half_width, 1e-08), 0.0, 1.0)
            bump = 0.5 * (1.0 - np.cos(np.pi * x))
            vr = np.maximum(vr, self.slow_speed_ms + bump * (peak - self.slow_speed_ms))
        vr += 25000.0 * np.sin(phi) - 15000.0 * np.sin(2.0 * phi + 0.5)
        return np.clip(vr, 250000.0, 900000.0)

    def _synthetic_polarity_map(self) -> np.ndarray:
        phi = self.phi_centers
        sign = np.sign(np.sin(phi - np.radians(20.0)))
        sign[sign == 0.0] = 1.0
        return sign

    def density_1au_m3(self) -> np.ndarray:
        v = self.vr_ms
        v0 = 400000.0
        frac = np.clip((v - v0) / 250000.0, -1.0, 1.0)
        n_cm3 = self.slow_density_cm3_1au - 0.5 * (self.slow_density_cm3_1au - self.fast_density_cm3_1au) * (frac + 1.0)
        n_cm3 = np.clip(n_cm3, self.fast_density_cm3_1au, self.slow_density_cm3_1au)
        return n_cm3 * 1000000.0

    def temperature_K(self) -> np.ndarray:
        v = self.vr_ms
        frac = np.clip((v - 350000.0) / 350000.0, 0.0, 1.0)
        return self.base_temperature_K * (0.9 + 0.5 * frac)

@dataclass
class BackgroundWSALite:
    grid: SphericalGrid
    boundary_map: WSALiteBoundaryMap
    include_corotation: bool = True

    def __post_init__(self):
        self.U = np.zeros((N_VARS_2D, self.grid.nr, self.grid.nphi), dtype=np.float64)
        self._build()

    def _build(self):
        r = self.grid.r_centers
        phi = self.grid.phi_centers
        vr0 = self.boundary_map.vr_ms
        n1au = self.boundary_map.density_1au_m3()
        T0 = self.boundary_map.temperature_K()
        br_sign = self.boundary_map.br_sign
        rho_1au = n1au * M_PROTON
        rho0 = rho_1au * (AU / R_INNER) ** 2
        br0_mag = B_1AU_T * (AU / R_INNER) ** 2
        br0 = br_sign * br0_mag
        rr = r[:, None]
        vr_2d = np.tile(vr0[None, :], (self.grid.nr, 1))
        rho_2d = rho0[None, :] * vr0[None, :] * R_INNER ** 2 / (np.maximum(vr_2d, 1e-12) * rr ** 2)
        rho_2d = np.maximum(rho_2d, 1e-25)
        T_2d = T0[None, :] * np.clip((R_INNER / rr) ** 0.1, 0.6, 1.4)
        p_2d = np.maximum(rho_2d / M_PROTON * K_B * T_2d, 1e-20)
        br_2d = br0[None, :] * (R_INNER / rr) ** 2
        bphi_2d = -br_2d * (OMEGA_SUN * rr / np.maximum(vr_2d, 1e-12))
        if self.include_corotation:
            vphi_2d = -OMEGA_SUN * rr * np.sin(phi[None, :] - np.pi)
        else:
            vphi_2d = np.zeros_like(vr_2d)
        self.U[I_RHO] = rho_2d
        self.U[I_VR] = vr_2d
        self.U[I_VPHI] = vphi_2d
        self.U[I_P] = p_2d
        self.U[I_BR] = br_2d
        self.U[I_BPHI] = bphi_2d
        self.U[I_PSI] = 0.0

    @classmethod
    def from_arrays(cls, grid: SphericalGrid, vr_ms: np.ndarray, br_sign: Optional[np.ndarray]=None, include_corotation: bool=True):
        boundary = WSALiteBoundaryMap(phi_centers=grid.phi_centers, vr_ms=vr_ms, br_sign=br_sign)
        return cls(grid=grid, boundary_map=boundary, include_corotation=include_corotation)

@dataclass
class WSALiteBoundaryMap3D:
    theta_centers: np.ndarray
    phi_centers: np.ndarray
    vr_ms: Optional[np.ndarray] = None
    br_sign: Optional[np.ndarray] = None
    slow_speed_ms: float = 400000.0
    fast_speed_ms: float = 650000.0
    base_temperature_K: float = 1500000.0
    slow_density_cm3_1au: float = 7.0
    fast_density_cm3_1au: float = 3.5
    hcs_tilt_deg: float = 15.0
    fast_lat_half_width_deg: float = 40.0

    def __post_init__(self):
        self.theta_centers = np.asarray(self.theta_centers, dtype=float)
        self.phi_centers = np.asarray(self.phi_centers, dtype=float)
        self.ntheta = len(self.theta_centers)
        self.nphi = len(self.phi_centers)
        self._THETA, self._PHI = np.meshgrid(self.theta_centers, self.phi_centers, indexing='ij')
        if self.vr_ms is None:
            self.vr_ms = self._synthetic_speed_map()
        else:
            self.vr_ms = np.asarray(self.vr_ms, dtype=float)
            if self.vr_ms.shape != (self.ntheta, self.nphi):
                raise ValueError(f'vr_ms must be shape ({self.ntheta}, {self.nphi}), got {self.vr_ms.shape}')
        if self.br_sign is None:
            self.br_sign = self._synthetic_polarity_map()
        else:
            self.br_sign = np.asarray(self.br_sign, dtype=float)
            if self.br_sign.shape != (self.ntheta, self.nphi):
                raise ValueError(f'br_sign must be shape ({self.ntheta}, {self.nphi})')
        self.br_sign = np.where(self.br_sign >= 0.0, 1.0, -1.0)
        self.vr_ms = np.clip(self.vr_ms, 250000.0, 900000.0)

    def _lat_envelope(self) -> np.ndarray:
        lat_rad = np.pi / 2.0 - self._THETA
        hw = np.radians(self.fast_lat_half_width_deg)
        x = np.clip(np.abs(lat_rad) / hw, 0.0, 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * x))

    def _angular_distance_phi(self, center_rad: float) -> np.ndarray:
        d = np.abs(self._PHI - center_rad)
        return np.where(d > np.pi, 2.0 * np.pi - d, d)

    def _synthetic_speed_map(self) -> np.ndarray:
        vr = np.full((self.ntheta, self.nphi), self.slow_speed_ms, dtype=float)
        lat_env = self._lat_envelope()
        streams = [(np.radians(50.0), np.radians(30.0), self.fast_speed_ms), (np.radians(230.0), np.radians(45.0), self.fast_speed_ms * 0.92)]
        for center, half_width, peak in streams:
            d = self._angular_distance_phi(center)
            x = np.clip((half_width - d) / max(half_width, 1e-08), 0.0, 1.0)
            bump = 0.5 * (1.0 - np.cos(np.pi * x))
            stream_speed = self.slow_speed_ms + bump * (peak - self.slow_speed_ms)
            stream_speed = self.slow_speed_ms + lat_env * (stream_speed - self.slow_speed_ms)
            vr = np.maximum(vr, stream_speed)
        phi_noise = 25000.0 * np.sin(self._PHI) - 15000.0 * np.sin(2.0 * self._PHI + 0.5)
        vr += phi_noise
        return np.clip(vr, 250000.0, 900000.0)

    def _synthetic_polarity_map(self) -> np.ndarray:
        tilt = np.radians(self.hcs_tilt_deg)
        hcs_lat = tilt * np.sin(self._PHI + np.radians(20.0))
        lat = np.pi / 2.0 - self._THETA
        sign = np.where(lat > hcs_lat, 1.0, -1.0)
        return sign

    def density_1au_m3(self) -> np.ndarray:
        v = self.vr_ms
        v0 = 400000.0
        frac = np.clip((v - v0) / 250000.0, -1.0, 1.0)
        n_cm3 = self.slow_density_cm3_1au - 0.5 * (self.slow_density_cm3_1au - self.fast_density_cm3_1au) * (frac + 1.0)
        n_cm3 = np.clip(n_cm3, self.fast_density_cm3_1au, self.slow_density_cm3_1au)
        return n_cm3 * 1000000.0

    def temperature_K(self) -> np.ndarray:
        v = self.vr_ms
        frac = np.clip((v - 350000.0) / 350000.0, 0.0, 1.0)
        return self.base_temperature_K * (0.9 + 0.5 * frac)

@dataclass
class BackgroundWSALite3D:
    grid: SphericalGrid3D
    boundary_map: WSALiteBoundaryMap3D
    include_corotation: bool = False

    def __post_init__(self):
        self.U = np.zeros((N_VARS, self.grid.nr, self.grid.ntheta, self.grid.nphi), dtype=np.float64)
        self._build()

    def _build(self):
        r = self.grid.r_centers
        theta = self.grid.theta_centers
        phi = self.grid.phi_centers
        vr0 = self.boundary_map.vr_ms
        n1au = self.boundary_map.density_1au_m3()
        T0 = self.boundary_map.temperature_K()
        br_sign = self.boundary_map.br_sign
        rho_1au = n1au * M_PROTON
        rho0 = rho_1au * (AU / R_INNER) ** 2
        br0_mag = B_1AU_T * (AU / R_INNER) ** 2
        br0 = br_sign * br0_mag
        rr = r[:, None, None]
        sint = np.sin(theta[None, :, None])
        phi_3d = phi[None, None, :]
        vr_3d = np.broadcast_to(vr0[None, :, :], (self.grid.nr, self.grid.ntheta, self.grid.nphi)).copy()
        rho_3d = rho0[None, :, :] * vr0[None, :, :] * R_INNER ** 2 / (np.maximum(vr_3d, 1e-12) * rr ** 2)
        rho_3d = np.maximum(rho_3d, 1e-25)
        T_3d = T0[None, :, :] * np.clip((R_INNER / rr) ** 0.1, 0.6, 1.4)
        p_3d = np.maximum(rho_3d / M_PROTON * K_B * T_3d, 1e-20)
        r_edges = self.grid.r_edges
        br_face = br0[None, :, :] * (R_INNER / r_edges[:, None, None]) ** 2
        br_3d = 0.5 * (br_face[:-1, :, :] + br_face[1:, :, :])
        bphi_3d = -br_3d * (OMEGA_SUN * rr * sint / np.maximum(vr_3d, 1e-12))
        dphi = self.grid.dphi
        dtheta = self.grid.dtheta
        theta_edges = self.grid.theta_edges
        cos_te = np.cos(theta_edges)
        sin_te_v = np.sin(theta_edges)
        cos_dff = cos_te[:-1] - cos_te[1:]
        r_edges = self.grid.r_edges
        rsqdiff = 0.5 * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
        Phi_r = (r_edges ** 2)[:, None, None] * cos_dff[None, :, None] * dphi * br_face
        Phi_phi = rsqdiff[:, None, None] * dtheta * bphi_3d
        dPhi_r = Phi_r[1:] - Phi_r[:-1]
        dPhi_phi = np.roll(Phi_phi, -1, axis=2) - Phi_phi
        residual = dPhi_r + dPhi_phi
        col_sum = residual.sum(axis=1, keepdims=True)
        residual_star = residual - col_sum / self.grid.ntheta
        Phi_th = np.zeros((self.grid.nr, self.grid.ntheta + 1, self.grid.nphi), dtype=np.float64)
        for j in range(self.grid.ntheta):
            Phi_th[:, j + 1, :] = Phi_th[:, j, :] - residual_star[:, j, :]
        south_err = Phi_th[:, -1, :].copy()
        frac = np.linspace(0.0, 1.0, self.grid.ntheta + 1)[None, :, None]
        Phi_th -= frac * south_err[:, None, :]
        Phi_th[:, 0, :] = 0.0
        Phi_th[:, -1, :] = 0.0
        A_th = rsqdiff[:, None, None] * np.where(np.abs(sin_te_v) > 1e-10, sin_te_v, 1e-10)[None, :, None] * dphi
        btheta_face = Phi_th / np.maximum(A_th, 1e-60)
        btheta_face[:, 0, :] = 0.0
        btheta_face[:, -1, :] = 0.0
        btheta_3d = 0.5 * (btheta_face[:, :-1, :] + btheta_face[:, 1:, :])
        btheta_3d[:, 0, :] = 0.0
        btheta_3d[:, -1, :] = 0.0
        self.Br_face = br_face
        self.Bth_face = btheta_face
        self.Bphi_face = bphi_3d.copy()
        if self.include_corotation:
            vphi_3d = -OMEGA_SUN * rr * sint * np.ones_like(phi_3d)
        else:
            vphi_3d = np.zeros((self.grid.nr, self.grid.ntheta, self.grid.nphi))
        vtheta_3d = np.zeros_like(vr_3d)
        self.U[I_RHO] = rho_3d
        self.U[I_VR] = vr_3d
        self.U[I_VPHI] = vphi_3d
        self.U[I_P] = p_3d
        self.U[I_BR] = br_3d
        self.U[I_BPHI] = bphi_3d
        self.U[I_PSI] = 0.0
        self.U[I_VTHETA] = vtheta_3d
        self.U[I_BTHETA] = btheta_3d

    @classmethod
    def from_arrays(cls, grid: SphericalGrid3D, vr_ms: np.ndarray, br_sign: Optional[np.ndarray]=None, include_corotation: bool=False):
        boundary = WSALiteBoundaryMap3D(theta_centers=grid.theta_centers, phi_centers=grid.phi_centers, vr_ms=vr_ms, br_sign=br_sign)
        return cls(grid=grid, boundary_map=boundary, include_corotation=include_corotation)

    def at_earth(self) -> dict:
        i, j, k = self.grid.earth_index_3d()
        from swcmep.utils.constants import m3_to_cm3
        return {'r_AU': self.grid.r_centers[i] / AU, 'v_r_kms': self.U[I_VR, i, j, k] / 1000.0, 'n_cm3': m3_to_cm3(self.U[I_RHO, i, j, k] / M_PROTON), 'p_Pa': self.U[I_P, i, j, k]}

    def summary(self) -> str:
        earth = self.at_earth()
        mem_gb = self.U.nbytes / 1000000000.0
        return f"BackgroundWSALite3D({self.grid.nr}r × {self.grid.ntheta}theta × {self.grid.nphi}phi)\n  At Earth (equator, phi=0):\n    v_r = {earth['v_r_kms']:.1f} km/s\n    n   = {earth['n_cm3']:.2f} cm⁻³\n  Memory : {mem_gb:.2f} GB\n"

def load_speed_map_csv(path: str):
    data = np.loadtxt(path, delimiter=',', comments='#')
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError('CSV must contain at least two columns: phi_deg, vr_kms')
    phi_rad = np.radians(data[:, 0])
    vr_ms = data[:, 1] * 1000.0
    return (phi_rad, vr_ms)

def remap_speed_to_grid(phi_src_rad: np.ndarray, vr_src_ms: np.ndarray, phi_target_rad: np.ndarray) -> np.ndarray:
    phi_src = np.mod(phi_src_rad, 2.0 * np.pi)
    phi_tgt = np.mod(phi_target_rad, 2.0 * np.pi)
    order = np.argsort(phi_src)
    phi_src = phi_src[order]
    vr_src_ms = vr_src_ms[order]
    phi_ext = np.concatenate([phi_src - 2.0 * np.pi, phi_src, phi_src + 2.0 * np.pi])
    vr_ext = np.concatenate([vr_src_ms, vr_src_ms, vr_src_ms])
    return np.interp(phi_tgt, phi_ext, vr_ext)
