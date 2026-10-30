from __future__ import annotations
import numpy as np
from swcmep.grid.spherical_grid import SphericalGrid3D
from swcmep.solar_wind.background_2d import I_P
from swcmep.solar_wind.background_wsa_lite import BackgroundWSALite3D, WSALiteBoundaryMap3D
from swcmep.utils.constants import I_EINT, N_VARS, R_INNER, R_OUTER

def make_background_3d(nr: int=256, ntheta: int=90, nphi: int=180, r_inner: float=R_INNER, r_outer: float=R_OUTER, log_radial: bool=True, slow_speed_ms: float=400000.0, fast_speed_ms: float=650000.0, hcs_tilt_deg: float=15.0, fast_lat_half_width_deg: float=40.0, include_corotation: bool=False, vr_ms: np.ndarray | None=None, br_sign: np.ndarray | None=None) -> tuple[np.ndarray, SphericalGrid3D]:
    grid = SphericalGrid3D(nr=nr, ntheta=ntheta, nphi=nphi, r_inner=r_inner, r_outer=r_outer, log_radial=log_radial)
    boundary = WSALiteBoundaryMap3D(theta_centers=grid.theta_centers, phi_centers=grid.phi_centers, vr_ms=vr_ms, br_sign=br_sign, slow_speed_ms=slow_speed_ms, fast_speed_ms=fast_speed_ms, hcs_tilt_deg=hcs_tilt_deg, fast_lat_half_width_deg=fast_lat_half_width_deg)
    bg = BackgroundWSALite3D(grid=grid, boundary_map=boundary, include_corotation=include_corotation)
    print(grid.summary())
    print(bg.summary())
    U = bg.U.copy()
    GAMMA_DEFAULT = 1.05
    if U.shape[0] < N_VARS:
        U_full = np.zeros((N_VARS, *U.shape[1:]), dtype=U.dtype)
        U_full[:U.shape[0]] = U
        U = U_full
    if N_VARS > I_EINT:
        U[I_EINT] = U[I_P] / (GAMMA_DEFAULT - 1.0)
    return (U, grid)
