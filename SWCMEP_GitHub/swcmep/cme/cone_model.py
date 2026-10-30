from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
import torch
from swcmep.grid.spherical_grid import SphericalGrid, SphericalGrid3D
from swcmep.solar_wind.background_2d import I_BPHI, I_BR, I_P, I_PSI, I_RHO, I_VPHI, I_VR
from swcmep.utils.constants import AU, I_BTHETA, I_EINT, I_VTHETA, K_B, M_PROTON
from swcmep.utils.validation import finite_float, integer_value, positive_float

@dataclass
class CMEParameters:
    speed_kms: float
    half_width_deg: float = 40.0
    longitude_deg: float = 0.0
    latitude_deg: float = 0.0
    duration_hours: float = 4.0
    density_enhancement: float = 5.0
    temperature_K: float = 2000000.0
    name: str = 'CME'
    inject_magnetic_shell: bool = False
    magnetic_enhancement: float = 1.2
    shell_layers: int = 2
    taper_deg: float = 10.0
    drag_start_radius_au: float = 0.15
    drag_speed_threshold: float = 0.2

    def __post_init__(self):
        for name in ('speed_kms', 'duration_hours', 'temperature_K'):
            setattr(self, name, positive_float(getattr(self, name), name))
        for name in ('half_width_deg', 'longitude_deg', 'latitude_deg', 'density_enhancement', 'magnetic_enhancement'):
            setattr(self, name, finite_float(getattr(self, name), name))
        if not 5 <= self.half_width_deg <= 90:
            raise ValueError('half_width_deg must be between 5 and 90 degrees')
        if not -90 <= self.latitude_deg <= 90:
            raise ValueError('latitude_deg must be between -90 and 90 degrees')
        if self.density_enhancement < 1:
            raise ValueError('density_enhancement must be >= 1')
        if self.magnetic_enhancement < 1:
            raise ValueError('magnetic_enhancement must be >= 1')
        self.shell_layers = integer_value(self.shell_layers, 'shell_layers')
        self.taper_deg = positive_float(self.taper_deg, 'taper_deg', allow_zero=True)
        self.drag_start_radius_au = positive_float(self.drag_start_radius_au, 'drag_start_radius_au', allow_zero=True)
        self.drag_speed_threshold = positive_float(self.drag_speed_threshold, 'drag_speed_threshold', allow_zero=True)

    @property
    def speed_ms(self) -> float:
        return self.speed_kms * 1000.0

    @property
    def half_width_rad(self) -> float:
        return np.radians(self.half_width_deg)

    @property
    def longitude_rad(self) -> float:
        return np.radians(self.longitude_deg)

    @property
    def latitude_rad(self) -> float:
        return np.radians(self.latitude_deg)

    @property
    def taper_rad(self) -> float:
        return np.radians(self.taper_deg)

    def transit_time_hours(self, distance_AU: float=1.0) -> float:
        return distance_AU * AU / self.speed_ms / 3600.0

    def summary(self) -> str:
        layers_used = min(max(self.shell_layers, 1), 6)
        lines = [f'CMEParameters: {self.name}', f'  Speed           : {self.speed_kms:.0f} km/s', f'  Half-width      : {self.half_width_deg:.1f} deg', f'  Longitude       : {self.longitude_deg:+.1f} deg from Sun-Earth line', f'  Latitude        : {self.latitude_deg:+.1f} deg from ecliptic', f'  Duration        : {self.duration_hours:.1f} hours', f'  Density factor  : {self.density_enhancement:.1f}x background', f'  Temperature     : {self.temperature_K / 1000000.0:.2f} MK', f'  Requested layers: {layers_used}', f"  B_phi inflow    : {('enabled' if self.inject_magnetic_shell else 'disabled')}"]
        if self.inject_magnetic_shell:
            lines.append(f'  B_phi factor    : {self.magnetic_enhancement:.2f}x background')
        lines.extend([f'  Ballistic ETA   : {self.transit_time_hours():.1f} hours', '  (Ballistic estimate only; MHD arrival accuracy requires validation)'])
        return '\n'.join(lines) + '\n'

@dataclass
class ConeModelInjector:
    grid: SphericalGrid
    U_background: np.ndarray
    cme: CMEParameters
    drag_coefficient: float = 1e-07
    _effective_half_width: float = field(init=False, repr=False)
    _drag_start_mask_np: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.drag_coefficient = positive_float(self.drag_coefficient, 'drag_coefficient', allow_zero=True)
        lat_factor = np.cos(self.cme.latitude_rad)
        self._effective_half_width = self.cme.half_width_rad * lat_factor
        arc_deg = np.degrees(self._effective_half_width * 2)
        phi = self.grid.phi_centers
        phi_nose = self.cme.longitude_rad % (2 * np.pi)
        delta_phi = np.abs(phi - phi_nose)
        delta_phi = np.where(delta_phi > np.pi, 2 * np.pi - delta_phi, delta_phi)
        n_cells = int((delta_phi <= self._effective_half_width).sum())
        print(f'CME cone: {n_cells} cells in injection zone ({arc_deg:.1f} deg arc at inner boundary)')
        self._drag_start_mask_np = self.grid.R >= self.cme.drag_start_radius_au * AU

    def is_injecting(self, t_seconds: float) -> bool:
        return 0.0 <= t_seconds < self.cme.duration_hours * 3600.0

    def timestep_limit(self, t_seconds: float) -> float:
        duration = self.cme.duration_hours * 3600.0
        if t_seconds < 0.0:
            return -t_seconds
        if t_seconds < duration:
            return min(duration / 20.0, duration - t_seconds)
        return math.inf

    def _temporal_envelope(self, t_seconds: float) -> float:
        t_total = self.cme.duration_hours * 3600.0
        ramp = 0.2 * t_total
        if t_seconds <= 0.0:
            return 0.0
        elif t_seconds < ramp:
            return float(0.5 * (1.0 - np.cos(np.pi * t_seconds / ramp)))
        elif t_seconds <= t_total - ramp:
            return 1.0
        elif t_seconds < t_total:
            return float(0.5 * (1.0 - np.cos(np.pi * (t_total - t_seconds) / ramp)))
        else:
            return 0.0

    def _angular_blend_np(self):
        phi = self.grid.phi_centers
        phi_nose = self.cme.longitude_rad % (2 * np.pi)
        delta_phi = np.abs(phi - phi_nose)
        delta_phi = np.where(delta_phi > np.pi, 2 * np.pi - delta_phi, delta_phi)
        hw = self._effective_half_width
        taper_w = max(self.cme.taper_rad, 1e-08)
        core = max(hw - taper_w, 0.0)
        blend = np.zeros_like(delta_phi)
        inside_core = delta_phi <= core
        in_taper = (delta_phi > core) & (delta_phi <= hw)
        blend[inside_core] = 1.0
        x = np.clip((hw - delta_phi) / taper_w, 0.0, 1.0)
        taper = 0.5 * (1.0 - np.cos(np.pi * x))
        blend[in_taper] = taper[in_taper]
        return blend

    def _angular_blend_gpu(self, device, dtype):
        return torch.tensor(self._angular_blend_np(), dtype=dtype, device=device)

    def _boundary_layer_weight(self, layer: int) -> float:
        return float(np.cos(0.5 * np.pi * layer / max(self.cme.shell_layers, 1)) ** 2)

    def _cme_pressure_np(self, rho_cme, p_bg):
        p_thermal = rho_cme / M_PROTON * K_B * self.cme.temperature_K
        p_density = p_bg * self.cme.density_enhancement
        return np.maximum(p_thermal, p_density)

    def _cme_pressure_gpu(self, rho_cme, p_bg):
        p_thermal = rho_cme / M_PROTON * K_B * self.cme.temperature_K
        p_density = p_bg * self.cme.density_enhancement
        return torch.maximum(p_thermal, p_density)

    def apply_boundary_gpu(self, U_gpu: torch.Tensor, t_seconds: float):
        if not self.is_injecting(t_seconds):
            return
        device = U_gpu.device
        dtype = U_gpu.dtype
        envelope = self._temporal_envelope(t_seconds)
        ang_blend = self._angular_blend_gpu(device, dtype)
        n_layers = min(max(self.cme.shell_layers, 1), 6, U_gpu.shape[1])
        with torch.no_grad():
            for layer in range(n_layers):
                bg_layer = min(layer, min(1, self.U_background.shape[1] - 1))
                layer_weight = self._boundary_layer_weight(layer)
                blend = envelope * layer_weight * ang_blend
                rho_bg = torch.tensor(self.U_background[I_RHO, bg_layer, :], dtype=dtype, device=device)
                v_bg = torch.tensor(self.U_background[I_VR, bg_layer, :], dtype=dtype, device=device)
                p_bg = torch.tensor(self.U_background[I_P, bg_layer, :], dtype=dtype, device=device)
                vphi_bg = torch.tensor(self.U_background[I_VPHI, bg_layer, :], dtype=dtype, device=device)
                bphi_bg = torch.tensor(self.U_background[I_BPHI, bg_layer, :], dtype=dtype, device=device)
                br_bg = torch.tensor(self.U_background[I_BR, bg_layer, :], dtype=dtype, device=device)
                rho_cme = rho_bg * self.cme.density_enhancement
                v_cme = torch.full_like(v_bg, self.cme.speed_ms)
                p_cme = self._cme_pressure_gpu(rho_cme, p_bg)
                U_gpu[I_RHO, layer, :] = rho_bg + blend * (rho_cme - rho_bg)
                U_gpu[I_VR, layer, :] = v_bg + blend * (v_cme - v_bg)
                U_gpu[I_VPHI, layer, :] = vphi_bg
                U_gpu[I_P, layer, :] = torch.clamp(p_bg + blend * (p_cme - p_bg), min=1e-20)
                U_gpu[I_BR, layer, :] = br_bg
                if self.cme.inject_magnetic_shell:
                    bphi_cme = bphi_bg * self.cme.magnetic_enhancement
                    U_gpu[I_BPHI, layer, :] = bphi_bg + blend * (bphi_cme - bphi_bg)
                else:
                    U_gpu[I_BPHI, layer, :] = bphi_bg
                U_gpu[I_PSI, layer, :] = 0.0

    def apply_boundary(self, U: np.ndarray, t_seconds: float):
        if not self.is_injecting(t_seconds):
            return
        envelope = self._temporal_envelope(t_seconds)
        ang_blend = self._angular_blend_np()
        n_layers = min(max(self.cme.shell_layers, 1), 2, U.shape[1])
        for layer in range(n_layers):
            bg_layer = min(layer, min(1, self.U_background.shape[1] - 1))
            layer_weight = self._boundary_layer_weight(layer)
            blend = envelope * layer_weight * ang_blend
            rho_bg = self.U_background[I_RHO, bg_layer, :]
            v_bg = self.U_background[I_VR, bg_layer, :]
            p_bg = self.U_background[I_P, bg_layer, :]
            vphi_bg = self.U_background[I_VPHI, bg_layer, :]
            bphi_bg = self.U_background[I_BPHI, bg_layer, :]
            br_bg = self.U_background[I_BR, bg_layer, :]
            rho_cme = rho_bg * self.cme.density_enhancement
            v_cme = np.full_like(v_bg, self.cme.speed_ms)
            p_cme = self._cme_pressure_np(rho_cme, p_bg)
            U[I_RHO, layer, :] = rho_bg + blend * (rho_cme - rho_bg)
            U[I_VR, layer, :] = v_bg + blend * (v_cme - v_bg)
            U[I_VPHI, layer, :] = vphi_bg
            U[I_P, layer, :] = np.maximum(p_bg + blend * (p_cme - p_bg), 1e-20)
            U[I_BR, layer, :] = br_bg
            if self.cme.inject_magnetic_shell:
                bphi_cme = bphi_bg * self.cme.magnetic_enhancement
                U[I_BPHI, layer, :] = bphi_bg + blend * (bphi_cme - bphi_bg)
            else:
                U[I_BPHI, layer, :] = bphi_bg
            U[I_PSI, layer, :] = 0.0

    def apply_gpu(self, U_gpu: torch.Tensor, t_seconds: float):
        self.apply_boundary_gpu(U_gpu, t_seconds)

    def apply(self, U: np.ndarray, t_seconds: float):
        self.apply_boundary(U, t_seconds)

    def apply_drag_gpu(self, U_gpu: torch.Tensor, dt: float):
        if self.drag_coefficient <= 0:
            return
        device = U_gpu.device
        dtype = U_gpu.dtype
        v_bg = torch.tensor(self.U_background[I_VR], dtype=dtype, device=device)
        rho_bg = torch.tensor(self.U_background[I_RHO], dtype=dtype, device=device)
        bphi_bg = torch.tensor(self.U_background[I_BPHI], dtype=dtype, device=device)
        drag_region = torch.tensor(self._drag_start_mask_np, dtype=torch.bool, device=device)
        v = U_gpu[I_VR]
        rho = torch.clamp(U_gpu[I_RHO], min=1e-30)
        bphi = U_gpu[I_BPHI]
        dv = v - v_bg
        fast_enough = dv > self.cme.drag_speed_threshold * torch.abs(v_bg)
        density_excess = rho > 1.05 * rho_bg
        magnetic_excess = torch.abs(bphi - bphi_bg) > 0.05 * torch.abs(bphi_bg) + 1e-20
        is_cme_like = fast_enough & (density_excess | magnetic_excess)
        gamma_si = self.drag_coefficient * 0.001
        a_drag = -gamma_si * dv * torch.abs(dv)
        dv_mag = torch.abs(dv)
        a_limit = dv_mag / (dt + 1e-20)
        a_drag = torch.clamp(a_drag, -a_limit, a_limit)
        with torch.no_grad():
            apply_mask = drag_region & is_cme_like
            U_gpu[I_VR] = torch.where(apply_mask, v + a_drag * dt, v)

@dataclass
class ConeModelInjector3D:
    grid: SphericalGrid3D
    U_background: np.ndarray
    cme: CMEParameters
    drag_coefficient: float = 1e-07
    gamma: float = 1.05
    boundary_layers: int = 3
    _cone_blend_np: np.ndarray = field(init=False, repr=False)
    _drag_mask_np: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.gamma = finite_float(self.gamma, 'gamma')
        if self.gamma <= 1.0:
            raise ValueError('gamma must be greater than 1')
        self.boundary_layers = integer_value(self.boundary_layers, 'boundary_layers')
        self.drag_coefficient = positive_float(self.drag_coefficient, 'drag_coefficient', allow_zero=True)
        theta_nose = np.pi / 2.0 - self.cme.latitude_rad
        phi_nose = self.cme.longitude_rad % (2.0 * np.pi)
        th = self.grid.theta_centers[:, np.newaxis]
        ph = self.grid.phi_centers[np.newaxis, :]
        cos_delta = np.sin(th) * np.sin(theta_nose) * np.cos(ph - phi_nose) + np.cos(th) * np.cos(theta_nose)
        cos_delta = np.clip(cos_delta, -1.0, 1.0)
        delta = np.arccos(cos_delta)
        hw = self.cme.half_width_rad
        taper_w = max(self.cme.taper_rad, 1e-08)
        core = max(hw - taper_w, 0.0)
        blend = np.zeros_like(delta)
        blend[delta <= core] = 1.0
        in_taper = (delta > core) & (delta <= hw)
        x = np.clip((hw - delta[in_taper]) / taper_w, 0.0, 1.0)
        blend[in_taper] = 0.5 * (1.0 - np.cos(np.pi * x))
        self._cone_blend_np = blend
        n_cells = int((delta <= hw).sum())
        solid_angle_sr = 2.0 * np.pi * (1.0 - np.cos(hw))
        effective_layers = min(self.boundary_layers, self.grid.nr)
        print(f'CME 3D cone: {n_cells} cells in injection zone  (half-width={self.cme.half_width_deg:.1f}°, solid angle={solid_angle_sr:.3f} sr)  nose=(lat={self.cme.latitude_deg:+.1f}°, lon={self.cme.longitude_deg:+.1f}°)  layers={effective_layers}')
        r3d = self.grid.r_centers[:, np.newaxis, np.newaxis]
        self._drag_mask_np = r3d >= self.cme.drag_start_radius_au * AU

    def is_injecting(self, t_seconds: float) -> bool:
        return 0.0 <= t_seconds < self.cme.duration_hours * 3600.0

    def timestep_limit(self, t_seconds: float) -> float:
        duration = self.cme.duration_hours * 3600.0
        if t_seconds < 0.0:
            return -t_seconds
        if t_seconds < duration:
            return min(duration / 20.0, duration - t_seconds)
        return math.inf

    def _temporal_envelope(self, t_seconds: float) -> float:
        t_total = self.cme.duration_hours * 3600.0
        ramp = 0.2 * t_total
        if t_seconds <= 0.0:
            return 0.0
        elif t_seconds < ramp:
            return float(0.5 * (1.0 - np.cos(np.pi * t_seconds / ramp)))
        elif t_seconds <= t_total - ramp:
            return 1.0
        elif t_seconds < t_total:
            return float(0.5 * (1.0 - np.cos(np.pi * (t_total - t_seconds) / ramp)))
        else:
            return 0.0

    def _cme_pressure_gpu(self, rho_cme: torch.Tensor, p_bg: torch.Tensor) -> torch.Tensor:
        p_thermal = rho_cme / M_PROTON * K_B * self.cme.temperature_K
        p_density = p_bg * self.cme.density_enhancement
        return torch.maximum(p_thermal, p_density)

    def apply_boundary_gpu_3d(self, U_gpu: torch.Tensor, t_seconds: float):
        if not self.is_injecting(t_seconds):
            return
        device = U_gpu.device
        dtype = U_gpu.dtype
        envelope = self._temporal_envelope(t_seconds)
        blend_2d = torch.tensor(self._cone_blend_np, dtype=dtype, device=device).unsqueeze(0)
        n_layers = min(max(self.boundary_layers, 1), U_gpu.shape[1])
        with torch.no_grad():
            for layer in range(n_layers):
                bg_layer = min(layer, self.U_background.shape[1] - 1)
                blend = envelope * blend_2d
                rho_bg = torch.tensor(self.U_background[I_RHO, bg_layer], dtype=dtype, device=device)
                v_bg = torch.tensor(self.U_background[I_VR, bg_layer], dtype=dtype, device=device)
                p_bg = torch.tensor(self.U_background[I_P, bg_layer], dtype=dtype, device=device)
                vphi_bg = torch.tensor(self.U_background[I_VPHI, bg_layer], dtype=dtype, device=device)
                br_bg = torch.tensor(self.U_background[I_BR, bg_layer], dtype=dtype, device=device)
                bphi_bg = torch.tensor(self.U_background[I_BPHI, bg_layer], dtype=dtype, device=device)
                rho_cme = rho_bg * self.cme.density_enhancement
                v_cme = torch.full_like(v_bg, self.cme.speed_ms)
                p_cme = self._cme_pressure_gpu(rho_cme, p_bg)
                b = blend[0]
                U_gpu[I_RHO, layer] = rho_bg + b * (rho_cme - rho_bg)
                U_gpu[I_VR, layer] = v_bg + b * (v_cme - v_bg)
                U_gpu[I_VPHI, layer] = vphi_bg
                U_gpu[I_VTHETA, layer] = torch.as_tensor(self.U_background[I_VTHETA, bg_layer], dtype=dtype, device=device) * (1.0 - b)
                U_gpu[I_P, layer] = torch.clamp(p_bg + b * (p_cme - p_bg), min=1e-20)
                if U_gpu.shape[0] > I_EINT:
                    U_gpu[I_EINT, layer] = U_gpu[I_P, layer] / (self.gamma - 1.0)
                U_gpu[I_BR, layer] = br_bg
                if self.cme.inject_magnetic_shell:
                    bphi_cme = bphi_bg * self.cme.magnetic_enhancement
                    U_gpu[I_BPHI, layer] = bphi_bg + b * (bphi_cme - bphi_bg)
                else:
                    U_gpu[I_BPHI, layer] = bphi_bg
                U_gpu[I_BTHETA, layer] = torch.as_tensor(self.U_background[I_BTHETA, bg_layer], dtype=dtype, device=device)
                U_gpu[I_PSI, layer] = 0.0

    def apply_boundary_gpu(self, U_gpu: torch.Tensor, t_seconds: float):
        self.apply_boundary_gpu_3d(U_gpu, t_seconds)

    def apply_drag_gpu_3d(self, U_gpu: torch.Tensor, dt: float):
        if self.drag_coefficient <= 0:
            return
        device = U_gpu.device
        dtype = U_gpu.dtype
        v_bg_np = self.U_background[I_VR]
        rho_bg_np = self.U_background[I_RHO]
        bphi_bg_np = self.U_background[I_BPHI]
        v_bg = torch.tensor(v_bg_np, dtype=dtype, device=device)
        rho_bg = torch.tensor(rho_bg_np, dtype=dtype, device=device)
        bphi_bg = torch.tensor(bphi_bg_np, dtype=dtype, device=device)
        drag_region = torch.tensor(self._drag_mask_np, dtype=torch.bool, device=device)
        v = U_gpu[I_VR]
        rho = torch.clamp(U_gpu[I_RHO], min=1e-30)
        bphi = U_gpu[I_BPHI]
        dv = v - v_bg
        fast_enough = dv > self.cme.drag_speed_threshold * torch.abs(v_bg)
        density_excess = rho > 1.05 * rho_bg
        magnetic_excess = torch.abs(bphi - bphi_bg) > 0.05 * torch.abs(bphi_bg) + 1e-20
        is_cme_like = fast_enough & (density_excess | magnetic_excess)
        gamma_si = self.drag_coefficient * 0.001
        a_drag = -gamma_si * dv * torch.abs(dv)
        a_limit = torch.abs(dv) / (dt + 1e-20)
        a_drag = torch.clamp(a_drag, -a_limit, a_limit)
        with torch.no_grad():
            mask = drag_region & is_cme_like
            U_gpu[I_VR] = torch.where(mask, v + a_drag * dt, v)

    def apply_drag_gpu(self, U_gpu: torch.Tensor, dt: float):
        self.apply_drag_gpu_3d(U_gpu, dt)

@dataclass(frozen=True)
class ScheduledCME3D:
    cme: CMEParameters
    start_seconds: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, 'start_seconds', positive_float(self.start_seconds, 'start_seconds', allow_zero=True))

    @property
    def end_seconds(self) -> float:
        return self.start_seconds + self.cme.duration_hours * 3600.0

@dataclass
class CompoundConeInjector3D:
    grid: SphericalGrid3D
    U_background: np.ndarray
    scheduled_cmes: tuple[ScheduledCME3D, ...] | list[ScheduledCME3D]
    drag_coefficient: float = 0.0
    gamma: float = 1.05
    boundary_layers: int = 3
    _injectors: tuple[ConeModelInjector3D, ...] = field(init=False, repr=False)

    def __post_init__(self):
        self.gamma = finite_float(self.gamma, 'gamma')
        if self.gamma <= 1.0:
            raise ValueError('gamma must be greater than 1')
        self.boundary_layers = integer_value(self.boundary_layers, 'boundary_layers')
        self.drag_coefficient = positive_float(self.drag_coefficient, 'drag_coefficient', allow_zero=True)
        if not self.scheduled_cmes:
            raise ValueError('at least one scheduled CME is required')
        if self.drag_coefficient != 0.0:
            raise ValueError('compound CME drag is not supported; set drag_coefficient=0. Interacting CMEs are evolved by the MHD solver.')
        scheduled = tuple(sorted(self.scheduled_cmes, key=lambda item: item.start_seconds))
        self.scheduled_cmes = scheduled
        self._injectors = tuple((ConeModelInjector3D(grid=self.grid, U_background=self.U_background, cme=item.cme, drag_coefficient=self.drag_coefficient, gamma=self.gamma, boundary_layers=self.boundary_layers) for item in scheduled))
        print(f'Compound CME boundary: {len(scheduled)} scheduled pulses, last injection ends at t={self.injection_end_seconds / 3600.0:.2f}h')

    @property
    def injection_end_seconds(self) -> float:
        return max((item.end_seconds for item in self.scheduled_cmes))

    def timestep_limit(self, t_seconds: float) -> float:
        return min((injector.timestep_limit(t_seconds - item.start_seconds) for item, injector in zip(self.scheduled_cmes, self._injectors)))

    def _active(self, t_seconds: float):
        active = []
        for item, injector in zip(self.scheduled_cmes, self._injectors):
            local_time = t_seconds - item.start_seconds
            if 0.0 <= local_time < item.cme.duration_hours * 3600.0:
                active.append((item, injector, local_time))
        return active

    def is_injecting(self, t_seconds: float) -> bool:
        return bool(self._active(t_seconds))

    def apply_boundary_gpu_3d(self, U_gpu: torch.Tensor, t_seconds: float):
        active = self._active(t_seconds)
        if not active:
            return
        device = U_gpu.device
        dtype = U_gpu.dtype
        n_layers = min(max(self.boundary_layers, 1), U_gpu.shape[1])
        with torch.no_grad():
            for layer in range(n_layers):
                bg_layer = min(layer, self.U_background.shape[1] - 1)
                rho_bg = torch.tensor(self.U_background[I_RHO, bg_layer], dtype=dtype, device=device)
                vr_bg = torch.tensor(self.U_background[I_VR, bg_layer], dtype=dtype, device=device)
                vphi_bg = torch.tensor(self.U_background[I_VPHI, bg_layer], dtype=dtype, device=device)
                p_bg = torch.tensor(self.U_background[I_P, bg_layer], dtype=dtype, device=device)
                br_bg = torch.tensor(self.U_background[I_BR, bg_layer], dtype=dtype, device=device)
                bphi_bg = torch.tensor(self.U_background[I_BPHI, bg_layer], dtype=dtype, device=device)
                U_gpu[I_RHO, layer] = rho_bg
                U_gpu[I_VR, layer] = vr_bg
                U_gpu[I_VPHI, layer] = vphi_bg
                vtheta_bg = torch.as_tensor(self.U_background[I_VTHETA, bg_layer], dtype=dtype, device=device)
                U_gpu[I_VTHETA, layer] = vtheta_bg
                U_gpu[I_P, layer] = p_bg
                U_gpu[I_BR, layer] = br_bg
                U_gpu[I_BPHI, layer] = bphi_bg
                U_gpu[I_BTHETA, layer] = torch.as_tensor(self.U_background[I_BTHETA, bg_layer], dtype=dtype, device=device)
                U_gpu[I_PSI, layer] = 0.0
                best_score = torch.full_like(vr_bg, -1.0)
                for item, injector, local_time in active:
                    envelope = injector._temporal_envelope(local_time)
                    if envelope <= 0.0:
                        continue
                    blend = envelope * torch.tensor(injector._cone_blend_np, dtype=dtype, device=device)
                    cme = item.cme
                    rho_cme = rho_bg * cme.density_enhancement
                    vr_cme = torch.full_like(vr_bg, cme.speed_ms)
                    p_cme = injector._cme_pressure_gpu(rho_cme, p_bg)
                    score = blend * torch.abs(vr_cme - vr_bg)
                    use = (blend > 0.0) & (score > best_score)
                    rho_candidate = rho_bg + blend * (rho_cme - rho_bg)
                    vr_candidate = vr_bg + blend * (vr_cme - vr_bg)
                    p_candidate = torch.clamp(p_bg + blend * (p_cme - p_bg), min=1e-20)
                    U_gpu[I_RHO, layer] = torch.where(use, rho_candidate, U_gpu[I_RHO, layer])
                    U_gpu[I_VR, layer] = torch.where(use, vr_candidate, U_gpu[I_VR, layer])
                    U_gpu[I_P, layer] = torch.where(use, p_candidate, U_gpu[I_P, layer])
                    if cme.inject_magnetic_shell:
                        bphi_cme = bphi_bg * cme.magnetic_enhancement
                        bphi_candidate = bphi_bg + blend * (bphi_cme - bphi_bg)
                        U_gpu[I_BPHI, layer] = torch.where(use, bphi_candidate, U_gpu[I_BPHI, layer])
                    if not cme.inject_magnetic_shell:
                        U_gpu[I_BPHI, layer] = torch.where(use, bphi_bg, U_gpu[I_BPHI, layer])
                    U_gpu[I_VTHETA, layer] = torch.where(use, vtheta_bg * (1.0 - blend), U_gpu[I_VTHETA, layer])
                    best_score = torch.where(use, score, best_score)
                if U_gpu.shape[0] > I_EINT:
                    U_gpu[I_EINT, layer] = U_gpu[I_P, layer] / (self.gamma - 1.0)

    def apply_boundary_gpu(self, U_gpu: torch.Tensor, t_seconds: float):
        self.apply_boundary_gpu_3d(U_gpu, t_seconds)

def run_cme_simulation(cme: CMEParameters, grid_nr: int=512, grid_nphi: int=360, T_corona: float=1500000.0, sim_hours: float=96.0, output_every_h: float=1.0, use_gpu: bool=True, use_drag: bool=True, drag_coefficient: float=1e-07, verbose: bool=True, relax_background_hours: float=24.0) -> dict:
    from swcmep.grid.spherical_grid import SphericalGrid
    from swcmep.solar_wind.background_wsa_lite import BackgroundWSALite, WSALiteBoundaryMap
    if verbose:
        print('=' * 60)
        print('SWCMEP CME Simulation (2D)')
        print('=' * 60)
        print(cme.summary())
        if use_drag:
            print(f'  Drag model    : enabled (gamma = {drag_coefficient:.1e} /km)')
    grid = SphericalGrid(nr=grid_nr, nphi=grid_nphi)
    boundary = WSALiteBoundaryMap(phi_centers=grid.phi_centers)
    bg = BackgroundWSALite(grid, boundary)
    injector = ConeModelInjector(grid, bg.U, cme, drag_coefficient=drag_coefficient if use_drag else 0.0)
    cfl = 0.3 if cme.speed_kms > 1500 else 0.5
    dtype = torch.float64 if cme.speed_kms > 1500 else torch.float32
    from swcmep.solar_wind.mhd_solver_gpu import MHDSolverGPU
    solver = MHDSolverGPU(grid, bg.U.copy(), gamma=1.05, cfl_factor=cfl, dtype=dtype, device=None if use_gpu else 'cpu')
    if hasattr(solver, 'use_well_balanced'):
        solver.use_well_balanced = True
    if hasattr(solver, 'use_boundary_sponge'):
        solver.use_boundary_sponge = True
    if relax_background_hours > 0.0:
        if verbose:
            print(f'\nRelaxing WSA-lite background for {relax_background_hours:.1f} h before CME injection...')
        solver.run(duration_seconds=relax_background_hours * 3600.0, output_every=max(3600.0, relax_background_hours * 3600.0), verbose=verbose, injector=None)
        if hasattr(solver, '_U_gpu'):
            U_relaxed = solver._U_gpu.detach().cpu().numpy().copy()
        else:
            U_relaxed = solver.U.copy()
        if hasattr(solver, 'set_equilibrium_state'):
            solver.set_equilibrium_state(U_relaxed)
        solver.t = 0.0
        solver.step = 0
        solver.dt = 0.0
        solver.U = U_relaxed.copy()
        injector.U_background = U_relaxed.copy()
    i_earth, j_earth = grid.earth_index()
    background_ref = injector.U_background
    v_background = background_ref[I_VR, i_earth, j_earth] / 1000.0
    n_background = background_ref[I_RHO, i_earth, j_earth] / M_PROTON / 1000000.0
    arrival_info = {'detected': False, 'time_hours': None, 'v_kms': None}

    def check_arrival(slvr, snap):
        U = snap['U']
        v_earth = U[I_VR, i_earth, j_earth] / 1000.0
        n_earth = U[I_RHO, i_earth, j_earth] / M_PROTON / 1000000.0
        if not arrival_info['detected'] and (v_earth > v_background * 1.2 and n_earth > n_background * 2.0):
            arrival_info['detected'] = True
            arrival_info['time_hours'] = snap['t'] / 3600.0
            arrival_info['v_kms'] = v_earth
            if verbose:
                print(f"\n  *** CME ARRIVAL at t={snap['t'] / 3600:.2f}h ***")
                print(f'      v={v_earth:.0f} km/s  n={n_earth:.1f} cm-3')
                print(f'      Ballistic estimate: {cme.transit_time_hours():.1f}h\n')
    snapshots = solver.run(duration_seconds=sim_hours * 3600.0, output_every=output_every_h * 3600.0, callback=check_arrival, verbose=verbose, injector=injector)
    if verbose:
        if arrival_info['detected']:
            print(f"\nCME arrived at t = {arrival_info['time_hours']:.1f} hours")
        else:
            print(f'\nCME did not reach Earth within {sim_hours:g} hours')
    return {'snapshots': snapshots, 'cme': cme, 'grid': grid, 'arrival': arrival_info, 'solver': solver}

def run_cme_simulation_3d(cme: CMEParameters, grid_nr: int=256, grid_ntheta: int=90, grid_nphi: int=180, sim_hours: float=96.0, output_every_h: float=1.0, use_drag: bool=True, drag_coefficient: float=1e-07, relax_background_hours: float=12.0, slow_speed_ms: float=400000.0, fast_speed_ms: float=650000.0, verbose: bool=True) -> dict:
    from swcmep.solar_wind.background_3d import make_background_3d
    from swcmep.solar_wind.mhd_solver_gpu import MHDSolverGPU3D
    if verbose:
        print('=' * 60)
        print('SWCMEP CME Simulation (3D)')
        print('=' * 60)
        print(cme.summary())
        if use_drag:
            print(f'  Drag model    : enabled (gamma = {drag_coefficient:.1e} /km)')
    U0, grid = make_background_3d(nr=grid_nr, ntheta=grid_ntheta, nphi=grid_nphi, slow_speed_ms=slow_speed_ms, fast_speed_ms=fast_speed_ms)
    cfl = 0.3 if cme.speed_kms > 1500 else 0.4
    dtype = torch.float64 if cme.speed_kms > 1500 else torch.float32
    solver = MHDSolverGPU3D(grid=grid, U=U0.copy(), gamma=1.05, cfl_factor=cfl, dtype=dtype)
    if relax_background_hours > 0.0:
        if verbose:
            print(f'\nRelaxing 3D background for {relax_background_hours:.1f} h before CME injection...')
        solver.run(duration_seconds=relax_background_hours * 3600.0, output_every=max(3600.0, relax_background_hours * 3600.0), verbose=verbose, injector=None)
        U_relaxed = solver._U_gpu.detach().cpu().numpy().copy()
        solver.set_equilibrium_state(U_relaxed)
        solver.t = 0.0
        solver.step = 0
        solver.dt = 0.0
        solver.U = U_relaxed.copy()
        if verbose:
            print('Relaxation complete. Freezing as equilibrium reference.\n')
    else:
        U_relaxed = solver.U.copy()
    injector = ConeModelInjector3D(grid=grid, U_background=U_relaxed, cme=cme, drag_coefficient=drag_coefficient if use_drag else 0.0, gamma=solver.gamma, boundary_layers=solver._resolved_guard_counts()[0])
    i_e, j_e, k_e = grid.earth_index_3d()
    v_bg_earth = U_relaxed[I_VR, i_e, j_e, k_e] / 1000.0
    n_bg_earth = U_relaxed[I_RHO, i_e, j_e, k_e] / M_PROTON / 1000000.0
    arrival_info = {'detected': False, 'time_hours': None, 'v_kms': None, 'n_cm3': None}

    def check_arrival(slvr, snap):
        U = snap['U']
        v_earth = U[I_VR, i_e, j_e, k_e] / 1000.0
        n_earth = U[I_RHO, i_e, j_e, k_e] / M_PROTON / 1000000.0
        if not arrival_info['detected'] and (v_earth > v_bg_earth * 1.2 and n_earth > n_bg_earth * 2.0):
            arrival_info['detected'] = True
            arrival_info['time_hours'] = snap['t'] / 3600.0
            arrival_info['v_kms'] = v_earth
            arrival_info['n_cm3'] = n_earth
            if verbose:
                print(f"\n  *** CME ARRIVAL (3D) at t={snap['t'] / 3600:.2f}h ***")
                print(f'      v={v_earth:.0f} km/s  n={n_earth:.1f} cm⁻³')
                print(f'      Ballistic estimate: {cme.transit_time_hours():.1f}h\n')
    snapshots = solver.run(duration_seconds=sim_hours * 3600.0, output_every=output_every_h * 3600.0, callback=check_arrival, verbose=verbose, injector=injector)
    if verbose:
        if arrival_info['detected']:
            print(f"\nCME arrived at t = {arrival_info['time_hours']:.1f} hours  v = {arrival_info['v_kms']:.0f} km/s")
        else:
            print(f'\nCME did not reach Earth within {sim_hours:g} hours')
    return {'snapshots': snapshots, 'cme': cme, 'grid': grid, 'arrival': arrival_info, 'solver': solver}
