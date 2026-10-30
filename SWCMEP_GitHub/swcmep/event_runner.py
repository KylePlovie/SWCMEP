from __future__ import annotations
import argparse
import csv
import io
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import torch
from swcmep.cme.cone_model import CMEParameters, ConeModelInjector3D
from swcmep.grid.spherical_grid import SphericalGrid3D
from swcmep.solar_wind.background_3d import make_background_3d
from swcmep.solar_wind.mhd_solver_gpu import MHDSolverGPU3D, SimulationError
from swcmep.utils.constants import AU, I_BPHI, I_BR, I_BTHETA, I_RHO, I_VR, M_PROTON, R_INNER
from swcmep.utils.io import atomic_write_json, atomic_write_text
from swcmep.utils.validation import positive_float
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'data' / 'events' / '2026-07-12T0212_CME-001.json'

def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError(f'UTC timestamp requires an offset or Z suffix: {value}')
    return parsed.astimezone(timezone.utc)

def format_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')

def angular_separation_deg(longitude_deg: float, latitude_deg: float) -> float:
    return math.degrees(math.acos(math.cos(math.radians(latitude_deg)) * math.cos(math.radians(longitude_deg))))

def _comparison_error_hours(predicted: datetime, reference: datetime) -> float:
    return (predicted - reference).total_seconds() / 3600.0

def _run_without_retaining_states(solver, **kwargs):
    expected_end = solver.t + kwargs['duration_seconds']
    result = solver.run(retain_snapshots=False, **kwargs)
    reason = getattr(solver, 'termination_reason', 'completed')
    if reason != 'completed' or solver.t < expected_end - 1e-06:
        raise SimulationError(f'simulation did not finish: {reason}, t={solver.t:g}s')
    return result
EARTH_SERIES_FIELDS = ('utc', 't_hours', 'v_kms', 'n_cm3', 'b_nt', 'arrival_metric')

def _save_equatorial_frame(path: Path, U: np.ndarray, grid: SphericalGrid3D, t_hours: float, utc: str, vr_scale_max: float) -> None:
    j_eq = grid.equatorial_slice()
    br = U[I_BR, :, j_eq, :]
    bphi = U[I_BPHI, :, j_eq, :]
    btheta = U[I_BTHETA, :, j_eq, :]
    np.savez_compressed(path, t_hours=np.asarray(t_hours, dtype=np.float64), utc=np.asarray(utc), vr_scale_max=np.asarray(vr_scale_max, dtype=np.float32), r_au=np.asarray(grid.r_centers / AU, dtype=np.float32), phi_rad=np.asarray(grid.phi_centers, dtype=np.float32), vr_kms=np.asarray(U[I_VR, :, j_eq, :] / 1000.0, dtype=np.float32), n_cm3=np.asarray(U[I_RHO, :, j_eq, :] / M_PROTON / 1000000.0, dtype=np.float32), b_nt=np.asarray(np.sqrt(br * br + bphi * bphi + btheta * btheta) * 1000000000.0, dtype=np.float32))

def run_event(config: dict, smoke: bool=False) -> dict:
    name = config['event_name']
    epoch = parse_utc(config['simulation_start_utc'])
    grid_cfg = dict(config['grid'])
    solver_cfg = dict(config['solver'])
    sim_hours = positive_float(config['simulation_hours'], 'simulation_hours')
    relax_hours = positive_float(config['relax_hours'], 'relax_hours', allow_zero=True)
    output_every_h = positive_float(config.get('output_every_hours', 1.0), 'output_every_hours')
    output_cfg = dict(config.get('outputs', {}))
    save_equatorial = bool(output_cfg.get('save_equatorial_frames', False))
    if sim_hours <= 0.0 or relax_hours < 0.0 or output_every_h <= 0.0:
        raise ValueError('simulation_hours and output_every_hours must be positive; relax_hours must be non-negative')
    if smoke:
        grid_cfg = {'nr': 32, 'ntheta': 15, 'nphi': 30}
        sim_hours = 12.0
        relax_hours = 0.25
    prefix = Path(config['output_prefix'])
    if smoke:
        prefix = prefix.with_name(prefix.name + '_smoke')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = prefix.with_name(prefix.name + '_frames')
    frame_manifest_path = frames_dir / 'manifest.json'
    frame_manifest: list[dict] = []
    if save_equatorial:
        frames_dir.mkdir(parents=True, exist_ok=True)
    print('=' * 72)
    print(f'SWCMEP 3D event — {name}')
    print('=' * 72)
    print(f'  Epoch       : {format_utc(epoch)} at 21.5 R_sun')
    print(f"  Grid        : {grid_cfg['nr']}r x {grid_cfg['ntheta']}theta x {grid_cfg['nphi']}phi")
    print(f'  Duration    : {sim_hours:.1f}h')
    print(f"  Scheme      : order {solver_cfg['spatial_order']}, shock flattening={solver_cfg['shock_flattening']}, threshold={solver_cfg['shock_threshold']}x")
    print()
    assumptions = config['assumptions']
    print('Unlisted ENLIL inputs represented by baseline assumptions:')
    print(f"  Pulse duration : {assumptions['duration_hours']:.1f}h")
    print(f"  Density factor : {assumptions['density_enhancement']:.1f}x")
    print(f"  Temperature    : {assumptions['temperature_K'] / 1000000.0:.2f} MK")
    print()
    event_times = [parse_utc(item['time_21_5_utc']) for item in config['cmes']]
    if not event_times:
        raise ValueError('at least one CME is required')
    if min(event_times) < epoch:
        raise ValueError('simulation epoch must not be later than a CME start')
    scheduled = []
    for item, start_utc in zip(config['cmes'], event_times):
        offset_s = (start_utc - epoch).total_seconds()
        cme = CMEParameters(speed_kms=float(item['speed_kms']), half_width_deg=float(item['half_width_deg']), longitude_deg=float(item['longitude_deg']), latitude_deg=float(item['latitude_deg']), duration_hours=float(item.get('duration_hours', assumptions['duration_hours'])), density_enhancement=float(item.get('density_enhancement', assumptions['density_enhancement'])), temperature_K=float(item.get('temperature_K', assumptions['temperature_K'])), shell_layers=3, name=item['analysis_id'])
        separation = angular_separation_deg(cme.longitude_deg, cme.latitude_deg)
        print(f'CME: {cme.name}')
        print(f'  Speed              : {cme.speed_kms:.0f}km/s')
        print(f'  Half-width         : {cme.half_width_deg:.1f}deg')
        print(f'  Longitude          : {cme.longitude_deg:+.1f}deg')
        print(f'  Latitude           : {cme.latitude_deg:+.1f}deg')
        print(f'  Assumed duration   : {cme.duration_hours:.1f}h')
        print(f'  Assumed density    : {cme.density_enhancement:.1f}x')
        print(f'  Assumed temperature: {cme.temperature_K / 1000000.0:.2f}MK')
        print(f'  Time at 21.5 R_sun : {format_utc(start_utc)}')
        print(f'  Injection offset   : +{offset_s / 3600.0:.2f}h')
        print(f'  Earth separation   : {separation:.2f}deg')
        print(f'  Cone-edge margin   : {cme.half_width_deg - separation:+.2f}deg')
        transit_h = (AU - R_INNER) / cme.speed_ms / 3600.0
        print(f'  21.5 R_sun ballistic: {transit_h:.2f}h')
        print(f'  Ballistic Earth UTC: {format_utc(start_utc + timedelta(hours=transit_h))}')
        print()
        scheduled.append((cme, offset_s))
    visual_speed_max = max(1000.0, 1.1 * max((cme.speed_kms for cme, _ in scheduled)))
    U0, solver_grid = make_background_3d(**grid_cfg)
    dtype_name = solver_cfg.get('dtype', 'float64')
    if dtype_name not in {'float32', 'float64'}:
        raise ValueError("solver.dtype must be 'float32' or 'float64'")
    dtype = torch.float64 if dtype_name == 'float64' else torch.float32
    solver = MHDSolverGPU3D(grid=solver_grid, U=U0.copy(), gamma=1.05, cfl_factor=float(solver_cfg['cfl']), dtype=dtype, spatial_order=int(solver_cfg['spatial_order']), shock_flattening=bool(solver_cfg['shock_flattening']), shock_threshold=float(solver_cfg['shock_threshold']))
    print(f'Relaxing background for {relax_hours:.2f}h...')
    relax_wall = time.time()
    _run_without_retaining_states(solver, duration_seconds=relax_hours * 3600.0, output_every=max(relax_hours * 3600.0, 1.0), verbose=False)
    solver.set_equilibrium_state(solver.U.copy())
    solver.t = 0.0
    solver.step = 0
    solver.dt = 0.0
    relaxed_background = solver.U.copy()
    print(f'  Done in {time.time() - relax_wall:.1f}s')
    if save_equatorial:
        initial_frame = frames_dir / 'frame_0000.npz'
        initial_utc = format_utc(epoch)
        _save_equatorial_frame(initial_frame, relaxed_background, solver_grid, 0.0, initial_utc, visual_speed_max)
        frame_manifest.append({'index': 0, 'file': initial_frame.name, 't_hours': 0.0, 'utc': initial_utc})
        atomic_write_json(frame_manifest_path, frame_manifest)
    injector_kwargs = {'grid': solver_grid, 'U_background': relaxed_background, 'drag_coefficient': float(assumptions.get('drag_coefficient', 0.0)), 'gamma': solver.gamma, 'boundary_layers': solver._resolved_guard_counts()[0]}
    if len(scheduled) == 1 and scheduled[0][1] == 0.0:
        injector = ConeModelInjector3D(cme=scheduled[0][0], **injector_kwargs)
        run_label = 'single-CME'
    else:
        from swcmep.cme.cone_model import CompoundConeInjector3D, ScheduledCME3D
        injector = CompoundConeInjector3D(scheduled_cmes=[ScheduledCME3D(cme=cme, start_seconds=offset_s) for cme, offset_s in scheduled], **injector_kwargs)
        run_label = 'multi-CME'
    i_earth, j_earth, k_earth = solver_grid.earth_index_3d()
    v_bg = relaxed_background[I_VR, i_earth, j_earth, k_earth] / 1000.0
    n_bg = relaxed_background[I_RHO, i_earth, j_earth, k_earth] / M_PROTON / 1000000.0
    previous_t_h = 0.0
    previous_metric = min(1.0 / 1.2, 1.0 / 2.0)
    arrival_time_h = None
    b_bg = float(np.linalg.norm(relaxed_background[[I_BR, I_BPHI, I_BTHETA], i_earth, j_earth, k_earth]) * 1000000000.0)
    series = [{'utc': format_utc(epoch), 't_hours': 0.0, 'v_kms': float(v_bg), 'n_cm3': float(n_bg), 'b_nt': b_bg, 'arrival_metric': previous_metric}]
    csv_path = prefix.with_name(prefix.name + '_earth.csv')
    json_path = prefix.with_name(prefix.name + '_summary.json')

    def save_series():
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=EARTH_SERIES_FIELDS)
        writer.writeheader()
        writer.writerows(series)
        atomic_write_text(csv_path, buffer.getvalue())
    save_series()
    peaks = {'v_kms': float(v_bg), 'n_cm3': float(n_bg), 'b_nt': b_bg}

    def callback(slv, snapshot):
        nonlocal previous_t_h, previous_metric, arrival_time_h
        if not slv.is_valid():
            raise FloatingPointError(f"invalid MHD state at t={snapshot['t'] / 3600.0:.2f}h")
        U = snapshot['U']
        t_h = snapshot['t'] / 3600.0
        stamp = epoch + timedelta(hours=t_h)
        v_earth = U[I_VR, i_earth, j_earth, k_earth] / 1000.0
        n_earth = U[I_RHO, i_earth, j_earth, k_earth] / M_PROTON / 1000000.0
        b_earth = np.sqrt(U[I_BR, i_earth, j_earth, k_earth] ** 2 + U[I_BPHI, i_earth, j_earth, k_earth] ** 2 + U[I_BTHETA, i_earth, j_earth, k_earth] ** 2) * 1000000000.0
        metric = min(v_earth / (1.2 * v_bg), n_earth / (2.0 * n_bg))
        if arrival_time_h is None and metric >= 1.0:
            span = metric - previous_metric
            fraction = np.clip((1.0 - previous_metric) / span, 0.0, 1.0) if span > 1e-12 else 1.0
            arrival_time_h = previous_t_h + fraction * (t_h - previous_t_h)
            arrival_utc = epoch + timedelta(hours=arrival_time_h)
            print(f'\n  *** MODELED EARTH SHOCK: {format_utc(arrival_utc)} ***')
            print(f'      interpolated t={arrival_time_h:.2f}h\n')
        peaks['v_kms'] = max(peaks['v_kms'], float(v_earth))
        peaks['n_cm3'] = max(peaks['n_cm3'], float(n_earth))
        peaks['b_nt'] = max(peaks['b_nt'], float(b_earth))
        series.append({'utc': format_utc(stamp), 't_hours': t_h, 'v_kms': float(v_earth), 'n_cm3': float(n_earth), 'b_nt': float(b_earth), 'arrival_metric': float(metric)})
        save_series()
        if save_equatorial:
            frame_index = len(frame_manifest)
            frame_path = frames_dir / f'frame_{frame_index:04d}.npz'
            _save_equatorial_frame(frame_path, U, solver_grid, t_h, format_utc(stamp), visual_speed_max)
            frame_manifest.append({'index': frame_index, 'file': frame_path.name, 't_hours': float(t_h), 'utc': format_utc(stamp)})
            atomic_write_json(frame_manifest_path, frame_manifest)
        print(f'  [earth] {format_utc(stamp)}  t={t_h:6.1f}h  v={v_earth:6.0f}km/s  n={n_earth:7.2f}cm^-3  |B|={b_earth:6.2f}nT  metric={metric:5.2f}')
        previous_t_h = t_h
        previous_metric = metric
    print(f'\nRunning {run_label} simulation for {sim_hours:.1f}h...')
    run_wall = time.time()
    run_error = None
    try:
        _run_without_retaining_states(solver, duration_seconds=sim_hours * 3600.0, output_every=output_every_h * 3600.0, callback=callback, injector=injector, verbose=False)
    except Exception as exc:
        run_error = exc
    wall_seconds = time.time() - run_wall
    modeled_arrival = epoch + timedelta(hours=arrival_time_h) if arrival_time_h is not None else None
    comparison = config.get('comparison', {})
    enlil_value = comparison.get('enlil_predicted_arrival_utc')
    observed_value = comparison.get('observed_shock_arrival_utc')
    enlil_arrival = parse_utc(enlil_value) if enlil_value else None
    observed_arrival = parse_utc(observed_value) if observed_value else None
    result = {'event_name': name, 'simulation_start_utc': format_utc(epoch), 'termination_reason': getattr(solver, 'termination_reason', None) or ('error' if run_error else 'completed'), 'error': str(run_error) if run_error else None, 'completed_hours': solver.t / 3600.0, 'modeled_arrival_utc': format_utc(modeled_arrival), 'modeled_arrival_hours': arrival_time_h, 'enlil_predicted_arrival_utc': format_utc(enlil_arrival), 'observed_shock_arrival_utc': format_utc(observed_arrival), 'swcmep_error_hours': _comparison_error_hours(modeled_arrival, observed_arrival) if modeled_arrival and observed_arrival else None, 'enlil_error_hours': _comparison_error_hours(enlil_arrival, observed_arrival) if enlil_arrival and observed_arrival else None, 'swcmep_minus_enlil_hours': _comparison_error_hours(modeled_arrival, enlil_arrival) if modeled_arrival and enlil_arrival else None, 'peak_earth': peaks, 'wall_seconds': wall_seconds, 'assumptions': assumptions, 'comparison_input': comparison, 'visualization': {'equatorial_frames': str(frames_dir) if save_equatorial else None, 'frame_count': len(frame_manifest)}}
    save_series()
    atomic_write_json(json_path, result)
    print('\n' + '=' * 72)
    print('Event results')
    print('=' * 72)
    print(f"  Termination      : {getattr(solver, 'termination_reason', 'completed')}")
    print(f'  Wall time        : {wall_seconds:.1f}s')
    print(f"  SWCMEP arrival   : {format_utc(modeled_arrival) or 'no impact detected'}")
    enlil_uncertainty = comparison.get('enlil_arrival_uncertainty_hours')
    enlil_display = format_utc(enlil_arrival) or 'not supplied'
    if enlil_arrival and enlil_uncertainty is not None:
        enlil_display += f' (+/- {float(enlil_uncertainty):g}h)'
    print(f'  ENLIL arrival    : {enlil_display}')
    print(f"  Observed shock   : {format_utc(observed_arrival) or 'not supplied'}")
    if modeled_arrival and enlil_arrival:
        print(f'  SWCMEP - ENLIL   : {_comparison_error_hours(modeled_arrival, enlil_arrival):+.2f}h')
    print(f"  Peak at Earth    : v={peaks['v_kms']:.0f}km/s  n={peaks['n_cm3']:.2f}cm^-3  |B|={peaks['b_nt']:.2f}nT")
    print(f'  Time series      : {csv_path}')
    print(f'  Summary          : {json_path}')
    print('=' * 72)
    if run_error is not None:
        raise run_error
    return result

def main():
    parser = argparse.ArgumentParser(description='Run a config-driven single- or multi-CME 3D event')
    parser.add_argument('config', nargs='?', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--smoke', action='store_true', help='run the configured boundary on a small 12-hour grid')
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error('supply an existing event JSON config path')
    with args.config.open('r', encoding='utf-8') as handle:
        config = json.load(handle)
    run_event(config, smoke=args.smoke)
if __name__ == '__main__':
    main()
