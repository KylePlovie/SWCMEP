import io
import json
from pathlib import Path
import numpy as np
import pytest
from swcmep.ui.render import render_earth_timeseries, render_equatorial_frame

def test_equatorial_renderer_produces_png(tmp_path):
    frame = tmp_path / 'frame.npz'
    r = np.linspace(0.1, 1.1, 8, dtype=np.float32)
    phi = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False, dtype=np.float32)
    values = 400.0 + 200.0 * np.outer(r, np.cos(phi) ** 2)
    np.savez_compressed(frame, t_hours=np.asarray(24.0), utc=np.asarray('2026-07-10T11:57:00Z'), r_au=r, phi_rad=phi, vr_kms=values, n_cm3=np.maximum(values / 100.0, 0.1), b_nt=np.maximum(values / 80.0, 0.1))
    output = render_equatorial_frame(frame, 'vr_kms')
    assert isinstance(output, io.BytesIO)
    assert output.read(8) == b'\x89PNG\r\n\x1a\n'

def test_earth_renderer_produces_png(tmp_path):
    csv_path = tmp_path / 'earth.csv'
    csv_path.write_text('utc,t_hours,v_kms,n_cm3,b_nt,arrival_metric\n2026-07-09T12:57:00Z,1,400,5,7,0.5\n2026-07-10T12:57:00Z,25,650,12,10,1.2\n', encoding='utf-8')
    summary_path = tmp_path / 'summary.json'
    summary_path.write_text(json.dumps({'modeled_arrival_utc': '2026-07-10T12:00:00Z'}), encoding='utf-8')
    config_path = tmp_path / 'event.json'
    config_path.write_text(json.dumps({'event_name': 'Synthetic event', 'comparison': {'enlil_predicted_arrival_utc': '2026-07-10T14:00:00Z', 'observed_shock_arrival_utc': '2026-07-10T15:00:00Z'}}), encoding='utf-8')
    output = render_earth_timeseries(csv_path, summary_path, config_path)
    assert output.read(8) == b'\x89PNG\r\n\x1a\n'

def test_dashboard_config_validation(tmp_path):
    pytest.importorskip('flask')
    from swcmep.ui.app import build_event_config
    payload = {'event_name': 'Test CME', 'analysis_id': 'TEST-CME-001', 'time_21_5_utc': '2026-07-09T11:57:00Z', 'longitude_deg': 20, 'latitude_deg': -8, 'speed_kms': 749, 'half_width_deg': 31, 'pulse_duration_hours': 6, 'density_enhancement': 4, 'temperature_mk': 2, 'drag_coefficient': 0, 'simulation_hours': 84, 'relax_hours': 6, 'output_every_hours': 1, 'nr': 128, 'ntheta': 45, 'nphi': 90, 'dtype': 'float64', 'cfl': 0.25, 'spatial_order': 2, 'shock_flattening': True, 'shock_threshold': 2}
    config = build_event_config(payload, tmp_path)
    assert len(config['cmes']) == 1
    assert config['outputs']['save_equatorial_frames'] is True
    assert config['output_prefix'] == str(tmp_path / 'result')

def test_dashboard_rejects_even_theta_grid(tmp_path):
    pytest.importorskip('flask')
    from swcmep.ui.app import build_event_config
    payload = {'event_name': 'Test CME', 'analysis_id': 'TEST-CME-001', 'time_21_5_utc': '2026-07-09T11:57:00Z', 'longitude_deg': 0, 'latitude_deg': 0, 'speed_kms': 800, 'half_width_deg': 30, 'pulse_duration_hours': 1, 'density_enhancement': 2, 'temperature_mk': 2, 'drag_coefficient': 0, 'simulation_hours': 24, 'relax_hours': 1, 'output_every_hours': 1, 'nr': 32, 'ntheta': 16, 'nphi': 30, 'dtype': 'float64', 'cfl': 0.25, 'spatial_order': 2, 'shock_flattening': True, 'shock_threshold': 2}
    with pytest.raises(ValueError, match='ntheta must be odd'):
        build_event_config(payload, tmp_path)

def test_dashboard_runner_falls_back_to_general_event_runner(tmp_path):
    pytest.importorskip('flask')
    from swcmep.ui.app import RunManager
    missing = tmp_path / 'run_cme_case_3d.py'
    fallback = tmp_path / 'run_compound_cme_3d.py'
    fallback.write_text('# test runner\n', encoding='utf-8')
    manager = RunManager(tmp_path / 'runs', runner_candidates=(missing, fallback))
    assert manager._runner() == fallback

def test_dashboard_reports_missing_runner_before_launch(tmp_path):
    pytest.importorskip('flask')
    from swcmep.ui.app import RunManager
    missing = Path(tmp_path / 'run_cme_case_3d.py')
    manager = RunManager(tmp_path / 'runs', runner_candidates=(missing,))
    with pytest.raises(ValueError, match='CME runner is missing'):
        manager._runner()

def test_dashboard_forces_utf8_for_redirected_solver_log(tmp_path):
    pytest.importorskip('flask')
    from swcmep.ui.app import _solver_environment
    env = _solver_environment(tmp_path)
    assert env['PYTHONUTF8'] == '1'
    assert env['PYTHONIOENCODING'] == 'utf-8'
    assert env['PYTHONUNBUFFERED'] == '1'
    assert env['MPLCONFIGDIR'] == str(tmp_path / '.matplotlib')
