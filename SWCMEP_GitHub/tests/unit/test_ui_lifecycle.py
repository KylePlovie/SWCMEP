import io
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
import pytest
from swcmep.ui.app import ActiveJob, RunManager, build_event_config, create_app
from swcmep.utils.io import atomic_write_json

@pytest.fixture
def payload():
    return {'event_name': 'Test', 'analysis_id': 'TEST-001', 'time_21_5_utc': '2026-07-09T11:57:00Z', 'longitude_deg': 0, 'latitude_deg': 0, 'speed_kms': 800, 'half_width_deg': 30, 'pulse_duration_hours': 1, 'density_enhancement': 2, 'temperature_mk': 2, 'drag_coefficient': 0, 'simulation_hours': 1, 'relax_hours': 0, 'output_every_hours': 1, 'nr': 32, 'ntheta': 9, 'nphi': 18, 'dtype': 'float64', 'cfl': 0.25, 'spatial_order': 2, 'shock_flattening': True, 'shock_threshold': 2}

@pytest.mark.parametrize('field,value', [('nr', 32.5), ('ntheta', 9.5), ('nr', float('inf')), ('nr', True), ('half_width_deg', 2), ('shock_flattening', 'false'), ('smoke', 'false'), ('speed_kms', float('nan'))])
def test_invalid_config_rejected_before_a_job_directory_exists(tmp_path, payload, field, value):
    manager = RunManager(tmp_path)
    payload[field] = value
    with pytest.raises(ValueError):
        manager.start(payload)
    assert list(tmp_path.iterdir()) == []

@pytest.mark.parametrize('reason,hours,expected', [('dt_collapse', 0.5, 'failed'), ('nonfinite_state', 0.5, 'failed'), ('completed', 0.5, 'failed'), ('completed', 1.0, 'completed')])
def test_status_checks_summary_and_duration_not_just_exit_code(tmp_path, payload, reason, hours, expected):
    manager = RunManager(tmp_path)
    job_id = manager._new_job_id()
    directory = tmp_path / job_id
    config = build_event_config(payload, directory)
    directory.mkdir()
    manager._write_metadata(directory, {'state': 'running', 'config': config})
    atomic_write_json(directory / 'result_summary.json', {'termination_reason': reason, 'completed_hours': hours})
    log = io.StringIO()
    manager._jobs[job_id] = ActiveJob(job_id, directory, SimpleNamespace(poll=lambda: 0), log)
    status = manager.status(job_id)
    assert status['state'] == expected
    assert status['completed_hours'] == hours
    assert log.closed
    assert manager.status(job_id)['state'] == expected

def test_restart_does_not_claim_an_unowned_process_is_running(tmp_path, payload):
    manager = RunManager(tmp_path)
    job_id = manager._new_job_id()
    directory = tmp_path / job_id
    directory.mkdir()
    manager._write_metadata(directory, {'state': 'running', 'config': build_event_config(payload, directory)})
    assert manager.status(job_id)['state'] == 'unmanaged'

def test_subprocess_launch_failure_closes_log_and_records_failure(tmp_path, payload, monkeypatch):
    manager = RunManager(tmp_path)
    captured = []

    def fail(*args, **kwargs):
        captured.append(kwargs['stdout'])
        raise OSError('test launch error')
    monkeypatch.setattr('swcmep.ui.app.subprocess.Popen', fail)
    with pytest.raises(ValueError, match='could not start solver'):
        manager.start(payload)
    assert captured[0].closed
    assert manager.list_jobs()[0]['state'] == 'failed'

def test_concurrent_launches_allow_only_one_process(tmp_path, payload, monkeypatch):
    manager = RunManager(tmp_path)
    starts = []

    def launch(*args, **kwargs):
        starts.append(args)
        return SimpleNamespace(poll=lambda: None, pid=123)
    monkeypatch.setattr('swcmep.ui.app.subprocess.Popen', launch)
    barrier = Barrier(2)

    def attempt():
        barrier.wait(timeout=5)
        try:
            return manager.start(payload)['state']
        except ValueError:
            return 'rejected'
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(attempt) for _ in range(2)]
        outcomes = sorted((f.result(timeout=10) for f in futures))
    assert outcomes == ['rejected', 'running']
    assert len(starts) == 1
    for job in manager._jobs.values():
        job.log_handle.close()

def test_unknown_job_image_is_404(tmp_path):
    client = create_app(tmp_path).test_client()
    response = client.get('/api/jobs/20260709T120000Z_deadbeef/frame/0.png')
    assert response.status_code == 404

def test_atomic_json_refuses_nan_without_replacing_prior_file(tmp_path):
    path = tmp_path / 'status.json'
    atomic_write_json(path, {'state': 'ok'})
    with pytest.raises(ValueError):
        atomic_write_json(path, {'time': float('nan')})
    assert json.loads(path.read_text()) == {'state': 'ok'}
