import csv
import json
from pathlib import Path
import numpy as np
import pytest
from swcmep.event_runner import run_event
from swcmep.solar_wind.mhd_solver_gpu import MHDSolverGPU3D

@pytest.fixture
def config(tmp_path):
    path = Path(__file__).parents[2] / 'data/events/2026-07-09T0753_CME-001.json'
    config = json.loads(path.read_text())
    config.update(grid={'nr': 32, 'ntheta': 9, 'nphi': 18}, simulation_hours=0.003, relax_hours=0, output_every_hours=0.001, output_prefix=str(tmp_path / 'event'), outputs={'save_equatorial_frames': True})
    return config

def test_real_event_exports_exact_times_and_finite_frames(config, tmp_path):
    result = run_event(config)
    assert result['termination_reason'] == 'completed'
    assert result['completed_hours'] == pytest.approx(0.003)
    rows = list(csv.DictReader((tmp_path / 'event_earth.csv').open()))
    assert [float(row['t_hours']) for row in rows] == pytest.approx([0, 0.001, 0.002, 0.003])
    frames = json.loads((tmp_path / 'event_frames/manifest.json').read_text())
    assert len(frames) == 4
    for frame in frames:
        with np.load(tmp_path / 'event_frames' / frame['file'], allow_pickle=False) as data:
            assert np.isfinite(data['vr_kms']).all()
            assert data['vr_kms'].shape == (32, 18)

def test_failed_run_preserves_partial_outputs_and_raises(config, tmp_path, monkeypatch):
    original = MHDSolverGPU3D._lax_friedrichs_step_gpu

    def fail_third_step(self, U, dt):
        if self.step == 2:
            raise RuntimeError('synthetic numerical failure')
        return original(self, U, dt)
    monkeypatch.setattr(MHDSolverGPU3D, '_lax_friedrichs_step_gpu', fail_third_step)
    with pytest.raises(RuntimeError, match='synthetic numerical failure'):
        run_event(config)
    summary = json.loads((tmp_path / 'event_summary.json').read_text())
    assert summary['termination_reason'] == 'error'
    assert summary['completed_hours'] == pytest.approx(0.002)
    rows = list(csv.DictReader((tmp_path / 'event_earth.csv').open()))
    assert len(rows) == 3
    assert float(rows[-1]['t_hours']) == pytest.approx(summary['completed_hours'])
