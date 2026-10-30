import json
import pytest
from swcmep.divergence_audit import run_audit
from swcmep.solar_wind.mhd_solver_gpu import MHDSolverGPU3D, SimulationError

@pytest.mark.parametrize('scenario', ['background', 'cme'])
def test_audit_samples_current_state_and_records_initial_faces_only(tmp_path, scenario):
    path = tmp_path / 'audit.json'
    report = run_audit(path, nr=16, ntheta=9, nphi=18, hours=0.005, output_hours=0.005, scenario=scenario)
    assert report == json.loads(path.read_text())
    assert report['status'] == 'completed'
    assert report['completed_seconds'] == 18.0
    assert len(report['samples']) == 2
    first, last = report['samples']
    assert first['time_seconds'] == 0.0
    assert last['time_seconds'] == 18.0
    assert last['operator'] == 'cell_centered'
    if scenario == 'background':
        assert last['active']['max'] == pytest.approx(first['active']['max'], abs=1e-13)
    else:
        assert last['max_radial_speed_change_m_per_s'] > 0
        assert last['max_magnetic_component_change_T'] > 0

def test_audit_preserves_partial_measurements_on_numerical_failure(tmp_path, monkeypatch):

    def fail(self):
        return 0.0
    monkeypatch.setattr(MHDSolverGPU3D, 'compute_dt', fail)
    path = tmp_path / 'failed.json'
    with pytest.raises(SimulationError, match='collapsed'):
        run_audit(path, nr=16, ntheta=9, nphi=18, hours=0.005)
    report = json.loads(path.read_text())
    assert report['status'] == 'failed'
    assert report['termination_reason'] == 'dt_collapse'
    assert report['completed_seconds'] == 0.0
    assert len(report['samples']) == 1
