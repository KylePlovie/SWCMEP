import numpy as np
import pytest
import torch
from swcmep.cme.cone_model import CMEParameters, CompoundConeInjector3D, ConeModelInjector3D, ScheduledCME3D, run_cme_simulation
from swcmep.solar_wind.background_3d import make_background_3d
from swcmep.solar_wind.mhd_solver_gpu import MHDSolverGPU3D, SimulationError
from swcmep.utils.constants import I_BPHI, I_BTHETA, I_RHO, I_VR, I_VTHETA

@pytest.fixture
def solver():
    U, grid = make_background_3d(nr=16, ntheta=9, nphi=18)
    return MHDSolverGPU3D(grid, U, device='cpu', dtype=torch.float64, use_boundary_sponge=False, pole_guard_cells=1)

def injector_for(solver, **kwargs):
    return ConeModelInjector3D(solver.grid, solver.U.copy(), CMEParameters(**kwargs), drag_coefficient=0)

@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1])
@pytest.mark.parametrize('field', ['duration_seconds', 'output_every'])
def test_rejects_invalid_run_intervals(solver, field, value):
    args = {'duration_seconds': 1, 'output_every': 1, field: value}
    with pytest.raises(ValueError):
        solver.run(**args, verbose=False)
    assert solver.t == 0

def test_zero_duration_and_exact_output_times(solver):
    assert solver.run(0, verbose=False) == []
    assert solver.termination_reason == 'completed'
    snapshots = solver.run(20, output_every=7, verbose=False)
    assert [s['t'] for s in snapshots] == [7, 14, 20]
    assert solver.dt == 6
    assert solver.dt_cfl >= solver.dt
    seen = []
    retained = solver.run(10, output_every=4, verbose=False, retain_snapshots=False, callback=lambda _, snap: seen.append(snap['t']))
    assert retained == []
    assert seen == [24, 28, 30]

def test_full_run_preserves_reference_including_outer_boundary(solver):
    reference = solver.U.copy()
    solver.run(3600, output_every=900, retain_snapshots=False, verbose=False)
    scales = np.array([1e-18, 1000000.0, 1000000.0, 1e-08, 1e-06, 1e-06, 1, 1000000.0, 1e-06, 1e-06])
    np.testing.assert_allclose(solver.U / scales[:, None, None, None], reference / scales[:, None, None, None], rtol=1e-10, atol=1e-13)

def test_cfl_is_computed_after_boundary_injection(solver, monkeypatch):
    inj = injector_for(solver, speed_kms=2200, duration_hours=1)
    solver.t = 1800
    baseline = solver.compute_dt()
    observed = []
    original_step = solver._lax_friedrichs_step_gpu

    def step(U, dt):
        observed.append((solver.dt_cfl, float(U[I_VR, 2, 4, 0])))
        return original_step(U, dt)
    monkeypatch.setattr(solver, '_lax_friedrichs_step_gpu', step)
    solver.run(1, injector=inj, verbose=False)
    assert observed[0][0] < baseline
    assert observed[0][1] == 2200000.0

@pytest.mark.parametrize('scheduled', [False, True])
def test_short_pulse_is_not_skipped(solver, monkeypatch, scheduled):
    cme = CMEParameters(1200, duration_hours=2 / 3600, half_width_deg=30)
    if scheduled:
        inj = CompoundConeInjector3D(solver.grid, solver.U.copy(), [ScheduledCME3D(cme, start_seconds=3)])
    else:
        inj = ConeModelInjector3D(solver.grid, solver.U.copy(), cme, drag_coefficient=0)
    speeds = []
    original_step = solver._lax_friedrichs_step_gpu

    def step(U, dt):
        speeds.append(float(U[I_VR, 2, 4, 0]))
        return original_step(U, dt)
    monkeypatch.setattr(solver, '_lax_friedrichs_step_gpu', step)
    solver.run(8, injector=inj, verbose=False)
    assert max(speeds) == pytest.approx(cme.speed_ms)
    assert solver.termination_reason == 'completed'

@pytest.mark.parametrize('bad_value', [float('nan'), -1.0])
def test_rejected_step_does_not_advance_clock_or_publish_bad_state(solver, monkeypatch, bad_value):
    before = solver.U.copy()

    def broken_step(U, dt):
        result = U.clone()
        result[I_RHO, 5, 4, 0] = bad_value
        return result
    monkeypatch.setattr(solver, '_lax_friedrichs_step_gpu', broken_step)
    with pytest.raises(SimulationError):
        solver.run(10, verbose=False)
    assert solver.t == 0 and solver.step == 0
    np.testing.assert_array_equal(solver.U, before)
    assert solver.termination_reason in ('nonfinite_state', 'invalid_state')

def test_cfl_collapse_is_failure_but_short_requested_duration_is_valid(solver, monkeypatch):
    solver.run(1e-08, verbose=False)
    assert solver.t == 1e-08
    monkeypatch.setattr(solver, 'compute_dt', lambda: 1e-08)
    with pytest.raises(SimulationError, match='collapsed'):
        solver.run(1, verbose=False)
    assert solver.termination_reason == 'dt_collapse'
    assert solver.t == 1e-08

def test_drag_hook_runs_once_per_accepted_step(solver, monkeypatch):
    inj = injector_for(solver, speed_kms=900, duration_hours=1)
    calls = []
    monkeypatch.setattr(inj, 'apply_drag_gpu', lambda U, dt: calls.append(dt))
    solver.run(10, output_every=4, injector=inj, verbose=False)
    assert calls == [4, 4, 2]

@pytest.mark.parametrize('compound', [False, True])
def test_zero_envelope_and_outside_cone_preserve_ambient_field(solver, compound):
    background = solver.U.copy()
    background[I_VTHETA] = 50
    cme = CMEParameters(800, duration_hours=1, half_width_deg=20)
    if compound:
        inj = CompoundConeInjector3D(solver.grid, background, [ScheduledCME3D(cme)])
    else:
        inj = ConeModelInjector3D(solver.grid, background, cme, drag_coefficient=0)
    U = torch.tensor(background)
    inj.apply_boundary_gpu_3d(U, 0)
    np.testing.assert_allclose(U[:9, :3], background[:9, :3], rtol=0, atol=0)
    inj.apply_boundary_gpu_3d(U, 1800)
    np.testing.assert_allclose(U[I_BTHETA, :3], background[I_BTHETA, :3], rtol=0, atol=0)
    assert float(U[I_VTHETA, 2, 4, 9]) == 50
    assert float(U[I_VTHETA, 2, 4, 0]) == 0

def test_compound_winner_does_not_inherit_losing_magnetic_shell(solver):
    slow = CMEParameters(800, duration_hours=1, inject_magnetic_shell=True, magnetic_enhancement=5)
    fast = CMEParameters(1800, duration_hours=1)
    inj = CompoundConeInjector3D(solver.grid, solver.U.copy(), [ScheduledCME3D(slow), ScheduledCME3D(fast)])
    inj.apply_boundary_gpu_3d(solver._U_gpu, 1800)
    assert float(solver._U_gpu[I_VR, 2, 4, 0]) == fast.speed_ms
    assert float(solver._U_gpu[I_BPHI, 2, 4, 0]) == solver.U[I_BPHI, 2, 4, 0]

def test_compound_injects_density_when_speed_matches_background(solver):
    cme = CMEParameters(solver.U[I_VR, 2, 4, 0] / 1000.0, density_enhancement=3, duration_hours=1)
    inj = CompoundConeInjector3D(solver.grid, solver.U.copy(), [ScheduledCME3D(cme)])
    inj.apply_boundary_gpu_3d(solver._U_gpu, 1800)
    assert float(solver._U_gpu[I_RHO, 2, 4, 0]) == pytest.approx(solver.U[I_RHO, 2, 4, 0] * 3, abs=0)

def test_compound_rejects_unimplemented_drag(solver):
    with pytest.raises(ValueError, match='compound CME drag is not supported'):
        CompoundConeInjector3D(solver.grid, solver.U, [ScheduledCME3D(CMEParameters(800))], drag_coefficient=1e-07)

def test_2d_cpu_runner_uses_real_mhd():
    result = run_cme_simulation(CMEParameters(800), grid_nr=16, grid_nphi=18, sim_hours=0.001, relax_background_hours=0, use_gpu=False, use_drag=False, verbose=False)
    assert result['solver'].device.type == 'cpu'
    assert result['solver'].termination_reason == 'completed'
    assert result['solver'].is_valid()
    assert len(result['snapshots']) == 1

@pytest.mark.parametrize('field', ['speed_kms', 'duration_hours', 'temperature_K', 'density_enhancement', 'latitude_deg', 'taper_deg'])
def test_nan_cme_parameters_are_rejected(field):
    with pytest.raises(ValueError, match='finite'):
        CMEParameters(**{'speed_kms': 800, field: float('nan')})

def test_fractional_shell_count_is_rejected():
    with pytest.raises(ValueError, match='integer'):
        CMEParameters(800, shell_layers=2.5)

def test_nonfinite_schedule_is_rejected():
    with pytest.raises(ValueError, match='finite'):
        ScheduledCME3D(CMEParameters(800), float('nan'))
