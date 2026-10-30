import numpy as np
import torch
from swcmep.cme.cone_model import CMEParameters, CompoundConeInjector3D, ConeModelInjector3D, ScheduledCME3D
from swcmep.solar_wind.background_2d import I_BPHI, I_VPHI
from swcmep.solar_wind.background_3d import make_background_3d
from swcmep.solar_wind.hlle_muscl import apply_pole_bcs_3d, prim_to_cons_3d, source_terms_3d
from swcmep.solar_wind.mhd_solver_gpu import MHDSolverGPU3D
from swcmep.utils.constants import I_BR, I_BTHETA, I_PSI, I_VR, I_VTHETA

def _small_solver(dtype=torch.float64, spatial_order=2, shock_flattening=False):
    U0, grid = make_background_3d(nr=16, ntheta=9, nphi=18)
    solver = MHDSolverGPU3D(grid=grid, U=U0, device='cpu', dtype=dtype, cfl_factor=0.2, inner_guard_cells=3, outer_guard_cells=3, pole_guard_cells=3, spatial_order=spatial_order, shock_flattening=shock_flattening)
    return solver

def test_default_background_uses_inertial_velocity_frame():
    U0, _ = make_background_3d(nr=16, ntheta=9, nphi=18)
    assert np.max(np.abs(U0[I_VPHI])) == 0.0
    assert np.max(np.abs(U0[I_BPHI])) > 0.0

def test_parker_spiral_uses_spherical_rotation_lever_arm():
    from swcmep.utils.constants import OMEGA_SUN
    U0, grid = make_background_3d(nr=16, ntheta=9, nphi=18)
    expected = -OMEGA_SUN * grid.R3 * grid.SINT / U0[I_VR]
    mask = np.abs(U0[I_BR]) > 1e-30
    np.testing.assert_allclose(U0[I_BPHI][mask] / U0[I_BR][mask], expected[mask], rtol=1e-13)

def test_smoke_grid_earth_monitor_is_outside_outer_guard():
    U0, grid = make_background_3d(nr=32, ntheta=9, nphi=18)
    solver = MHDSolverGPU3D(grid=grid, U=U0, device='cpu', dtype=torch.float64, cfl_factor=0.2, inner_guard_cells=3, outer_guard_cells=3, pole_guard_cells=3)
    _, outer_guard, _ = solver._resolved_guard_counts()
    i_earth, _, _ = solver.grid.earth_index_3d()
    assert i_earth < solver.grid.nr - outer_guard

def test_equilibrium_residual_is_stationary_to_roundoff():
    solver = _small_solver()
    U_before = solver._U_gpu.clone()
    dt = min(solver.compute_dt(), 1.0)
    U_after = solver._lax_friedrichs_step_gpu(U_before, dt)
    torch.testing.assert_close(U_after, U_before, rtol=2e-11, atol=1e-25)

def test_theta_momentum_is_not_globally_erased():
    solver = _small_solver()
    i, j, k = (5, 4, 3)
    solver._U_gpu[I_VTHETA, i, j, k] = 2000.0
    dt = min(solver.compute_dt(), 0.001)
    U_after = solver._lax_friedrichs_step_gpu(solver._U_gpu, dt)
    assert abs(float(U_after[I_VTHETA, i, j, k])) > 1000.0

def test_pole_bc_does_not_zero_adjacent_row():
    U = torch.zeros((10, 5, 7, 4), dtype=torch.float64)
    U[I_VTHETA, :, 1, :] = 2000.0
    U[I_VTHETA, :, -2, :] = -2000.0
    apply_pole_bcs_3d(U)
    assert torch.all(U[I_VTHETA, :, 1, :] == 2000.0)
    assert torch.all(U[I_VTHETA, :, -2, :] == -2000.0)

def test_glm_metric_sources_match_spherical_gradient():
    gamma = 1.05
    nr, ntheta, nphi = (5, 7, 4)
    U = torch.zeros((10, nr, ntheta, nphi), dtype=torch.float64)
    U[0] = 1.0
    U[3] = 1.0
    U[9] = 1.0 / (gamma - 1.0)
    U[I_PSI] = 2.0
    W = prim_to_cons_3d(U, gamma)
    r = torch.linspace(1.0, 2.0, nr, dtype=torch.float64).reshape(-1, 1, 1)
    theta = (torch.arange(ntheta, dtype=torch.float64) + 0.5) * (torch.pi / ntheta)
    cot = (torch.cos(theta) / torch.sin(theta)).reshape(1, ntheta, 1)
    zero_gravity = torch.zeros_like(r)
    S = source_terms_3d(W, r, cot, zero_gravity, gamma, c_h=1.0, ct_active=False, apply_powell=False)
    torch.testing.assert_close(S[I_BR], (4.0 / r).expand_as(S[I_BR]))
    torch.testing.assert_close(S[I_BTHETA], (2.0 * cot / r).expand_as(S[I_BTHETA]))

def test_cfl_responds_to_alfven_speed_without_cap():
    solver = _small_solver()
    dt_background = solver.compute_dt()
    solver._U_gpu[I_BR] *= 10.0
    solver._U_gpu[I_BPHI] *= 10.0
    solver._U_gpu[I_BTHETA] *= 10.0
    dt_strong_field = solver.compute_dt()
    assert dt_strong_field < 0.25 * dt_background

def test_injector_drives_first_active_cell_through_guard_face():
    solver = _small_solver()
    U_background = solver._U_gpu.cpu().numpy().copy()
    cme = CMEParameters(speed_kms=1200.0, half_width_deg=30.0, duration_hours=1.0, density_enhancement=2.0, shell_layers=4)
    injector = ConeModelInjector3D(grid=solver.grid, U_background=U_background, cme=cme, drag_coefficient=0.0, gamma=solver.gamma, boundary_layers=solver.inner_guard_cells)
    first_active, _, _ = solver._resolved_guard_counts()
    j = solver.grid.ntheta // 2
    k = 0
    v_before = float(solver._U_gpu[I_VR, first_active, j, k])
    injector.apply_boundary_gpu_3d(solver._U_gpu, 0.5 * 3600.0)
    assert float(solver._U_gpu[I_VR, first_active - 1, j, k]) == cme.speed_ms
    assert float(solver._U_gpu[I_VR, first_active, j, k]) == v_before
    dt = min(solver.compute_dt(), 1.0)
    U_after = solver._lax_friedrichs_step_gpu(solver._U_gpu, dt)
    assert float(U_after[I_VR, first_active, j, k]) > v_before

def test_first_order_hlle_keeps_strong_inflow_finite():
    solver = _small_solver(spatial_order=1)
    cme = CMEParameters(speed_kms=2125.0, half_width_deg=40.0, duration_hours=1.0, density_enhancement=4.0, shell_layers=3)
    injector = ConeModelInjector3D(grid=solver.grid, U_background=solver._U_gpu.cpu().numpy().copy(), cme=cme, drag_coefficient=0.0, gamma=solver.gamma, boundary_layers=solver.inner_guard_cells)
    injector.apply_boundary_gpu_3d(solver._U_gpu, 0.5 * 3600.0)
    dt = min(solver.compute_dt(), 1.0)
    U_after = solver._lax_friedrichs_step_gpu(solver._U_gpu, dt)
    assert bool(torch.isfinite(U_after).all())
    assert bool(torch.all(U_after[0] > 0.0))
    assert bool(torch.all(U_after[3] > 0.0))

def test_hybrid_hlle_keeps_strong_inflow_finite():
    solver = _small_solver(spatial_order=2, shock_flattening=True)
    cme = CMEParameters(speed_kms=2125.0, half_width_deg=40.0, duration_hours=1.0, density_enhancement=4.0, shell_layers=3)
    injector = ConeModelInjector3D(grid=solver.grid, U_background=solver._U_gpu.cpu().numpy().copy(), cme=cme, drag_coefficient=0.0, gamma=solver.gamma, boundary_layers=solver.inner_guard_cells)
    injector.apply_boundary_gpu_3d(solver._U_gpu, 0.5 * 3600.0)
    dt = min(solver.compute_dt(), 1.0)
    U_after = solver._lax_friedrichs_step_gpu(solver._U_gpu, dt)
    assert bool(torch.isfinite(U_after).all())
    assert bool(torch.all(U_after[0] > 0.0))
    assert bool(torch.all(U_after[3] > 0.0))

def test_compound_injector_preserves_fast_core_and_slow_flank():
    solver = _small_solver()
    background = solver._U_gpu.cpu().numpy().copy()
    fast = CMEParameters(speed_kms=1200.0, half_width_deg=20.0, longitude_deg=0.0, duration_hours=1.0, density_enhancement=4.0)
    slow_wide = CMEParameters(speed_kms=600.0, half_width_deg=60.0, longitude_deg=40.0, duration_hours=1.0, density_enhancement=4.0)
    compound = CompoundConeInjector3D(grid=solver.grid, U_background=background, scheduled_cmes=(ScheduledCME3D(fast, start_seconds=0.0), ScheduledCME3D(slow_wide, start_seconds=0.0)), gamma=solver.gamma, boundary_layers=solver.inner_guard_cells)
    first_active, _, _ = solver._resolved_guard_counts()
    guard = first_active - 1
    j_earth = solver.grid.ntheta // 2
    k_earth = solver.grid.phi_index(0.0)
    compound.apply_boundary_gpu_3d(solver._U_gpu, 0.5 * 3600.0)
    assert float(solver._U_gpu[I_VR, guard, j_earth, k_earth]) == fast.speed_ms
    fast_mask = compound._injectors[0]._cone_blend_np
    slow_mask = compound._injectors[1]._cone_blend_np
    slow_only = np.argwhere((slow_mask > 0.0) & (fast_mask == 0.0))[0]
    j_flank, k_flank = (int(slow_only[0]), int(slow_only[1]))
    v_background = background[I_VR, guard, j_flank, k_flank]
    v_after = float(solver._U_gpu[I_VR, guard, j_flank, k_flank])
    assert v_after > v_background
    assert v_after <= slow_wide.speed_ms
