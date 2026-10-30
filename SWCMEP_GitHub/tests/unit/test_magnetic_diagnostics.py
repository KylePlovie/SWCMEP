import numpy as np
import pytest
import torch
from swcmep.grid.spherical_grid import SphericalGrid3D
from swcmep.solar_wind.ct_mhd_3d import compute_divB_face
from swcmep.solar_wind.magnetic_diagnostics import centered_divergence_fields, face_divergence_fields, summarize_divergence

def face_case(nphi=12, scale=1.0):
    re = torch.tensor([1.0, 1.3, 2.0, 3.0, 5.0], dtype=torch.float64) * scale
    te = torch.linspace(0.0, torch.pi, 8, dtype=torch.float64)
    br = torch.zeros(5, 7, nphi, dtype=torch.float64)
    btheta = torch.zeros(4, 8, nphi, dtype=torch.float64)
    bphi = torch.zeros(4, 7, nphi, dtype=torch.float64)
    return (br, btheta, bphi, re, te, 2.0 * np.pi / nphi)

@pytest.mark.parametrize('scale', [1.0, 150000000000.0])
def test_radial_linear_field_has_exact_divergence_three_times_slope(scale):
    br, bt, bp, re, te, dphi = face_case(scale=scale)
    slope = 2e-09 / scale
    br[:] = slope * re[:, None, None]
    fields = face_divergence_fields(br, bt, bp, re, te, dphi)
    torch.testing.assert_close(fields['divergence'], torch.full_like(bp, 3.0 * slope), rtol=2e-14, atol=0)

def test_inverse_square_radial_field_is_divergence_free():
    br, bt, bp, re, te, dphi = face_case(scale=150000000000.0)
    br[:] = 5e-09 * (re[0] / re[:, None, None]) ** 2
    fields = face_divergence_fields(br, bt, bp, re, te, dphi)
    assert float(fields['normalized'].max()) < 1e-14

def test_normalized_divergence_is_invariant_to_length_units():
    case = face_case()
    br, bt, bp, re, te, dphi = case
    br[:] = re[:, None, None]
    original = face_divergence_fields(*case)
    scaled = face_divergence_fields(br, bt, bp, re * 150000000000.0, te, dphi)
    torch.testing.assert_close(scaled['normalized'], original['normalized'], rtol=2e-14, atol=0)
    torch.testing.assert_close(scaled['divergence'] * 150000000000.0, original['divergence'], rtol=2e-14, atol=0)

def test_theta_flux_uses_spherical_face_area():
    br, bt, bp, re, _, dphi = face_case()
    te = torch.linspace(0.2, 2.8, 8, dtype=torch.float64)
    bt[:] = torch.tan(te / 2)[None, :, None]
    fields = face_divergence_fields(br, bt, bp, re, te, dphi)
    expected = 1.5 * (re[1:] ** 2 - re[:-1] ** 2) / (re[1:] ** 3 - re[:-1] ** 3)
    torch.testing.assert_close(fields['theta'], expected[:, None, None].expand_as(bp))

@pytest.mark.parametrize('nphi', [11, 12])
def test_phi_faces_wrap_with_left_face_convention(nphi):
    br, bt, bp, re, te, dphi = face_case(nphi)
    phi_edges = (torch.arange(nphi, dtype=torch.float64) - 0.5) * dphi
    bp[:] = torch.sin(phi_edges)
    fields = face_divergence_fields(br, bt, bp, re, te, dphi)
    assert bool((fields['phi'][..., 0] > 0).all())
    torch.testing.assert_close(fields['phi'].sum(dim=2), torch.zeros_like(bp[:, :, 0]), atol=1e-14, rtol=0)
    rolled = face_divergence_fields(br, bt, torch.roll(bp, 3, dims=2), re, te, dphi)
    torch.testing.assert_close(rolled['divergence'], torch.roll(fields['divergence'], 3, dims=2))

def test_normalization_uses_staggered_phi_average_and_shortest_width():
    br, bt, bp, re, te, dphi = face_case(nphi=120)
    bp[:] = torch.arange(120, dtype=torch.float64) + 1
    fields = face_divergence_fields(br, bt, bp, re, te, dphi)
    expected_magnitude = 0.5 * (bp + torch.roll(bp, -1, dims=2))
    torch.testing.assert_close(fields['magnetic_magnitude'], expected_magnitude)
    rc = 0.5 * (re[:-1] + re[1:])
    theta0 = 0.5 * (te[0] + te[1])
    expected_width = rc * torch.sin(theta0) * dphi
    torch.testing.assert_close(fields['length_scale'][:, 0, 0], expected_width)

def test_legacy_face_api_uses_corrected_measurement():
    case = face_case(scale=150000000000.0)
    br, bt, bp, re, te, dphi = case
    br[:] = 1e-20 * re[:, None, None]
    fields = face_divergence_fields(*case)
    result = compute_divB_face(br, bt, bp, re[:, None, None], te, torch.diff(re), te[1] - te[0], dphi)
    assert result == (float(fields['normalized'].max()), float(fields['normalized'].mean()))

def test_centered_phi_derivative_uses_current_cell_locations():
    grid = SphericalGrid3D(nr=8, ntheta=9, nphi=18)
    br = torch.as_tensor(1.0 / grid.R3 ** 2).clone()
    bt = torch.zeros_like(br)
    bp = torch.as_tensor(np.sin(grid.PHI3)).clone()
    fields = centered_divergence_fields(br, bt, bp, grid.r_centers, grid.theta_centers, grid.dr, grid.dtheta, grid.dphi)
    expected = np.cos(grid.PHI3) * np.sin(grid.dphi) / grid.dphi / (grid.R3 * grid.SINT)
    torch.testing.assert_close(fields['phi'], torch.as_tensor(expected[1:-1, 1:-1]))
    assert float(fields['radial'].abs().max()) < 1e-45

def test_summary_maps_cropped_peak_back_to_original_grid():
    grid = SphericalGrid3D(nr=8, ntheta=9, nphi=18)
    shape = (6, 7, 18)
    values = torch.zeros(shape, dtype=torch.float64)
    values[2, 3, 17] = 7.0
    fields = {key: values.clone() for key in ('normalized', 'divergence', 'radial', 'theta', 'phi', 'magnetic_magnitude', 'length_scale')}
    result = summarize_divergence(fields, grid, region=(slice(1, 4), slice(2, 6), slice(None)), offset=(1, 1, 0))
    assert result['max'] == 7.0
    assert result['worst_cell']['index'] == [3, 4, 17]
    assert result['cell_count'] == 3 * 4 * 18

def test_solver_diagnoses_current_cells_without_using_stale_faces():
    from swcmep.solar_wind.background_3d import make_background_3d
    from swcmep.solar_wind.mhd_solver_gpu import MHDSolverGPU3D
    from swcmep.utils.constants import I_BPHI
    state, grid = make_background_3d(nr=16, ntheta=9, nphi=18)
    solver = MHDSolverGPU3D(grid, state, device='cpu', dtype=torch.float64)
    before = solver.divergence_report()
    faces_before = solver._Bphi_face.clone()
    solver._U_gpu[I_BPHI, 5, 4] += 1e-06 * torch.as_tensor(np.sin(grid.phi_centers))
    after = solver.divergence_report()
    assert after['active']['mean'] != pytest.approx(before['active']['mean'])
    assert solver.compute_divB() == (after['active']['max'], after['active']['mean'])
    torch.testing.assert_close(solver._Bphi_face, faces_before, rtol=0, atol=0)
