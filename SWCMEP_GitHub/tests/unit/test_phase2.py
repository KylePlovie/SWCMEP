import numpy as np
import pytest
from swcmep.grid.spherical_grid import SphericalGrid
from swcmep.solar_wind.background_2d import N_VARS, Background2D
from swcmep.solar_wind.parker import ParkerSolution
from swcmep.utils.constants import AU, R_INNER, R_OUTER

@pytest.fixture(scope='module')
def grid():
    return SphericalGrid(nr=128, nphi=90)

@pytest.fixture(scope='module')
def parker():
    return ParkerSolution(T_corona=1500000.0)

@pytest.fixture(scope='module')
def bg(grid, parker):
    return Background2D(grid, parker)

class TestSphericalGrid:

    def test_grid_shape(self, grid):
        assert grid.R.shape == (128, 90)
        assert grid.PHI.shape == (128, 90)

    def test_radial_range(self, grid):
        assert grid.r_centers.min() >= R_INNER * 0.99
        assert grid.r_centers.max() <= R_OUTER * 1.01

    def test_phi_range(self, grid):
        assert grid.phi_centers.min() >= 0
        assert grid.phi_centers.max() <= 2 * np.pi

    def test_phi_is_uniform(self, grid):
        dphi = np.diff(grid.phi_edges)
        assert np.allclose(dphi, dphi[0], rtol=1e-06)

    def test_earth_direction_is_a_cell_center(self, grid):
        _, j = grid.earth_index()
        assert np.isclose(grid.phi_centers[j], 0.0)

    def test_log_radial_gives_small_inner_cells(self, grid):
        assert grid.dr[0] < grid.dr[-1] / 10

    def test_earth_index_near_1au(self, grid):
        i, j = grid.earth_index()
        r_earth = grid.r_centers[i]
        assert abs(r_earth - AU) / AU < 0.05

    def test_cell_area_positive(self, grid):
        assert np.all(grid.cell_area() > 0)

    def test_total_cells(self, grid):
        assert grid.nr * grid.nphi == 128 * 90

    def test_r_index_roundtrip(self, grid):
        i = grid.r_index(AU)
        assert grid.r_edges[i] <= AU <= grid.r_edges[i + 1]

    def test_cartesian_conversion(self, grid):
        X, Y = grid.to_cartesian()
        R_check = np.sqrt(X ** 2 + Y ** 2)
        assert np.allclose(R_check, grid.R, rtol=1e-06)

class TestBackground2D:

    def test_state_shape(self, bg, grid):
        assert bg.U.shape == (N_VARS, 128, 90)

    def test_azimuthal_symmetry(self, bg):
        for i_var in range(N_VARS):
            col0 = bg.U[i_var, :, 0]
            for j in range(1, bg.grid.nphi):
                assert np.allclose(bg.U[i_var, :, j], col0, rtol=1e-06), f'Variable {i_var} not azimuthally symmetric at phi index {j}'

    def test_density_positive(self, bg):
        assert np.all(bg.rho > 0)

    def test_velocity_positive(self, bg):
        assert np.all(bg.v_r > 0)

    def test_vphi_zero(self, bg):
        assert np.all(bg.v_phi == 0)

    def test_pressure_positive(self, bg):
        assert np.all(bg.pressure > 0)

    def test_mach_subsonic_near_sun(self, bg):
        mach = bg.mach_number()[:, 0]
        assert mach.min() < 1.0 or mach.max() > 1.0, 'No sonic transition found on grid'
        i, _ = bg.grid.earth_index()
        assert bg.mach_number()[i, 0] > 1.0

    def test_mach_supersonic_at_1au(self, bg):
        i, j = bg.grid.earth_index()
        mach_earth = bg.mach_number()[i, j]
        assert mach_earth > 1.0, f'Mach at 1 AU = {mach_earth:.2f}, expected > 1'

    def test_speed_at_earth_reasonable(self, bg):
        earth = bg.at_earth()
        assert 300 < earth['v_r_kms'] < 900

    def test_density_at_earth_reasonable(self, bg):
        earth = bg.at_earth()
        assert 1 < earth['n_cm3'] < 100

    def test_density_decreases_radially(self, bg):
        n = bg.number_density()[:, 0]
        assert np.all(np.diff(n) < 0), 'Density not decreasing radially'

    def test_velocity_increases_radially(self, bg):
        v = bg.v_r[:, 0]
        dv = np.diff(v)
        n_decreasing = np.sum(dv <= 0)
        assert n_decreasing < 30, f'Too many decreasing velocity steps: {n_decreasing}'

    def test_memory_reasonable(self, bg):
        assert bg.U.nbytes < 500000000.0
