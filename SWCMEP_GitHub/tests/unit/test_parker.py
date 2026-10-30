import numpy as np
import pytest
from swcmep.solar_wind.parker import ParkerSolution
from swcmep.utils.constants import AU, K_B, M_PROTON, M_SUN, R_SUN, G

class TestParkerPhysics:

    @pytest.fixture(scope='class')
    @staticmethod
    def sol_1p5():
        return ParkerSolution(T_corona=1500000.0)

    @pytest.fixture(scope='class')
    @staticmethod
    def sol_2p0():
        return ParkerSolution(T_corona=2000000.0)

    def test_sound_speed_formula(self, sol_1p5):
        expected = np.sqrt(K_B * 1500000.0 / M_PROTON)
        assert abs(sol_1p5.c_s - expected) / expected < 1e-06

    def test_critical_radius_formula(self, sol_1p5):
        expected = G * M_SUN / (2.0 * sol_1p5.c_s ** 2)
        assert abs(sol_1p5.r_c - expected) / expected < 1e-06

    def test_critical_radius_in_reasonable_range(self, sol_1p5):
        r_c_rsun = sol_1p5.r_c / R_SUN
        assert 2.0 < r_c_rsun < 20.0, f'r_c = {r_c_rsun:.1f} R_sun outside expected range'

    def test_speed_at_1au_within_observed_range(self, sol_1p5):
        v_1au = sol_1p5.v_at(AU) / 1000.0
        assert 300 < v_1au < 900, f'v(1 AU) = {v_1au:.0f} km/s outside 300–900 km/s'

    def test_higher_temperature_gives_faster_wind(self, sol_1p5, sol_2p0):
        v_1p5 = sol_1p5.v_at(AU)
        v_2p0 = sol_2p0.v_at(AU)
        assert v_2p0 > v_1p5, f'Expected faster wind for T=2 MK, got {v_2p0 / 1000.0:.0f} < {v_1p5 / 1000.0:.0f} km/s'

    def test_velocity_monotonically_increasing(self, sol_1p5):
        v_outer = sol_1p5.v[sol_1p5.r > sol_1p5.r_c]
        dv = np.diff(v_outer)
        assert np.sum(dv <= 0) <= 1, 'Velocity not monotonically increasing beyond critical point'

    def test_density_decreases_with_distance(self, sol_1p5):
        n_diff = np.diff(sol_1p5.n)
        assert np.sum(n_diff >= 0) <= 1, 'Density not monotonically decreasing'

    def test_mass_flux_conservation(self, sol_1p5):
        flux = sol_1p5.r ** 2 * sol_1p5.n * sol_1p5.v
        flux_variation = (flux.max() - flux.min()) / flux.mean()
        assert flux_variation < 0.01, f'Mass flux varies by {flux_variation * 100:.2f}% — conservation violated'

    def test_density_at_1au_order_of_magnitude(self, sol_1p5):
        n_1au_cm3 = sol_1p5.n_at(AU) / 1000000.0
        assert 1.0 < n_1au_cm3 < 100.0, f'n(1 AU) = {n_1au_cm3:.2f} cm^-3 outside 1–100'

    def test_solution_covers_1au(self, sol_1p5):
        assert sol_1p5.r.max() >= AU, 'Solution does not reach 1 AU'

    def test_solution_starts_inside_critical_point(self, sol_1p5):
        assert sol_1p5.r.min() > sol_1p5.r_c
        assert abs(sol_1p5.r.min() - sol_1p5.r_c) / sol_1p5.r_c < 0.001

    def test_v_at_critical_point_equals_sound_speed(self, sol_1p5):
        v_rc = sol_1p5.v_at(sol_1p5.r_c)
        relative_error = abs(v_rc - sol_1p5.c_s) / sol_1p5.c_s
        assert relative_error < 0.01, f'v at critical point {v_rc / 1000.0:.2f} km/s ≠ c_s {sol_1p5.c_s / 1000.0:.2f} km/s (error: {relative_error * 100:.2f}%)'

class TestParkerInterface:

    def test_default_construction(self):
        sol = ParkerSolution()
        assert sol.T_corona == 1500000.0

    def test_v_at_returns_float(self):
        sol = ParkerSolution()
        assert isinstance(sol.v_at(AU), float)

    def test_n_at_returns_float(self):
        sol = ParkerSolution()
        assert isinstance(sol.n_at(AU), float)

    def test_summary_returns_string(self):
        sol = ParkerSolution()
        s = sol.summary()
        assert isinstance(s, str)
        assert 'km/s' in s

    def test_r_and_v_same_length(self):
        sol = ParkerSolution()
        assert len(sol.r) == len(sol.v) == len(sol.n)
