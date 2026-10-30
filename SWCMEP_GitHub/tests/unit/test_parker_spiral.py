import pytest
from swcmep.solar_wind.parker import ParkerSolution
from swcmep.solar_wind.parker_spiral import ParkerSpiral
from swcmep.utils.constants import AU

@pytest.fixture(scope='module')
def spiral():
    sol = ParkerSolution(T_corona=1500000.0)
    return ParkerSpiral(sol)

@pytest.fixture(scope='module')
def spiral_fast():
    sol = ParkerSolution(T_corona=2000000.0)
    return ParkerSpiral(sol)

class TestSpiralPhysics:

    def test_Br_falls_as_inverse_r_squared(self, spiral):
        r1, r2 = (0.5 * AU, 1.0 * AU)
        Br1 = abs(spiral.B_r_at(r1))
        Br2 = abs(spiral.B_r_at(r2))
        ratio = Br1 / Br2
        expected = (r2 / r1) ** 2
        assert abs(ratio - expected) / expected < 0.01, f'B_r ratio {ratio:.3f} ≠ expected {expected:.3f} (1/r^2 law violated)'

    def test_Bmag_at_1au_close_to_observed(self, spiral):
        Bmag_nT = abs(spiral.B_mag_at(AU)) / 1e-09
        assert 2.0 < Bmag_nT < 10.0, f'|B|(1 AU) = {Bmag_nT:.2f} nT, expected 2–10 nT'

    def test_spiral_angle_at_1au_slow_wind(self, spiral):
        psi = spiral.spiral_angle_at(AU)
        assert 35 < psi < 55, f'Spiral angle {psi:.1f}° outside expected 35–55° for slow wind'

    def test_fast_wind_less_wound_than_slow(self, spiral, spiral_fast):
        psi_slow = spiral.spiral_angle_at(AU)
        psi_fast = spiral_fast.spiral_angle_at(AU)
        assert psi_fast < psi_slow, f'Fast wind angle {psi_fast:.1f}° should be < slow wind {psi_slow:.1f}°'

    def test_Bphi_dominates_at_large_r(self, spiral):
        Br = abs(spiral.B_r_at(AU))
        Bphi = abs(spiral.B_phi_at(AU))
        assert Bphi > Br, f'|B_phi| {Bphi / 1e-09:.2f} nT should exceed |B_r| {Br / 1e-09:.2f} nT at 1 AU'

    def test_Br_dominates_near_sun(self, spiral):
        r_near = 0.05 * AU
        Br = abs(spiral.B_r_at(r_near))
        Bphi = abs(spiral.B_phi_at(r_near))
        assert Br > Bphi, f'|B_r| should dominate near Sun, got |B_r|={Br / 1e-09:.1f} < |B_phi|={Bphi / 1e-09:.1f} nT'

    def test_spiral_angle_increases_with_distance(self, spiral):
        psi_inner = spiral.spiral_angle_at(0.3 * AU)
        psi_outer = spiral.spiral_angle_at(1.0 * AU)
        assert psi_outer > psi_inner, 'Spiral angle should increase with distance'

    def test_Bphi_negative(self, spiral):
        assert spiral.B_phi_at(AU) < 0, 'B_phi should be negative (trailing spiral)'

class TestSpiralInterface:

    def test_summary_contains_angle(self, spiral):
        s = spiral.summary()
        assert '°' in s

    def test_arrays_same_length_as_parker(self, spiral):
        assert len(spiral.B_r) == len(spiral.parker.r)
        assert len(spiral.B_phi) == len(spiral.parker.r)

    def test_query_methods_return_float(self, spiral):
        assert isinstance(spiral.B_r_at(AU), float)
        assert isinstance(spiral.B_phi_at(AU), float)
        assert isinstance(spiral.B_mag_at(AU), float)
        assert isinstance(spiral.spiral_angle_at(AU), float)
