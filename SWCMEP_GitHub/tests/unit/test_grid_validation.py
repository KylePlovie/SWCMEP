import numpy as np
import pytest
from swcmep.grid.spherical_grid import SphericalGrid, SphericalGrid3D

def test_exact_spherical_volumes():
    grid = SphericalGrid3D(nr=16, ntheta=9, nphi=18)
    expected = 4 * np.pi * (grid.r_outer ** 3 - grid.r_inner ** 3) / 3
    assert grid.cell_volume().sum() == pytest.approx(expected, rel=1e-14)

def test_radial_endpoints_and_out_of_domain():
    grid = SphericalGrid(nr=16, nphi=18)
    assert grid.r_index(grid.r_inner) == 0
    assert grid.r_index(grid.r_outer) == grid.nr - 1
    with pytest.raises(ValueError, match='outside'):
        grid.r_index(grid.r_inner / 2)

@pytest.mark.parametrize('kwargs', [{'nr': 16.5}, {'nr': 0}, {'nphi': 0}, {'r_inner': float('nan')}, {'r_inner': -1}, {'ntheta': 1}, {'theta_outer': 4}])
def test_invalid_grid_rejected(kwargs):
    with pytest.raises(ValueError):
        SphericalGrid3D(**{'nr': 16, 'ntheta': 9, 'nphi': 18, **kwargs})
