import numpy as np
import pytest
from swcmep.grid.spherical_grid import SphericalGrid
from swcmep.solar_wind.background_2d import Background2D
from swcmep.solar_wind.mhd_solver import RHO, VR, MHDSolver
from swcmep.solar_wind.parker import ParkerSolution

@pytest.fixture(scope='module')
def small_solver():
    grid = SphericalGrid(nr=64, nphi=45)
    parker = ParkerSolution(T_corona=1500000.0)
    bg = Background2D(grid, parker)
    return MHDSolver(grid, bg.U.copy(), gamma=1.05, cfl_factor=0.5)

class TestCFL:

    def test_dt_positive(self, small_solver):
        dt = small_solver.compute_dt()
        assert dt > 0

    def test_dt_finite(self, small_solver):
        dt = small_solver.compute_dt()
        assert np.isfinite(dt)

    def test_dt_reasonable_magnitude(self, small_solver):
        dt = small_solver.compute_dt()
        assert 1.0 < dt < 3600.0, f'dt = {dt:.1f}s seems unreasonable'

    def test_cfl_factor_scales_dt(self):
        grid = SphericalGrid(nr=64, nphi=45)
        parker = ParkerSolution(T_corona=1500000.0)
        bg = Background2D(grid, parker)
        s1 = MHDSolver(grid, bg.U.copy(), cfl_factor=0.5)
        s2 = MHDSolver(grid, bg.U.copy(), cfl_factor=0.25)
        assert abs(s1.compute_dt() / s2.compute_dt() - 2.0) < 0.01

class TestSolverPhysics:

    def test_initial_state_valid(self, small_solver):
        assert small_solver.is_valid()

    def test_sound_speed_positive(self, small_solver):
        cs = small_solver.sound_speed(small_solver.U)
        assert np.all(cs > 0)

    def test_sound_speed_finite(self, small_solver):
        cs = small_solver.sound_speed(small_solver.U)
        assert np.all(np.isfinite(cs))

    def test_one_step_stays_valid(self, small_solver):
        dt = small_solver.compute_dt()
        U_new = small_solver._lax_friedrichs_step(small_solver.U, dt)
        small_solver._apply_boundaries(U_new)
        assert np.all(np.isfinite(U_new)), 'NaN/Inf after one step'
        assert np.all(U_new[RHO] > 0), 'Negative density after one step'

    def test_short_run_stays_valid(self):
        grid = SphericalGrid(nr=64, nphi=45)
        parker = ParkerSolution(T_corona=1500000.0)
        bg = Background2D(grid, parker)
        solver = MHDSolver(grid, bg.U.copy(), gamma=1.05)
        solver.run(duration_seconds=3600, output_every=3600, verbose=False)
        assert solver.is_valid(), 'Solver state invalid after 1 hour run'

    def test_background_is_steady(self):
        grid = SphericalGrid(nr=64, nphi=45)
        parker = ParkerSolution(T_corona=1500000.0)
        bg = Background2D(grid, parker)
        solver = MHDSolver(grid, bg.U.copy(), gamma=1.05)
        v_initial = solver.U[VR, :, 0].copy()
        solver.run(duration_seconds=7200, output_every=7200, verbose=False)
        v_final = solver.U[VR, :, 0]
        interior = slice(2, -2)
        rel_change = np.abs(v_final[interior] - v_initial[interior]) / (np.abs(v_initial[interior]) + 1e-10)
        max_change = rel_change.max()
        assert max_change < 0.3, f'Background changed by {max_change * 100:.1f}% — not steady'

    def test_mass_approximately_conserved(self):
        grid = SphericalGrid(nr=64, nphi=45)
        parker = ParkerSolution(T_corona=1500000.0)
        bg = Background2D(grid, parker)
        solver = MHDSolver(grid, bg.U.copy(), gamma=1.05)
        mass_initial = solver.total_mass()
        solver.run(duration_seconds=3600, output_every=3600, verbose=False)
        mass_final = solver.total_mass()
        rel_change = abs(mass_final - mass_initial) / mass_initial
        assert rel_change < 0.25, f'Mass changed by {rel_change * 100:.1f}% — conservation violated'

    def test_velocity_at_earth_reasonable_after_run(self):
        grid = SphericalGrid(nr=64, nphi=45)
        parker = ParkerSolution(T_corona=1500000.0)
        bg = Background2D(grid, parker)
        solver = MHDSolver(grid, bg.U.copy(), gamma=1.05)
        solver.run(duration_seconds=3600, output_every=3600, verbose=False)
        v_earth = solver._v_at_earth()
        assert 200 < v_earth < 1200, f'v(Earth) = {v_earth:.0f} km/s after run — unreasonable'

class TestSolverInterface:

    def test_snapshots_returned(self):
        grid = SphericalGrid(nr=64, nphi=45)
        parker = ParkerSolution(T_corona=1500000.0)
        bg = Background2D(grid, parker)
        solver = MHDSolver(grid, bg.U.copy())
        snaps = solver.run(duration_seconds=1800, output_every=600, verbose=False)
        assert len(snaps) >= 2

    def test_snapshot_has_required_keys(self):
        grid = SphericalGrid(nr=64, nphi=45)
        parker = ParkerSolution(T_corona=1500000.0)
        bg = Background2D(grid, parker)
        solver = MHDSolver(grid, bg.U.copy())
        snaps = solver.run(duration_seconds=600, output_every=600, verbose=False)
        snap = snaps[-1]
        for key in ('t', 'step', 'U', 'dt'):
            assert key in snap, f"Missing key '{key}' in snapshot"

    def test_time_advances(self):
        grid = SphericalGrid(nr=64, nphi=45)
        parker = ParkerSolution(T_corona=1500000.0)
        bg = Background2D(grid, parker)
        solver = MHDSolver(grid, bg.U.copy())
        t0 = solver.t
        solver.run(duration_seconds=600, output_every=600, verbose=False)
        assert solver.t > t0

    def test_step_count_increases(self):
        grid = SphericalGrid(nr=64, nphi=45)
        parker = ParkerSolution(T_corona=1500000.0)
        bg = Background2D(grid, parker)
        solver = MHDSolver(grid, bg.U.copy())
        solver.run(duration_seconds=600, output_every=600, verbose=False)
        assert solver.step > 0
