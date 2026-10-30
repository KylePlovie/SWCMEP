import argparse
from pathlib import Path
import torch
from swcmep.cme.cone_model import CMEParameters, ConeModelInjector3D
from swcmep.grid.spherical_grid import SphericalGrid3D
from swcmep.solar_wind.background_wsa_lite import BackgroundWSALite3D, WSALiteBoundaryMap3D
from swcmep.solar_wind.magnetic_diagnostics import centered_divergence_fields, face_divergence_fields, summarize_divergence
from swcmep.solar_wind.mhd_solver_gpu import MHDSolverGPU3D
from swcmep.utils.constants import I_BPHI, I_BR, I_BTHETA, I_EINT, I_P, I_VR
from swcmep.utils.io import atomic_write_json
from swcmep.utils.validation import positive_float

def run_audit(output, *, nr=32, ntheta=15, nphi=30, hours=1.0, output_hours=0.25, scenario='background', dtype=torch.float64, device='cpu'):
    hours = positive_float(hours, 'hours', allow_zero=True)
    output_hours = positive_float(output_hours, 'output_hours')
    if scenario not in ('background', 'cme'):
        raise ValueError('scenario must be background or cme')
    grid = SphericalGrid3D(nr=nr, ntheta=ntheta, nphi=nphi)
    boundary = WSALiteBoundaryMap3D(grid.theta_centers, grid.phi_centers)
    background = BackgroundWSALite3D(grid, boundary)
    background.U[I_EINT] = background.U[I_P] / (1.05 - 1.0)
    face_fields = face_divergence_fields(*(torch.as_tensor(field) for field in (background.Br_face, background.Bth_face, background.Bphi_face)), grid.r_edges, grid.theta_edges, grid.dphi)
    raw_fields = centered_divergence_fields(*(torch.as_tensor(background.U[index]) for index in (I_BR, I_BTHETA, I_BPHI)), grid.r_centers, grid.theta_centers, grid.dr, grid.dtheta, grid.dphi)
    solver = MHDSolverGPU3D(grid, background.U, dtype=dtype, device=device, cfl_factor=0.2)
    initial = solver._U_gpu.clone()
    report = {'grid': [nr, ntheta, nphi], 'dtype': str(dtype), 'device': str(solver.device), 'scenario': scenario, 'requested_hours': hours, 'output_hours': output_hours, 'use_ct': solver.use_ct, 'equilibrium_residual': solver.use_well_balanced, 'guard_cells_inner_outer_pole': list(solver._resolved_guard_counts()), 'status': 'running', 'initial_background_centered_active': summarize_divergence(raw_fields, grid, region=solver._active_divergence_region(), offset=(1, 1, 0)), 'initial_stored_faces_full_domain': summarize_divergence(face_fields, grid), 'notes': ['Face data are measured only at initialization; GLM does not evolve the stored faces.', 'The background builder currently stores Bphi centers as faces; its face placement needs correction.', 'Face and centered operators differ. Their values are not interchangeable.', 'The equilibrium residual preserves the reference; a steady run is not physical relaxation.'], 'samples': []}
    del face_fields, raw_fields, background

    def sample(current, snapshot=None):
        measurement = current.divergence_report()
        measurement['max_magnetic_component_change_T'] = float((current._U_gpu[[I_BR, I_BTHETA, I_BPHI]] - initial[[I_BR, I_BTHETA, I_BPHI]]).abs().max())
        measurement['max_radial_speed_change_m_per_s'] = float((current._U_gpu[I_VR] - initial[I_VR]).abs().max())
        report['samples'].append(measurement)
        atomic_write_json(Path(output), report)
    sample(solver)
    injector = None
    if scenario == 'cme':
        injector = ConeModelInjector3D(grid, solver.U.copy(), CMEParameters(speed_kms=1200.0, half_width_deg=30.0, duration_hours=1.0, density_enhancement=2.0), drag_coefficient=0.0, boundary_layers=solver.inner_guard_cells)
    try:
        solver.run(hours * 3600.0, output_every=output_hours * 3600.0, injector=injector, callback=sample, retain_snapshots=False, verbose=False)
        report['status'] = 'completed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = str(exc)
        raise
    finally:
        report['termination_reason'] = solver.termination_reason
        report['completed_seconds'] = solver.t
        atomic_write_json(Path(output), report)
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('divergence_audit.json'))
    parser.add_argument('--nr', type=int, default=32)
    parser.add_argument('--ntheta', type=int, default=15)
    parser.add_argument('--nphi', type=int, default=30)
    parser.add_argument('--hours', type=float, default=1.0)
    parser.add_argument('--output-hours', type=float, default=0.25)
    parser.add_argument('--scenario', choices=('background', 'cme'), default='background')
    parser.add_argument('--dtype', choices=('float32', 'float64'), default='float64')
    parser.add_argument('--device', default='cpu')
    args = vars(parser.parse_args())
    args['dtype'] = getattr(torch, args['dtype'])
    report = run_audit(**args)
    first, last = (report['samples'][0], report['samples'][-1])
    print(f"Active centered divergence: {first['active']['max']:.6g} -> {last['active']['max']:.6g}")
    print(f"Worst final cell: {last['active']['worst_cell']['index']}; report: {args['output']}")
if __name__ == '__main__':
    main()
