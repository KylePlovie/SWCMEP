from __future__ import annotations
import argparse

def run() -> None:
    from swcmep.cme.cone_model import CMEParameters, run_cme_simulation_3d
    parser = argparse.ArgumentParser(description='Run a tiny SWCMEP 3D CME smoke simulation')
    parser.add_argument('--nr', type=int, default=16)
    parser.add_argument('--ntheta', type=int, default=9)
    parser.add_argument('--nphi', type=int, default=18)
    parser.add_argument('--speed-kms', type=float, default=800.0)
    parser.add_argument('--sim-hours', type=float, default=0.05)
    args = parser.parse_args()
    cme = CMEParameters(speed_kms=args.speed_kms, half_width_deg=20.0, duration_hours=min(args.sim_hours, 0.05), density_enhancement=2.0, temperature_K=1500000.0, shell_layers=2)
    nr = max(args.nr, 16)
    ntheta = max(args.ntheta, 9)
    nphi = max(args.nphi, 18)
    result = run_cme_simulation_3d(cme, grid_nr=nr, grid_ntheta=ntheta, grid_nphi=nphi, sim_hours=args.sim_hours, output_every_h=args.sim_hours, relax_background_hours=0.0, use_drag=False, verbose=True)
    print(f"Snapshots: {len(result['snapshots'])}")
    print(f"Valid final state: {result['solver'].is_valid()}")

def plot() -> None:
    from swcmep.solar_wind.background_2d import _cli
    _cli()

def fetch() -> None:
    raise SystemExit('No data-fetch backend is implemented yet; use an event JSON config.')
