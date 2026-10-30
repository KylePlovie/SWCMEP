# SWCMEP

Solar Wind and CME Projector is an experimental Python project for exploring solar wind and coronal mass ejection propagation on spherical grids. It combines a PyTorch MHD solver, synthetic background wind maps, cone CME injection, and a Flask dashboard.

## Run locally

Requires Python 3.10 or newer. From this directory:

```sh
python -m pip install -e ".[web]"
python -m swcmep.ui
```

Open http://127.0.0.1:8765. The solver can run on CPU or CUDA hardware.

## Development

```sh
python -m pip install -e ".[dev,web]"
python -m pytest -q
```

The source version passed 159 tests on CPU. 
The numerical solver is in `swcmep/solar_wind`, grid geometry in `swcmep/grid`, CME injection in `swcmep/cme`, and the dashboard in `swcmep/ui`. Event execution uses `python -m swcmep.event_runner`. Magnetic diagnostics are available through `python -m swcmep.divergence_audit`.

## Project status

This is a research and learning project, not an operational space-weather forecast system or a reproduction of ENLIL. The WSA-lite boundary is synthetic. The initial magnetic field has a known divergence issue near the current sheet. Constrained transport remains experimental and disabled by default. Physical relaxation, resolution convergence, and arrival accuracy need further validation. Completed runs and passing software tests do not establish forecast accuracy.

