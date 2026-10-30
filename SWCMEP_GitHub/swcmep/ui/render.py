from __future__ import annotations
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
FIELD_SPECS = {'vr_kms': ('Radial speed', 'km/s', 'turbo'), 'n_cm3': ('Proton density', 'cm$^{-3}$', 'magma'), 'b_nt': ('Magnetic-field magnitude', 'nT', 'viridis')}

def _png_buffer(fig) -> io.BytesIO:
    output = io.BytesIO()
    fig.savefig(output, format='png', dpi=135, bbox_inches='tight')
    plt.close(fig)
    output.seek(0)
    return output

def render_equatorial_frame(path: Path, field: str='vr_kms') -> io.BytesIO:
    if field not in FIELD_SPECS:
        raise ValueError(f'unknown map field: {field}')
    with np.load(path, allow_pickle=False) as frame:
        r_au = np.asarray(frame['r_au'], dtype=float)
        phi = np.asarray(frame['phi_rad'], dtype=float)
        values = np.asarray(frame[field], dtype=float)
        t_hours = float(frame['t_hours'])
        utc = str(frame['utc'])
        vr_scale_max = float(frame['vr_scale_max']) if 'vr_scale_max' in frame else 1200.0
    wrapped_phi = np.concatenate([phi, [2.0 * np.pi]])
    wrapped_values = np.concatenate([values, values[:, :1]], axis=1)
    label, units, cmap = FIELD_SPECS[field]
    mesh_kwargs = {'cmap': cmap, 'rasterized': True}
    if field == 'vr_kms':
        mesh_kwargs.update(vmin=250.0, vmax=max(vr_scale_max, 700.0))
    elif field == 'n_cm3':
        mesh_kwargs['norm'] = LogNorm(vmin=0.1, vmax=10000.0, clip=True)
    else:
        mesh_kwargs['norm'] = LogNorm(vmin=0.1, vmax=1000.0, clip=True)
    fig, ax = plt.subplots(figsize=(8.1, 6.6), subplot_kw={'projection': 'polar'})
    mesh = ax.pcolormesh(wrapped_phi, r_au, wrapped_values, shading='nearest', **mesh_kwargs)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)
    ax.set_ylim(float(r_au.min()), min(float(r_au.max()), 1.1))
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1 AU'])
    ax.grid(color='0.45', alpha=0.35, linewidth=0.7)
    ax.scatter([0.0], [1.0], s=42, color='#4aa3ff', edgecolor='white', linewidth=0.8, zorder=5, label='Earth')
    ax.scatter([0.0], [float(r_au.min())], s=70, color='#ffd34d', edgecolor='#9d6f00', linewidth=0.8, zorder=5, label='Sun')
    ax.legend(loc='upper right', bbox_to_anchor=(1.16, 1.12), framealpha=0.85)
    ax.set_title(f'{label} — {utc}  (t={t_hours:.1f} h)', pad=20)
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.1, shrink=0.78)
    colorbar.set_label(f'{label} ({units})')
    fig.tight_layout()
    return _png_buffer(fig)

def _read_series(path: Path) -> dict[str, list]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, list] = {'utc': [], 'v_kms': [], 'n_cm3': [], 'b_nt': [], 'arrival_metric': []}
    with path.open('r', encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            result['utc'].append(datetime.fromisoformat(row['utc'].replace('Z', '+00:00')))
            for key in ('v_kms', 'n_cm3', 'b_nt', 'arrival_metric'):
                result[key].append(float(row[key]))
    if not result['utc']:
        raise FileNotFoundError(path)
    return result

def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)

def _event_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace('Z', '+00:00')) if value else None

def render_earth_timeseries(csv_path: Path, summary_path: Path, config_path: Path) -> io.BytesIO:
    data = _read_series(csv_path)
    summary = _load_json(summary_path)
    config = _load_json(config_path)
    times = data['utc']
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.0), sharex=True)
    series = (('v_kms', 'Radial speed (km/s)', '#e45756'), ('n_cm3', 'Density (cm$^{-3}$)', '#4c78a8'), ('b_nt', '|B| (nT)', '#72b7b2'))
    for ax, (key, label, color) in zip(axes, series):
        ax.plot(times, data[key], color=color, linewidth=1.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    comparison = config.get('comparison', {})
    markers = ((summary.get('modeled_arrival_utc'), 'SWCMEP', '#e45756', '-'), (comparison.get('enlil_predicted_arrival_utc'), 'ENLIL', '#f2cf5b', '--'), (comparison.get('observed_shock_arrival_utc'), 'Observed', '#54a24b', ':'))
    for value, label, color, style in markers:
        stamp = _event_time(value)
        if stamp is None:
            continue
        for ax in axes:
            ax.axvline(stamp, color=color, linestyle=style, linewidth=1.5, label=label)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        axes[0].legend(unique.values(), unique.keys(), loc='best', ncol=3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%MZ', tz=timezone.utc))
    axes[-1].set_xlabel('UTC')
    fig.suptitle(config.get('event_name', 'SWCMEP Earth time series'))
    fig.tight_layout()
    return _png_buffer(fig)
