from __future__ import annotations
import argparse
import json
import math
import os
import re
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from flask import Flask, Response, jsonify, render_template, request, send_file
from swcmep.utils.io import atomic_write_json
from swcmep.utils.validation import finite_float, integer_value
from .render import render_earth_timeseries, render_equatorial_frame
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(os.environ.get('SWCMEP_RUN_ROOT', Path.cwd() / 'data' / 'ui_runs')).resolve()
PACKAGED_RUNNER = Path(__file__).resolve().parents[1] / 'event_runner.py'
RUNNER_CANDIDATES = (PROJECT_ROOT / 'run_cme_case_3d.py', PROJECT_ROOT / 'run_compound_cme_3d.py', PACKAGED_RUNNER)
JOB_ID_RE = re.compile('^[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$')

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')

def _solver_environment(job_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['MPLCONFIGDIR'] = str(job_dir / '.matplotlib')
    return env

def _parse_utc(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} is required')
    try:
        parsed = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError(f'{field} must be an ISO-8601 timestamp') from exc
    if parsed.tzinfo is None:
        raise ValueError(f'{field} must include Z or a UTC offset')
    return parsed.astimezone(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')

def _number(payload: dict, name: str, minimum: float, maximum: float, *, integer: bool=False) -> float | int:
    try:
        value = integer_value(payload[name], name) if integer else finite_float(payload[name], name)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if value < minimum or value > maximum:
        raise ValueError(f'{name} must be between {minimum:g} and {maximum:g}')
    return value

def _boolean(payload: dict, name: str, default: bool) -> bool:
    value = payload.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f'{name} must be true or false')
    return value

def _optional_utc(payload: dict, name: str) -> str | None:
    value = payload.get(name)
    return _parse_utc(value, name) if value else None

def build_event_config(payload: dict, job_dir: Path) -> dict:
    time_21_5 = _parse_utc(payload.get('time_21_5_utc'), 'time_21_5_utc')
    event_name = str(payload.get('event_name', '')).strip()
    analysis_id = str(payload.get('analysis_id', '')).strip()
    if not event_name:
        raise ValueError('event_name is required')
    if not analysis_id:
        raise ValueError('analysis_id is required')
    if len(event_name) > 120 or len(analysis_id) > 120:
        raise ValueError('event_name and analysis_id must be 120 characters or fewer')
    ntheta = int(_number(payload, 'ntheta', 9, 181, integer=True))
    if ntheta % 2 == 0:
        raise ValueError('ntheta must be odd so a cell is centered on the equator')
    dtype = str(payload.get('dtype', 'float64'))
    if dtype not in {'float32', 'float64'}:
        raise ValueError('dtype must be float32 or float64')
    spatial_order = int(_number(payload, 'spatial_order', 1, 2, integer=True))
    output_prefix = job_dir / 'result'
    config = {'event_name': event_name, 'simulation_start_utc': time_21_5, 'simulation_hours': _number(payload, 'simulation_hours', 0.1, 240.0), 'relax_hours': _number(payload, 'relax_hours', 0.0, 48.0), 'output_every_hours': _number(payload, 'output_every_hours', 0.25, 12.0), 'grid': {'nr': _number(payload, 'nr', 16, 512, integer=True), 'ntheta': ntheta, 'nphi': _number(payload, 'nphi', 18, 360, integer=True)}, 'solver': {'dtype': dtype, 'cfl': _number(payload, 'cfl', 0.05, 0.5), 'spatial_order': spatial_order, 'shock_flattening': _boolean(payload, 'shock_flattening', True), 'shock_threshold': _number(payload, 'shock_threshold', 1.05, 20.0)}, 'assumptions': {'duration_hours': _number(payload, 'pulse_duration_hours', 0.05, 48.0), 'density_enhancement': _number(payload, 'density_enhancement', 1.0, 20.0), 'temperature_K': _number(payload, 'temperature_mk', 0.05, 20.0) * 1000000.0, 'drag_coefficient': _number(payload, 'drag_coefficient', 0.0, 0.0001), 'source': 'Dashboard input'}, 'cmes': [{'analysis_id': analysis_id, 'time_21_5_utc': time_21_5, 'longitude_deg': _number(payload, 'longitude_deg', -180.0, 180.0), 'latitude_deg': _number(payload, 'latitude_deg', -90.0, 90.0), 'speed_kms': _number(payload, 'speed_kms', 100.0, 4000.0), 'half_width_deg': _number(payload, 'half_width_deg', 5.0, 90.0)}], 'comparison': {'enlil_predicted_arrival_utc': _optional_utc(payload, 'enlil_predicted_arrival_utc'), 'observed_shock_arrival_utc': _optional_utc(payload, 'observed_shock_arrival_utc'), 'observation_source': str(payload.get('observation_source', '')).strip() or None}, 'outputs': {'save_equatorial_frames': True}, 'output_prefix': str(output_prefix), 'ui': {'smoke': _boolean(payload, 'smoke', False)}}
    return config

@dataclass
class ActiveJob:
    job_id: str
    directory: Path
    process: subprocess.Popen
    log_handle: Any
    cancelled: bool = False

class RunManager:

    def __init__(self, run_root: Path=RUN_ROOT, runner_candidates: tuple[Path, ...]=RUNNER_CANDIDATES):
        self.run_root = Path(run_root).resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.runner_candidates = runner_candidates
        self._jobs: dict[str, ActiveJob] = {}
        self._lock = threading.RLock()

    def _runner(self) -> Path:
        for candidate in self.runner_candidates:
            if candidate.is_file():
                return candidate
        expected = ' or '.join((str(path) for path in self.runner_candidates))
        raise ValueError('the CME runner is missing; restore ' + expected)

    def _new_job_id(self) -> str:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        return f'{stamp}_{uuid.uuid4().hex[:8]}'

    def _job_dir(self, job_id: str) -> Path:
        if not JOB_ID_RE.fullmatch(job_id):
            raise KeyError(job_id)
        return self.run_root / job_id

    @staticmethod
    def _actual_prefix(config: dict) -> Path:
        prefix = Path(config['output_prefix'])
        if config.get('ui', {}).get('smoke'):
            prefix = prefix.with_name(prefix.name + '_smoke')
        return prefix

    def start(self, payload: dict) -> dict:
        with self._lock:
            for job_id in list(self._jobs):
                self.status(job_id)
            if any((job.process.poll() is None for job in self._jobs.values())):
                raise ValueError('another simulation is already running; wait for it or cancel it')
            runner = self._runner()
            job_id = self._new_job_id()
            job_dir = self.run_root / job_id
            config = build_event_config(payload, job_dir)
            job_dir.mkdir(parents=True, exist_ok=False)
            config_path = job_dir / 'event.json'
            atomic_write_json(config_path, config)
            metadata = {'job_id': job_id, 'state': 'starting', 'created_utc': _utc_now(), 'config': config}
            self._write_metadata(job_dir, metadata)
            log_handle = (job_dir / 'run.log').open('w', encoding='utf-8', buffering=1)
            command = [sys.executable, '-m', 'swcmep.event_runner'] if runner == PACKAGED_RUNNER else [sys.executable, str(runner)]
            command.append(str(config_path))
            if config['ui']['smoke']:
                command.append('--smoke')
            try:
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log_handle, stderr=subprocess.STDOUT, env=_solver_environment(job_dir), creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
            except OSError as exc:
                log_handle.close()
                metadata.update(state='failed', error=str(exc), finished_utc=_utc_now())
                self._write_metadata(job_dir, metadata)
                raise ValueError(f'could not start solver: {exc}') from exc
            self._jobs[job_id] = ActiveJob(job_id, job_dir, process, log_handle)
            metadata.update(state='running', pid=process.pid)
            self._write_metadata(job_dir, metadata)
            return self.status(job_id)

    @staticmethod
    def _write_metadata(job_dir: Path, metadata: dict) -> None:
        atomic_write_json(job_dir / 'job.json', metadata)

    def _load_metadata(self, job_dir: Path) -> dict:
        try:
            with (job_dir / 'job.json').open('r', encoding='utf-8') as handle:
                return json.load(handle)
        except FileNotFoundError as exc:
            raise KeyError(job_dir.name) from exc

    def status(self, job_id: str) -> dict:
        with self._lock:
            job_dir = self._job_dir(job_id)
            if not job_dir.is_dir():
                raise KeyError(job_id)
            metadata = self._load_metadata(job_dir)
            config = metadata['config']
            prefix = self._actual_prefix(config)
            summary_path = prefix.with_name(prefix.name + '_summary.json')
            manifest_path = prefix.with_name(prefix.name + '_frames') / 'manifest.json'
            manifest = []
            if manifest_path.exists():
                try:
                    with manifest_path.open('r', encoding='utf-8') as handle:
                        manifest = json.load(handle)
                except (OSError, json.JSONDecodeError):
                    manifest = []
            summary = None
            if summary_path.exists():
                try:
                    with summary_path.open('r', encoding='utf-8') as handle:
                        summary = json.load(handle)
                except (OSError, json.JSONDecodeError):
                    summary = None
            active = self._jobs.get(job_id)
            if active is not None:
                returncode = active.process.poll()
                if returncode is None:
                    state = 'running'
                else:
                    target = 12.0 if config.get('ui', {}).get('smoke') else float(config['simulation_hours'])
                    completed = bool(summary and summary.get('termination_reason') == 'completed' and (float(summary.get('completed_hours', 0)) >= target - 1e-06))
                    state = 'cancelled' if active.cancelled else 'completed' if returncode == 0 and completed else 'failed'
                    active.log_handle.close()
                    self._jobs.pop(job_id, None)
                    metadata.update(state=state, returncode=returncode, finished_utc=_utc_now())
                    self._write_metadata(job_dir, metadata)
            else:
                state = metadata.get('state', 'unknown')
                if state in {'running', 'starting'}:
                    state = 'unmanaged'
            smoke = bool(config.get('ui', {}).get('smoke'))
            target_hours = 12.0 if smoke else float(config['simulation_hours'])
            completed_hours = float(summary.get('completed_hours', 0.0)) if summary else float(manifest[-1]['t_hours']) if manifest else 0.0
            return {'job_id': job_id, 'state': state, 'created_utc': metadata.get('created_utc'), 'event_name': config['event_name'], 'smoke': smoke, 'completed_hours': completed_hours, 'target_hours': target_hours, 'progress': min(completed_hours / max(target_hours, 1e-09), 1.0), 'frame_count': len(manifest), 'latest_frame': len(manifest) - 1 if manifest else None, 'summary': summary}

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            active = self._jobs.get(job_id)
        if active is None or active.process.poll() is not None:
            return self.status(job_id)
        active.cancelled = True
        try:
            active.process.terminate()
        except ProcessLookupError:
            pass
        return self.status(job_id)

    def log_tail(self, job_id: str, limit_bytes: int=80000) -> str:
        path = self._job_dir(job_id) / 'run.log'
        if not path.exists():
            return ''
        with path.open('rb') as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(size - limit_bytes, 0))
            data = handle.read()
        return data.decode('utf-8', errors='replace')

    def manifest(self, job_id: str) -> list[dict]:
        metadata = self._load_metadata(self._job_dir(job_id))
        prefix = self._actual_prefix(metadata['config'])
        path = prefix.with_name(prefix.name + '_frames') / 'manifest.json'
        if not path.exists():
            return []
        with path.open('r', encoding='utf-8') as handle:
            return json.load(handle)

    def frame_path(self, job_id: str, index: int) -> Path:
        manifest = self.manifest(job_id)
        if index < 0 or index >= len(manifest):
            raise KeyError(index)
        metadata = self._load_metadata(self._job_dir(job_id))
        prefix = self._actual_prefix(metadata['config'])
        frames_dir = prefix.with_name(prefix.name + '_frames')
        path = frames_dir / manifest[index]['file']
        if not path.is_file() or path.parent != frames_dir:
            raise KeyError(index)
        return path

    def result_paths(self, job_id: str) -> tuple[Path, Path, Path]:
        metadata = self._load_metadata(self._job_dir(job_id))
        prefix = self._actual_prefix(metadata['config'])
        return (prefix.with_name(prefix.name + '_earth.csv'), prefix.with_name(prefix.name + '_summary.json'), self._job_dir(job_id) / 'event.json')

    def list_jobs(self) -> list[dict]:
        jobs = []
        for path in sorted(self.run_root.glob('*/job.json'), reverse=True)[:30]:
            try:
                jobs.append(self.status(path.parent.name))
            except (KeyError, OSError, json.JSONDecodeError):
                continue
        return jobs

def create_app(run_root: Path | None=None) -> Flask:
    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False
    manager = RunManager(run_root or RUN_ROOT)
    app.extensions['swcmep_runs'] = manager

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/api/jobs')
    def list_jobs():
        return jsonify(manager.list_jobs())

    @app.post('/api/jobs')
    def create_job():
        try:
            payload = request.get_json(force=True)
            if not isinstance(payload, dict):
                raise ValueError('request body must be a JSON object')
            return (jsonify(manager.start(payload)), 201)
        except ValueError as exc:
            return (jsonify({'error': str(exc)}), 400)

    @app.get('/api/jobs/<job_id>')
    def job_status(job_id: str):
        try:
            return jsonify(manager.status(job_id))
        except KeyError:
            return (jsonify({'error': 'job not found'}), 404)

    @app.post('/api/jobs/<job_id>/cancel')
    def cancel_job(job_id: str):
        try:
            return jsonify(manager.cancel(job_id))
        except KeyError:
            return (jsonify({'error': 'job not found'}), 404)

    @app.get('/api/jobs/<job_id>/log')
    def job_log(job_id: str):
        try:
            return Response(manager.log_tail(job_id), mimetype='text/plain')
        except KeyError:
            return Response('job not found', status=404, mimetype='text/plain')

    @app.get('/api/jobs/<job_id>/frames')
    def frames(job_id: str):
        try:
            return jsonify(manager.manifest(job_id))
        except (KeyError, OSError, json.JSONDecodeError):
            return jsonify([])

    @app.get('/api/jobs/<job_id>/frame/<int:index>.png')
    def frame_image(job_id: str, index: int):
        field = request.args.get('field', 'vr_kms')
        try:
            image = render_equatorial_frame(manager.frame_path(job_id, index), field)
        except KeyError:
            return (jsonify({'error': 'frame not found'}), 404)
        except ValueError as exc:
            return (jsonify({'error': str(exc)}), 400)
        return send_file(image, mimetype='image/png', max_age=0)

    @app.get('/api/jobs/<job_id>/earth.png')
    def earth_image(job_id: str):
        try:
            csv_path, summary_path, config_path = manager.result_paths(job_id)
            image = render_earth_timeseries(csv_path, summary_path, config_path)
        except (KeyError, FileNotFoundError):
            return (jsonify({'error': 'Earth time series is not available yet'}), 404)
        return send_file(image, mimetype='image/png', max_age=0)
    return app

def main() -> None:
    parser = argparse.ArgumentParser(description='Start the local SWCMEP dashboard')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    app = create_app()
    print(f'SWCMEP dashboard: http://{args.host}:{args.port}')
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
if __name__ == '__main__':
    main()
