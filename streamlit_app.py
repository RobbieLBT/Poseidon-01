from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.animation import PillowWriter
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml

import mission_analysis as ma


REPO_ROOT = Path(__file__).resolve().parent
SIM_CONFIG_PATH = REPO_ROOT / 'sim_config.yaml'
SCENARIO_DIR = REPO_ROOT / 'scenarios' / 'mission_profiles'
MAP_DIR = REPO_ROOT / 'environment' / 'maps'
WEATHER_DIR = REPO_ROOT / 'environment' / 'weather'
LOG_ROOT = REPO_ROOT / 'logs' / 'simulator_run'

WEATHER_MODES = ['min', 'minus_1sd', 'minus_2sd']
CONTROL_OPTIONS = {'direct': 'direct_pursuit', 'damped': 'damped_pursuit'}
DEFAULT_SCENARIO_FILE = 'scenario-mvp-02-short-scan.yaml'
DEFAULT_MAX_TIME_S = 1200.0
ANIMATION_MAX_FRAMES = 180

SCENARIO_TO_MAP = {
    'scenario-mvp-01-auv-patrol.yaml': 'Pamlico_Sound.png',
    'scenario-mvp-02-short-scan.yaml': 'Pamlico_Sound.png',
    'scenario-mvp-03-scan.yaml': 'Pamlico_Sound.png',
    'scenario-mvp-04-deep-scan.yaml': 'Hudson_Canyon.png',
    'scenario-mvp-05-ssv-dash.yaml': 'Hampton_Roads.png',
    'scenario-mvp-06-ssv-short-dash.yaml': 'Hampton_Roads.png',
}

APP_REPO_URL = os.environ.get(
    'POSEIDON_REPO_URL',
    'https://github.com/RobbieLBT/Poseidon-01',
)
APP_REPO_BRANCH = os.environ.get('POSEIDON_REPO_BRANCH', 'main')


def github_repo_web_url(repo_url: str) -> str:
    text = str(repo_url or '').strip()
    if not text:
        return ''
    if text.endswith('.git'):
        text = text[:-4]
    return text.rstrip('/')


def github_raw_base_url(repo_url: str, branch: str) -> str:
    web_url = github_repo_web_url(repo_url)
    prefix = 'https://github.com/'
    if web_url.startswith(prefix):
        return 'https://raw.githubusercontent.com/' + web_url[len(prefix):].strip('/') + f'/{branch.strip("/")}'
    return web_url.rstrip('/')


def build_default_map_png_urls() -> Dict[str, str]:
    raw_base = github_raw_base_url(APP_REPO_URL, APP_REPO_BRANCH)
    names = ['Virginia_Beach', 'Pamlico_Sound', 'Hudson_Canyon', 'Hampton_Roads']
    out: Dict[str, str] = {}
    for name in names:
        url = f'{raw_base}/environment/maps/{name}.png'
        out[name] = url
        out[f'{name}.png'] = url
        out[name.replace('_', ' ')] = url
    return out


DEFAULT_MAP_PNG_URLS: Dict[str, str] = build_default_map_png_urls()

st.set_page_config(page_title='Poseidon Mission Sandbox', layout='wide')


@dataclass
class RunArtifacts:
    run_dir: Path
    scenario_path: Path
    sim_config_path: Path
    scenario: Dict[str, Any]
    sim_cfg: Dict[str, Any]
    planned_paths: Dict[str, List[Tuple[float, float, float]]]
    truth_histories: Dict[str, Dict[str, List[float]]]
    estimate_histories: Dict[str, Dict[str, List[float]]]
    vehicle_logs: Dict[str, pd.DataFrame]
    stdout_summary: str
    map_png_path: Optional[Path]
    wall_clock_s: float = 0.0
    compute_energy_wh_est: Optional[float] = None


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Expected mapping in {path}')
    return data


def save_yaml(data: Mapping[str, Any], path: Path) -> None:
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(dict(data), f, sort_keys=False)


def pretty_scenario_name(path: Path) -> str:
    stem = path.stem.replace('scenario-mvp-', '')
    return stem.replace('-', ' ').title()


def discover_scenarios() -> Dict[str, Path]:
    scenario_paths = sorted(SCENARIO_DIR.glob('scenario-mvp-*.yaml'))
    ordered: List[Path] = []
    default_path = SCENARIO_DIR / DEFAULT_SCENARIO_FILE
    if default_path in scenario_paths:
        ordered.append(default_path)
    ordered.extend([p for p in scenario_paths if p != default_path])
    return {pretty_scenario_name(path): path for path in ordered}


def discover_png_maps() -> Dict[str, Path]:
    if not MAP_DIR.exists():
        return {}
    return {p.stem.replace('_', ' '): p for p in sorted(MAP_DIR.glob('*.png'))}


def default_map_name_for_scenario(scenario_path: Path) -> Optional[str]:
    return SCENARIO_TO_MAP.get(scenario_path.name)


def default_map_for_scenario(scenario_path: Path) -> Optional[Path]:
    map_name = default_map_name_for_scenario(scenario_path)
    if not map_name:
        return None
    return MAP_DIR / map_name


def remote_png_url_for_map(map_png_path: Optional[Path] = None, map_name: str = '') -> Optional[str]:
    candidates: List[str] = []
    if map_png_path is not None:
        candidates.extend([map_png_path.name, map_png_path.stem, map_png_path.stem.replace('_', ' ')])
    if map_name:
        candidates.extend([map_name, Path(map_name).stem, Path(map_name).stem.replace('_', ' ')])
    for key in candidates:
        url = DEFAULT_MAP_PNG_URLS.get(key)
        if url:
            return url
    return None


def guess_geotiff_for_map(map_png_path: Optional[Path]) -> Optional[Path]:
    if map_png_path is None:
        return None
    for suffix in ('.tif', '.tiff'):
        candidate = map_png_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def detect_weather_source() -> Optional[Path]:
    preferred = WEATHER_DIR / 'noaa_44014_2012.json'
    if preferred.exists():
        return preferred
    candidates = sorted(WEATHER_DIR.glob('*.json'))
    return candidates[0] if candidates else None


def relative_to_repo(path: Optional[Path]) -> str:
    if path is None:
        return ''
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace('\\', '/')
    except Exception:
        return str(path)


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in ('', None):
        return default
    return bool(value)


def _merge_legacy_underwater_profile_blocks(cfg: Dict[str, Any]) -> None:
    profiles = cfg.get('profiles', {})
    if not isinstance(profiles, dict):
        return

    legacy_drift = profiles.get('drift') if isinstance(profiles.get('drift'), dict) else None
    legacy_sensors = profiles.get('sensors') if isinstance(profiles.get('sensors'), dict) else None
    legacy_estimation = profiles.get('estimation') if isinstance(profiles.get('estimation'), dict) else None

    if not any((legacy_drift, legacy_sensors, legacy_estimation)):
        return

    for profile_name, profile_cfg in profiles.items():
        if not isinstance(profile_cfg, dict):
            continue
        if str(profile_cfg.get('vehicle_mode', '')).lower() != 'underwater':
            continue
        if legacy_drift and not isinstance(profile_cfg.get('drift'), dict):
            profile_cfg['drift'] = copy.deepcopy(legacy_drift)
        if legacy_sensors and not isinstance(profile_cfg.get('sensors'), dict):
            profile_cfg['sensors'] = copy.deepcopy(legacy_sensors)
        if legacy_estimation and not isinstance(profile_cfg.get('estimation'), dict):
            profile_cfg['estimation'] = copy.deepcopy(legacy_estimation)


def build_runtime_sim_config(
    base_cfg: Mapping[str, Any],
    weather_mode: str,
    controller_label: str,
    imu_drift_enabled: bool,
    dvl_sensor_enabled: bool,
    bathy_sensor_enabled: bool,
    bathy_blend_gain: Optional[float],
    map_png_path: Optional[Path],
    run_dir: Path,
) -> Tuple[Dict[str, Any], List[str]]:
    cfg = copy.deepcopy(dict(base_cfg))
    notes: List[str] = []

    _merge_legacy_underwater_profile_blocks(cfg)

    weather_source = detect_weather_source()
    forcing = cfg.setdefault('environment_forcing', {})
    if weather_source is not None:
        forcing['mode'] = 'json_scenario'
        forcing['source'] = relative_to_repo(weather_source)
        forcing['scenario'] = weather_mode
    else:
        notes.append('No weather JSON found under environment/weather; leaving forcing source unchanged.')

    controller_name = CONTROL_OPTIONS[controller_label]

    profiles = cfg.setdefault('profiles', {})
    vehicles = cfg.setdefault('vehicles', {})
    underwater_profiles: set[str] = set()

    for profile_name, profile_cfg in profiles.items():
        if not isinstance(profile_cfg, dict):
            continue
        profile_cfg['controller'] = controller_name
        if str(profile_cfg.get('vehicle_mode', '')).lower() == 'underwater':
            underwater_profiles.add(profile_name)
            drift_cfg = profile_cfg.setdefault('drift', {})
            drift_cfg['enabled'] = bool(imu_drift_enabled)

            sensors_cfg = profile_cfg.setdefault('sensors', {})
            imu_cfg = sensors_cfg.setdefault('imu', {})
            imu_cfg['enabled'] = bool(imu_drift_enabled)

            dvl_cfg = sensors_cfg.setdefault('dvl', {})
            dvl_cfg['enabled'] = bool(dvl_sensor_enabled)

            bathy_cfg = sensors_cfg.setdefault('bathymetry_cone', {})
            bathy_cfg['enabled'] = bool(bathy_sensor_enabled)

            est_cfg = profile_cfg.setdefault('estimation', {})
            est_cfg['bathymetry_localization_enabled'] = bool(bathy_sensor_enabled)
            if bathy_blend_gain is not None:
                est_cfg['blend_gain'] = float(bathy_blend_gain)

            geotiff_path = guess_geotiff_for_map(map_png_path)
            if geotiff_path is not None:
                bathy_cfg['geotiff_path'] = relative_to_repo(geotiff_path)
            elif bathy_sensor_enabled:
                notes.append('Bathymetry sensor enabled, but no matching GeoTIFF was found beside the selected PNG.')

    for vehicle_cfg in vehicles.values():
        if not isinstance(vehicle_cfg, dict):
            continue
        sim_profile = str(vehicle_cfg.get('sim_profile', ''))
        if sim_profile in underwater_profiles:
            vehicle_cfg['waypoint_acceptance'] = vehicle_cfg.get('waypoint_acceptance', 'estimated')

    logging_cfg = cfg.setdefault('logging', {})
    logging_cfg['enabled'] = True
    logging_cfg['output_dir'] = relative_to_repo(run_dir)
    logging_cfg['one_csv_per_vehicle'] = True

    viz_cfg = cfg.setdefault('visualization', {})
    viz_cfg['enable_2d'] = True
    viz_cfg['enable_3d'] = True

    cfg['__source_path__'] = str(SIM_CONFIG_PATH)
    return cfg, notes


def build_preview_figure(scenario: Mapping[str, Any]) -> go.Figure:
    fig = go.Figure()
    vehicles = scenario.get('vehicles', {})

    for vehicle_id, vehicle_cfg in vehicles.items():
        x_vals: List[float] = []
        y_vals: List[float] = []

        pose = vehicle_cfg.get('initial_state', {}).get('pose', {})
        if pose.get('frame') == 'local_ned':
            value = pose.get('value', [0.0, 0.0, 0.0])
            x_vals.append(float(value[0]))
            y_vals.append(float(value[1]))

        for route_item in vehicle_cfg.get('route', []):
            geometry = route_item.get('geometry', {})
            if geometry.get('type') == 'polyline':
                for point in geometry.get('points', []):
                    x_vals.append(float(point[0]))
                    y_vals.append(float(point[1]))
            elif 'position' in route_item:
                point = route_item.get('position', {}).get('value', [0.0, 0.0, 0.0])
                x_vals.append(float(point[0]))
                y_vals.append(float(point[1]))

        if not x_vals:
            continue

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                name=f'{vehicle_id} planned',
                line={'dash': 'dash'},
            )
        )

    fig.update_layout(
        title='Mission Preview',
        xaxis_title='X [m]',
        yaxis_title='Y [m]',
        legend_title='Path',
        height=520,
        xaxis={'scaleanchor': 'y', 'scaleratio': 1.0},
        margin={'l': 20, 'r': 20, 't': 50, 'b': 20},
    )
    return fig


def histories_to_plotly_2d(
    planned_paths: Mapping[str, List[Tuple[float, float, float]]],
    truth_histories: Mapping[str, Mapping[str, Sequence[float]]],
    estimate_histories: Mapping[str, Mapping[str, Sequence[float]]],
) -> go.Figure:
    fig = go.Figure()
    for name, plan in planned_paths.items():
        if plan:
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in plan],
                    y=[p[1] for p in plan],
                    mode='lines+markers',
                    name=f'{name} planned',
                    line={'dash': 'dash'},
                )
            )
    for name, hist in truth_histories.items():
        fig.add_trace(
            go.Scatter(
                x=list(hist.get('x', [])),
                y=list(hist.get('y', [])),
                mode='lines',
                name=f'{name} true',
            )
        )
    for name, hist in estimate_histories.items():
        fig.add_trace(
            go.Scatter(
                x=list(hist.get('x', [])),
                y=list(hist.get('y', [])),
                mode='lines',
                name=f'{name} estimated',
                line={'dash': 'dot'},
            )
        )

    fig.update_layout(
        title='Executed Mission: Planned vs True vs Estimated (2D)',
        xaxis_title='X [m]',
        yaxis_title='Y [m]',
        height=650,
        xaxis={'scaleanchor': 'y', 'scaleratio': 1.0},
        margin={'l': 20, 'r': 20, 't': 50, 'b': 20},
    )
    return fig


def histories_to_plotly_3d(
    planned_paths: Mapping[str, List[Tuple[float, float, float]]],
    truth_histories: Mapping[str, Mapping[str, Sequence[float]]],
    estimate_histories: Mapping[str, Mapping[str, Sequence[float]]],
) -> go.Figure:
    fig = go.Figure()
    for name, plan in planned_paths.items():
        if plan:
            fig.add_trace(
                go.Scatter3d(
                    x=[p[0] for p in plan],
                    y=[p[1] for p in plan],
                    z=[p[2] for p in plan],
                    mode='lines+markers',
                    name=f'{name} planned',
                    line={'dash': 'dash'},
                )
            )
    for name, hist in truth_histories.items():
        fig.add_trace(
            go.Scatter3d(
                x=list(hist.get('x', [])),
                y=list(hist.get('y', [])),
                z=list(hist.get('z', [])),
                mode='lines',
                name=f'{name} true',
            )
        )
    for name, hist in estimate_histories.items():
        fig.add_trace(
            go.Scatter3d(
                x=list(hist.get('x', [])),
                y=list(hist.get('y', [])),
                z=list(hist.get('z', [])),
                mode='lines',
                name=f'{name} estimated',
                line={'dash': 'dot'},
            )
        )
    fig.update_layout(
        title='Executed Mission: Planned vs True vs Estimated (3D Isometric)',
        scene={
            'xaxis_title': 'X [m]',
            'yaxis_title': 'Y [m]',
            'zaxis_title': 'Z [m, local NED]',
            'camera': {'eye': {'x': 1.6, 'y': -1.6, 'z': 1.0}},
            'aspectmode': 'data',
            'xaxis': {'autorange': True},
            'yaxis': {'autorange': True},
            'zaxis': {'autorange': True},
        },
        height=720,
        margin={'l': 0, 'r': 0, 't': 50, 'b': 0},
    )
    try:
        fig.update_scenes(xaxis_autorange=True, yaxis_autorange=True, zaxis_autorange=True)
    except Exception:
        pass
    return fig


def build_timeseries_figure(vehicle_logs: Mapping[str, pd.DataFrame], column: str, title: str, yaxis: str) -> go.Figure:
    fig = go.Figure()
    for vehicle_name, df in vehicle_logs.items():
        if column not in df.columns:
            continue
        samples = df[df.get('row_type', pd.Series(index=df.index, dtype=object)).fillna('sample') == 'sample'].copy()
        if 'timestamp_s' not in samples.columns or samples.empty:
            continue
        series = pd.to_numeric(samples[column], errors='coerce')
        if series.notna().sum() == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=pd.to_numeric(samples['timestamp_s'], errors='coerce'),
                y=series,
                mode='lines',
                name=vehicle_name,
            )
        )
    fig.update_layout(title=title, xaxis_title='Time [s]', yaxis_title=yaxis, height=360, margin={'l': 20, 'r': 20, 't': 50, 'b': 20})
    return fig


def build_position_error_timeseries(vehicle_logs: Mapping[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for vehicle_name, df in vehicle_logs.items():
        samples = df[df.get('row_type', pd.Series(index=df.index, dtype=object)).fillna('sample') == 'sample'].copy()
        if samples.empty or 'timestamp_s' not in samples.columns:
            continue
        needed = ['true_x', 'true_y', 'true_z', 'est_x', 'est_y', 'est_z']
        if not all(col in samples.columns for col in needed):
            continue
        t = pd.to_numeric(samples['timestamp_s'], errors='coerce')
        tx = pd.to_numeric(samples['true_x'], errors='coerce')
        ty = pd.to_numeric(samples['true_y'], errors='coerce')
        tz = pd.to_numeric(samples['true_z'], errors='coerce')
        ex = pd.to_numeric(samples['est_x'], errors='coerce')
        ey = pd.to_numeric(samples['est_y'], errors='coerce')
        ez = pd.to_numeric(samples['est_z'], errors='coerce')
        err = ((tx - ex) ** 2 + (ty - ey) ** 2 + (tz - ez) ** 2) ** 0.5
        if err.notna().sum() == 0:
            continue
        fig.add_trace(go.Scatter(x=t, y=err, mode='lines', name=vehicle_name))
    fig.update_layout(title='Position Error', xaxis_title='Time [s]', yaxis_title='Error [m]', height=360, margin={'l': 20, 'r': 20, 't': 50, 'b': 20})
    return fig




def _sample_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.get('row_type', pd.Series(index=df.index, dtype=object)).fillna('sample') == 'sample'].copy()


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors='coerce').fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _object_series(df: pd.DataFrame, column: str, default: str = '') -> pd.Series:
    if column in df.columns:
        return df[column].fillna(default)
    return pd.Series(default, index=df.index, dtype=object)


def build_dvl_modifier_timeseries(vehicle_logs: Mapping[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for vehicle_name, df in vehicle_logs.items():
        samples = _sample_rows(df)
        if samples.empty or 'timestamp_s' not in samples.columns:
            continue
        if 'dvl_innovation_norm_mps' not in samples.columns:
            continue
        t = pd.to_numeric(samples['timestamp_s'], errors='coerce')
        innovation = _numeric_series(samples, 'dvl_innovation_norm_mps', 0.0)
        vel_gain = _numeric_series(samples, 'dvl_vel_blend_gain', 1.0)
        pos_gain = _numeric_series(samples, 'dvl_pos_blend_gain', 0.0)
        gain_sum = vel_gain + pos_gain
        modifier = innovation * gain_sum
        accepted = _numeric_series(samples, 'dvl_accepted', 0.0)
        modifier = modifier.where(accepted > 0, 0.0)
        if modifier.notna().sum() == 0:
            continue
        fig.add_trace(go.Scatter(x=t, y=modifier, mode='lines', name=vehicle_name))
    fig.update_layout(
        title='DVL Modifier',
        xaxis_title='Time [s]',
        yaxis_title='Effective DVL correction [m/s-equivalent]',
        height=360,
        margin={'l': 20, 'r': 20, 't': 50, 'b': 20},
    )
    return fig


def build_bathy_modifier_timeseries(vehicle_logs: Mapping[str, pd.DataFrame]) -> go.Figure:
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    for vehicle_name, df in vehicle_logs.items():
        samples = _sample_rows(df)
        if samples.empty or 'timestamp_s' not in samples.columns:
            continue
        t = pd.to_numeric(samples['timestamp_s'], errors='coerce')
        corr_x = _numeric_series(samples, 'correction_x', 0.0)
        corr_y = _numeric_series(samples, 'correction_y', 0.0)
        modifier = (corr_x.pow(2) + corr_y.pow(2)).pow(0.5)
        if modifier.notna().sum() > 0:
            fig.add_trace(
                go.Scatter(x=t, y=modifier, mode='lines', name=f'{vehicle_name} modifier'),
                secondary_y=False,
            )

        status_series = _object_series(samples, 'bathy_status', '')
        accepted = _numeric_series(samples, 'bathy_accepted', 0.0)
        enabled = _numeric_series(samples, 'bathy_enabled', 0.0)
        status_numeric = []
        for is_enabled, is_accepted, status in zip(enabled, accepted, status_series):
            if is_enabled <= 0 or str(status).strip().lower() in {'', 'disabled'}:
                status_numeric.append(0)
            elif is_accepted > 0 or str(status).strip().lower() in {'accepted', 'applied'}:
                status_numeric.append(2)
            else:
                status_numeric.append(1)
        fig.add_trace(
            go.Scatter(
                x=t, y=status_numeric, mode='lines', line_shape='hv',
                name=f'{vehicle_name} status',
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title='Bathymetry Modifier and Status',
        xaxis_title='Time [s]',
        height=360,
        margin={'l': 20, 'r': 20, 't': 50, 'b': 20},
        legend_title='Trace',
    )
    fig.update_yaxes(title_text='Bathy correction magnitude [m]', secondary_y=False)
    fig.update_yaxes(
        title_text='Bathy state',
        secondary_y=True,
        tickmode='array',
        tickvals=[0, 1, 2],
        ticktext=['off', 'skipped', 'applied'],
        range=[-0.2, 2.2],
    )
    return fig


def compute_kpis(vehicle_logs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    mission_time_s = 0.0

    for vehicle_name, df in vehicle_logs.items():
        samples = df[df.get('row_type', pd.Series(index=df.index, dtype=object)).fillna('sample') == 'sample'].copy()
        if samples.empty or 'timestamp_s' not in samples.columns:
            continue

        t = pd.to_numeric(samples['timestamp_s'], errors='coerce')
        mission_time_s = max(mission_time_s, float(t.max()) if t.notna().any() else 0.0)

        integrated_thrust = float('nan')
        if 'thrust' in samples.columns:
            thrust = pd.to_numeric(samples['thrust'], errors='coerce')
            valid = pd.DataFrame({'t': t, 'thrust': thrust}).dropna()
            if len(valid) >= 2:
                integrated_thrust = float((valid['thrust'].abs() * valid['t'].diff().fillna(0.0)).sum())
            elif len(valid) == 1:
                integrated_thrust = 0.0

        if 'distance_traveled_m' in samples.columns:
            dist = pd.to_numeric(samples['distance_traveled_m'], errors='coerce')
            total_distance_m = float(dist.dropna().iloc[-1]) if dist.notna().any() else 0.0
        else:
            x = pd.to_numeric(samples.get('true_x'), errors='coerce')
            y = pd.to_numeric(samples.get('true_y'), errors='coerce')
            z = pd.to_numeric(samples.get('true_z'), errors='coerce')
            diffs = ((x.diff().fillna(0.0) ** 2) + (y.diff().fillna(0.0) ** 2) + (z.diff().fillna(0.0) ** 2)) ** 0.5
            total_distance_m = float(diffs.sum())

        avg_abs_err_m = float('nan')
        final_abs_err_m = float('nan')
        max_abs_err_m = float('nan')
        needed = ['true_x', 'true_y', 'true_z', 'est_x', 'est_y', 'est_z']
        if all(col in samples.columns for col in needed):
            tx = pd.to_numeric(samples['true_x'], errors='coerce')
            ty = pd.to_numeric(samples['true_y'], errors='coerce')
            tz = pd.to_numeric(samples['true_z'], errors='coerce')
            ex = pd.to_numeric(samples['est_x'], errors='coerce')
            ey = pd.to_numeric(samples['est_y'], errors='coerce')
            ez = pd.to_numeric(samples['est_z'], errors='coerce')
            err = ((tx - ex) ** 2 + (ty - ey) ** 2 + (tz - ez) ** 2) ** 0.5
            valid_err = err.dropna()
            if not valid_err.empty:
                avg_abs_err_m = float(valid_err.mean())
                final_abs_err_m = float(valid_err.iloc[-1])
                max_abs_err_m = float(valid_err.max())

        rows.append({
            'vehicle': vehicle_name,
            'integrated_thrust': integrated_thrust,
            'distance_traveled_m': total_distance_m,
            'avg_abs_position_error_m': avg_abs_err_m,
            'final_abs_position_error_m': final_abs_err_m,
            'max_abs_position_error_m': max_abs_err_m,
        })

    out = pd.DataFrame(rows)
    out.attrs['mission_time_s'] = mission_time_s
    return out

def slice_history(history: Mapping[str, Sequence[float]], step: int) -> Dict[str, List[float]]:
    if step <= 1:
        return {k: list(v) for k, v in history.items()}
    time_key = next(iter(history.keys()), None)
    n = len(history.get(time_key, [])) if time_key is not None else 0
    keep = list(range(0, n, step))
    if n and keep[-1] != n - 1:
        keep.append(n - 1)
    return {k: [list(v)[i] for i in keep if i < len(v)] for k, v in history.items()}


def downsample_histories(
    histories: Mapping[str, Mapping[str, Sequence[float]]],
    max_frames: int = ANIMATION_MAX_FRAMES,
) -> Tuple[Dict[str, Dict[str, List[float]]], int]:
    if not histories:
        return {}, 1
    max_len = max(len(hist.get('x', [])) for hist in histories.values())
    if max_len <= max_frames:
        return {name: {k: list(v) for k, v in hist.items()} for name, hist in histories.items()}, 1
    step = max(1, math.ceil(max_len / max_frames))
    return {name: slice_history(hist, step) for name, hist in histories.items()}, step


def write_animation_gif(
    output_path: Path,
    animate_fn,
    planned_paths: Mapping[str, List[Tuple[float, float, float]]],
    truth_histories: Mapping[str, Mapping[str, Sequence[float]]],
    estimate_histories: Mapping[str, Mapping[str, Sequence[float]]],
    dt: float,
    playback_speed: float,
) -> Path:
    reduced_truth, step = downsample_histories(truth_histories)
    reduced_est = {name: slice_history(hist, step) for name, hist in estimate_histories.items()}
    animation = animate_fn(
        planned_paths,
        reduced_truth,
        reduced_est,
        dt=float(dt) * float(step),
        playback_speed=float(playback_speed),
    )
    fps = max(1, min(20, int(round(4.0 * max(playback_speed, 0.25)))))
    animation.save(str(output_path), writer=PillowWriter(fps=fps))
    plt.close(animation._fig)
    return output_path


def run_simulation(
    scenario_path: Path,
    base_sim_cfg: Mapping[str, Any],
    weather_mode: str,
    controller_label: str,
    imu_drift_enabled: bool,
    dvl_sensor_enabled: bool,
    bathy_sensor_enabled: bool,
    bathy_blend_gain: Optional[float],
    map_png_path: Optional[Path],
    playback_speed: float,
    max_time_s: float,
    output_gif_2d: bool,
    output_gif_3d: bool,
    assumed_compute_power_w: float = 0.0,
) -> RunArtifacts:
    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = LOG_ROOT / f'streamlit_{run_stamp}'
    run_dir.mkdir(parents=True, exist_ok=True)

    runtime_sim_cfg, notes = build_runtime_sim_config(
        base_cfg=base_sim_cfg,
        weather_mode=weather_mode,
        controller_label=controller_label,
        imu_drift_enabled=imu_drift_enabled,
        dvl_sensor_enabled=dvl_sensor_enabled,
        bathy_sensor_enabled=bathy_sensor_enabled,
        bathy_blend_gain=bathy_blend_gain,
        map_png_path=map_png_path,
        run_dir=run_dir,
    )
    sim_cfg_path = run_dir / 'sim_config.runtime.yaml'
    save_yaml(runtime_sim_cfg, sim_cfg_path)

    scenario = ma.load_yaml_file(scenario_path)
    sim_cfg = ma.load_sim_config(sim_cfg_path)
    animator = ma.ScenarioAnimator(
        scenario=scenario,
        sim_cfg=sim_cfg,
        time_step_s=float(sim_cfg.get('global', {}).get('integration_dt_s', ma.DEFAULT_TIME_STEP_S)),
        output_dir=run_dir,
    )
    t0_wall = time.perf_counter()
    animator.run(max_time_s=max_time_s)
    wall_clock_s = float(time.perf_counter() - t0_wall)
    compute_energy_wh_est = None
    if float(assumed_compute_power_w) > 0.0:
        compute_energy_wh_est = float(assumed_compute_power_w) * wall_clock_s / 3600.0

    planned_paths = dict(animator.planned_paths)
    truth_histories = {name: state.truth_history for name, state in animator.vehicle_states.items()}
    estimate_histories = {name: state.estimate_history for name, state in animator.vehicle_states.items()}
    vehicle_logs = {name: pd.DataFrame(state.log_rows) for name, state in animator.vehicle_states.items()}

    dt = float(sim_cfg.get('global', {}).get('log_dt_s', sim_cfg.get('logging', {}).get('sample_interval_s', 1.0)))
    if output_gif_2d:
        write_animation_gif(run_dir / 'mission_2d.gif', ma.animate_mission_2d, planned_paths, truth_histories, estimate_histories, dt, playback_speed)
    if output_gif_3d:
        write_animation_gif(run_dir / 'mission_3d.gif', ma.animate_mission_3d, planned_paths, truth_histories, estimate_histories, dt, playback_speed)

    stdout_summary = '\n'.join([
        f'Run directory: {relative_to_repo(run_dir)}',
        f'Scenario: {relative_to_repo(scenario_path)}',
        f'Controller: {CONTROL_OPTIONS[controller_label]}',
        f'Weather mode: {weather_mode}',
        f'IMU drift enabled: {imu_drift_enabled}',
        f'DVL sensor enabled: {dvl_sensor_enabled}',
        f'Bathymetry sensor enabled: {bathy_sensor_enabled}',
        f'Wall-clock compute duration: {wall_clock_s:.3f} s',
        *(([f'Estimated compute energy: {compute_energy_wh_est:.6f} Wh'] if compute_energy_wh_est is not None else [])),
        *notes,
    ])

    return RunArtifacts(
        run_dir=run_dir,
        scenario_path=scenario_path,
        sim_config_path=sim_cfg_path,
        scenario=scenario,
        sim_cfg=sim_cfg,
        planned_paths=planned_paths,
        truth_histories=truth_histories,
        estimate_histories=estimate_histories,
        vehicle_logs=vehicle_logs,
        stdout_summary=stdout_summary,
        map_png_path=map_png_path,
        wall_clock_s=wall_clock_s,
        compute_energy_wh_est=compute_energy_wh_est,
    )


def ensure_state() -> None:
    st.session_state.setdefault('last_run', None)


def main() -> None:
    ensure_state()
    st.title('Poseidon Mission Sandbox')

    st.markdown(
    """
    ### A maritime modeling simulation environment for software-defined autonomous systems.

    [GitHub repository](%s)

    This tool blends vehicle physics, guidance and control strategies, sensor suites, and real-world geographic and weather data to perform comprehensive mission analysis on autonomous maritime concepts.
        """ % github_repo_web_url(APP_REPO_URL)
    )

    if not SIM_CONFIG_PATH.exists():
        st.error(f'Missing sim config: {SIM_CONFIG_PATH}')
        st.stop()
    if not SCENARIO_DIR.exists():
        st.error(f'Missing scenario directory: {SCENARIO_DIR}')
        st.stop()

    scenarios = discover_scenarios()
    if not scenarios:
        st.error(f'No scenarios found in {SCENARIO_DIR}')
        st.stop()

    maps = discover_png_maps()
    base_sim_cfg = load_yaml(SIM_CONFIG_PATH)

    scenario_labels = list(scenarios.keys())
    default_index = 0
    for i, path in enumerate(scenarios.values()):
        if path.name == DEFAULT_SCENARIO_FILE:
            default_index = i
            break

    with st.sidebar:
        st.header('Mission setup')
        scenario_label = st.selectbox('Scenario', scenario_labels, index=default_index)
        scenario_path = scenarios[scenario_label]

        default_map_path = default_map_for_scenario(scenario_path)
        default_map_name = default_map_name_for_scenario(scenario_path) or ''
        map_options: Dict[str, Path] = dict(maps)
        if default_map_name:
            map_options.setdefault(default_map_name.replace('_', ' ').replace('.png', ''), MAP_DIR / default_map_name)

        map_labels = list(map_options.keys())
        selected_map_path = default_map_path
        map_label = default_map_name.replace('_', ' ').replace('.png', '') if default_map_name else ''
        if map_labels:
            selected_index = 0
            default_display_label = default_map_name.replace('_', ' ').replace('.png', '') if default_map_name else ''
            if default_display_label in map_labels:
                selected_index = map_labels.index(default_display_label)
            map_label = st.selectbox('Map PNG', map_labels, index=selected_index)
            selected_map_path = map_options[map_label]

        weather_mode = st.selectbox('Weather Mode', WEATHER_MODES, index=0)
        controller_label = st.selectbox('Control Strategy', list(CONTROL_OPTIONS.keys()), index=1)
        imu_drift_enabled = st.checkbox('AUV IMU Drift', value=True)
        dvl_sensor_enabled = st.checkbox('AUV DVL', value=False)
        bathy_sensor_enabled = st.checkbox('AUV Bathymetry Sensor', value=True)
        bathy_blend_gain = st.slider('Bathy Blend Gain', min_value=0.0, max_value=1.0, value=0.20, step=0.01, help='Dominant bathymetry aggressiveness term. Higher values pull the estimate harder toward the bathy match.')
        playback_speed = st.slider('Playback Speed', min_value=0.25, max_value=50.0, value=5.0, step=0.25)
        max_time_s = st.number_input('Max Simulation Time [s]', min_value=10.0, max_value=50000.0, value=DEFAULT_MAX_TIME_S, step=10.0)
        output_gif_2d = st.checkbox('Export 2D GIF', value=False)
        output_gif_3d = st.checkbox('Export 3D GIF', value=False)
        assumed_compute_power_w = st.number_input('Assumed compute power [W] for energy proxy', min_value=0.0, max_value=2000.0, value=0.0, step=5.0)

        run_clicked = st.button('Run mission', type='primary', use_container_width=True)

    preview_col, info_col = st.columns([2, 1])
    scenario = load_yaml(scenario_path)
    with preview_col:
        st.plotly_chart(build_preview_figure(scenario), use_container_width=True)
    with info_col:
        st.subheader("Paths")
        st.code(
            '\n'.join([
                f'scenario: {relative_to_repo(scenario_path)}',
                f'sim config: {relative_to_repo(SIM_CONFIG_PATH)}',
                f'maps: {relative_to_repo(MAP_DIR)}',
                f'map PNG source: {github_raw_base_url(APP_REPO_URL, APP_REPO_BRANCH)}/environment/maps',
                f'weather: {relative_to_repo(WEATHER_DIR)}',
                f'logs: {relative_to_repo(LOG_ROOT)}',
            ]),
            language='text',
        )
        selected_map_name = selected_map_path.name if selected_map_path is not None else default_map_name_for_scenario(scenario_path)
        selected_map_url = remote_png_url_for_map(selected_map_path, selected_map_name or map_label)
        if selected_map_url:
            st.image(selected_map_url, caption=selected_map_name or map_label, use_container_width=True)
        elif selected_map_path is not None and selected_map_path.exists():
            st.image(str(selected_map_path), caption=selected_map_path.name, use_container_width=True)
        else:
            st.info('No static PNG available for the selected scenario. Verify the PNG is committed under environment/maps/.')
        st.caption("Bathymetry Data: NOAA DEM Global Mosaic")
        st.caption("Weather Data: NOAA NDBC Station 44014")


    if run_clicked:
        with st.spinner('Running mission analysis...'):
            st.session_state.last_run = run_simulation(
                scenario_path=scenario_path,
                base_sim_cfg=base_sim_cfg,
                weather_mode=weather_mode,
                controller_label=controller_label,
                imu_drift_enabled=imu_drift_enabled,
                dvl_sensor_enabled=dvl_sensor_enabled,
                bathy_sensor_enabled=bathy_sensor_enabled,
                bathy_blend_gain=float(bathy_blend_gain),
                map_png_path=selected_map_path,
                playback_speed=playback_speed,
                max_time_s=float(max_time_s),
                output_gif_2d=output_gif_2d,
                output_gif_3d=output_gif_3d,
                assumed_compute_power_w=float(assumed_compute_power_w),
            )

    artifacts: Optional[RunArtifacts] = st.session_state.last_run
    if artifacts is None:
        st.info('Configure the mission and click Run mission.')
        return

    st.divider()

    gif2d = artifacts.run_dir / 'mission_2d.gif'
    gif3d = artifacts.run_dir / 'mission_3d.gif'
    st.subheader('Mission animations')
    gif_cols = st.columns(2)
    with gif_cols[0]:
        if gif2d.exists():
            st.image(str(gif2d), caption='2D mission animation', use_container_width=True)
        else:
            st.info('2D mission GIF was not exported for this run.')
    with gif_cols[1]:
        if gif3d.exists():
            st.image(str(gif3d), caption='3D mission animation', use_container_width=True)
        else:
            st.info('3D mission GIF was not exported for this run.')

    kpi_df = compute_kpis(artifacts.vehicle_logs)
    mission_time_s = float(kpi_df.attrs.get('mission_time_s', 0.0))

    st.subheader('Mission KPIs')
    top = st.columns(3)
    top[0].metric('Total mission time', f'{mission_time_s:.1f} s')
    top[1].metric('Wall-clock compute duration', f'{artifacts.wall_clock_s:.3f} s')
    if artifacts.compute_energy_wh_est is not None:
        top[2].metric('Compute energy proxy', f'{artifacts.compute_energy_wh_est:.6f} Wh')

    if not kpi_df.empty:
        for _, row in kpi_df.iterrows():
            st.markdown(f"**{row['vehicle']}**")
            cols = st.columns(5)
            cols[0].metric('Energy expended (integrated thrust)', f"{row['integrated_thrust']:.2f} thrust*s")
            cols[1].metric('Distance traveled', f"{row['distance_traveled_m']:.1f} m")
            cols[2].metric('Avg abs positioning error', f"{row['avg_abs_position_error_m']:.4f} m")
            cols[3].metric('Final abs positioning error', f"{row['final_abs_position_error_m']:.4f} m")
            cols[4].metric('Max abs positioning error', f"{row['max_abs_position_error_m']:.4f} m")

    st.subheader('State time series')
    ts_top_left, ts_top_right = st.columns(2)
    with ts_top_left:
        st.plotly_chart(build_position_error_timeseries(artifacts.vehicle_logs), use_container_width=True)
    with ts_top_right:
        st.plotly_chart(build_timeseries_figure(artifacts.vehicle_logs, 'thrust', 'Thrust', 'Thrust command'), use_container_width=True)

    ts_bottom_left, ts_bottom_right = st.columns(2)
    with ts_bottom_left:
        st.plotly_chart(build_timeseries_figure(artifacts.vehicle_logs, 'delta_act_deg', 'Rudder Position', 'deg'), use_container_width=True)
    with ts_bottom_right:
        st.plotly_chart(build_timeseries_figure(artifacts.vehicle_logs, 'delta_cmd_deg', 'Rudder Command', 'deg'), use_container_width=True)

    ts_third_left, ts_third_right = st.columns(2)
    with ts_third_left:
        st.plotly_chart(build_dvl_modifier_timeseries(artifacts.vehicle_logs), use_container_width=True)
    with ts_third_right:
        st.plotly_chart(build_bathy_modifier_timeseries(artifacts.vehicle_logs), use_container_width=True)

    exec_cols = st.columns(2)
    with exec_cols[0]:
        st.plotly_chart(
            histories_to_plotly_2d(artifacts.planned_paths, artifacts.truth_histories, artifacts.estimate_histories),
            use_container_width=True,
        )
    with exec_cols[1]:
        st.plotly_chart(
            histories_to_plotly_3d(artifacts.planned_paths, artifacts.truth_histories, artifacts.estimate_histories),
            use_container_width=True,
        )

    st.subheader('Log preview')
    preview_frames: List[pd.DataFrame] = []
    for vehicle_name, df in artifacts.vehicle_logs.items():
        if df.empty:
            continue
        sample_df = df.copy().head(50)
        sample_df.insert(0, 'vehicle', vehicle_name)
        preview_frames.append(sample_df)
    if preview_frames:
        st.dataframe(pd.concat(preview_frames, ignore_index=True), use_container_width=True)
    else:
        st.info('No vehicle log rows available from the run.')

    st.subheader('Run output')
    st.code(artifacts.stdout_summary, language='text')
    st.code(relative_to_repo(artifacts.sim_config_path), language='text')


if __name__ == '__main__':
    main()
