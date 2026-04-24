from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class VectorScenario:
    label: str
    wind_speed_mps: float
    wind_dir_deg: float
    wind_xy_mps: Tuple[float, float]
    current_speed_mps: float
    current_dir_deg: float
    current_xy_mps: Tuple[float, float]
    current_z_mps: float = 0.0


@dataclass
class ForcingSummary:
    wind_column: str
    wind_dir_column: str
    current_column: str
    current_dir_column: str
    row_count: int
    wind_mean_mps: float
    wind_std_mps: float
    current_mean_mps: float
    current_std_mps: float
    scenarios: List[VectorScenario]


def _normalize_columns(columns: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for col in columns:
        key = ''.join(ch.lower() for ch in col if ch.isalnum())
        out[key] = col
    return out


def _find_column(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    norm = _normalize_columns(columns)
    for candidate in candidates:
        key = ''.join(ch.lower() for ch in candidate if ch.isalnum())
        if key in norm:
            return norm[key]
    return None


def _coerce_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .replace({'MM': np.nan, '99': np.nan, '999': np.nan, '9999': np.nan, 'NaN': np.nan, 'nan': np.nan})
    )
    return pd.to_numeric(cleaned, errors='coerce')


def _clip_nonnegative(value: float) -> float:
    return max(0.0, float(value))


def _dir_to_xy(speed_mps: float, direction_deg: float, convention: str) -> Tuple[float, float]:
    """
    Returns (vx_east, vy_north).

    convention='to'   : direction points toward motion.
    convention='from' : direction points from source; vector is reversed.

    Bearings are assumed clockwise from true north.
    """
    theta = math.radians(direction_deg)
    east = speed_mps * math.sin(theta)
    north = speed_mps * math.cos(theta)
    if convention == 'from':
        east *= -1.0
        north *= -1.0
    return (east, north)


def _quantile_direction(df: pd.DataFrame, speed_col: str, dir_col: str, target_speed: float) -> float:
    valid = df[[speed_col, dir_col]].dropna().sort_values(speed_col)
    if valid.empty:
        raise ValueError(f'No valid rows available for {speed_col}/{dir_col}')
    idx = (valid[speed_col] - target_speed).abs().idxmin()
    return float(valid.loc[idx, dir_col])


def load_buoy_csv(path: Path) -> pd.DataFrame:
    """Load a NOAA-style CSV.

    Handles either:
    - a normal header row, or
    - a commented header row beginning with '#'.
    """
    lines = path.read_text(encoding='utf-8').splitlines()
    if not lines:
        raise ValueError('CSV file is empty')

    header = None
    data_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            candidate = stripped.lstrip('#').strip()
            if ',' in candidate and header is None:
                header = [item.strip() for item in candidate.split(',')]
            elif ',' not in candidate and header is None and len(candidate.split()) >= 4:
                header = candidate.split()
            continue
        data_lines.append(line)

    if header is not None:
        from io import StringIO
        payload = '\n'.join(data_lines)
        if ',' in payload.splitlines()[0]:
            df = pd.read_csv(StringIO(payload), header=None, names=header)
        else:
            df = pd.read_csv(StringIO(payload), delim_whitespace=True, header=None, names=header)
    else:
        df = pd.read_csv(path)

    if df.empty:
        raise ValueError('CSV parsed empty')
    return df


def summarize_forcing(
    csv_path: Path,
    wind_speed_col: Optional[str] = None,
    wind_dir_col: Optional[str] = None,
    current_speed_col: Optional[str] = None,
    current_dir_col: Optional[str] = None,
    wind_units: str = 'mps',
    current_units: str = 'mps',
    wind_direction_convention: str = 'from',
    current_direction_convention: str = 'to',
) -> ForcingSummary:
    df = load_buoy_csv(csv_path)

    wind_speed_col = wind_speed_col or _find_column(df.columns, ['continuous_winds__WSPD', 'standard_met__WSPD', 'WSPD', 'wind_speed', 'windspd', 'wind_spd', 'wspd'])
    wind_dir_col = wind_dir_col or _find_column(df.columns, ['continuous_winds__WDIR', 'standard_met__WDIR', 'WDIR', 'wind_dir', 'wdir', 'mwd'])
    current_speed_col = current_speed_col or _find_column(df.columns, ['adcp_currents__SPD01', 'CURSPD', 'current_speed', 'cspd', 'spd', 'speed'])
    current_dir_col = current_dir_col or _find_column(df.columns, ['adcp_currents__DIR01', 'CURDIR', 'current_dir', 'cdir', 'dir', 'direction'])

    missing = [
        name
        for name, value in [
            ('wind speed', wind_speed_col),
            ('wind direction', wind_dir_col),
            ('current speed', current_speed_col),
            ('current direction', current_dir_col),
        ]
        if value is None
    ]
    if missing:
        raise ValueError(
            'Could not infer required columns: ' + ', '.join(missing) + '. '
            f'Available columns: {list(df.columns)}'
        )

    df = df.copy()
    df[wind_speed_col] = _coerce_numeric(df[wind_speed_col])
    df[wind_dir_col] = _coerce_numeric(df[wind_dir_col])
    df[current_speed_col] = _coerce_numeric(df[current_speed_col])
    df[current_dir_col] = _coerce_numeric(df[current_dir_col])

    if wind_units.lower() in {'kts', 'kt', 'knots'}:
        df[wind_speed_col] *= 0.514444
    elif wind_units.lower() in {'cmps', 'cm/s'}:
        df[wind_speed_col] *= 0.01

    if current_units.lower() in {'kts', 'kt', 'knots'}:
        df[current_speed_col] *= 0.514444
    elif current_units.lower() in {'cmps', 'cm/s'}:
        df[current_speed_col] *= 0.01

    wind_valid = df[[wind_speed_col, wind_dir_col]].dropna()
    current_valid = df[[current_speed_col, current_dir_col]].dropna()
    if wind_valid.empty or current_valid.empty:
        raise ValueError('Not enough valid wind/current rows after cleaning missing values')

    wind_mean = float(wind_valid[wind_speed_col].mean())
    wind_std = float(wind_valid[wind_speed_col].std(ddof=1)) if len(wind_valid) > 1 else 0.0
    current_mean = float(current_valid[current_speed_col].mean())
    current_std = float(current_valid[current_speed_col].std(ddof=1)) if len(current_valid) > 1 else 0.0

    scenario_defs = [
        ('min', float(wind_valid[wind_speed_col].min()), float(current_valid[current_speed_col].min())),
        ('minus_2sd', _clip_nonnegative(wind_mean - 2.0 * wind_std), _clip_nonnegative(current_mean - current_std)),
        ('minus_1sd', _clip_nonnegative(wind_mean - 1.0 * wind_std), _clip_nonnegative(current_mean - current_std)),
        ('mean', _clip_nonnegative(wind_mean), _clip_nonnegative(current_mean - current_std)),
        ('plus_1sd', _clip_nonnegative(wind_mean + 1.0 * wind_std), _clip_nonnegative(current_mean - current_std)),
        ('plus_2sd', _clip_nonnegative(wind_mean + 2.0 * wind_std), _clip_nonnegative(current_mean + 2.0 * current_std)),
    ]

    scenarios: List[VectorScenario] = []
    for label, wind_speed, current_speed in scenario_defs:
        wind_dir = _quantile_direction(wind_valid, wind_speed_col, wind_dir_col, wind_speed)
        current_dir = _quantile_direction(current_valid, current_speed_col, current_dir_col, current_speed)
        scenarios.append(
            VectorScenario(
                label=label,
                wind_speed_mps=wind_speed,
                wind_dir_deg=wind_dir,
                wind_xy_mps=_dir_to_xy(wind_speed, wind_dir, wind_direction_convention),
                current_speed_mps=current_speed,
                current_dir_deg=current_dir,
                current_xy_mps=_dir_to_xy(current_speed, current_dir, current_direction_convention),
            )
        )

    return ForcingSummary(
        wind_column=wind_speed_col,
        wind_dir_column=wind_dir_col,
        current_column=current_speed_col,
        current_dir_column=current_dir_col,
        row_count=int(min(len(wind_valid), len(current_valid))),
        wind_mean_mps=wind_mean,
        wind_std_mps=wind_std,
        current_mean_mps=current_mean,
        current_std_mps=current_std,
        scenarios=scenarios,
    )


def build_env_blocks(summary: ForcingSummary) -> Dict[str, Dict[str, object]]:
    blocks: Dict[str, Dict[str, object]] = {}
    for scenario in summary.scenarios:
        blocks[scenario.label] = {
            'environment': {
                'surface_current_enabled': True,
                'surface_current_xy': [round(scenario.current_xy_mps[0], 4), round(scenario.current_xy_mps[1], 4)],
                'underwater_current_enabled': True,
                'underwater_current_xy': [round(scenario.current_xy_mps[0], 4), round(scenario.current_xy_mps[1], 4)],
                'underwater_current_z_enabled': False,
                'underwater_current_z': 0.0,
                'wind_xy_reference': [round(scenario.wind_xy_mps[0], 4), round(scenario.wind_xy_mps[1], 4)],
            }
        }
    return blocks


def _to_jsonable(summary: ForcingSummary) -> Dict[str, object]:
    data = asdict(summary)
    data['env_blocks'] = build_env_blocks(summary)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize NOAA buoy CSV into simple environmental forcing vectors.')
    parser.add_argument('csv_path', type=Path)
    parser.add_argument('--wind-speed-col', default=None)
    parser.add_argument('--wind-dir-col', default=None)
    parser.add_argument('--current-speed-col', default=None)
    parser.add_argument('--current-dir-col', default=None)
    parser.add_argument('--wind-units', default='mps', choices=['mps', 'kts', 'kt', 'knots', 'cmps', 'cm/s'])
    parser.add_argument('--current-units', default='cm/s', choices=['mps', 'kts', 'kt', 'knots', 'cmps', 'cm/s'])
    parser.add_argument('--wind-direction-convention', default='from', choices=['from', 'to'])
    parser.add_argument('--current-direction-convention', default='to', choices=['from', 'to'])
    parser.add_argument('--json-out', type=Path, default=None)
    args = parser.parse_args()

    summary = summarize_forcing(
        csv_path=args.csv_path,
        wind_speed_col=args.wind_speed_col,
        wind_dir_col=args.wind_dir_col,
        current_speed_col=args.current_speed_col,
        current_dir_col=args.current_dir_col,
        wind_units=args.wind_units,
        current_units=args.current_units,
        wind_direction_convention=args.wind_direction_convention,
        current_direction_convention=args.current_direction_convention,
    )

    print(f'Rows used: {summary.row_count}')
    print(f'Wind mean/std (mps): {summary.wind_mean_mps:.3f} / {summary.wind_std_mps:.3f}')
    print(f'Current mean/std (mps): {summary.current_mean_mps:.3f} / {summary.current_std_mps:.3f}')
    print('')
    for scenario in summary.scenarios:
        print(f'[{scenario.label}]')
        print(f'  wind:    speed={scenario.wind_speed_mps:.3f} mps dir={scenario.wind_dir_deg:.1f} deg xy={tuple(round(v,4) for v in scenario.wind_xy_mps)}')
        print(f'  current: speed={scenario.current_speed_mps:.3f} mps dir={scenario.current_dir_deg:.1f} deg xy={tuple(round(v,4) for v in scenario.current_xy_mps)} z=0.0')
        print('')

    payload = _to_jsonable(summary)
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f'Wrote {args.json_out}')
    else:
        print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
