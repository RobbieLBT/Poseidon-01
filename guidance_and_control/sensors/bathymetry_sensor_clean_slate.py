from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rasterio
import requests


_GOOGLE_DRIVE_FILE_ID_RE = re.compile(r"(?:/file/d/|[?&]id=)([A-Za-z0-9_-]+)")


def _extract_google_drive_file_id(url: str) -> Optional[str]:
    match = _GOOGLE_DRIVE_FILE_ID_RE.search(str(url))
    return match.group(1) if match else None


def _filename_for_remote_url(url: str, cache_dir: Path) -> Path:
    drive_id = _extract_google_drive_file_id(url)
    if drive_id:
        return cache_dir / f"google_drive_{drive_id}.tiff"
    suffix = ".tiff"
    lowered = url.lower().split("?", 1)[0]
    for candidate in (".tif", ".tiff"):
        if lowered.endswith(candidate):
            suffix = candidate
            break
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"remote_bathymetry_{digest}{suffix}"


def _download_remote_geotiff(url: str, *, cache_dir: str | Path = "/tmp", timeout_s: float = 120.0) -> Path:
    """Download an HTTP/HTTPS GeoTIFF to a local cache and return the cached path.

    Public Google Drive file links are supported without a Google API key. The file
    or parent folder must still be shared as "Anyone with the link can view".
    """
    cache_path = Path(cache_dir).expanduser()
    cache_path.mkdir(parents=True, exist_ok=True)
    out_path = _filename_for_remote_url(url, cache_path)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    session = requests.Session()
    drive_id = _extract_google_drive_file_id(url)
    download_url = "https://drive.google.com/uc?export=download&id=" + drive_id if drive_id else url
    response = session.get(download_url, stream=True, timeout=float(timeout_s))

    # Large Google Drive files can return a confirmation interstitial. Follow the
    # confirm token if present; otherwise write the response as-is.
    content_type = response.headers.get("content-type", "").lower()
    if drive_id and "text/html" in content_type:
        html = response.text
        token_match = re.search(r"confirm=([0-9A-Za-z_-]+)", html)
        if token_match:
            response = session.get(
                "https://drive.google.com/uc?export=download"
                f"&confirm={token_match.group(1)}&id={drive_id}",
                stream=True,
                timeout=float(timeout_s),
            )
        else:
            raise RuntimeError(
                "Google Drive returned an HTML page instead of a TIFF. "
                "Confirm the file is shared as 'Anyone with the link can view'."
            )

    response.raise_for_status()
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with tmp_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    tmp_path.replace(out_path)
    return out_path


def resolve_bathymetry_source(path_or_url: str | Path, *, cache_dir: str | Path = "/tmp", timeout_s: float = 120.0) -> Path:
    raw = str(path_or_url).strip()
    if raw.startswith(("http://", "https://")):
        return _download_remote_geotiff(raw, cache_dir=cache_dir, timeout_s=timeout_s)
    return Path(raw).expanduser()


@dataclass
class ConeVolumeConfig:
    enabled: bool = True
    num_beams_azimuth: int = 16
    num_beams_radial: int = 4
    max_slant_range_m: float = 100.0
    min_slant_range_m: float = 1.0
    half_angle_deg: float = 35.0
    update_rate_hz: float = 2.0
    range_sigma_m: float = 0.35
    dropout_prob: float = 0.0
    min_altitude_m: float = 0.5
    max_altitude_m: float = 100.0
    debug_print: bool = False

    @property
    def half_angle_rad(self) -> float:
        return math.radians(float(self.half_angle_deg))


@dataclass
class BathyCorrectionConfig:
    enabled: bool = True
    search_radius_m: float = 40.0
    search_step_m: float = 4.0
    blend_gain: float = 0.35
    max_step_m: float = 3.0
    min_valid_returns: int = 8
    min_texture_m: float = 0.75
    max_texture_m: float = 8.0
    min_gradient_norm: float = 0.01
    max_rmse_m: float = 3.0
    min_rmse_improvement_m: float = 0.25
    imu_drift_attenuation_max: float = 0.65
    confidence_smoothing: float = 0.85


@dataclass
class ConeVolumeMeasurement:
    vehicle_xyz_m: Tuple[float, float, float]
    yaw_rad: float
    sample_xy_m: np.ndarray
    sample_bottom_z_m: np.ndarray
    sample_slant_range_m: np.ndarray
    sample_vertical_range_m: np.ndarray
    valid: np.ndarray
    altitude_m: float
    texture_m: float
    gradient_norm: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            'vehicle_xyz_m': self.vehicle_xyz_m,
            'yaw_rad': self.yaw_rad,
            'sample_xy_m': self.sample_xy_m,
            'sample_bottom_z_m': self.sample_bottom_z_m,
            'sample_slant_range_m': self.sample_slant_range_m,
            'sample_vertical_range_m': self.sample_vertical_range_m,
            'valid': self.valid,
            'altitude_m': self.altitude_m,
            'texture_m': self.texture_m,
            'gradient_norm': self.gradient_norm,
        }


class BathymetryGrid:
    def __init__(self, elevation_m: np.ndarray, dx_m: float, dy_m: float, *, source_path: Optional[Path] = None) -> None:
        self.elevation_m = np.asarray(elevation_m, dtype=float)
        self.ny, self.nx = self.elevation_m.shape
        self.dx_m = float(dx_m)
        self.dy_m = float(dy_m)
        self.source_path = Path(source_path) if source_path is not None else None
        self.width_m = max(0.0, (self.nx - 1) * self.dx_m)
        self.height_m = max(0.0, (self.ny - 1) * self.dy_m)
        self.grad_y_m, self.grad_x_m = np.gradient(self.elevation_m, self.dy_m, self.dx_m)

    @classmethod
    def from_geotiff(cls, path: str | Path) -> 'BathymetryGrid':
        path = resolve_bathymetry_source(path)
        with rasterio.open(path) as src:
            band = src.read(1).astype(float)
            nodata = src.nodata
            transform = src.transform
            bounds = src.bounds

            if nodata is not None:
                band[np.isclose(band, float(nodata))] = np.nan

            dx_native = abs(float(transform.a))
            dy_native = abs(float(transform.e))

            if src.crs is not None and getattr(src.crs, 'is_geographic', False):
                lat_mid_deg = 0.5 * (float(bounds.top) + float(bounds.bottom))
                lat_mid_rad = math.radians(lat_mid_deg)
                meters_per_deg_lat = 111320.0
                meters_per_deg_lon = 111320.0 * math.cos(lat_mid_rad)
                dx_native = dx_native * meters_per_deg_lon
                dy_native = dy_native * meters_per_deg_lat

            local = np.flipud(band)

        return cls(local, dx_m=dx_native, dy_m=dy_native, source_path=path)
    
    def contains(self, x_m: float, y_m: float) -> bool:
        return 0.0 <= x_m <= self.width_m and 0.0 <= y_m <= self.height_m

    def _interp(self, arr: np.ndarray, x_m: float, y_m: float) -> float:
        if not self.contains(x_m, y_m):
            return float('nan')
        fx = x_m / self.dx_m
        fy = y_m / self.dy_m
        x0 = int(math.floor(fx))
        y0 = int(math.floor(fy))
        x1 = min(x0 + 1, self.nx - 1)
        y1 = min(y0 + 1, self.ny - 1)
        tx = fx - x0
        ty = fy - y0
        v00 = arr[y0, x0]
        v10 = arr[y0, x1]
        v01 = arr[y1, x0]
        v11 = arr[y1, x1]
        if np.isnan([v00, v10, v01, v11]).any():
            return float('nan')
        v0 = (1.0 - tx) * v00 + tx * v10
        v1 = (1.0 - tx) * v01 + tx * v11
        return float((1.0 - ty) * v0 + ty * v1)

    def sample_elevation(self, x_m: float, y_m: float) -> float:
        return self._interp(self.elevation_m, x_m, y_m)

    def sample_gradient(self, x_m: float, y_m: float) -> Tuple[float, float]:
        return self._interp(self.grad_x_m, x_m, y_m), self._interp(self.grad_y_m, x_m, y_m)

    def describe_resolution(self) -> Dict[str, float]:
        return {
            'dx_m': float(self.dx_m),
            'dy_m': float(self.dy_m),
            'width_m': float(self.width_m),
            'height_m': float(self.height_m),
            'nx': int(self.nx),
            'ny': int(self.ny),
        }


@dataclass
class BathyCorrection:
    confidence: float
    correction_x: float
    correction_y: float
    correction_z: float
    texture: float
    status: str

def _wrap_angle(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def _clamp_norm(vec_xy: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec_xy))
    if norm <= max_norm or norm <= 1e-9:
        return vec_xy
    return vec_xy * (max_norm / norm)


def _confidence_from_texture(texture_m: float, gradient_norm: float, cfg: BathyCorrectionConfig) -> float:
    tex = _clamp((texture_m - cfg.min_texture_m) / max(1e-6, (cfg.max_texture_m - cfg.min_texture_m)), 0.0, 1.0)
    grad = 0.0 if gradient_norm <= 0.0 else _clamp(gradient_norm / max(cfg.min_gradient_norm, 1e-6), 0.0, 1.0)
    return float(tex * math.sqrt(max(0.0, grad)))


def simulate_cone_volume_measurement(
    grid: BathymetryGrid,
    vehicle_xyz_m: Tuple[float, float, float],
    yaw_rad: float,
    cfg: ConeVolumeConfig,
    rng: Optional[np.random.Generator] = None,
) -> ConeVolumeMeasurement:
    rng = rng if rng is not None else np.random.default_rng()
    x0, y0, z0 = map(float, vehicle_xyz_m)
    center_bottom_z = grid.sample_elevation(x0, y0)
    if math.isnan(center_bottom_z):
        return ConeVolumeMeasurement(vehicle_xyz_m, yaw_rad, np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool), float('nan'), 0.0, 0.0)

    altitude_m = float(z0 - center_bottom_z)
    if not (cfg.min_altitude_m <= altitude_m <= cfg.max_altitude_m):
        return ConeVolumeMeasurement(vehicle_xyz_m, yaw_rad, np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool), altitude_m, 0.0, 0.0)

    radii = np.linspace(0.25, 1.0, cfg.num_beams_radial) * min(cfg.max_slant_range_m * math.sin(cfg.half_angle_rad), altitude_m * math.tan(cfg.half_angle_rad))
    azimuths = np.linspace(0.0, 2.0 * math.pi, cfg.num_beams_azimuth, endpoint=False)

    points_xy = []
    bottoms = []
    slant = []
    vertical = []
    valid = []

    for r in radii:
        for az in azimuths:
            beam_az = yaw_rad + az
            sx = x0 + r * math.cos(beam_az)
            sy = y0 + r * math.sin(beam_az)
            bz = grid.sample_elevation(sx, sy)
            if math.isnan(bz) or rng.random() < cfg.dropout_prob:
                points_xy.append((sx, sy))
                bottoms.append(float('nan'))
                slant.append(float('nan'))
                vertical.append(float('nan'))
                valid.append(False)
                continue
            vr = float(z0 - bz)
            sr = math.hypot(r, vr)
            if sr < cfg.min_slant_range_m or sr > cfg.max_slant_range_m:
                points_xy.append((sx, sy))
                bottoms.append(float('nan'))
                slant.append(float('nan'))
                vertical.append(float('nan'))
                valid.append(False)
                continue
            measured_sr = sr + float(rng.normal(0.0, cfg.range_sigma_m))
            points_xy.append((sx, sy))
            bottoms.append(bz)
            slant.append(measured_sr)
            vertical.append(vr)
            valid.append(True)

    points_xy_arr = np.asarray(points_xy, dtype=float)
    bottoms_arr = np.asarray(bottoms, dtype=float)
    slant_arr = np.asarray(slant, dtype=float)
    vertical_arr = np.asarray(vertical, dtype=float)
    valid_arr = np.asarray(valid, dtype=bool)

    if np.count_nonzero(valid_arr) > 1:
        texture = float(np.nanstd(bottoms_arr[valid_arr]))
        gx, gy = grid.sample_gradient(x0, y0)
        gradient_norm = float(math.hypot(gx, gy)) if not (math.isnan(gx) or math.isnan(gy)) else 0.0
    else:
        texture = 0.0
        gradient_norm = 0.0

    return ConeVolumeMeasurement(
        vehicle_xyz_m=(x0, y0, z0),
        yaw_rad=float(yaw_rad),
        sample_xy_m=points_xy_arr,
        sample_bottom_z_m=bottoms_arr,
        sample_slant_range_m=slant_arr,
        sample_vertical_range_m=vertical_arr,
        valid=valid_arr,
        altitude_m=altitude_m,
        texture_m=texture,
        gradient_norm=gradient_norm,
    )


def estimate_bathy_correction(
    grid: BathymetryGrid,
    measurement: ConeVolumeMeasurement,
    prior_xy_m: Tuple[float, float],
    drift_xy_m: Tuple[float, float],
    cfg: BathyCorrectionConfig,
    prev_confidence: float = 0.0,
) -> Dict[str, Any]:
    prior = np.asarray(prior_xy_m, dtype=float)
    drift_xy = np.asarray(drift_xy_m, dtype=float)
    valid_idx = np.flatnonzero(measurement.valid)

    if not cfg.enabled:
        return {'accepted': False, 'correction_xy_m': np.zeros(2), 'confidence': 0.0, 'status': 'disabled'}
    if valid_idx.size < cfg.min_valid_returns:
        return {'accepted': False, 'correction_xy_m': np.zeros(2), 'confidence': 0.0, 'status': 'insufficient_returns'}

    raw_conf = _confidence_from_texture(measurement.texture_m, measurement.gradient_norm, cfg)
    confidence = cfg.confidence_smoothing * prev_confidence + (1.0 - cfg.confidence_smoothing) * raw_conf
    if confidence <= 1e-3:
        return {'accepted': False, 'correction_xy_m': np.zeros(2), 'confidence': confidence, 'status': 'flat_bottom'}

    rel_xy = measurement.sample_xy_m[valid_idx] - np.asarray(measurement.vehicle_xyz_m[:2], dtype=float)
    measured_bottom = measurement.sample_bottom_z_m[valid_idx]

    best_score = float('inf')
    best_xy = prior.copy()
    prior_score = float('inf')

    offsets = np.arange(-cfg.search_radius_m, cfg.search_radius_m + 0.5 * cfg.search_step_m, cfg.search_step_m)
    for dx in offsets:
        for dy in offsets:
            candidate = prior + np.array([dx, dy], dtype=float)
            pred = np.array([grid.sample_elevation(candidate[0] + dxy[0], candidate[1] + dxy[1]) for dxy in rel_xy], dtype=float)
            if np.isnan(pred).any():
                continue
            rmse = float(np.sqrt(np.mean((pred - measured_bottom) ** 2)))
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                prior_score = rmse
            if rmse < best_score:
                best_score = rmse
                best_xy = candidate.copy()

    if not math.isfinite(best_score):
        return {'accepted': False, 'correction_xy_m': np.zeros(2), 'confidence': confidence, 'status': 'no_candidate'}

    improvement = prior_score - best_score if math.isfinite(prior_score) else 0.0
    if best_score > cfg.max_rmse_m:
        return {'accepted': False, 'correction_xy_m': np.zeros(2), 'confidence': confidence, 'status': 'rmse_too_high', 'best_score': best_score}
    if improvement < cfg.min_rmse_improvement_m:
        return {'accepted': False, 'correction_xy_m': np.zeros(2), 'confidence': confidence, 'status': 'poor_improvement', 'best_score': best_score, 'prior_score': prior_score}

    innovation = best_xy - prior
    correction = _clamp_norm(cfg.blend_gain * confidence * innovation, cfg.max_step_m)

    drift_attenuation = _clamp(cfg.imu_drift_attenuation_max * confidence, 0.0, cfg.imu_drift_attenuation_max)
    drift_correction = -drift_attenuation * drift_xy
    total_correction = _clamp_norm(correction + drift_correction, cfg.max_step_m)

    return {
        'accepted': True,
        'status': 'accepted',
        'best_xy_m': best_xy,
        'best_score': best_score,
        'prior_score': prior_score,
        'score_improvement': improvement,
        'raw_confidence': raw_conf,
        'confidence': confidence,
        'innovation_xy_m': innovation,
        'map_match_correction_xy_m': correction,
        'drift_correction_xy_m': drift_correction,
        'correction_xy_m': total_correction,
        'corrected_xy_m': prior + total_correction,
        'texture_m': measurement.texture_m,
        'gradient_norm': measurement.gradient_norm,
        'valid_returns': int(valid_idx.size),
    }


def apply_bathy_to_estimate(estimate: Any, bathy_update: Dict[str, Any]) -> Any:
    if not bathy_update.get('accepted', False):
        return estimate
    corr = np.asarray(bathy_update['correction_xy_m'], dtype=float)
    estimate.x = float(estimate.x + corr[0])
    estimate.y = float(estimate.y + corr[1])
    return estimate


# Example integration point for guidance_core/nav_env.
def bathy_update_step(
    grid: BathymetryGrid,
    estimate: Any,
    truth: Any,
    drift_xy_m: Tuple[float, float],
    yaw_rad: float,
    sensor_cfg: ConeVolumeConfig,
    corr_cfg: BathyCorrectionConfig,
    prev_confidence: float,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    measurement = simulate_cone_volume_measurement(
        grid=grid,
        vehicle_xyz_m=(float(truth.x), float(truth.y), float(truth.z)),
        yaw_rad=float(yaw_rad),
        cfg=sensor_cfg,
        rng=rng,
    )
    update = estimate_bathy_correction(
        grid=grid,
        measurement=measurement,
        prior_xy_m=(float(estimate.x), float(estimate.y)),
        drift_xy_m=drift_xy_m,
        cfg=corr_cfg,
        prev_confidence=prev_confidence,
    )
    apply_bathy_to_estimate(estimate, update)
    return {
        'measurement': measurement,
        'update': update,
        'confidence': float(update.get('confidence', 0.0)),
    }
