from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a bathymetry GeoTIFF for sanity-checking. Reports raster resolution, "
            "min/max surface height, and a global texture characterization based on the "
            "same local-standard-deviation idea used by the bathymetry sensor."
        )
    )
    parser.add_argument("tiff", help="Path to the bathymetry GeoTIFF")
    parser.add_argument(
        "--texture-radius-m",
        type=float,
        default=25.0,
        help="Neighborhood radius in meters for local texture estimation (default: 25)",
    )
    parser.add_argument(
        "--stride-px",
        type=int,
        default=16,
        help="Sample every N pixels when building the texture map (default: 16)",
    )
    parser.add_argument(
        "--min-valid",
        type=int,
        default=8,
        help="Minimum valid cells required in a texture neighborhood (default: 8)",
    )
    parser.add_argument(
        "--min-texture-m",
        type=float,
        default=0.75,
        help="Texture threshold below which terrain is considered flat/smooth (default: 0.75)",
    )
    parser.add_argument(
        "--max-texture-m",
        type=float,
        default=8.0,
        help="Texture threshold at or above which terrain is considered rough (default: 8.0)",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default=None,
        help="Optional output PNG path for an elevation + texture analysis figure",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra debug information about CRS, pixel size, and texture window size",
    )
    return parser.parse_args()


def _native_unit_to_meters(crs) -> float:
    if crs is None:
        return 1.0

    for attr in ("linear_units_factor", "units_factor"):
        value = getattr(crs, attr, None)
        if value is not None:
            if isinstance(value, tuple) and len(value) >= 2:
                try:
                    return float(value[-1])
                except (TypeError, ValueError):
                    pass
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    unit_name = str(getattr(crs, "linear_units", "") or getattr(crs, "units", "")).lower()
    if "foot" in unit_name or "feet" in unit_name or unit_name == "ft":
        return 0.3048
    if "us survey foot" in unit_name:
        return 1200.0 / 3937.0
    return 1.0


def load_bathymetry(path: str | Path, *, debug: bool = False) -> Tuple[np.ndarray, Dict[str, object]]:
    path = Path(path)
    with rasterio.open(path) as src:
        band = src.read(1).astype(float)
        nodata = src.nodata
        transform = src.transform
        bounds = src.bounds
        crs = src.crs

        if nodata is not None:
            band[np.isclose(band, float(nodata))] = np.nan

        dx_native = abs(float(transform.a))
        dy_native = abs(float(transform.e))

        if dx_native == 0.0 or dy_native == 0.0:
            dx_m = 1.0
            dy_m = 1.0
            unit_label = "fallback 1 m/pixel (missing or invalid geotransform)"
        elif crs is not None and getattr(crs, "is_geographic", False):
            lat_mid_deg = 0.5 * (float(bounds.top) + float(bounds.bottom))
            lat_mid_rad = math.radians(lat_mid_deg)
            meters_per_deg_lat = 111320.0
            meters_per_deg_lon = 111320.0 * math.cos(lat_mid_rad)
            dx_m = dx_native * meters_per_deg_lon
            dy_m = dy_native * meters_per_deg_lat
            unit_label = "degrees -> meters"
        else:
            unit_scale = _native_unit_to_meters(crs)
            dx_m = dx_native * unit_scale
            dy_m = dy_native * unit_scale
            unit_label = f"projected/native * {unit_scale:g}"

        elevation_m = np.flipud(band)

        meta: Dict[str, object] = {
            "path": str(path),
            "crs": str(crs) if crs is not None else "None",
            "bounds": bounds,
            "transform": transform,
            "dx_native": dx_native,
            "dy_native": dy_native,
            "dx_m": dx_m,
            "dy_m": dy_m,
            "nx": int(src.width),
            "ny": int(src.height),
            "width_m": max(0.0, (src.width - 1) * dx_m),
            "height_m": max(0.0, (src.height - 1) * dy_m),
            "nodata": nodata,
            "unit_label": unit_label,
        }

    if debug:
        print(f"DEBUG crs={meta['crs']}")
        print(f"DEBUG transform={meta['transform']}")
        print(f"DEBUG pixel_native=({meta['dx_native']}, {meta['dy_native']})")
        print(f"DEBUG pixel_m=({meta['dx_m']}, {meta['dy_m']})")
        print(f"DEBUG size_px=({meta['nx']}, {meta['ny']})")
        print(f"DEBUG bounds={meta['bounds']}")
        print(f"DEBUG unit_handling={meta['unit_label']}")

    return elevation_m, meta

def circular_offsets(
    dx_m: float,
    dy_m: float,
    radius_m: float,
    *,
    max_candidates: int = 250_000,
    debug: bool = False,
) -> np.ndarray:
    if dx_m <= 0.0 or dy_m <= 0.0:
        raise ValueError(f"Invalid pixel size: dx_m={dx_m}, dy_m={dy_m}")
    if radius_m <= 0.0:
        raise ValueError(f"Texture radius must be positive, got {radius_m}")

    rx = max(1, int(math.ceil(radius_m / dx_m)))
    ry = max(1, int(math.ceil(radius_m / dy_m)))
    est_box = (2 * rx + 1) * (2 * ry + 1)

    if debug:
        print(f"DEBUG circular_offsets radius_m={radius_m}")
        print(f"DEBUG circular_offsets rx={rx} ry={ry} est_box={est_box}")

    if est_box > max_candidates:
        raise ValueError(
            "Texture window is too large before sampling begins. "
            f"Computed rx={rx}, ry={ry}, est_box={est_box}. "
            "This usually means the TIFF pixel size or CRS units are being interpreted incorrectly. "
            "Try --debug, inspect dx_m/dy_m, reduce --texture-radius-m, or fix the raster CRS metadata."
        )

    offsets = []
    for oy in range(-ry, ry + 1):
        for ox in range(-rx, rx + 1):
            if math.hypot(ox * dx_m, oy * dy_m) <= radius_m:
                offsets.append((oy, ox))

    if not offsets:
        raise ValueError("No offsets generated for texture computation. Check dx_m, dy_m, and radius_m.")

    return np.asarray(offsets, dtype=np.int32)


def compute_texture_map(
    elevation_m: np.ndarray,
    dx_m: float,
    dy_m: float,
    radius_m: float,
    stride_px: int,
    min_valid: int,
    *,
    debug: bool = False,
) -> np.ndarray:
    if stride_px <= 0:
        raise ValueError(f"stride_px must be positive, got {stride_px}")
    if min_valid <= 0:
        raise ValueError(f"min_valid must be positive, got {min_valid}")

    offsets = circular_offsets(dx_m, dy_m, radius_m, debug=debug)
    ny, nx = elevation_m.shape
    texture_map = np.full((ny, nx), np.nan, dtype=float)

    ys = range(0, ny, stride_px)
    xs = range(0, nx, stride_px)

    for y in ys:
        for x in xs:
            if np.isnan(elevation_m[y, x]):
                continue

            samples = []
            for oy, ox in offsets:
                yy = y + int(oy)
                xx = x + int(ox)
                if 0 <= yy < ny and 0 <= xx < nx:
                    value = elevation_m[yy, xx]
                    if not np.isnan(value):
                        samples.append(value)

            if len(samples) >= min_valid:
                texture_map[y, x] = float(np.nanstd(np.asarray(samples, dtype=float)))

    return texture_map


def summarize_elevation(elevation_m: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(elevation_m)
    valid_count = int(np.count_nonzero(valid))
    total_count = int(elevation_m.size)

    if valid_count == 0:
        raise ValueError("Raster contains no valid elevation cells.")

    values = elevation_m[valid]
    return {
        "valid_count": valid_count,
        "total_count": total_count,
        "coverage_pct": 100.0 * valid_count / max(1, total_count),
        "min_m": float(np.nanmin(values)),
        "max_m": float(np.nanmax(values)),
        "mean_m": float(np.nanmean(values)),
        "std_m": float(np.nanstd(values)),
    }


def summarize_texture(texture_map: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(texture_map)
    valid_count = int(np.count_nonzero(valid))

    if valid_count == 0:
        return {
            "valid_count": 0,
            "min_m": float("nan"),
            "max_m": float("nan"),
            "mean_m": float("nan"),
            "median_m": float("nan"),
            "std_m": float("nan"),
            "p90_m": float("nan"),
        }

    values = texture_map[valid]
    return {
        "valid_count": valid_count,
        "min_m": float(np.nanmin(values)),
        "max_m": float(np.nanmax(values)),
        "mean_m": float(np.nanmean(values)),
        "median_m": float(np.nanmedian(values)),
        "std_m": float(np.nanstd(values)),
        "p90_m": float(np.nanpercentile(values, 90.0)),
    }


def characterize_texture(texture_stats: Dict[str, float], min_texture_m: float, max_texture_m: float) -> str:
    median_tex = float(texture_stats["median_m"])
    p90_tex = float(texture_stats["p90_m"])

    if not np.isfinite(median_tex):
        return "unknown (no valid texture samples)"
    if p90_tex < min_texture_m:
        return "flat / low-feature seabed"
    if median_tex < min_texture_m:
        return "mostly smooth with occasional features"
    if median_tex < 0.5 * (min_texture_m + max_texture_m):
        return "moderately textured"
    if median_tex < max_texture_m:
        return "highly textured"
    return "rough / strongly varying relief"


def maybe_save_plot(
    elevation_m: np.ndarray,
    texture_map: np.ndarray,
    meta: Dict[str, object],
    output_png: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    width_m = float(meta["width_m"])
    height_m = float(meta["height_m"])
    extent = [0.0, width_m, 0.0, height_m]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    im0 = axes[0].imshow(elevation_m, origin="lower", extent=extent)
    axes[0].set_title("Surface / Bathymetry Height")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    fig.colorbar(im0, ax=axes[0], shrink=0.85, label="height [m]")

    im1 = axes[1].imshow(texture_map, origin="lower", extent=extent)
    axes[1].set_title("Local Texture (std-dev of height)")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    fig.colorbar(im1, ax=axes[1], shrink=0.85, label="texture [m]")

    fig.suptitle(Path(str(meta["path"])).name)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def print_report(
    meta: Dict[str, object],
    elev_stats: Dict[str, float],
    texture_stats: Dict[str, float],
    texture_label: str,
    args: argparse.Namespace,
) -> None:
    print("=" * 72)
    print(f"Bathymetry analysis: {Path(args.tiff).name}")
    print("=" * 72)
    print(f"CRS:                {meta['crs']}")
    print(f"Pixel size native:  ({meta['dx_native']:.6g}, {meta['dy_native']:.6g})")
    print(f"Pixel size meters:  ({meta['dx_m']:.6g}, {meta['dy_m']:.6g})")
    print(f"Raster size [px]:   ({meta['nx']}, {meta['ny']})")
    print(f"Map size [m]:       ({float(meta['width_m']):.3f}, {float(meta['height_m']):.3f})")
    print(f"Coverage:           {elev_stats['valid_count']}/{elev_stats['total_count']} ({elev_stats['coverage_pct']:.2f}%)")
    print("-")
    print(f"Min surface height: {elev_stats['min_m']:.3f} m")
    print(f"Max surface height: {elev_stats['max_m']:.3f} m")
    print(f"Mean height:        {elev_stats['mean_m']:.3f} m")
    print(f"Height std-dev:     {elev_stats['std_m']:.3f} m")
    print("-")
    print(f"Texture radius:     {float(args.texture_radius_m):.3f} m")
    print(f"Texture stride:     {int(args.stride_px)} px")
    print(f"Texture samples:    {texture_stats['valid_count']}")
    print(f"Texture min:        {texture_stats['min_m']:.3f} m")
    print(f"Texture max:        {texture_stats['max_m']:.3f} m")
    print(f"Texture mean:       {texture_stats['mean_m']:.3f} m")
    print(f"Texture median:     {texture_stats['median_m']:.3f} m")
    print(f"Texture p90:        {texture_stats['p90_m']:.3f} m")
    print(f"Texture std-dev:    {texture_stats['std_m']:.3f} m")
    print(f"Global character:   {texture_label}")
    print("=" * 72)


def main() -> int:
    args = parse_args()

    elevation_m, meta = load_bathymetry(args.tiff, debug=args.debug)
    elev_stats = summarize_elevation(elevation_m)

    try:
        texture_map = compute_texture_map(
            elevation_m=elevation_m,
            dx_m=float(meta["dx_m"]),
            dy_m=float(meta["dy_m"]),
            radius_m=float(args.texture_radius_m),
            stride_px=int(args.stride_px),
            min_valid=int(args.min_valid),
            debug=args.debug,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    texture_stats = summarize_texture(texture_map)
    texture_label = characterize_texture(
        texture_stats,
        min_texture_m=float(args.min_texture_m),
        max_texture_m=float(args.max_texture_m),
    )

    print_report(meta, elev_stats, texture_stats, texture_label, args)

    if args.output_png:
        maybe_save_plot(elevation_m, texture_map, meta, args.output_png)
        print(f"Saved analysis figure to: {args.output_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
