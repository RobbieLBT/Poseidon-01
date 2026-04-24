from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio


def load_bathymetry(path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
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

        if crs is not None and getattr(crs, "is_geographic", False):
            lat_mid_deg = 0.5 * (float(bounds.top) + float(bounds.bottom))
            lat_mid_rad = math.radians(lat_mid_deg)
            meters_per_deg_lat = 111320.0
            meters_per_deg_lon = 111320.0 * math.cos(lat_mid_rad)
            dx_m = dx_native * meters_per_deg_lon
            dy_m = dy_native * meters_per_deg_lat
        
        
        else:
            dx_m = dx_native
            dy_m = dy_native
        print(f"DEBUG dx_m={meta['dx_m']}, dy_m={meta['dy_m']}, crs={meta['crs']}")

        # Match BathymetryGrid.from_geotiff local orientation.
        local = np.flipud(band)

        meta = {
            "crs": crs,
            "bounds": bounds,
            "width": src.width,
            "height": src.height,
            "transform": transform,
            "dx_native": dx_native,
            "dy_native": dy_native,
            "dx_m": dx_m,
            "dy_m": dy_m,
        }

    return local, meta


def circular_offsets(dx_m: float, dy_m: float, radius_m: float, max_offsets: int = 200000) -> np.ndarray:
    if dx_m <= 0.0 or dy_m <= 0.0:
        raise ValueError(f"Invalid pixel size: dx_m={dx_m}, dy_m={dy_m}")

    rx = max(1, int(math.ceil(radius_m / dx_m)))
    ry = max(1, int(math.ceil(radius_m / dy_m)))

    est_box = (2 * rx + 1) * (2 * ry + 1)
    if est_box > max_offsets:
        raise ValueError(
            f"Texture window is too large: rx={rx}, ry={ry}, "
            f"~{est_box} candidate offsets. "
            f"This usually means the TIFF pixel size/CRS units are being interpreted incorrectly. "
            f"Try a smaller --texture-radius-m or inspect CRS/pixel size."
        )

    offsets = []
    for oy in range(-ry, ry + 1):
        for ox in range(-rx, rx + 1):
            dist_m = math.hypot(ox * dx_m, oy * dy_m)
            if dist_m <= radius_m:
                offsets.append((oy, ox))

    return np.asarray(offsets, dtype=np.int32)


def compute_texture_map(
    elevation_m: np.ndarray,
    dx_m: float,
    dy_m: float,
    radius_m: float,
    stride_px: int,
    min_valid: int,
) -> np.ndarray:
    offsets = circular_offsets(dx_m, dy_m, radius_m)
    ny, nx = elevation_m.shape
    texture_map = np.full((ny, nx), np.nan, dtype=float)

    stride_px = max(1, int(stride_px))
    for y in range(0, ny, stride_px):
        for x in range(0, nx, stride_px):
            vals = []
            for oy, ox in offsets:
                yy = y + oy
                xx = x + ox
                if 0 <= yy < ny and 0 <= xx < nx:
                    v = elevation_m[yy, xx]
                    if not np.isnan(v):
                        vals.append(v)
            if len(vals) >= min_valid:
                texture_map[y, x] = float(np.std(np.asarray(vals, dtype=float)))

    if stride_px > 1:
        # Expand sampled cells to full raster using nearest sampled index.
        sample_ys = np.arange(0, ny, stride_px)
        sample_xs = np.arange(0, nx, stride_px)
        sampled = texture_map[np.ix_(sample_ys, sample_xs)]
        y_idx = np.clip(np.round(np.arange(ny) / stride_px).astype(int), 0, sampled.shape[0] - 1)
        x_idx = np.clip(np.round(np.arange(nx) / stride_px).astype(int), 0, sampled.shape[1] - 1)
        texture_map = sampled[np.ix_(y_idx, x_idx)]

    return texture_map


def characterize_texture(texture_map: np.ndarray, min_texture_m: float, max_texture_m: float) -> str:
    valid = texture_map[np.isfinite(texture_map)]
    if valid.size == 0:
        return "no valid texture samples"

    p50 = float(np.percentile(valid, 50))
    p90 = float(np.percentile(valid, 90))

    if p90 < min_texture_m:
        return "mostly flat / low-feature bottom"
    if p50 < min_texture_m and p90 < max_texture_m:
        return "mostly smooth with some usable structure"
    if p50 < max_texture_m and p90 < max_texture_m:
        return "moderately textured / map-match friendly"
    return "highly textured / rough bottom"


def compute_gradient_map(elevation_m: np.ndarray, dy_m: float, dx_m: float) -> np.ndarray:
    grad_y, grad_x = np.gradient(elevation_m, dy_m, dx_m)
    return np.hypot(grad_x, grad_y)


def summarize_map(
    elevation_m: np.ndarray,
    texture_map: np.ndarray,
    gradient_map: np.ndarray,
    meta: Dict[str, object],
    texture_min_m: float,
    texture_max_m: float,
) -> str:
    valid_elev = elevation_m[np.isfinite(elevation_m)]
    valid_tex = texture_map[np.isfinite(texture_map)]
    valid_grad = gradient_map[np.isfinite(gradient_map)]

    width_m = (int(meta["width"]) - 1) * float(meta["dx_m"])
    height_m = (int(meta["height"]) - 1) * float(meta["dy_m"])

    lines = [
        f"CRS: {meta['crs']}",
        f"Pixel size native: {meta['dx_native']:.6f}, {meta['dy_native']:.6f}",
        f"Pixel size approx meters: {meta['dx_m']:.3f}, {meta['dy_m']:.3f}",
        f"Width x Height pixels: {meta['width']}, {meta['height']}",
        f"Width x Height meters: {width_m:.1f}, {height_m:.1f}",
        f"Bounds: {meta['bounds']}",
        f"Valid cells: {valid_elev.size} / {elevation_m.size} ({100.0 * valid_elev.size / max(1, elevation_m.size):.2f}%)",
        "",
        f"Min surface height: {np.nanmin(valid_elev):.3f} m",
        f"Max surface height: {np.nanmax(valid_elev):.3f} m",
        f"Mean surface height: {np.nanmean(valid_elev):.3f} m",
        f"Std surface height: {np.nanstd(valid_elev):.3f} m",
        "",
        "Texture summary (same underlying idea as bathymetry sensor: local std-dev of bottom elevation):",
        f"  texture p10 / p50 / p90: {np.percentile(valid_tex, 10):.3f} / {np.percentile(valid_tex, 50):.3f} / {np.percentile(valid_tex, 90):.3f} m",
        f"  min / max local texture: {np.nanmin(valid_tex):.3f} / {np.nanmax(valid_tex):.3f} m",
        f"  global characterization: {characterize_texture(texture_map, texture_min_m, texture_max_m)}",
        f"  fraction below min_texture ({texture_min_m:.2f} m): {100.0 * np.mean(valid_tex < texture_min_m):.2f}%",
        f"  fraction in usable band [{texture_min_m:.2f}, {texture_max_m:.2f}] m: {100.0 * np.mean((valid_tex >= texture_min_m) & (valid_tex <= texture_max_m)):.2f}%",
        f"  fraction above max_texture ({texture_max_m:.2f} m): {100.0 * np.mean(valid_tex > texture_max_m):.2f}%",
        "",
        "Gradient summary:",
        f"  gradient norm p50 / p90 / max: {np.percentile(valid_grad, 50):.4f} / {np.percentile(valid_grad, 90):.4f} / {np.nanmax(valid_grad):.4f}",
    ]
    return "\n".join(lines)


def maybe_write_png(output_png: Path, elevation_m: np.ndarray, texture_map: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    im0 = axes[0].imshow(elevation_m, origin="lower")
    axes[0].set_title("Surface Height / Bathymetry")
    axes[0].set_xlabel("x pixel")
    axes[0].set_ylabel("y pixel")
    fig.colorbar(im0, ax=axes[0], shrink=0.85, label="height (m)")

    im1 = axes[1].imshow(texture_map, origin="lower")
    axes[1].set_title("Local Texture Map")
    axes[1].set_xlabel("x pixel")
    axes[1].set_ylabel("y pixel")
    fig.colorbar(im1, ax=axes[1], shrink=0.85, label="local std-dev (m)")

    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Human-facing bathymetry TIFF sanity checker with texture characterization.",
    )
    parser.add_argument("tiff", type=Path, help="Path to GeoTIFF / bathymetry raster")
    parser.add_argument(
        "--texture-radius-m",
        type=float,
        default=25.0,
        help="Radius for local texture calculation in meters",
    )
    parser.add_argument(
        "--stride-px",
        type=int,
        default=8,
        help="Sampling stride in pixels for faster whole-map analysis",
    )
    parser.add_argument(
        "--min-valid",
        type=int,
        default=8,
        help="Minimum valid cells required in a local neighborhood",
    )
    parser.add_argument(
        "--min-texture-m",
        type=float,
        default=0.75,
        help="Lower texture threshold copied from bathy correction heuristics",
    )
    parser.add_argument(
        "--max-texture-m",
        type=float,
        default=8.0,
        help="Upper texture threshold copied from bathy correction heuristics",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Optional path to save a 2-panel analysis PNG",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    elevation_m, meta = load_bathymetry(args.tiff)
    texture_map = compute_texture_map(
        elevation_m=elevation_m,
        dx_m=float(meta["dx_m"]),
        dy_m=float(meta["dy_m"]),
        radius_m=float(args.texture_radius_m),
        stride_px=int(args.stride_px),
        min_valid=int(args.min_valid),
    )
    gradient_map = compute_gradient_map(elevation_m, float(meta["dy_m"]), float(meta["dx_m"]))

    report = summarize_map(
        elevation_m=elevation_m,
        texture_map=texture_map,
        gradient_map=gradient_map,
        meta=meta,
        texture_min_m=float(args.min_texture_m),
        texture_max_m=float(args.max_texture_m),
    )
    print(report)

    if args.output_png is not None:
        maybe_write_png(args.output_png, elevation_m, texture_map)
        print(f"\nSaved analysis PNG to: {args.output_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
