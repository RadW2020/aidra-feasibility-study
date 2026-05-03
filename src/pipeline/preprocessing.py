"""
Preprocesamiento de imagenes SAR Sentinel-1 GRD.

Responsabilidades:
1. Calibracion radiometrica: DN (digital number) -> sigma0 (backscatter en dB)
2. Filtrado de speckle (ruido inherente en SAR) mediante filtro Lee
3. Recorte al area de interes (AOI)
4. Tiling: dividir imagen grande en tiles manejables para inferencia
5. Generacion de tiles SAR sinteticos para testing

Dependencias:
- rasterio (lectura/escritura GeoTIFF)
- numpy (operaciones matriciales)
- scipy.ndimage (filtro de speckle Lee)
- xml.etree.ElementTree (lectura de annotation XML)

Notas:
- Las imagenes Sentinel-1 GRD vienen en formato TIFF con metadatos XML
- La calibracion usa los coeficientes del archivo annotation XML del producto
- El filtro de speckle Lee con ventana 7x7 es el estandar para vessel detection
- Los tiles deben tener tamano fijo (ej: 640x640 px) con overlap (ej: 64 px)
  para evitar perder detecciones en los bordes
"""

from __future__ import annotations

import contextlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rasterio.windows
from scipy.ndimage import uniform_filter

from src.observability.loki_logger import StructuredLogger

_log = StructuredLogger("aidra.preprocessing")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_sigma0(tiff_path: Path, annotation_xml: Path) -> np.ndarray:
    """Convert raw digital numbers (DN) to sigma-nought in decibels.

    The calibration lookup vector (LUT) is read from the Sentinel-1
    annotation/calibration XML.  Each pixel is calibrated as::

        sigma0_linear = (DN ** 2) / calibration_lut ** 2
        sigma0_db     = 10 * log10(sigma0_linear)

    Args:
        tiff_path: Path to the measurement TIFF file.
        annotation_xml: Path to the calibration XML file that contains
            ``<sigmaNought>`` calibration vectors.

    Returns:
        A ``float32`` array of sigma0 values in dB.
    """
    _log.info(
        "Starting radiometric calibration",
        extra={
            "tiff": str(tiff_path),
            "annotation": str(annotation_xml),
        },
    )

    # --- Read calibration LUT from XML ---
    tree = ET.parse(annotation_xml)
    root = tree.getroot()

    sigma_vectors: list[list[float]] = []
    for cal_vec in root.iter("calibrationVector"):
        sigma_el = cal_vec.find("sigmaNought")
        if sigma_el is not None and sigma_el.text:
            values = [float(v) for v in sigma_el.text.strip().split()]
            sigma_vectors.append(values)

    # --- Read the TIFF image ---
    with rasterio.open(tiff_path) as src:
        dn = src.read(1).astype(np.float32)

    rows, cols = dn.shape

    # --- Build 2-D calibration array ---
    if sigma_vectors:
        # Stack vectors and interpolate to match image dimensions.
        # Each vector covers all columns at a specific azimuth line.
        lut_stack = np.array(sigma_vectors, dtype=np.float32)

        # Resample columns if vector length differs from image width
        if lut_stack.shape[1] != cols:
            x_src = np.linspace(0, cols - 1, lut_stack.shape[1])
            x_dst = np.arange(cols, dtype=np.float32)
            resampled_rows = []
            for row_vec in lut_stack:
                resampled_rows.append(np.interp(x_dst, x_src, row_vec))
            lut_stack = np.array(resampled_rows, dtype=np.float32)

        # Resample rows to match image height
        if lut_stack.shape[0] != rows:
            y_src = np.linspace(0, rows - 1, lut_stack.shape[0])
            y_dst = np.arange(rows, dtype=np.float32)
            resampled_cols: list[np.ndarray] = []
            for c in range(cols):
                resampled_cols.append(
                    np.interp(y_dst, y_src, lut_stack[:, c])
                )
            calibration_lut = np.column_stack(resampled_cols).astype(np.float32)
        else:
            calibration_lut = lut_stack
    else:
        _log.warning(
            "No sigmaNought vectors found in annotation XML; "
            "using unity calibration"
        )
        calibration_lut = np.ones((rows, cols), dtype=np.float32)

    # --- Apply calibration ---
    # Avoid division by zero
    calibration_lut = np.where(calibration_lut == 0, 1.0, calibration_lut)

    sigma0_linear = (dn ** 2) / (calibration_lut ** 2)
    # Clamp to small positive before log to avoid -inf / nan
    sigma0_linear = np.clip(sigma0_linear, 1e-10, None)
    sigma0_db = (10.0 * np.log10(sigma0_linear)).astype(np.float32)

    _log.info(
        "Calibration complete",
        extra={
            "shape": list(sigma0_db.shape),
            "min_db": float(np.min(sigma0_db)),
            "max_db": float(np.max(sigma0_db)),
        },
    )
    return sigma0_db


# ---------------------------------------------------------------------------
# Speckle filter
# ---------------------------------------------------------------------------


def apply_lee_filter(image: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Apply Lee speckle filter to a SAR image.

    The Lee filter reduces multiplicative speckle noise while preserving
    edges and bright targets (vessels).  The algorithm:

    1. Compute local mean in a sliding window.
    2. Compute local variance in a sliding window.
    3. Estimate noise variance from the overall image.
    4. Compute weighting factor ``k = var_local / (var_local + var_noise)``.
    5. Output: ``filtered = mean_local + k * (original - mean_local)``.

    When ``k ~ 0`` (homogeneous area) the output tends to the local mean
    (maximum smoothing).  When ``k ~ 1`` (edge / target) the output tends
    to the original value (no smoothing).

    Args:
        image: 2-D ``float32`` SAR image (calibrated sigma0).
        window_size: Side length of the square averaging window (must be
            odd).  Default is 7 (standard for vessel detection).

    Returns:
        Filtered ``float32`` image with same shape as input.
    """
    if window_size % 2 == 0:
        window_size += 1
        _log.warning(
            "Lee filter window_size must be odd; adjusted",
            extra={"window_size": window_size},
        )

    img = image.astype(np.float64)

    # Local statistics via uniform (box) filter
    mean_local = uniform_filter(img, size=window_size, mode="reflect")
    mean_sq = uniform_filter(img ** 2, size=window_size, mode="reflect")
    var_local = mean_sq - mean_local ** 2
    var_local = np.clip(var_local, 0.0, None)

    # Estimate noise variance as mean of local variances (assumes
    # speckle-dominated regions dominate the image, which holds for SAR
    # sea scenes).
    var_noise = float(np.mean(var_local))
    if var_noise <= 0:
        _log.warning("Noise variance is zero; returning original image")
        return image.copy()

    # Weighting factor
    k = var_local / (var_local + var_noise)
    k = np.clip(k, 0.0, 1.0)

    filtered = (mean_local + k * (img - mean_local)).astype(np.float32)

    _log.info(
        "Lee filter applied",
        extra={"window_size": window_size, "shape": list(image.shape)},
    )
    return filtered


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------


def create_tiles(
    image: np.ndarray,
    tile_size: int = 640,
    overlap: int = 64,
    geo_transform: tuple[float, ...] | None = None,
) -> list[dict[str, Any]]:
    """Split a large image into tiles with optional overlap.

    Tiles at the right and bottom edges are zero-padded to ``tile_size``
    so every tile has uniform dimensions suitable for batch inference.

    Args:
        image: 2-D array (rows x cols).
        tile_size: Width and height of each tile in pixels.
        overlap: Number of overlapping pixels between adjacent tiles.
        geo_transform: Affine geo-transform tuple
            ``(origin_x, pixel_w, 0, origin_y, 0, pixel_h)`` used to
            compute geographic bounds of each tile.

    Returns:
        A list of dicts, each with keys:
        - ``array``: ``np.ndarray`` of shape ``(tile_size, tile_size)``.
        - ``row_offset``: Row start index in the source image.
        - ``col_offset``: Column start index in the source image.
        - ``geo_bounds``: ``dict`` with ``lat_min, lat_max, lon_min, lon_max``
          (set to ``None`` when ``geo_transform`` is not provided).
    """
    rows, cols = image.shape[:2]
    step = tile_size - overlap
    if step <= 0:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than tile_size ({tile_size})"
        )

    tiles: list[dict[str, Any]] = []

    for row_start in range(0, rows, step):
        for col_start in range(0, cols, step):
            row_end = min(row_start + tile_size, rows)
            col_end = min(col_start + tile_size, cols)

            tile = np.zeros((tile_size, tile_size), dtype=image.dtype)
            tile[: row_end - row_start, : col_end - col_start] = image[
                row_start:row_end, col_start:col_end
            ]

            # Compute geographic bounds if transform is available
            geo_bounds: dict[str, float | None]
            if geo_transform is not None:
                geo_bounds = _tile_geo_corners(  # type: ignore[assignment]
                    geo_transform, col_start, row_start, tile_size
                )
            else:
                geo_bounds = {
                    "lon_min": None,
                    "lon_max": None,
                    "lat_min": None,
                    "lat_max": None,
                }

            tiles.append(
                {
                    "array": tile,
                    "row_offset": row_start,
                    "col_offset": col_start,
                    "geo_bounds": geo_bounds,
                }
            )

    _log.info(
        "Tiling complete",
        extra={
            "image_shape": list(image.shape),
            "tile_size": tile_size,
            "overlap": overlap,
            "num_tiles": len(tiles),
        },
    )
    return tiles


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------


def _parse_calibration_lut(
    annotation_xml: Path, num_cols: int
) -> np.ndarray | None:
    """Parse calibration LUT from annotation XML as a 1-D array.

    Returns a 1-D float32 array of length *num_cols* (interpolated from
    the sigmaNought vectors), or ``None`` if parsing fails.
    """
    try:
        tree = ET.parse(annotation_xml)
        root = tree.getroot()
        for cal_vec in root.iter("calibrationVector"):
            sigma_el = cal_vec.find("sigmaNought")
            if sigma_el is not None and sigma_el.text:
                values = np.array(
                    [float(v) for v in sigma_el.text.strip().split()],
                    dtype=np.float32,
                )
                if len(values) != num_cols:
                    x_src = np.linspace(0, num_cols - 1, len(values))
                    x_dst = np.arange(num_cols, dtype=np.float32)
                    values = np.interp(x_dst, x_src, values).astype(np.float32)
                return values
    except Exception as exc:
        _log.warning("Failed to parse calibration LUT", extra={"error": str(exc)})
    return None


def _calibrate_tile_linear(
    dn_tile: np.ndarray, cal_row: np.ndarray | None
) -> np.ndarray:
    """Calibrate a single tile from DN to linear sigma0 (power scale).

    Returns linear-power sigma0 because downstream consumers (Lee filter,
    CFAR) assume Rayleigh-distributed positive values.  Convert to dB only
    for visualization or YOLO input.
    """
    dn = dn_tile.astype(np.float32)
    if cal_row is not None:
        lut = np.where(cal_row == 0, 1.0, cal_row)
        sigma0_linear = (dn ** 2) / (lut ** 2)
    else:
        sigma0_linear = dn ** 2
    return np.clip(sigma0_linear, 1e-10, None).astype(np.float32)


def _load_gcps(product_dir: Path) -> list[dict[str, float]] | None:
    """Load Ground Control Points from Sentinel-1 annotation XML.

    GCPs map pixel (line, pixel) to geographic (lat, lon) coordinates.
    They are in annotation/s1*-vv*.xml under <geolocationGridPoint>.
    """
    # Find annotation XML (not calibration)
    for xml_path in product_dir.rglob("annotation/s1*.xml"):
        if "calibration" in str(xml_path):
            continue
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            gcps = []
            for point in root.iter("geolocationGridPoint"):
                line_el = point.find("line")
                pixel_el = point.find("pixel")
                lat_el = point.find("latitude")
                lon_el = point.find("longitude")
                if all(x is not None and x.text for x in [line_el, pixel_el, lat_el, lon_el]):
                    gcps.append({
                        "line": float(line_el.text),
                        "pixel": float(pixel_el.text),
                        "lat": float(lat_el.text),
                        "lon": float(lon_el.text),
                    })
            if gcps:
                _log.info("Loaded GCPs", extra={"count": len(gcps), "xml": xml_path.name})
                return gcps
        except Exception as exc:
            _log.warning("Failed to parse GCPs", extra={"xml": str(xml_path), "error": str(exc)})
    return None


def _build_pixel_to_geo_transform(
    gcps: list[dict[str, float]], img_height: int, img_width: int
) -> tuple[float, ...] | None:
    """Build a 6-element affine geo_transform from GCPs.

    The transform is the standard GDAL/rasterio form::

        lon = origin_x + col*pixel_w + row*rot_lon
        lat = origin_y + col*rot_lat + row*pixel_h

    Slots [2] (rot_lon) and [4] (rot_lat) capture the orbit rotation —
    Sentinel-1 descending passes over Iberia rotate ~13°, so dropping
    these terms (the previous behaviour) introduced kilometre-scale
    drift at the swath corners.

    Returns ``(origin_x, pixel_w, rot_lon, origin_y, rot_lat, pixel_h)``
    or ``None`` if fewer than 4 GCPs are available.
    """
    if not gcps or len(gcps) < 4:
        return None

    lines = np.array([g["line"] for g in gcps])
    pixels = np.array([g["pixel"] for g in gcps])
    lats = np.array([g["lat"] for g in gcps])
    lons = np.array([g["lon"] for g in gcps])

    # lon = a*pixel + b*line + c   (a=pixel_w, b=rot_lon, c=origin_x)
    # lat = d*pixel + e*line + f   (d=rot_lat, e=pixel_h, f=origin_y)
    A = np.column_stack([pixels, lines, np.ones(len(gcps))])
    lon_coeffs, *_ = np.linalg.lstsq(A, lons, rcond=None)
    lat_coeffs, *_ = np.linalg.lstsq(A, lats, rcond=None)

    pixel_w = float(lon_coeffs[0])
    rot_lon = float(lon_coeffs[1])
    origin_x = float(lon_coeffs[2])
    rot_lat = float(lat_coeffs[0])
    pixel_h = float(lat_coeffs[1])
    origin_y = float(lat_coeffs[2])

    # Residuals — useful to spot products where a linear affine isn't enough
    lon_pred = pixels * pixel_w + lines * rot_lon + origin_x
    lat_pred = pixels * rot_lat + lines * pixel_h + origin_y
    lon_res = float(np.sqrt(np.mean((lons - lon_pred) ** 2)))
    lat_res = float(np.sqrt(np.mean((lats - lat_pred) ** 2)))

    _log.info(
        "Geo transform from GCPs",
        extra={
            "origin": [origin_x, origin_y],
            "pixel_size": [pixel_w, pixel_h],
            "rotation": [rot_lon, rot_lat],
            "rmse_deg": [lon_res, lat_res],
            "lat_range": [float(lats.min()), float(lats.max())],
            "lon_range": [float(lons.min()), float(lons.max())],
        },
    )

    return (origin_x, pixel_w, rot_lon, origin_y, rot_lat, pixel_h)


# ---------------------------------------------------------------------------
# Affine helpers — apply the full 6-element geo_transform incl. rotation
# ---------------------------------------------------------------------------


def affine_pixel_to_geo(
    geo_transform: tuple[float, ...],
    col: float,
    row: float,
) -> tuple[float, float]:
    """Apply a 6-element affine to map (col, row) → (lon, lat)."""
    origin_x, pixel_w, rot_lon, origin_y, rot_lat, pixel_h = geo_transform[:6]
    lon = origin_x + col * pixel_w + row * rot_lon
    lat = origin_y + col * rot_lat + row * pixel_h
    return lon, lat


def affine_geo_to_pixel(
    geo_transform: tuple[float, ...],
    lon: float,
    lat: float,
) -> tuple[float, float]:
    """Invert a 6-element affine to map (lon, lat) → (col, row).

    Solves the 2×2 system::

        pixel_w * col + rot_lon * row = lon - origin_x
        rot_lat * col + pixel_h * row = lat - origin_y
    """
    origin_x, pixel_w, rot_lon, origin_y, rot_lat, pixel_h = geo_transform[:6]
    det = pixel_w * pixel_h - rot_lon * rot_lat
    if abs(det) < 1e-20:
        # Degenerate transform — fall back to axis-aligned division
        col = (lon - origin_x) / pixel_w if pixel_w else 0.0
        row = (lat - origin_y) / pixel_h if pixel_h else 0.0
        return col, row
    dx = lon - origin_x
    dy = lat - origin_y
    col = (pixel_h * dx - rot_lon * dy) / det
    row = (-rot_lat * dx + pixel_w * dy) / det
    return col, row


def _tile_geo_corners(
    geo_transform: tuple[float, ...],
    col_offset: int,
    row_offset: int,
    tile_size: int,
) -> dict[str, float]:
    """Return the lat/lon bounding box of a (possibly rotated) tile.

    The tile is a rectangle in pixel space; under a rotated affine its
    geographic outline is also rotated.  We return the axis-aligned
    bbox of the four projected corners so downstream code keeps working
    with a simple ``lon_min/lon_max/lat_min/lat_max`` dict.
    """
    corners = [
        (col_offset, row_offset),
        (col_offset + tile_size, row_offset),
        (col_offset + tile_size, row_offset + tile_size),
        (col_offset, row_offset + tile_size),
    ]
    lons = []
    lats = []
    for c, r in corners:
        lon, lat = affine_pixel_to_geo(geo_transform, c, r)
        lons.append(lon)
        lats.append(lat)
    return {
        "lon_min": float(min(lons)),
        "lon_max": float(max(lons)),
        "lat_min": float(min(lats)),
        "lat_max": float(max(lats)),
    }


def _is_sentinel2(product_dir: Path) -> bool:
    """Check if a product directory is Sentinel-2 (has GRANULE/ structure)."""
    return any(product_dir.rglob("GRANULE/*/IMG_DATA"))


def _find_s2_bands(product_dir: Path) -> dict[str, Path] | None:
    """Find Sentinel-2 10m band files (B02=blue, B03=green, B04=red)."""
    bands: dict[str, Path] = {}
    # S2 L2A structure: GRANULE/*/IMG_DATA/R10m/*_B0X_10m.jp2
    for jp2 in product_dir.rglob("R10m/*.jp2"):
        name = jp2.stem.upper()
        if "_B04_" in name:
            bands["red"] = jp2
        elif "_B03_" in name:
            bands["green"] = jp2
        elif "_B02_" in name:
            bands["blue"] = jp2
    # Also check non-R10m paths (older format)
    if len(bands) < 3:
        for jp2 in product_dir.rglob("*.jp2"):
            name = jp2.stem.upper()
            if "_B04" in name and "red" not in bands:
                bands["red"] = jp2
            elif "_B03" in name and "green" not in bands:
                bands["green"] = jp2
            elif "_B02" in name and "blue" not in bands:
                bands["blue"] = jp2
    if len(bands) == 3:
        _log.info("Found S2 bands", extra={k: v.name for k, v in bands.items()})
        return bands
    return None


def preprocess_s2_full(
    product_dir: Path,
    aoi_bbox: list[float] | None = None,
    tile_size: int = 640,
    overlap: int = 64,
) -> dict[str, Any]:
    """Preprocess Sentinel-2 optical product into RGB tiles.

    Reads B04 (red), B03 (green), B02 (blue) at 10m resolution,
    composes RGB image, and tiles it for YOLO inference.
    No calibration or speckle filter needed (optical, not SAR).
    """
    product_dir = Path(product_dir)
    _log.info("Starting S2 optical preprocessing", extra={"product_dir": str(product_dir)})

    bands = _find_s2_bands(product_dir)
    if bands is None:
        raise FileNotFoundError(f"Could not find S2 RGB bands in {product_dir}")

    # Read geo metadata from red band (all 10m bands have same extent)
    with rasterio.open(bands["red"]) as src:
        crs = str(src.crs) if src.crs else "EPSG:32630"
        transform = src.transform
        img_height = src.height
        img_width = src.width
        geo_transform = (
            transform.c, transform.a, transform.b,
            transform.f, transform.d, transform.e,
        )

    # Build CRS transformer to WGS84 (S2 is usually UTM)
    from pyproj import Transformer
    to_wgs84 = None
    if crs and crs != "EPSG:4326":
        try:
            to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            _log.info("CRS transformer created", extra={"from": crs, "to": "EPSG:4326"})
        except Exception as exc:
            _log.warning("Could not create CRS transformer", extra={"error": str(exc)})

    _log.info("S2 image metadata", extra={"shape": [img_height, img_width], "crs": crs})

    # Windowed tiling: read RGB per tile directly
    step = tile_size - overlap
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) must be < tile_size ({tile_size})")

    tiles: list[dict[str, Any]] = []

    red_src = rasterio.open(bands["red"])
    green_src = rasterio.open(bands["green"])
    blue_src = rasterio.open(bands["blue"])

    try:
        for row_offset in range(0, img_height, step):
            for col_offset in range(0, img_width, step):
                r_end = min(row_offset + tile_size, img_height)
                c_end = min(col_offset + tile_size, img_width)

                win = rasterio.windows.Window(
                    col_off=col_offset,
                    row_off=row_offset,
                    width=c_end - col_offset,
                    height=r_end - row_offset,
                )

                r = red_src.read(1, window=win).astype(np.float32)
                g = green_src.read(1, window=win).astype(np.float32)
                b = blue_src.read(1, window=win).astype(np.float32)

                # Normalize to 0-255 (S2 L2A values are 0-10000 reflectance)
                scale = 255.0 / 3000.0  # typical max for visual stretch
                r = np.clip(r * scale, 0, 255).astype(np.uint8)
                g = np.clip(g * scale, 0, 255).astype(np.uint8)
                b = np.clip(b * scale, 0, 255).astype(np.uint8)

                # Stack RGB and pad to tile_size
                rgb = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                h, w = r.shape
                rgb[:h, :w, 0] = r
                rgb[:h, :w, 1] = g
                rgb[:h, :w, 2] = b

                # Geo bounds — project the 4 rotated corners (S2 may have a
                # non-zero rotation when the granule isn't axis-aligned).
                corner_xys: list[tuple[float, float]] = []
                for cc, rr in (
                    (col_offset, row_offset),
                    (col_offset + tile_size, row_offset),
                    (col_offset + tile_size, row_offset + tile_size),
                    (col_offset, row_offset + tile_size),
                ):
                    corner_xys.append(affine_pixel_to_geo(geo_transform, cc, rr))

                if to_wgs84 is not None:
                    lonlats = [
                        to_wgs84.transform(x, y) for x, y in corner_xys
                    ]
                else:
                    lonlats = corner_xys

                lons = [pt[0] for pt in lonlats]
                lats = [pt[1] for pt in lonlats]
                geo_bounds = {
                    "lon_min": float(min(lons)),
                    "lon_max": float(max(lons)),
                    "lat_min": float(min(lats)),
                    "lat_max": float(max(lats)),
                }

                tiles.append({
                    "array": rgb,
                    "row_offset": row_offset,
                    "col_offset": col_offset,
                    "geo_bounds": geo_bounds,
                    "geo_transform": geo_transform,
                })
    finally:
        red_src.close()
        green_src.close()
        blue_src.close()

    _log.info("S2 preprocessing complete", extra={
        "num_tiles": len(tiles),
        "original_shape": [img_height, img_width],
    })

    return {
        "tiles": tiles,
        "metadata": {
            "product_dir": str(product_dir),
            "original_shape": (img_height, img_width),
            "sensor": "sentinel-2",
            "calibration": "rgb_normalized",
            "filter": "none",
            "tile_size": tile_size,
            "overlap": overlap,
            "num_tiles": len(tiles),
            "crs": crs,
            "geo_transform": geo_transform,
        },
    }


def preprocess_full(
    product_dir: Path,
    aoi_bbox: list[float] | None = None,
    tile_size: int = 640,
    overlap: int = 64,
) -> dict[str, Any]:
    """Run preprocessing on a Sentinel-1 (SAR) or Sentinel-2 (optical) product.

    Auto-detects the sensor type:
    - Sentinel-2: reads RGB bands (B04, B03, B02) and composes color tiles
    - Sentinel-1: windowed SAR processing with calibration + Lee filter

    Steps per tile (S1):
    1. Read tile window from TIFF via ``rasterio.windows.Window``.
    2. Apply radiometric calibration (DN -> sigma0 dB) using the LUT.
    3. Apply Lee speckle filter (7x7 window).
    4. Store tile with geo-referencing metadata.

    Args:
        product_dir: Root directory of the extracted S1 product.
        aoi_bbox: Optional ``[lon_min, lat_min, lon_max, lat_max]`` to crop.
        tile_size: Tile side length in pixels.
        overlap: Overlap between adjacent tiles in pixels.

    Returns:
        A dict with keys ``tiles`` (list of tile dicts) and ``metadata``.
    """
    product_dir = Path(product_dir)

    # Auto-detect sensor type
    if _is_sentinel2(product_dir):
        _log.info("Detected Sentinel-2 product, using optical preprocessing")
        return preprocess_s2_full(product_dir, aoi_bbox, tile_size, overlap)

    _log.info(
        "Starting SAR preprocessing (windowed)",
        extra={"product_dir": str(product_dir)},
    )

    # --- 1. Locate files ---
    tiff_path = _find_file(product_dir, "*.tiff") or _find_file(
        product_dir, "*.tif"
    )
    if tiff_path is None:
        raise FileNotFoundError(
            f"No TIFF measurement file found in {product_dir}"
        )

    annotation_xml = _find_calibration_xml(product_dir)

    # --- 2. Open TIFF and read metadata (no pixel data yet) ---
    with rasterio.open(tiff_path) as src:
        crs = str(src.crs) if src.crs else "EPSG:4326"
        transform = src.transform
        img_height = src.height
        img_width = src.width
        original_shape = (img_height, img_width)

    # --- 2b. Build geo_transform from GCPs (S1 GRD has no embedded CRS) ---
    # NOTE on Terrain Correction: AIDRA currently uses a linear affine
    # fit over the GCPs as its geo-referencing step. A proper
    # Range-Doppler TC against a DEM (SRTM 1") would live in
    # `src/pipeline/terrain_correction.py:apply_terrain_correction`
    # but is NOT YET wired in (deferred — see RISK_REGISTER and the
    # SAR audit in score-against-rubric). Acceptable for flat-sea AOI
    # like the Strait of Gibraltar; insufficient over coast/relief.
    gcps = _load_gcps(product_dir)
    geo_transform: tuple[float, ...] | None = None
    if gcps:
        geo_transform = _build_pixel_to_geo_transform(gcps, img_height, img_width)
        crs = "EPSG:4326"

    if geo_transform is None:
        # Fallback: try rasterio transform (may be identity/pixel coords)
        if transform and transform.a != 1.0:
            geo_transform = (
                transform.c, transform.a, transform.b,
                transform.f, transform.d, transform.e,
            )
        else:
            _log.warning("No valid geo_transform — coordinates will be in pixel space")

    _log.info(
        "Image metadata read",
        extra={
            "shape": [img_height, img_width],
            "crs": crs,
            "has_gcps": gcps is not None and len(gcps) > 0 if gcps else False,
            "geo_transform": list(geo_transform) if geo_transform else None,
        },
    )

    # --- 3. Parse calibration LUT (1-D, per-column) ---
    cal_row: np.ndarray | None = None
    if annotation_xml is not None:
        cal_row = _parse_calibration_lut(annotation_xml, img_width)
        if cal_row is not None:
            _log.info("Calibration LUT loaded", extra={"cols": len(cal_row)})
        else:
            _log.warning("Could not parse calibration LUT; using raw DN")
    else:
        _log.warning("No calibration XML found; using raw DN")

    # --- 4. Compute AOI pixel window if bbox provided ---
    row_start_aoi, col_start_aoi = 0, 0
    read_height, read_width = img_height, img_width
    if aoi_bbox is not None:
        lon_min, lat_min, lon_max, lat_max = aoi_bbox
        # Project all four corners through the (possibly rotated) affine
        # and take the axis-aligned pixel bbox of the result.
        cols, rows = [], []
        for lon, lat in (
            (lon_min, lat_min), (lon_max, lat_min),
            (lon_max, lat_max), (lon_min, lat_max),
        ):
            c, r = affine_geo_to_pixel(geo_transform, lon, lat)
            cols.append(c)
            rows.append(r)
        col_start_aoi = max(0, int(min(cols)))
        col_end_aoi = min(img_width, int(max(cols)) + 1)
        row_start_aoi = max(0, int(min(rows)))
        row_end_aoi = min(img_height, int(max(rows)) + 1)
        read_height = max(0, row_end_aoi - row_start_aoi)
        read_width = max(0, col_end_aoi - col_start_aoi)
        _log.info(
            "AOI crop",
            extra={
                "bbox": aoi_bbox,
                "pixel_window": [row_start_aoi, col_start_aoi, read_height, read_width],
            },
        )

    # --- 5. Calculate valid data footprint (fast downsampled pass) ---
    valid_footprint = None
    if geo_transform is not None:
        try:
            with rasterio.open(tiff_path) as src:
                # Read heavily downsampled (factor 50) to find data boundaries
                ds = 50
                overview_data = src.read(
                    1,
                    out_shape=(img_height // ds, img_width // ds),
                )
            valid_footprint = _calculate_valid_footprint(
                overview_data,
                (
                    geo_transform[0],
                    geo_transform[1] * ds,
                    geo_transform[2] * ds,
                    geo_transform[3],
                    geo_transform[4] * ds,
                    geo_transform[5] * ds,
                ),
            )
            if valid_footprint:
                _log.info("Valid data footprint calculated")
        except Exception as exc:
            _log.warning("Could not calculate footprint", extra={"error": str(exc)})

    # --- 6. Read tiles directly via windowed I/O ---
    step = tile_size - overlap
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) must be < tile_size ({tile_size})")

    tiles: list[dict[str, Any]] = []
    lee_margin = 4  # Extra margin for Lee filter border effects (half of 7)

    with rasterio.open(tiff_path) as src:
        for row_offset in range(0, read_height, step):
            for col_offset in range(0, read_width, step):
                # Compute read window with margin for Lee filter
                r_start = row_start_aoi + row_offset - lee_margin
                c_start = col_start_aoi + col_offset - lee_margin
                r_size = tile_size + 2 * lee_margin
                c_size = tile_size + 2 * lee_margin

                # Clamp to image bounds
                r_start_clamped = max(0, r_start)
                c_start_clamped = max(0, c_start)
                r_end = min(img_height, r_start + r_size)
                c_end = min(img_width, c_start + c_size)

                win = rasterio.windows.Window(
                    col_off=c_start_clamped,
                    row_off=r_start_clamped,
                    width=c_end - c_start_clamped,
                    height=r_end - r_start_clamped,
                )

                # Read DN for this window only
                dn_chunk = src.read(1, window=win)

                # Calibrate to LINEAR sigma0 (Lee + CFAR assume linear power)
                chunk_cal = (
                    cal_row[c_start_clamped:c_end] if cal_row is not None else None
                )
                calibrated_linear = _calibrate_tile_linear(dn_chunk, chunk_cal)

                # Lee filter on linear-scale data (multiplicative noise model)
                filtered = apply_lee_filter(calibrated_linear, window_size=7)

                # Extract the center tile (remove Lee margin)
                margin_top = r_start_clamped - r_start if r_start < 0 else lee_margin
                margin_left = c_start_clamped - c_start if c_start < 0 else lee_margin
                center = filtered[
                    margin_top : margin_top + tile_size,
                    margin_left : margin_left + tile_size,
                ]

                # Pad if at edge
                tile = np.zeros((tile_size, tile_size), dtype=np.float32)
                tile[: center.shape[0], : center.shape[1]] = center

                # Geo bounds — compute from the four rotated corners
                abs_row = row_start_aoi + row_offset
                abs_col = col_start_aoi + col_offset
                geo_bounds = _tile_geo_corners(
                    geo_transform, abs_col, abs_row, tile_size
                )

                tiles.append({
                    "array": tile,
                    "row_offset": abs_row,
                    "col_offset": abs_col,
                    "geo_bounds": geo_bounds,
                    "geo_transform": geo_transform,
                })

    _log.info(
        "Windowed preprocessing complete",
        extra={
            "num_tiles": len(tiles),
            "original_shape": list(original_shape),
            "read_area": [read_height, read_width],
        },
    )

    metadata: dict[str, Any] = {
        "product_dir": str(product_dir),
        "original_shape": original_shape,
        "calibration": "sigma0_linear" if cal_row is not None else "raw_dn_squared",
        "filter": "lee_7x7_linear",
        "tile_size": tile_size,
        "overlap": overlap,
        "num_tiles": len(tiles),
        "crs": crs,
        "geo_transform": geo_transform,
        "valid_footprint": valid_footprint,
    }

    # SAR product metadata (manifest.safe + annotation XML)
    sar_meta = parse_sar_metadata(product_dir)
    if sar_meta:
        metadata["sar"] = sar_meta

    # I-SAR-1: marca quality=invalid si falta cualquier paso critico.
    quality, reasons = _evaluate_scene_quality(
        cal_row=cal_row,
        gcps=gcps,
        geo_transform=geo_transform,
        valid_footprint=valid_footprint,
        num_tiles=len(tiles),
    )
    metadata["quality"] = quality
    metadata["quality_reasons"] = reasons
    if quality == "invalid":
        _log.warning(
            "Scene marked quality=invalid (I-SAR-1)",
            extra={"reasons": reasons, "product_dir": str(product_dir)},
        )

    return {"tiles": tiles, "metadata": metadata}


def _evaluate_scene_quality(
    cal_row: np.ndarray | None,
    gcps: list[Any] | None,
    geo_transform: tuple[float, ...] | None,
    valid_footprint: dict[str, Any] | None,
    num_tiles: int,
) -> tuple[str, list[str]]:
    """Decide quality flag for a preprocessed scene (I-SAR-1).

    Returns ``("valid"|"invalid", reasons)``. ``invalid`` se asigna si
    falta algun pre-requisito del pipeline (LUT, GCPs, geo_transform,
    footprint, tiles). El engine usa esto para skipear deteccion en
    escenas no fiables sin perder el registro.
    """
    reasons: list[str] = []
    if cal_row is None:
        reasons.append("missing_calibration_lut")
    if not gcps:
        reasons.append("missing_gcps")
    if geo_transform is None:
        reasons.append("missing_geo_transform")
    if not valid_footprint:
        reasons.append("missing_valid_footprint")
    if num_tiles == 0:
        reasons.append("no_tiles_generated")
    return ("invalid" if reasons else "valid", reasons)


# ---------------------------------------------------------------------------
# SAR metadata parser (Sentinel-1 SAFE)
# ---------------------------------------------------------------------------


def parse_sar_metadata(product_dir: Path) -> dict[str, Any]:
    """Extrae metadatos SAR desde el SAFE de Sentinel-1.

    Lee ``manifest.safe`` (XML) cuando esta disponible. Tolera fallos:
    devuelve un dict parcial con las claves que se hayan podido
    resolver. Cierra parte del criterio Q3 (metadatos GEOINT).

    Returns
    -------
    dict[str, Any]
        Claves posibles:
        ``incidence_angle`` (float, grados),
        ``polarisation`` (str, e.g. "VV+VH"),
        ``orbit_direction`` (``"ASCENDING"`` / ``"DESCENDING"``),
        ``relative_orbit`` (int),
        ``product_type`` (``"GRD"`` / ``"SLC"`` / ``"OCN"``),
        ``pixel_spacing`` (float, metros).
    """
    meta: dict[str, Any] = {}
    product_dir = Path(product_dir)

    # 1. From product directory name (best-effort, always available)
    name = product_dir.name
    if "_GRD" in name:
        meta["product_type"] = "GRD"
    elif "_SLC" in name:
        meta["product_type"] = "SLC"
    elif "_OCN" in name:
        meta["product_type"] = "OCN"

    # 2. manifest.safe (rich metadata) — search recursively because S1
    # products are sometimes nested in a <name>.SAFE/ subdir.
    manifest = next(product_dir.rglob("manifest.safe"), None)
    if manifest is not None:
        try:
            tree = ET.parse(manifest)
            root = tree.getroot()
            text = ET.tostring(root, encoding="unicode")

            # Crude but robust: match SAFE namespace fields by local name.
            # Avoids juggling ns prefixes that vary across product versions.
            def _find_local(local: str) -> str | None:
                for elem in root.iter():
                    tag = elem.tag.rsplit("}", 1)[-1]
                    if tag == local and elem.text:
                        return elem.text.strip()
                return None

            pass_dir = _find_local("pass")
            if pass_dir:
                meta["orbit_direction"] = pass_dir.upper()

            rel_orbit = _find_local("relativeOrbitNumber")
            if rel_orbit and rel_orbit.lstrip("-").isdigit():
                meta["relative_orbit"] = int(rel_orbit)

            prod_type = _find_local("productType")
            if prod_type:
                meta["product_type"] = prod_type.upper()

            pol_modes = []
            for elem in root.iter():
                tag = elem.tag.rsplit("}", 1)[-1]
                if tag == "transmitterReceiverPolarisation" and elem.text:
                    pol_modes.append(elem.text.strip().upper())
            if pol_modes:
                meta["polarisation"] = "+".join(sorted(set(pol_modes)))

            # heuristic for ground range pixel spacing
            if "rangePixelSpacing>" in text:
                idx = text.index("rangePixelSpacing>")
                end = text.index("<", idx)
                snippet = text[idx + len("rangePixelSpacing>") : end]
                with contextlib.suppress(ValueError):
                    meta["pixel_spacing"] = float(snippet)
        except (ET.ParseError, OSError) as exc:
            _log.warning("manifest.safe parse failed: %s", exc)

    # 3. annotation XML (incidence angle as average over geolocation grid)
    ann = next(product_dir.rglob("annotation/*.xml"), None)
    if ann is not None:
        try:
            tree = ET.parse(ann)
            root = tree.getroot()
            angles: list[float] = []
            for elem in root.iter():
                tag = elem.tag.rsplit("}", 1)[-1]
                if tag == "incidenceAngle" and elem.text:
                    try:
                        angles.append(float(elem.text))
                    except ValueError:
                        continue
            if angles:
                meta["incidence_angle"] = float(sum(angles) / len(angles))
            if "polarisation" not in meta:
                for elem in root.iter():
                    tag = elem.tag.rsplit("}", 1)[-1]
                    if tag == "polarisation" and elem.text:
                        meta["polarisation"] = elem.text.strip().upper()
                        break
        except (ET.ParseError, OSError) as exc:
            _log.warning("annotation XML parse failed: %s", exc)

    return meta


# ---------------------------------------------------------------------------
# Synthetic SAR tile generation (testing)
# ---------------------------------------------------------------------------


def generate_synthetic_sar_tile(
    size: int = 640,
    num_vessels: int = 5,
    noise_mean: float = 0.3,
    vessel_amplitude: float = 5.0,
    seed: int = 42,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Generate a synthetic SAR tile with simulated vessel signatures.

    The background uses a Rayleigh distribution (standard model for SAR
    sea clutter), and vessels are represented as 2-D Gaussian bright
    points with random size and position.

    This is intended for unit testing and benchmarking the detection
    pipeline without requiring real Sentinel-1 data.

    Args:
        size: Tile side length in pixels.
        num_vessels: Number of simulated vessels.
        noise_mean: Scale parameter for Rayleigh background noise.
        vessel_amplitude: Peak amplitude of vessel Gaussian signatures.
        seed: Random seed for reproducibility.

    Returns:
        A tuple ``(image, ground_truth)`` where *image* is a
        ``float32`` array of shape ``(size, size)`` and *ground_truth*
        is a list of dicts with ``bbox``, ``center``, ``width``,
        ``height`` for each vessel.
    """
    rng = np.random.default_rng(seed)

    # Background: Rayleigh-distributed clutter (models SAR sea surface)
    background = rng.rayleigh(scale=noise_mean, size=(size, size)).astype(
        np.float32
    )

    ground_truth: list[dict[str, Any]] = []
    for _ in range(num_vessels):
        # Random position (keep away from edges to fit Gaussian fully)
        cx = int(rng.integers(50, size - 50))
        cy = int(rng.integers(50, size - 50))

        # Random vessel extent (half-widths 3-15 px)
        w = int(rng.integers(3, 15))
        h = int(rng.integers(3, 15))

        # 2-D Gaussian bright point
        y_lo, y_hi = max(0, cy - h), min(size, cy + h)
        x_lo, x_hi = max(0, cx - w), min(size, cx + w)

        y_grid, x_grid = np.ogrid[y_lo:y_hi, x_lo:x_hi]
        sigma_x = max(w / 3.0, 0.5)
        sigma_y = max(h / 3.0, 0.5)
        gaussian = np.exp(
            -(
                (x_grid - cx) ** 2 / (2.0 * sigma_x ** 2)
                + (y_grid - cy) ** 2 / (2.0 * sigma_y ** 2)
            )
        )
        background[y_lo:y_hi, x_lo:x_hi] += vessel_amplitude * gaussian

        ground_truth.append(
            {
                "bbox": [cx - w, cy - h, cx + w, cy + h],
                "center": [cx, cy],
                "width": w * 2,
                "height": h * 2,
            }
        )

    return background, ground_truth


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _calculate_valid_footprint(image: np.ndarray, geo_transform: tuple[float, ...]) -> dict[str, Any] | None:
    """Professional approach: Detect the valid data boundary.

    Creates a GeoJSON polygon of the area containing actual sensor data,
    excluding the black "NoData" margins where artifacts typically live.
    """
    try:
        from shapely.geometry import MultiPoint, mapping

        # Image is already downsampled; just find valid (non-zero) pixels
        # S1 GRD has 0 in NoData areas (edges of swath)
        threshold = 0 if image.dtype in (np.uint8, np.uint16, np.int16) else 1e-6
        rows, cols = np.where(image > threshold)
        if len(rows) < 100:
            return None

        # Convert pixel coords to geographic coordinates (full affine)
        origin_x, px_w, rot_lon, origin_y, rot_lat, px_h = geo_transform[:6]
        lons = origin_x + cols * px_w + rows * rot_lon
        lats = origin_y + cols * rot_lat + rows * px_h

        # Subsample points for convex hull (too many is slow)
        step = max(1, len(lons) // 5000)
        points = [
            (float(lon), float(lat))
            for lon, lat in zip(lons[::step], lats[::step], strict=False)
        ]
        poly = MultiPoint(points).convex_hull

        # Apply a small negative buffer (e.g., 200 meters) to be safe
        # (Assuming ~0.0001 degrees per 10 meters)
        safe_poly = poly.buffer(-0.002)

        return mapping(safe_poly)
    except ImportError:
        _log.warning("Shapely not installed; skipping footprint calculation")
        return None
    except Exception as exc:
        _log.error("Failed to calculate footprint", extra={"error": str(exc)})
        return None


def _find_file(directory: Path, pattern: str) -> Path | None:
    """Find the first file matching a glob pattern recursively."""
    matches = list(directory.rglob(pattern))
    return matches[0] if matches else None


def _find_calibration_xml(product_dir: Path) -> Path | None:
    """Locate the calibration annotation XML in an S1 product tree.

    Sentinel-1 products store calibration data under
    ``annotation/calibration/calibration-*.xml``.
    """
    # Primary location
    matches = list(product_dir.rglob("calibration-*.xml"))
    if matches:
        return matches[0]
    # Fallback: any XML with 'calibration' in the name
    matches = list(product_dir.rglob("*calibration*.xml"))
    return matches[0] if matches else None


def _crop_to_aoi(
    image: np.ndarray,
    geo_transform: tuple[float, ...],
    aoi_bbox: list[float],
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Crop an image array to an AOI bounding box.

    Args:
        image: 2-D array.
        geo_transform: ``(origin_x, pixel_w, 0, origin_y, 0, pixel_h)``.
        aoi_bbox: ``[lon_min, lat_min, lon_max, lat_max]``.

    Returns:
        Tuple of (cropped_image, new_geo_transform).
    """
    origin_x, pixel_w, rot1, origin_y, rot2, pixel_h = geo_transform[:6]
    lon_min, lat_min, lon_max, lat_max = aoi_bbox
    rows, cols = image.shape

    # Project the four AOI corners through the inverse affine and take
    # the axis-aligned pixel bbox.  A linear inversion that ignores the
    # rotation slots silently mis-crops rotated S1 products.
    proj_cols: list[float] = []
    proj_rows: list[float] = []
    for lon, lat in (
        (lon_min, lat_min), (lon_max, lat_min),
        (lon_max, lat_max), (lon_min, lat_max),
    ):
        c, r = affine_geo_to_pixel(geo_transform, lon, lat)
        proj_cols.append(c)
        proj_rows.append(r)
    col_start = max(0, int(min(proj_cols)))
    col_end = min(cols, int(max(proj_cols)) + 1)
    row_start = max(0, int(min(proj_rows)))
    row_end = min(rows, int(max(proj_rows)) + 1)

    # Ensure valid range
    if col_start >= col_end or row_start >= row_end:
        _log.warning(
            "AOI does not intersect image; returning original",
            extra={"aoi_bbox": aoi_bbox},
        )
        return image, geo_transform

    cropped = image[row_start:row_end, col_start:col_end]
    # New origin is the projected location of pixel (col_start, row_start)
    new_origin_x = origin_x + col_start * pixel_w + row_start * rot1
    new_origin_y = origin_y + col_start * rot2 + row_start * pixel_h
    new_geo_transform = (
        new_origin_x,
        pixel_w,
        rot1,
        new_origin_y,
        rot2,
        pixel_h,
    )

    _log.info(
        "Cropped to AOI",
        extra={
            "aoi_bbox": aoi_bbox,
            "cropped_shape": list(cropped.shape),
        },
    )
    return cropped, new_geo_transform
