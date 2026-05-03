"""
Limpieza de archivos temporales de productos satelitales.

Las imagenes Sentinel-1 ocupan ~500 MB - 1 GB cada una.  Con 24 GB de RAM
y 200 GB de disco, es critico limpiar productos descargados una vez
procesados para evitar quedarse sin espacio.

Este modulo proporciona:
- Eliminacion de un producto individual.
- Limpieza por antigueedad (productos mas viejos que N horas).
- Consulta de uso de disco.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

from src.observability.loki_logger import StructuredLogger

_log = StructuredLogger("aidra.cleanup")


async def cleanup_product(product_dir: Path) -> None:
    """Delete an entire product directory tree.

    Removes the directory and all its contents (measurement TIFFs,
    annotation XMLs, zip files, etc.).  This is safe to call even if the
    directory does not exist.

    Args:
        product_dir: Path to the product directory to remove.
    """
    product_dir = Path(product_dir)
    if not product_dir.exists():
        _log.info(
            "Product directory does not exist, nothing to clean",
            extra={"path": str(product_dir)},
        )
        return

    # Compute size before deletion for logging
    size_mb = _dir_size_mb(product_dir)

    # Run the blocking I/O in a thread to stay async-friendly
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, shutil.rmtree, product_dir)

    _log.info(
        "Product directory deleted",
        extra={"path": str(product_dir), "freed_mb": round(size_mb, 2)},
    )


async def cleanup_old_products(
    images_dir: Path,
    max_age_hours: int = 24,
) -> int:
    """Delete product directories older than *max_age_hours*.

    A product's age is determined by the modification time of its
    top-level directory.  Both extracted product directories and leftover
    ``.zip`` files are cleaned.

    Args:
        images_dir: Root directory where products are stored.
        max_age_hours: Maximum allowed age in hours.  Products older
            than this are deleted.

    Returns:
        Number of products (directories + zip files) deleted.
    """
    images_dir = Path(images_dir)
    if not images_dir.exists():
        _log.warning(
            "Images directory does not exist",
            extra={"path": str(images_dir)},
        )
        return 0

    cutoff = time.time() - (max_age_hours * 3600)
    deleted = 0

    for entry in images_dir.iterdir():
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue

        if mtime >= cutoff:
            continue

        if entry.is_dir():
            size_mb = _dir_size_mb(entry)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, shutil.rmtree, entry)
            _log.info(
                "Deleted old product directory",
                extra={
                    "path": str(entry),
                    "age_hours": round((time.time() - mtime) / 3600, 1),
                    "freed_mb": round(size_mb, 2),
                },
            )
            deleted += 1
        elif entry.is_file() and entry.suffix == ".zip":
            size_mb = entry.stat().st_size / (1024 * 1024)
            entry.unlink()
            _log.info(
                "Deleted old zip file",
                extra={
                    "path": str(entry),
                    "age_hours": round((time.time() - mtime) / 3600, 1),
                    "freed_mb": round(size_mb, 2),
                },
            )
            deleted += 1

    _log.info(
        "Old product cleanup complete",
        extra={
            "images_dir": str(images_dir),
            "max_age_hours": max_age_hours,
            "deleted_count": deleted,
        },
    )
    return deleted


def get_disk_usage(path: Path) -> dict[str, float]:
    """Return disk usage statistics for the filesystem containing *path*.

    Args:
        path: Any path on the target filesystem (file or directory).

    Returns:
        A dict with:
        - ``total_gb``: Total filesystem capacity in GiB.
        - ``used_gb``: Used space in GiB.
        - ``free_gb``: Available space in GiB.
        - ``percent``: Usage percentage (0-100).
    """
    path = Path(path)
    usage = shutil.disk_usage(path)

    total_gb = round(usage.total / (1024 ** 3), 2)
    used_gb = round(usage.used / (1024 ** 3), 2)
    free_gb = round(usage.free / (1024 ** 3), 2)
    percent = round((usage.used / usage.total) * 100, 1) if usage.total else 0.0

    stats: dict[str, float] = {
        "total_gb": total_gb,
        "used_gb": used_gb,
        "free_gb": free_gb,
        "percent": percent,
    }

    _log.info(
        "Disk usage queried",
        extra={"path": str(path), **stats},
    )
    return stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dir_size_mb(directory: Path) -> float:
    """Compute total size of all files in a directory tree (in MB)."""
    total = 0
    try:
        for f in directory.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except OSError:
        pass
    return total / (1024 * 1024)
