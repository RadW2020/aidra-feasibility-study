"""
Model registry, constraint profiles, and search zone endpoints.

Provides read-only access to the catalogue of registered ML models,
available constraint profiles, and predefined search zones.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.db.connection import db
from src.db.models import ModelInfo
from src.db.queries import SELECT_ALL_MODELS

logger = logging.getLogger("aidra.api.models_api")

router = APIRouter(tags=["models"])

# ---------------------------------------------------------------------------
# Constraint profiles (static configuration; mirrors TECHNICAL_SPEC 9.2)
# ---------------------------------------------------------------------------

PROFILES: list[dict] = [
    {
        "name": "ground",
        "display_name": "Ground Station",
        "description": "Sin restricciones, estacion terrena",
        "cpu_limit": 4.0,
        "memory_limit_mb": 24576,
        "simulates": "Ground processing station (baseline)",
    },
    {
        "name": "sat-high",
        "display_name": "Satellite High-End",
        "description": "Satelite gama alta (Xilinx Zynq / Unibap iX10)",
        "cpu_limit": 2.0,
        "memory_limit_mb": 4096,
        "simulates": "High-end satellite processor (e.g. Xilinx Zynq UltraScale+)",
    },
    {
        "name": "sat-mid",
        "display_name": "Satellite Mid-Range",
        "description": "Satelite gama media",
        "cpu_limit": 1.0,
        "memory_limit_mb": 2048,
        "simulates": "Mid-range satellite processor",
    },
    {
        "name": "sat-low",
        "display_name": "Satellite Low-End / CubeSat",
        "description": "Satelite gama baja o CubeSat",
        "cpu_limit": 0.5,
        "memory_limit_mb": 1024,
        "simulates": "Low-end processor / CubeSat (e.g. Raspberry Pi class)",
    },
    {
        "name": "sat-extreme",
        "display_name": "Extreme Constraint",
        "description": "Limite inferior: donde se rompe el pipeline",
        "cpu_limit": 0.25,
        "memory_limit_mb": 512,
        "simulates": "Extreme constraint - find breaking point",
    },
]

# ---------------------------------------------------------------------------
# Search zones (static configuration; mirrors TECHNICAL_SPEC 5.5)
# ---------------------------------------------------------------------------

SEARCH_ZONES: list[dict] = [
    {
        "name": "gibraltar",
        "display_name": "Estrecho de Gibraltar",
        "bbox": [-5.8, 35.7, -5.2, 36.2],
        "description": "Alto trafico maritimo, estrecho natural",
    },
    {
        "name": "mediterranean_west",
        "display_name": "Mediterraneo Occidental",
        "bbox": [-1.0, 36.5, 4.0, 39.5],
        "description": "Ruta comercial principal, costas Espana-Argelia",
    },
    {
        "name": "suez_approach",
        "display_name": "Aproximacion Canal de Suez",
        "bbox": [32.0, 29.5, 34.0, 31.5],
        "description": "Zona de espera, alta densidad de barcos",
    },
    {
        "name": "english_channel",
        "display_name": "Canal de la Mancha",
        "bbox": [-2.0, 49.5, 2.0, 51.5],
        "description": "Ruta comercial Europa del Norte",
    },
    {
        "name": "north_adriatic",
        "display_name": "Norte del Adriatico",
        "bbox": [12.0, 44.5, 14.0, 45.8],
        "description": "Zona portuaria, Venecia-Trieste",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_model_info(row) -> ModelInfo:  # type: ignore[no-untyped-def]
    """Convert an asyncpg Record to a ModelInfo model."""
    return ModelInfo(
        id=row["id"],
        name=row["name"],
        version=row["version"],
        format=row["format"],
        file_hash=row["file_hash"],
        size_mb=row["size_mb"],
        base_model=row.get("base_model"),
        compression_technique=row.get("compression_technique", "none"),
        num_params=row.get("num_params"),
        input_size=list(row["input_size"]) if row.get("input_size") else [640, 640],
        classes=list(row["classes"]) if row.get("classes") else ["vessel"],
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    """List all registered models and their compression variants.

    Returns entries from the ``models_registry`` table including
    model name, version, format, file hash, size, compression
    technique, and architecture metadata.
    """
    try:
        rows = await db.fetch(SELECT_ALL_MODELS)
        return [_row_to_model_info(r) for r in rows]
    except Exception as exc:
        logger.error("Failed to list models: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query models registry: {exc}",
        ) from exc


@router.get("/profiles")
async def list_profiles() -> list[dict]:
    """List all available constraint profiles.

    Returns the five predefined hardware-simulation profiles:
    ``ground``, ``sat-high``, ``sat-mid``, ``sat-low``, and
    ``sat-extreme``, each with CPU/RAM limits and a description
    of the hardware it simulates.
    """
    return PROFILES


@router.get("/zones")
async def list_zones() -> list[dict]:
    """List all predefined search zones.

    Each zone includes a name, display name, bounding box
    (``[lon_min, lat_min, lon_max, lat_max]``), and a description
    of the maritime area.
    """
    return SEARCH_ZONES


# ---------------------------------------------------------------------------
# POST /models/fetch — bring a weight into the models volume without SSH
# ---------------------------------------------------------------------------

_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.(onnx|pt)$")


class ModelFetchRequest(BaseModel):
    """Download a weight from an https URL into ``Settings.models_dir``.

    The models directory is a Docker volume on the deployment, so weights
    cannot be shipped in the image; this is the repo-provided, API-invokable
    way to place one (e.g. a GitHub Release asset). The SHA256 is mandatory
    and verified before the file is exposed; the card gate (I-MOD-4 /
    I-AIA-1) is checked BEFORE downloading; the registry scan afterwards
    records compressed variants as ``candidate`` (I-MOD-3).
    """

    filename: str = Field(pattern=_SAFE_FILENAME.pattern, description="Target file name under models_dir.")
    url: str = Field(pattern=r"^https://", max_length=2048)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    overwrite: bool = False


def _download(url: str, dest: Path, chunk: int = 1 << 20) -> str:
    """Stream ``url`` into ``dest`` and return its SHA256 (module-level for tests)."""
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "aidra-models-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r, dest.open("wb") as out:
        while data := r.read(chunk):
            h.update(data)
            out.write(data)
    return h.hexdigest()


@router.post("/models/fetch")
async def fetch_model(request: ModelFetchRequest) -> dict[str, Any]:
    from src.config import Settings
    from src.models.manager import ModelManager, initial_variant_status

    settings = Settings()
    models_dir = Path(settings.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / request.filename
    if target.exists() and not request.overwrite:
        raise HTTPException(status_code=409, detail=f"{request.filename} already exists (set overwrite=true)")

    manager = ModelManager(models_dir=models_dir, db=db)
    stem = target.stem
    try:
        manager._require_model_card(stem, target)  # I-MOD-4 / I-AIA-1, before any download
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"No MODEL_CARD for {stem}: {exc}") from exc

    tmp = models_dir / f".{request.filename}.part"
    try:
        digest = _download(request.url, tmp)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=502, detail=f"download failed: {type(exc).__name__}: {exc}") from exc
    if digest != request.sha256:
        tmp.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=f"SHA256 mismatch: expected {request.sha256[:16]}…, got {digest[:16]}…; file discarded",
        )
    shutil.move(str(tmp), str(target))
    size_mb = round(target.stat().st_size / (1024 * 1024), 2)

    registered = []
    try:
        registered = await manager.scan_and_register()
    except Exception:
        logger.warning("scan_and_register after fetch failed", exc_info=True)
    _name, version, _fmt = ModelManager._parse_model_name(request.filename)
    technique = next((t for _s, v, t in manager_suffixes() if v == version), "none")
    logger.info("Fetched model %s (%.2f MB, sha256 %s…)", request.filename, size_mb, digest[:12])
    return {
        "filename": request.filename,
        "path": str(target),
        "sha256": digest,
        "size_mb": size_mb,
        "registered_models": len(registered),
        "initial_status": initial_variant_status(technique),
    }


def manager_suffixes() -> list[tuple[str, str, str]]:
    from src.models.manager import _COMPRESSION_SUFFIXES

    return list(_COMPRESSION_SUFFIXES)
