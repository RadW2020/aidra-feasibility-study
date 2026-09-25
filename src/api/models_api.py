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
# Constraint profiles and search zones
# ---------------------------------------------------------------------------
#
# Derived from the definitions the engine actually uses
# (src/profiles/definitions.py, src/pipeline/ingestion.py) instead of hand
# copies, so what the API advertises is what the trigger accepts.


def profiles_catalog() -> list[dict]:
    from src.profiles.definitions import PROFILE_ORDER
    from src.profiles.definitions import PROFILES as _PROFILES

    return [
        {
            "name": p.name,
            "display_name": p.display_name,
            "description": p.description,
            "cpu_limit": p.cpu_limit,
            "memory_limit_mb": p.memory_limit_mb,
            "simulates": p.simulates,
            "tdp_watts": p.tdp_watts,
        }
        for p in (_PROFILES[n] for n in PROFILE_ORDER)
    ]


def zones_catalog() -> list[dict]:
    from src.pipeline.ingestion import SEARCH_ZONES as _ZONES

    return [
        {
            "name": key,
            "display_name": z["name"],
            "bbox": list(z["bbox"]),
            "description": z["description"],
        }
        for key, z in _ZONES.items()
    ]


# Module-level views kept for importers of the old constants.
PROFILES: list[dict] = profiles_catalog()
SEARCH_ZONES: list[dict] = zones_catalog()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_card(name: str, file_path: str | None) -> bool | None:
    """Whether the AI Act card gate (I-AIA-1) would accept this model."""
    try:
        from src.config import Settings
        from src.models.manager import model_card_exists

        return model_card_exists(Settings().models_dir, name, Path(file_path or f"{name}.pt"))
    except Exception:  # noqa: BLE001 - informational field
        return None


def _row_to_model_info(row) -> ModelInfo:  # type: ignore[no-untyped-def]
    """Convert an asyncpg Record to a ModelInfo model."""
    classes = list(row["classes"]) if row.get("classes") else ["vessel"]
    name = row["name"]
    return ModelInfo(
        id=row["id"],
        name=name,
        version=row["version"],
        format=row["format"],
        file_hash=row["file_hash"],
        size_mb=row["size_mb"],
        base_model=row.get("base_model"),
        compression_technique=row.get("compression_technique", "none"),
        num_params=row.get("num_params"),
        input_size=list(row["input_size"]) if row.get("input_size") else [640, 640],
        classes=classes,
        status=row.get("status") or "active",
        rejection_reason=row.get("rejection_reason"),
        sar_compatible=name.startswith("cfar") or bool({c.lower() for c in classes} & {"ship", "vessel"}),
        has_model_card=_has_card(name, row.get("file_path")),
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
    return profiles_catalog()


@router.get("/zones")
async def list_zones() -> list[dict]:
    """List all predefined search zones.

    Each zone includes a name, display name, bounding box
    (``[lon_min, lat_min, lon_max, lat_max]``), and a description
    of the maritime area.
    """
    return zones_catalog()


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
