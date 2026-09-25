"""Discovery endpoints: what exists, what the words mean, what is configured.

* ``GET /api/catalog`` — zones (pipeline search zones *and* Tip & Cue zones,
  with the mapping between them), constraint profiles, registered models
  with status and gate checks, and the defaults a run falls back to.
* ``GET /api/vocabulary`` — every enumeration with its meaning
  (``src/vocabulary.py``).
* ``GET /api/config`` — the effective ``Settings`` with secrets censored,
  plus the ``settings_hash`` D3 bundles record and the running commit.

A client should be able to plan any call from these three answers without
reading the source or asking a human.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from src import vocabulary
from src.api.models_api import list_models, profiles_catalog, zones_catalog
from src.config import Settings

router = APIRouter(tags=["discovery"])


@router.get("/catalog")
async def get_catalog() -> dict[str, Any]:
    """Zones, profiles, models (with status) and defaults in one call."""
    from src.tipcue.zones import DEFAULT_ZONES

    settings = Settings()
    models = [m.model_dump(mode="json") for m in await list_models()]
    return {
        "zones": {
            "search_zones": zones_catalog(),
            "tipcue_zones": [
                {
                    "id": z.id,
                    "name": z.name,
                    "bbox": list(z.bbox),
                    "search_zone": z.search_zone,
                    "active": z.active,
                    "description": z.description,
                }
                for z in DEFAULT_ZONES
            ],
            "note": (
                "Pipeline runs take a search zone. Tip & Cue zones are operational areas; "
                "cues on them search the scenes of their search_zone."
            ),
        },
        "profiles": profiles_catalog(),
        "models": models,
        "builtin_detectors": [
            {"name": "cfar-default", "note": "CA-CFAR, always available; window from Settings (cfar_*)."}
        ],
        "defaults": {
            "zone": settings.default_zone,
            "model": settings.default_model,
            "model_version": settings.default_model_version,
            "profile": settings.default_profile,
            "confidence_threshold": settings.confidence_threshold,
            "iou_threshold": settings.iou_threshold,
        },
        "see_also": {"vocabulary": "/api/vocabulary", "config": "/api/config", "openapi": "/openapi.json"},
    }


@router.get("/vocabulary")
async def get_vocabulary() -> dict[str, Any]:
    """Every enumerated value the API returns or accepts, with its meaning."""
    return vocabulary.as_document()


@router.get("/config")
async def get_config() -> dict[str, Any]:
    """Effective configuration, secrets censored.

    ``settings_hash`` is computed exactly like the one in a D3 bundle's
    manifest, so it tells whether a bundle was produced under the
    configuration this deployment runs now.
    """
    from src.traceability.bundler import censored_settings_snapshot
    from src.traceability.hasher import get_commit_sha

    data, _payload, digest = censored_settings_snapshot(Settings())
    return {
        "commit_sha": get_commit_sha(),
        "settings_hash": digest,
        "settings": data,
        "note": "Secrets (passwords, tokens, URL credentials) are masked as ***.",
    }
