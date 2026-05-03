"""
Model registry, constraint profiles, and search zone endpoints.

Provides read-only access to the catalogue of registered ML models,
available constraint profiles, and predefined search zones.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

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
