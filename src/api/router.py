"""
Router principal que agrupa todos los sub-routers bajo el prefijo ``/api``.

Cada sub-modulo define su propio ``APIRouter`` con tags para la
documentacion Swagger.  Este archivo simplemente los agrega en un
unico router que ``main.py`` monta en la aplicacion FastAPI.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api import (
    benchmarks,
    detections,
    health,
    interpretability,
    metrics,
    models_api,
    ogc_features,
    orbital,
    pipeline,
    stac,
    tasking,
    tipcue_replay,
    traceability,
    validation,
)

router = APIRouter(prefix="/api")

router.include_router(health.router)
router.include_router(detections.router)
router.include_router(pipeline.router)
router.include_router(benchmarks.router)
router.include_router(traceability.router)
router.include_router(tasking.router)
router.include_router(metrics.router)
router.include_router(models_api.router)
router.include_router(orbital.router)
router.include_router(stac.router)
router.include_router(ogc_features.router)
router.include_router(tipcue_replay.router)
router.include_router(interpretability.router)
router.include_router(validation.router)
