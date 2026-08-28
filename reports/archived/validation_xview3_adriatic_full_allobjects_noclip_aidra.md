> **SUPERSEDED (2026-08-28)**: GT = all xView3 HIGH+MEDIUM objects (incl. fixed infrastructure) instead of is_vessel=True, and no valid-data-boundary clipping. First full-pipeline run; its per-scene prediction counts are identical to the vessels-only rerun (reproducibility evidence).

## Métricas de validación (D2 — `scripts/run_validation.py`)

- **Match mode**: distancia al centro ≤ 20 px (xView3-SAR convention)
- **AP**: 0.0371
- **Pd (recall)**: 0.4477
- **FAR / km²**: 0.0361
- **Precision**: 0.0734
- **F1**: 0.1261
- Escenas evaluadas: 11
- Ground-truth total: 2993
- Predicciones (post-confidence ≥ 0.25): 18257
- Área cubierta: 468575.2 km²
- Pipeline path: `full`, tile 640px / overlap 64px
- Provenance: commit `2cae5abb7dd2`, model_hash `18aec1bb3caf7dd2`, settings_hash `3ab0c3dfc0689266`, seed 42

**Sea-only (I-DET-2, on_land excluded from preds and GT):** AP 0.0469 · Pd 0.5382 · FAR/km² 0.0347 · Precision 0.0757 · GT 2471 · preds 17580
