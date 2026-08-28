> **SUPERSEDED (2026-08-28)**: GT = all xView3 HIGH+MEDIUM objects instead of is_vessel=True. Detector-only path (no Lee, no sea mask, no clipping).

## Métricas de validación (D2 — `scripts/run_validation.py`)

- **Match mode**: distancia al centro ≤ 20 px (xView3-SAR convention)
- **AP**: 0.0367
- **Pd (recall)**: 0.1490
- **FAR / km²**: 0.0037
- **Precision**: 0.2036
- **F1**: 0.1721
- Escenas evaluadas: 11
- Ground-truth total: 2993
- Predicciones (post-confidence ≥ 0.25): 2191
- Área cubierta: 468575.2 km²
- Pipeline path: `detector`, tile 640px / overlap 64px
- Provenance: commit `d6d36d4b658d`, model_hash `18aec1bb3caf7dd2`, settings_hash `3ab0c3dfc0689266`, seed 42

**Sea-only (I-DET-2, on_land excluded from preds and GT):** AP 0.0367 · Pd 0.1490 · FAR/km² 0.0037 · Precision 0.2036 · GT 2993 · preds 2191
