> **SUPERSEDED (2026-08-28)**: run without valid-data-boundary clipping; see JSON superseded_reason.

## Métricas de validación (D2 — `scripts/run_validation.py`)

- **Match mode**: distancia al centro ≤ 20 px (xView3-SAR convention)
- **AP**: 0.0000
- **Pd (recall)**: 0.0000
- **FAR / km²**: 0.0000
- **Precision**: 0.0000
- **F1**: 0.0000
- Escenas evaluadas: 11
- Ground-truth total: 1997
- Predicciones (post-confidence ≥ 0.25): 0
- Área cubierta: 468575.2 km²
- Pipeline path: `full`, tile 640px / overlap 64px
- Provenance: commit `d6d36d4b658d`, model_hash `18aec1bb3caf7dd2`, settings_hash `3ab0c3dfc0689266`, seed 42

**Sea-only (I-DET-2, on_land excluded from preds and GT):** AP 0.0000 · Pd 0.0000 · FAR/km² 0.0000 · Precision 0.0000 · GT 1566 · preds 0
