> **SUPERSEDED (2026-08-28)**: GT = all xView3 HIGH+MEDIUM objects instead of is_vessel=True. Detector-only path (no Lee, no sea mask, no clipping).

## Métricas de validación (D2 — `scripts/run_validation.py`)

- **Match mode**: distancia al centro ≤ 20 px (xView3-SAR convention)
- **AP**: 0.0315
- **Pd (recall)**: 0.5005
- **FAR / km²**: 0.1187
- **Precision**: 0.0262
- **F1**: 0.0498
- Escenas evaluadas: 11
- Ground-truth total: 2993
- Predicciones (post-confidence ≥ 0.25): 57117
- Área cubierta: 468575.2 km²
- Pipeline path: `detector`, tile 640px / overlap 64px
- Provenance: commit `80b0687552fc`, model_hash `n/a`, settings_hash `3ab0c3dfc0689266`, seed 42

**Sea-only (I-DET-2, on_land excluded from preds and GT):** AP 0.0315 · Pd 0.5005 · FAR/km² 0.1187 · Precision 0.0262 · GT 2993 · preds 57117
