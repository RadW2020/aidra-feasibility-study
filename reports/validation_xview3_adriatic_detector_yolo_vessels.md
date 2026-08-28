## Métricas de validación (D2 — `scripts/run_validation.py`)

- **Match mode**: distancia al centro ≤ 20 px (xView3-SAR convention)
- **AP**: 0.0255
- **Pd (recall)**: 0.1432
- **FAR / km²**: 0.0041
- **Precision**: 0.1305
- **F1**: 0.1366
- Escenas evaluadas: 11
- Ground-truth total: 1997
- Predicciones (post-confidence ≥ 0.25): 2191
- Área cubierta: 468575.2 km²
- Pipeline path: `detector`, tile 640px / overlap 64px
- Provenance: commit `80b0687552fc`, model_hash `18aec1bb3caf7dd2`, settings_hash `3ab0c3dfc0689266`, seed 42

**Sea-only (I-DET-2, on_land excluded from preds and GT):** AP 0.0255 · Pd 0.1432 · FAR/km² 0.0041 · Precision 0.1305 · GT 1997 · preds 2191
