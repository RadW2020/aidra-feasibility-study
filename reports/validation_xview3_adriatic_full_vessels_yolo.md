## Métricas de validación (D2 — `scripts/run_validation.py`)

- **Match mode**: distancia al centro ≤ 20 px (xView3-SAR convention)
- **AP**: 0.0148
- **Pd (recall)**: 0.0931
- **FAR / km²**: 0.0027
- **Precision**: 0.1290
- **F1**: 0.1082
- Escenas evaluadas: 11
- Ground-truth total: 1997
- Predicciones (post-confidence ≥ 0.25): 1442
- Área cubierta: 468575.2 km²
- Pipeline path: `full`, tile 640px / overlap 64px
- Provenance: commit `8ce9ca2e0ff2`, model_hash `18aec1bb3caf7dd2`, settings_hash `3ab0c3dfc0689266`, seed 42

**Sea-only (I-DET-2, on_land excluded from preds and GT):** AP 0.0254 · Pd 0.1162 · FAR/km² 0.0015 · Precision 0.2011 · GT 1566 · preds 905
