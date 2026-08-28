## Métricas de validación (D2 — `scripts/run_validation.py`)

- **Match mode**: distancia al centro ≤ 20 px (xView3-SAR convention)
- **AP**: 0.0120
- **Pd (recall)**: 0.4337
- **FAR / km²**: 0.1200
- **Precision**: 0.0152
- **F1**: 0.0293
- Escenas evaluadas: 11
- Ground-truth total: 1997
- Predicciones (post-confidence ≥ 0.25): 57117
- Área cubierta: 468575.2 km²
- Pipeline path: `detector`, tile 640px / overlap 64px
- Provenance: commit `8ce9ca2e0ff2`, model_hash `n/a`, settings_hash `3ab0c3dfc0689266`, seed 42

**Sea-only (I-DET-2, on_land excluded from preds and GT):** AP 0.0120 · Pd 0.4337 · FAR/km² 0.1200 · Precision 0.0152 · GT 1997 · preds 57117
