## Métricas de validación (D2 — `scripts/run_validation.py`)

- **Match mode**: distancia al centro ≤ 20 px (xView3-SAR convention)
- **AP**: 0.1028
- **Pd (recall)**: 0.3560
- **FAR / km²**: 0.0080
- **Precision**: 0.1594
- **F1**: 0.2202
- Escenas evaluadas: 11
- Ground-truth total: 1997
- Predicciones (post-confidence ≥ 0.25): 4460
- Área cubierta: 468575.2 km²
- Pipeline path: `full`, tile 640px / overlap 64px
- Provenance: commit `8ce9ca2e0ff2`, model_hash `18aec1bb3caf7dd2`, settings_hash `3ab0c3dfc0689266`, seed 42

**Sea-only (I-DET-2, on_land excluded from preds and GT):** AP 0.1424 · Pd 0.4476 · FAR/km² 0.0067 · Precision 0.1819 · GT 1566 · preds 3854
