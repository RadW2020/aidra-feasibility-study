## Métricas de validación (D2 — `scripts/run_validation.py`)

- **Match mode**: distancia al centro ≤ 20 px (xView3-SAR convention)
- **AP**: 0.0256
- **Pd (recall)**: 0.1432
- **FAR / km²**: 0.0040
- **Precision**: 0.1313
- **F1**: 0.1370
- Escenas evaluadas: 11
- Ground-truth total: 1997
- Predicciones (post-confidence ≥ 0.25): 2178
- Área cubierta: 468575.2 km²
- Pipeline path: `full`, tile 640px / overlap 64px
- Provenance: commit `9156a3651ec7`, model_hash `18aec1bb3caf7dd2`, settings_hash `de2212a22bc6138a`, seed 42

**Sea-only (I-DET-2, on_land excluded from preds and GT):** AP 0.0412 · Pd 0.1782 · FAR/km² 0.0023 · Precision 0.2029 · GT 1566 · preds 1375
