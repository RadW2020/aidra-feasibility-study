## Métricas de validación (D2 — `scripts/run_validation.py`)

- **Match mode**: distancia al centro ≤ 20 px (xView3-SAR convention)
- **AP**: 0.1100
- **Pd (recall)**: 0.3605
- **FAR / km²**: 0.0083
- **Precision**: 0.1562
- **F1**: 0.2180
- Escenas evaluadas: 11
- Ground-truth total: 1997
- Predicciones (post-confidence ≥ 0.25): 4610
- Área cubierta: 468575.2 km²
- Pipeline path: `full`, tile 640px / overlap 64px
- Provenance: commit `9156a3651ec7`, model_hash `18aec1bb3caf7dd2`, settings_hash `de2212a22bc6138a`, seed 42

**Sea-only (I-DET-2, on_land excluded from preds and GT):** AP 0.1495 · Pd 0.4521 · FAR/km² 0.0065 · Precision 0.1884 · GT 1566 · preds 3757
