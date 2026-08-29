## Métricas de validación (D2 — `scripts/run_validation.py`)

- **Match mode**: distancia al centro ≤ 20 px (xView3-SAR convention)
- **AP**: 0.1176
- **Pd (recall)**: 0.3555
- **FAR / km²**: 0.0079
- **Precision**: 0.1615
- **F1**: 0.2222
- Escenas evaluadas: 11
- Ground-truth total: 1997
- Predicciones (post-confidence ≥ 0.25): 4395
- Área cubierta: 468575.2 km²
- Pipeline path: `full`, tile 640px / overlap 64px
- Provenance: commit `40bb5ef977d4`, model_hash `dfe4c9669c0c19c1`, settings_hash `de2212a22bc6138a`, seed 42

**Sea-only (I-DET-2, on_land excluded from preds and GT):** AP 0.1561 · Pd 0.4483 · FAR/km² 0.0064 · Precision 0.1895 · GT 1566 · preds 3704
