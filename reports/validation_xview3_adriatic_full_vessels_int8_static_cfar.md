## Métricas de validación (D2 — `scripts/run_validation.py`)

- **Match mode**: distancia al centro ≤ 20 px (xView3-SAR convention)
- **AP**: 0.1344
- **Pd (recall)**: 0.3490
- **FAR / km²**: 0.0067
- **Precision**: 0.1826
- **F1**: 0.2397
- Escenas evaluadas: 11
- Ground-truth total: 1997
- Predicciones (post-confidence ≥ 0.25): 3818
- Área cubierta: 468575.2 km²
- Pipeline path: `full`, tile 640px / overlap 64px
- Provenance: commit `40bb5ef977d4`, model_hash `dfe4c9669c0c19c1`, settings_hash `de2212a22bc6138a`, seed 42

**Sea-only (I-DET-2, on_land excluded from preds and GT):** AP 0.1707 · Pd 0.4413 · FAR/km² 0.0065 · Precision 0.1857 · GT 1566 · preds 3721
