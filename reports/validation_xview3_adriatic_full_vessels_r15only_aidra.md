## Métricas de validación (D2 — `scripts/run_validation.py`)

- **Match mode**: distancia al centro ≤ 20 px (xView3-SAR convention)
- **AP**: 0.0994
- **Pd (recall)**: 0.3630
- **FAR / km²**: 0.0087
- **Precision**: 0.1515
- **F1**: 0.2137
- Escenas evaluadas: 11
- Ground-truth total: 1997
- Predicciones (post-confidence ≥ 0.25): 4787
- Área cubierta: 468575.2 km²
- Pipeline path: `full`, tile 640px / overlap 64px
- Provenance: commit `40bb5ef977d4`, model_hash `18aec1bb3caf7dd2`, settings_hash `4efd73b776eab54d`, seed 42

**Sea-only (I-DET-2, on_land excluded from preds and GT):** AP 0.1391 · Pd 0.4553 · FAR/km² 0.0069 · Precision 0.1813 · GT 1566 · preds 3932
