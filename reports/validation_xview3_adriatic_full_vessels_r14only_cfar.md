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
- Provenance: commit `59e028b4667d`, model_hash `18aec1bb3caf7dd2`, settings_hash `a6ba34bbe2e3ad59`, seed 42

**Sea-only (I-DET-2, on_land excluded from preds and GT):** AP 0.1707 · Pd 0.4413 · FAR/km² 0.0065 · Precision 0.1857 · GT 1566 · preds 3721
