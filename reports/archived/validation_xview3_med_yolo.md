> **SUPERSEDED (2026-08-28, R13)** by `reports/validation_xview3_adriatic_full_vessels_yolo.json`. Reason: detector-only (no Lee, no sea mask, no edge filter); AP without monotone envelope; no provenance anchors. Kept for audit.

## Métricas de validación (D2 — `scripts/run_validation.py`)

- **Match mode**: distancia al centro ≤ 20 px (xView3-SAR convention)
- **mAP**: 0.0242
- **Pd (recall)**: 0.1432
- **FAR / km²**: 0.0041
- **Precision**: 0.1305
- Escenas evaluadas: 11
- Ground-truth total: 1997
- Predicciones (post-confidence ≥ 0.25): 2191
- Área cubierta: 468575.2 km²
