> **SUPERSEDED (2026-08-28, R13)** by `reports/validation_xview3_adriatic_full_vessels_cfar.json`. Reason: conf 0.10 / tiles 1024, detector-only (no Lee, no sea mask, no edge filter); not comparable with the YOLO report (conf 0.25 / tiles 640); AP without monotone envelope; no provenance anchors. Kept for audit.

## Métricas de validación (D2 — `scripts/run_validation.py`)

- **Match mode**: distancia al centro ≤ 20 px (xView3-SAR convention)
- **mAP**: 0.0104
- **Pd (recall)**: 0.4226
- **FAR / km²**: 0.1157
- **Precision**: 0.0153
- Escenas evaluadas: 11
- Ground-truth total: 1997
- Predicciones (post-confidence ≥ 0.10): 55064
- Área cubierta: 468575.2 km²
