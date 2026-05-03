---
description: Auditoría rápida de trazabilidad (SHA256, execution_log, run_id, configs)
---

Lanza al agente `traceability-curator` en **Modo A — Auditoría**.

Verifica:
- `execution_log` cubre las columnas críticas (`image_hash`, `model_hash`, `output_hash`, `input_params_hash`).
- `pending → final` flow en `src/pipeline/engine.py`.
- `run_id` propagado a logs Loki y métricas Prometheus.
- Configs versionadas con commit SHA por run.
- Muestra de SHA256 cuadra con artefactos reales.

Output: reporte estructurado con HALLAZGOS por severidad y RECOMENDACIONES con `file:line`.
