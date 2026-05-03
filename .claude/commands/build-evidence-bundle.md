---
description: Empaqueta el bundle de evidencia D3 (logs, configs, métricas, muestras, manifest SHA256)
argument-hint: [from_date] [to_date] [scope]
---

Construye el paquete de evidencia D3 para el rango: `$ARGUMENTS` (si vacío, últimos 30 días).

Lanzar al agente `traceability-curator` en **Modo B** (bundle D3) con esos parámetros.

Recordatorios:

- Solo entran runs con `image_hash`, `model_hash`, `output_hash`, `input_params_hash`, `pipeline_version`, commit SHA y config serializada.
- Cada artefacto del bundle queda con SHA256 en `MANIFEST.json`.
- El bundle se almacena en `evidence_bundles/<fecha>_<scope>.tar.gz`, dentro UE.
- Verificar al final con `python -m src.traceability.verifier --bundle <path>`.

Si hay runs candidatos sin trazabilidad completa, **listarlos pero no incluirlos** en el bundle. Reportar al usuario para decidir si se hace re-run.

Output: ruta al `.tar.gz`, SHA root del bundle, número de runs incluidos vs excluidos, comando de verificación.
