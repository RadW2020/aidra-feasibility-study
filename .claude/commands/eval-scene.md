---
description: Auditoría completa sobre una escena Sentinel-1 (preprocesado + detección + trazabilidad)
argument-hint: <scene_id_or_path>
---

Realiza una auditoría completa sobre la escena: `$ARGUMENTS`.

Pasos:

1. **Localizar la escena**: si el argumento es un path, usarlo; si es un ID, buscar en `download/` o consultar `execution_log` por `image_id` similar.
2. **Lanzar `sar-preproc-auditor`** sobre la escena para verificar la cadena de preprocesado (orbit/calib/speckle/TC, edge filter, footprint clip).
3. **Lanzar `detection-quality-reviewer`** para validar las detecciones asociadas a la escena: invariantes I-DET-1..4, on_land flag, cluster_anomaly.
4. **Lanzar `traceability-curator` (modo auditoría)** para validar SHA256, run_id, columnas críticas en `execution_log`.
5. **Consolidar**: generar un único reporte combinando hallazgos de los tres agentes, agrupados por SEVERIDAD.
6. **Recomendaciones**: lista priorizada de acciones mínimas, con `file_path:line_number`.

NO modifiques código. Solo audita y reporta. Si la escena no existe, devuelve un error claro indicando dónde buscaste.
