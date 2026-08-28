-- 017: num_valid_targets en execution_log (R11, I-DET-2).
--
-- `num_detections` cuenta TODO lo persistido, incluido el ruido CFAR
-- sobre tierra (88% del total en la auditoria de 2026-05-08, 55% en la
-- era pre-mascara-afin, ~2% despues). Los dashboards operacionales y el
-- modelo declarado de "producto reducido" del dashboard 06 necesitan el
-- recuento de objetivos de mar validos, separado del raw. Marcar, no
-- borrar: `num_detections` se conserva intacto para auditoria.
--
-- Backfill idempotente desde `detections.quality_verdict` para los runs
-- exitosos ya persistidos. Los runs nuevos lo rellenan en
-- `PipelineEngine._save_detections` -> `ExecutionRecorder.update`.

ALTER TABLE execution_log
    ADD COLUMN IF NOT EXISTS num_valid_targets INTEGER;

COMMENT ON COLUMN execution_log.num_valid_targets IS
    'Detecciones persistidas con quality_verdict = valid_sea_target (I-DET-2). '
    'num_detections sigue siendo el total raw incluyendo land_artifact.';

WITH valid AS (
    SELECT
        e.id,
        COUNT(d.id) FILTER (WHERE d.quality_verdict = 'valid_sea_target')::integer
            AS num_valid_targets
    FROM execution_log e
    LEFT JOIN detections d ON d.execution_id = e.id
    WHERE e.status = 'success'
    GROUP BY e.id
)
UPDATE execution_log e
SET num_valid_targets = v.num_valid_targets
FROM valid v
WHERE e.id = v.id
  AND e.num_valid_targets IS DISTINCT FROM v.num_valid_targets;
