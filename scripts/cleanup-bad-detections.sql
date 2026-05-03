-- Maintenance: drop garbage detections produced before the
-- linear-power CFAR + land-mask + rotation-aware geocoding fixes.
--
-- Usage (PostGIS DB):
--   psql -h <host> -U <user> -d <db> -f scripts/cleanup-bad-detections.sql
--
-- The script is wrapped in a transaction with explicit ROLLBACK at the
-- end so it shows what would change without committing.  Switch the
-- final ROLLBACK to COMMIT once the SELECT counts look right.

\set ON_ERROR_STOP on

BEGIN;

-- 1) Inspect: which executions produced unrealistic vessel counts?
--    Anything >100 detections in the Gibraltar zone with the legacy
--    CFAR-only source is almost certainly the bug-affected runs.
SELECT
    el.id                     AS execution_id,
    el.created_at,
    el.search_zone,
    el.image_id,
    el.num_detections,
    el.avg_confidence,
    COUNT(d.id) FILTER (WHERE d.source = 'cfar')  AS cfar_count,
    COUNT(d.id) FILTER (WHERE d.source = 'yolo')  AS yolo_count,
    COUNT(d.id) FILTER (WHERE d.source = 'fused') AS fused_count
FROM execution_log el
LEFT JOIN detections d ON d.execution_id = el.id
GROUP BY el.id
HAVING COUNT(d.id) > 100
   AND COUNT(d.id) FILTER (WHERE d.source IN ('yolo', 'fused')) = 0
ORDER BY el.created_at DESC;

-- 2) Stage the doomed executions in a temp table so we can see the
--    impact before deleting.
CREATE TEMP TABLE bad_executions AS
SELECT el.id
FROM execution_log el
JOIN detections d ON d.execution_id = el.id
GROUP BY el.id
HAVING COUNT(d.id) > 100
   AND COUNT(d.id) FILTER (WHERE d.source IN ('yolo', 'fused')) = 0;

SELECT COUNT(*) AS executions_to_invalidate FROM bad_executions;
SELECT COUNT(*) AS detections_to_delete
FROM detections
WHERE execution_id IN (SELECT id FROM bad_executions);

-- 3) Delete detections (cascades from execution_log if you prefer to
--    drop the whole execution; here we keep the execution_log row so
--    the audit trail survives, just mark it invalid).
DELETE FROM detections
WHERE execution_id IN (SELECT id FROM bad_executions);

UPDATE execution_log
   SET status = 'invalid',
       error_message = COALESCE(error_message, '') ||
                       ' [cleanup] purged after CFAR linear-power fix',
       num_detections = 0,
       avg_confidence = NULL,
       max_confidence = NULL,
       min_confidence = NULL
 WHERE id IN (SELECT id FROM bad_executions);

-- 4) Verify the impact.
SELECT
    (SELECT COUNT(*) FROM detections)          AS detections_remaining,
    (SELECT COUNT(*) FROM execution_log
        WHERE status = 'invalid')              AS invalidated_executions;

-- Switch this to COMMIT once you're happy with the report above.
ROLLBACK;
