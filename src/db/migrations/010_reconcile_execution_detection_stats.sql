-- ====================================================
-- AIDRA execution/detection count reconciliation
-- PostgreSQL 16
-- Migration: 010_reconcile_execution_detection_stats
--
-- Historical execution_log rows stored raw detector counts before edge
-- clipping/persistence. The pipeline now records persisted detections;
-- this migration aligns legacy successful runs with the rows actually
-- available for audit in detections.
-- ====================================================

WITH persisted AS (
    SELECT
        e.id,
        COUNT(d.id)::integer AS num_detections,
        AVG(d.confidence)::real AS avg_confidence,
        MAX(d.confidence)::real AS max_confidence,
        MIN(d.confidence)::real AS min_confidence
    FROM execution_log e
    LEFT JOIN detections d ON d.execution_id = e.id
    WHERE e.status = 'success'
    GROUP BY e.id
)
UPDATE execution_log e
SET
    num_detections = p.num_detections,
    avg_confidence = p.avg_confidence,
    max_confidence = p.max_confidence,
    min_confidence = p.min_confidence
FROM persisted p
WHERE e.id = p.id
  AND (
      e.num_detections IS DISTINCT FROM p.num_detections
      OR e.avg_confidence IS DISTINCT FROM p.avg_confidence
      OR e.max_confidence IS DISTINCT FROM p.max_confidence
      OR e.min_confidence IS DISTINCT FROM p.min_confidence
  );
