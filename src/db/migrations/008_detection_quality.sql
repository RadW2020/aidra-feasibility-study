-- ====================================================
-- AIDRA Detection quality verdicts
-- PostgreSQL 16 + PostGIS
-- Migration: 008_detection_quality
--
-- Keeps raw detector output auditable while making the operationally
-- useful subset explicit:
--   valid_sea_target  -> sea, non-anomalous, YOLO/fused
--   candidate         -> sea, non-anomalous, CFAR-only candidate
--   land_artifact     -> land hit retained for audit, excluded from sea metrics
--   cluster_artifact  -> dense cluster retained for audit, likely artefact
--   outside_footprint -> center falls outside the scene bbox
-- ====================================================

ALTER TABLE detections
    ADD COLUMN IF NOT EXISTS quality_verdict TEXT NOT NULL DEFAULT 'candidate';

COMMENT ON COLUMN detections.quality_verdict IS
    'Operational quality label: valid_sea_target, candidate, land_artifact, cluster_artifact, outside_footprint.';

UPDATE detections d
SET quality_verdict = CASE
    WHEN e.image_bbox IS NOT NULL AND NOT ST_Covers(e.image_bbox, d.center_geo)
        THEN 'outside_footprint'
    WHEN d.on_land
        THEN 'land_artifact'
    WHEN d.cluster_anomaly
        THEN 'cluster_artifact'
    WHEN d.source IN ('yolo', 'fused')
        THEN 'valid_sea_target'
    ELSE 'candidate'
END
FROM execution_log e
WHERE e.id = d.execution_id;

CREATE INDEX IF NOT EXISTS idx_detections_quality_verdict
    ON detections(quality_verdict);

CREATE INDEX IF NOT EXISTS idx_detections_valid_targets
    ON detections(execution_id, source, confidence DESC)
    WHERE quality_verdict = 'valid_sea_target';
