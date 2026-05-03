-- ====================================================
-- AIDRA Database Indexes
-- PostgreSQL 16 + PostGIS 3.4
-- Migration: 002_indexes
-- ====================================================

-- Indices para execution_log
CREATE INDEX idx_execution_log_created_at ON execution_log(created_at DESC);
CREATE INDEX idx_execution_log_profile ON execution_log(constraint_profile);
CREATE INDEX idx_execution_log_model ON execution_log(model_name, model_version);
CREATE INDEX idx_execution_log_status ON execution_log(status);
CREATE INDEX idx_execution_log_image_id ON execution_log(image_id);
CREATE INDEX idx_execution_log_trigger ON execution_log(trigger_type);
CREATE INDEX idx_execution_log_bbox ON execution_log USING GIST(image_bbox);

-- Indices para detections
CREATE INDEX idx_detections_execution ON detections(execution_id);
CREATE INDEX idx_detections_confidence ON detections(confidence DESC);
CREATE INDEX idx_detections_source ON detections(source);
CREATE INDEX idx_detections_geo ON detections USING GIST(center_geo);
CREATE INDEX idx_detections_bbox ON detections USING GIST(bbox_geo);
CREATE INDEX idx_detections_created ON detections(created_at DESC);

-- Indices para models_registry
CREATE INDEX idx_models_name ON models_registry(name);
CREATE INDEX idx_models_technique ON models_registry(compression_technique);
