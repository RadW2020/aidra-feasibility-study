-- AIDRA eval fixture: a small, fully known world the scenarios reason about.
--
-- Shapes mirror production rows (live API, 2026-09): the static INT8 variant
-- aborting on sat-mid at the ~2.7 GB runtime floor, the orphan reaper, the
-- scheduled-scan dedup skip, a Copernicus outage on a Tip & Cue run, the same
-- scene persisted once per profile, and both legs of the compression triplet.
-- Timestamps are relative to NOW() so "last week" scenarios stay valid.
-- Ids are fixed: scenarios and graders reference them.

TRUNCATE api_audit_log, api_idempotency_keys, tasking_queue, detections,
         validation_runs, execution_log, models_registry RESTART IDENTITY CASCADE;

-- ---------------------------------------------------------------- models
INSERT INTO models_registry (id, name, version, format, file_path, file_hash, size_mb,
                             base_model, compression_technique, classes, status, rejection_reason) VALUES
('a0000000-0000-4000-8000-000000000001', 'vesseltracker-sar-yolov8', 'v1.0', 'pytorch',
 '/eval/models/vesseltracker-sar-yolov8.pt', '18aec1bb3caf7dd2c5ace8d397e241c485e917c28df248eefe794578c996d671',
 49.62, NULL, 'none', '{ship}', 'active', NULL),
('a0000000-0000-4000-8000-000000000002', 'vesseltracker-sar-yolov8', 'int8-static', 'onnx',
 '/eval/models/vesseltracker-sar-yolov8-int8-static.onnx', 'dfe4c9669c0c19c1f03d71fbae4bc5aaa2119fefa89f78656b665e622e194521',
 25.31, 'vesseltracker-sar-yolov8', 'static_int8', '{ship}', 'active', NULL),
('a0000000-0000-4000-8000-000000000003', 'vesseltracker-sar-yolov8', 'int8-dynamic', 'onnx',
 '/eval/models/vesseltracker-sar-yolov8-int8-dynamic.onnx', 'ea0ee6dacd5d389ab5a3778362061776cd52307f0978d6c7db0980935f21573b',
 25.06, 'vesseltracker-sar-yolov8', 'dynamic_int8', '{ship}', 'rejected',
 'I-MOD-3: x4.5 detections vs baseline FP32 on the same scene (9490 vs ~2100); massive false positives from dynamic_int8.'),
('a0000000-0000-4000-8000-000000000004', 'yolov8n', 'v1.0', 'onnx',
 '/eval/models/yolov8n.onnx', 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36',
 6.25, NULL, 'none', '{person,car,boat}', 'active', NULL);

-- ------------------------------------------------------------- executions
-- E1 ground FP32 success on the Gibraltar scene.
INSERT INTO execution_log (id, created_at, image_id, image_title, image_hash, image_sensing_date, image_size_mb,
    search_zone, model_name, model_version, model_hash, model_size_mb, model_format, compression_technique,
    confidence_threshold, iou_threshold, constraint_profile, num_detections, num_valid_targets,
    total_duration_ms, inference_p50_ms, inference_p95_ms, peak_ram_mb, output_hash, input_params_hash,
    status, trigger_type, commit_sha, image_bbox) VALUES
('e1000000-0000-4000-8000-000000000001', NOW() - INTERVAL '2 days', 'S1A_IW_GRDH_GIB_0923', 'S1A_IW_GRDH_1SDV_20260923T063012_GIB',
 '4b5dfec3' || repeat('0', 56), NOW() - INTERVAL '2 days 3 hours', 1702.4, 'gibraltar', 'vesseltracker-sar-yolov8', 'v1.0',
 '18aec1bb3caf7dd2c5ace8d397e241c485e917c28df248eefe794578c996d671', 49.62, 'pytorch', 'none',
 0.25, 0.45, 'ground', 6, 2, 3126000, 2333, 2380, 2610, 'c1' || repeat('0', 62), 'd1' || repeat('0', 62),
 'success', 'scheduled', '703c9dd' || repeat('0', 33),
 ST_GeomFromText('POLYGON((-6.2 35.4,-4.9 35.4,-4.9 36.6,-6.2 36.6,-6.2 35.4))', 4326)),
-- E2 same scene, static INT8 on sat-high (fits with band streaming).
('e1000000-0000-4000-8000-000000000002', NOW() - INTERVAL '2 days' + INTERVAL '1 hour', 'S1A_IW_GRDH_GIB_0923', 'S1A_IW_GRDH_1SDV_20260923T063012_GIB',
 '4b5dfec3' || repeat('0', 56), NOW() - INTERVAL '2 days 3 hours', 1702.4, 'gibraltar', 'vesseltracker-sar-yolov8', 'int8-static',
 'dfe4c9669c0c19c1f03d71fbae4bc5aaa2119fefa89f78656b665e622e194521', 25.31, 'onnx', 'static_int8',
 0.25, 0.45, 'sat-high', 6, 2, 378000, 254, 369, 2700, 'c2' || repeat('0', 62), 'd2' || repeat('0', 62),
 'success', 'manual', '703c9dd' || repeat('0', 33),
 ST_GeomFromText('POLYGON((-6.2 35.4,-4.9 35.4,-4.9 36.6,-6.2 36.6,-6.2 35.4))', 4326)),
-- E3 same scene, static INT8 on sat-mid: memory-budget abort.
('e1000000-0000-4000-8000-000000000003', NOW() - INTERVAL '2 days' + INTERVAL '2 hours', 'S1A_IW_GRDH_GIB_0923', 'S1A_IW_GRDH_1SDV_20260923T063012_GIB',
 '4b5dfec3' || repeat('0', 56), NOW() - INTERVAL '2 days 3 hours', 1702.4, 'gibraltar', 'vesseltracker-sar-yolov8', 'int8-static',
 'dfe4c9669c0c19c1f03d71fbae4bc5aaa2119fefa89f78656b665e622e194521', 25.31, 'onnx', 'static_int8',
 0.25, 0.45, 'sat-mid', 0, NULL, 95000, NULL, NULL, 2650, '', 'd3' || repeat('0', 62),
 'error', 'manual', '703c9dd' || repeat('0', 33), NULL);
UPDATE execution_log SET error_message =
  'MemoryError: memory budget exceeded: peak RSS 2650 MB > 2048 MB budget of profile ''sat-mid'' (enforcement=abort)'
  WHERE id = 'e1000000-0000-4000-8000-000000000003';

INSERT INTO execution_log (id, created_at, image_id, image_hash, search_zone, model_name, model_version, model_hash,
    model_size_mb, confidence_threshold, iou_threshold, constraint_profile, output_hash, input_params_hash,
    status, trigger_type, triggered_by, error_message, notes, commit_sha) VALUES
-- E4 reaped orphan (container restarted mid-run).
('e1000000-0000-4000-8000-000000000004', NOW() - INTERVAL '1 day', 'S1A_IW_GRDH_GIB_0924', 'pending', 'gibraltar',
 'vesseltracker-sar-yolov8', 'v1.0', '18aec1bb3caf7dd2c5ace8d397e241c485e917c28df248eefe794578c996d671', 49.62,
 0.25, 0.45, 'ground', '', 'd4' || repeat('0', 62), 'failed', 'scheduled', NULL,
 'reaped: stuck in running for >240 minutes', NULL, '703c9dd' || repeat('0', 33)),
-- E5 scheduled scan found nothing new (dedup against E1).
('e1000000-0000-4000-8000-000000000005', NOW() - INTERVAL '6 hours', 'S1A_IW_GRDH_GIB_0923', 'pending', 'gibraltar',
 'vesseltracker-sar-yolov8', 'v1.0', '18aec1bb3caf7dd2c5ace8d397e241c485e917c28df248eefe794578c996d671', 49.62,
 0.25, 0.45, 'ground', '', 'd1' || repeat('0', 62), 'skipped', 'scheduled', NULL, NULL,
 'skipped: scene S1A_IW_GRDH_GIB_0923 already processed by execution e1000000-0000-4000-8000-000000000001 with the same model and parameters (6 detections)',
 '703c9dd' || repeat('0', 33)),
-- E6 Tip & Cue follow-up of E1 that hit a Copernicus outage.
('e1000000-0000-4000-8000-000000000006', NOW() - INTERVAL '3 hours', 'pending', 'pending', 'gibraltar',
 'vesseltracker-sar-yolov8', 'v1.0', '18aec1bb3caf7dd2c5ace8d397e241c485e917c28df248eefe794578c996d671', 49.62,
 0.25, 0.45, 'ground', '', 'd6' || repeat('0', 62), 'error', 'cue', 'e1000000-0000-4000-8000-000000000001',
 'IngestionError: Copernicus search failed: 503 Service Unavailable', NULL, '703c9dd' || repeat('0', 33)),
-- E8 an older English Channel run, outside "last week".
('e1000000-0000-4000-8000-000000000008', NOW() - INTERVAL '20 days', 'S1B_IW_GRDH_ENG_0905', 'e8' || repeat('0', 62),
 'english_channel', 'vesseltracker-sar-yolov8', 'v1.0', '18aec1bb3caf7dd2c5ace8d397e241c485e917c28df248eefe794578c996d671',
 49.62, 0.25, 0.45, 'ground', 'c8' || repeat('0', 62), 'd8' || repeat('0', 62), 'success', 'scheduled', NULL,
 NULL, NULL, '6841794' || repeat('0', 33));
UPDATE execution_log SET num_detections = 2, num_valid_targets = 2 WHERE id = 'e1000000-0000-4000-8000-000000000008';

-- ------------------------------------------------------------- detections
-- The same six objects persisted by E1 (ground) and E2 (sat-high): counting
-- rows double-counts; one_run_per_scene keeps E1's.
INSERT INTO detections (id, execution_id, created_at, center_geo, bbox_pixel, confidence, source, cfar_snr,
                        yolo_score, class_name, tile_index, on_land, cluster_anomaly, quality_verdict)
SELECT ('d' || e.n || '000000-0000-4000-8000-00000000000' || o.k)::uuid, e.id, e.created_at,
       ST_SetSRID(ST_MakePoint(o.lon, o.lat), 4326), ARRAY[o.k * 10.0, 20.0, o.k * 10.0 + 30.0, 50.0]::real[],
       o.conf - e.n * 0.01, o.src, o.snr, o.yolo, 'ship', 500 + o.k, o.land, o.cluster, o.verdict
FROM (VALUES (1, 'e1000000-0000-4000-8000-000000000001'::uuid, NOW() - INTERVAL '2 days'),
             (2, 'e1000000-0000-4000-8000-000000000002'::uuid, NOW() - INTERVAL '2 days' + INTERVAL '1 hour'))
     AS e(n, id, created_at),
     (VALUES (1, -5.45, 35.95, 0.94, 'fused', 68.4, 0.88, false, false, 'valid_sea_target'),
             (2, -5.40, 35.90, 0.81, 'yolo', NULL, 0.81, false, false, 'valid_sea_target'),
             (3, -5.55, 36.02, 0.66, 'cfar', 21.0, NULL, false, false, 'candidate'),
             (4, -5.33, 35.85, 0.61, 'cfar', 18.5, NULL, false, false, 'candidate'),
             (5, -5.35, 36.14, 0.58, 'cfar', 30.2, NULL, true, false, 'land_artifact'),
             (6, -5.50, 35.99, 0.52, 'yolo', NULL, 0.52, false, true, 'cluster_artifact'))
     AS o(k, lon, lat, conf, src, snr, yolo, land, cluster, verdict);

INSERT INTO detections (id, execution_id, created_at, center_geo, bbox_pixel, confidence, source, class_name,
                        tile_index, on_land, cluster_anomaly, quality_verdict) VALUES
('d8000000-0000-4000-8000-000000000001', 'e1000000-0000-4000-8000-000000000008', NOW() - INTERVAL '20 days',
 ST_SetSRID(ST_MakePoint(1.2, 50.9), 4326), '{10,10,40,40}', 0.9, 'fused', 'ship', 12, false, false, 'valid_sea_target'),
('d8000000-0000-4000-8000-000000000002', 'e1000000-0000-4000-8000-000000000008', NOW() - INTERVAL '20 days',
 ST_SetSRID(ST_MakePoint(1.4, 51.0), 4326), '{50,50,80,80}', 0.7, 'yolo', 'ship', 13, false, false, 'valid_sea_target');

-- ---------------------------------------------------------- Tip & Cue queue
INSERT INTO tasking_queue (id, created_at, trigger_type, triggered_by, target_bbox, target_zone, priority, reason,
                           status, execution_id, result_status, attempts) VALUES
('c0000000-0000-4000-8000-000000000001', NOW() - INTERVAL '2 days', 'cue', 'e1000000-0000-4000-8000-000000000001',
 ST_GeomFromText('POLYGON((-5.6 35.8,-5.3 35.8,-5.3 36.1,-5.6 36.1,-5.6 35.8))', 4326), 'gibraltar_strait', 2,
 'high_confidence_in_gibraltar_strait_cluster_detected (2 detections, avg_conf=0.88)', 'completed',
 'e1000000-0000-4000-8000-000000000006', 'error', 1),
('c0000000-0000-4000-8000-000000000002', NOW() - INTERVAL '40 minutes', 'cue', NULL,
 ST_GeomFromText('POLYGON((-5.6 35.8,-5.3 35.8,-5.3 36.1,-5.6 36.1,-5.6 35.8))', 4326), 'gibraltar_strait', 2,
 'manual: operator follow-up of the fused cluster', 'pending', NULL, NULL, 0);

-- ------------------------------------------------------- validation (xView3)
INSERT INTO validation_runs (id, created_at, model_name, model_version, model_hash, compression_technique, dataset,
    dataset_split, match_mode, iou_threshold, center_tolerance_px, confidence_threshold, num_scenes, num_ground_truth,
    num_predictions, true_positives, false_positives, false_negatives, total_area_km2, map_at_iou, pd_recall,
    far_per_km2, precision, notes, commit_sha, pipeline_path, provenance_json) VALUES
('f0000000-0000-4000-8000-000000000001', NOW() - INTERVAL '27 days', 'vesseltracker-sar-yolov8', 'v1.0',
 '18aec1bb3caf7dd2c5ace8d397e241c485e917c28df248eefe794578c996d671', 'none', 'xview3-sar/validation/adriatic',
 'validation', 'center', 0.5, 20, 0.25, 11, 1997, 2178, 286, 1892, 1711, 468575.2, 0.0256, 0.1432, 0.00404, 0.1313,
 'r14r15 defaults (fusion=center, yolo_input=unfiltered)', '9156a3651ec736d270dcf402f9ec3ba19e2edd9d', 'full',
 '{"settings_hash": "de2212a22bc6138ae1525fd19cae0cae748fdafe33e5626ea71a8028c7cb6350"}'),
('f0000000-0000-4000-8000-000000000002', NOW() - INTERVAL '26 days', 'vesseltracker-sar-yolov8', 'v1.0',
 '18aec1bb3caf7dd2c5ace8d397e241c485e917c28df248eefe794578c996d671', 'none', 'xview3-sar/validation/adriatic',
 'validation', 'center', 0.5, 20, 0.25, 11, 1997, 1442, 186, 1256, 1811, 468575.2, 0.0148, 0.0931, 0.00268, 0.1290,
 'baseline 2026-08-28 (fusion=iou, yolo_input=filtered)', '8ce9ca2e0ff267e0e48b0f568c9e8ce80fb6d7f8', 'full',
 '{"settings_hash": "3ab0c3dfc0689266dc84557fe0278b450d92507557f30181e68b08f8cf58b131"}'),
('f0000000-0000-4000-8000-000000000003', NOW() - INTERVAL '27 days', 'vesseltracker-sar-yolov8', 'int8-static',
 'dfe4c9669c0c19c1f03d71fbae4bc5aaa2119fefa89f78656b665e622e194521', 'static_int8', 'xview3-sar/validation/adriatic',
 'validation', 'center', 0.5, 20, 0.25, 11, 1997, 1985, 294, 1691, 1703, 468575.2, 0.0317, 0.1472, 0.00361, 0.1481,
 'compression triplet quality leg vs r14r15 FP32', '40bb5ef977d4a36bbc6090ea6e603b0ead386774', 'full',
 '{"settings_hash": "de2212a22bc6138ae1525fd19cae0cae748fdafe33e5626ea71a8028c7cb6350"}');

-- ------------------------------------------------------------- audit trail
INSERT INTO api_audit_log (created_at, request_id, actor, actor_scope, authenticated, client, method, path, operation,
    status_code, outcome, error_code, duration_ms, resource_type, resource_id) VALUES
(NOW() - INTERVAL '2 days' + INTERVAL '59 minutes', 'req-seed-1', 'claude-agent', 'run', true, 'aidra-mcp/1.0.0 via claude-code/2.1',
 'POST', '/api/pipeline/trigger', 'POST /api/pipeline/trigger', 200, 'success', NULL, 41.2,
 'execution', 'e1000000-0000-4000-8000-000000000002'),
(NOW() - INTERVAL '1 day', 'req-seed-2', 'claude-agent', 'run', true, 'aidra-mcp/1.0.0 via claude-code/2.1',
 'POST', '/api/models/fetch', 'POST /api/models/fetch', 403, 'rejected', 'insufficient_scope', 0.9, NULL, NULL);
