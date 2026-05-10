"""
Consultas SQL parametrizadas.

Todas las queries usan $1, $2... (parametros asyncpg, no f-strings).
Nunca concatenar strings SQL.
"""

# ====================================================================
# execution_log
# ====================================================================

INSERT_EXECUTION = """
    INSERT INTO execution_log (
        id, image_id, image_title, image_hash, image_bbox,
        image_sensing_date, image_size_mb, search_zone,
        model_name, model_version, model_hash, model_size_mb,
        model_format, compression_technique,
        confidence_threshold, iou_threshold, constraint_profile,
        cpu_limit, memory_limit_mb, tile_size, tile_overlap,
        num_detections, avg_confidence, max_confidence, min_confidence,
        total_duration_ms, download_ms, preprocessing_ms, inference_ms,
        postprocessing_ms, peak_ram_mb, avg_ram_mb, cpu_usage_pct,
        num_tiles, output_hash, input_params_hash,
        status, error_message, trigger_type, triggered_by,
        pipeline_version, hostname, commit_sha
    ) VALUES (
        $1, $2, $3, $4, CASE WHEN $5::text IS NOT NULL THEN ST_GeomFromGeoJSON($5::text) ELSE NULL END,
        $6, $7, $8,
        $9, $10, $11, $12,
        $13, $14,
        $15, $16, $17,
        $18, $19, $20, $21,
        $22, $23, $24, $25,
        $26, $27, $28, $29,
        $30, $31, $32, $33,
        $34, $35, $36,
        $37, $38, $39, $40,
        $41, $42, $43
    )
"""

SELECT_EXECUTION_BY_ID = """
    SELECT *, ST_AsGeoJSON(image_bbox) AS image_bbox_geojson
    FROM execution_log
    WHERE id = $1
"""

SELECT_EXECUTIONS = """
    SELECT *, ST_AsGeoJSON(image_bbox) AS image_bbox_geojson
    FROM execution_log
    WHERE ($1::text IS NULL OR constraint_profile = $1)
      AND ($2::text IS NULL OR model_name = $2)
      AND ($3::text IS NULL OR status = $3)
      AND ($4::timestamptz IS NULL OR created_at >= $4)
      AND ($5::timestamptz IS NULL OR created_at <= $5)
    ORDER BY created_at DESC
    LIMIT $6 OFFSET $7
"""

COUNT_EXECUTIONS = """
    SELECT COUNT(*)
    FROM execution_log
    WHERE ($1::text IS NULL OR constraint_profile = $1)
      AND ($2::text IS NULL OR model_name = $2)
      AND ($3::text IS NULL OR status = $3)
"""

# Reaper: marks executions stuck in pending/running past a threshold as
# failed. The CTE captures the previous status before the UPDATE so the
# annotation in error_message is accurate (PostgreSQL evaluates the SET
# expression with pre-update column values, but the CTE form makes the
# intent explicit and survives schema reorderings).
REAP_ORPHAN_EXECUTIONS = """
    WITH targets AS (
        SELECT id, status AS prior_status, created_at
        FROM execution_log
        WHERE status IN ('pending', 'running')
          AND created_at < NOW() - make_interval(mins => $1::int)
    )
    UPDATE execution_log e
    SET status = 'failed',
        error_message = NULLIF(
            TRIM(BOTH ' | ' FROM
                COALESCE(e.error_message, '') ||
                CASE WHEN COALESCE(e.error_message, '') = '' THEN '' ELSE ' | ' END ||
                'reaped: stuck in ' || t.prior_status || ' for >' || $1::int || ' minutes'
            ),
            ''
        )
    FROM targets t
    WHERE e.id = t.id
    RETURNING e.id, t.prior_status, t.created_at
"""

# ====================================================================
# detections
# ====================================================================

INSERT_DETECTION = """
    INSERT INTO detections (
        id, execution_id,
        center_geo, bbox_geo, bbox_pixel,
        confidence, source, cfar_snr, yolo_score, class_name,
        tile_index, tile_row_offset, tile_col_offset,
        on_land, cluster_anomaly, thumbnail_path, quality_verdict
    ) VALUES (
        $1, $2,
        ST_SetSRID(ST_MakePoint($3, $4), 4326),
        CASE WHEN $5::text IS NOT NULL THEN ST_GeomFromGeoJSON($5::text) ELSE NULL END,
        $6,
        $7, $8, $9, $10, $11,
        $12, $13, $14,
        $15, $16, $17, $18
    )
"""

INSERT_DETECTIONS_BATCH = """
    INSERT INTO detections (
        execution_id, center_geo, bbox_geo, bbox_pixel,
        confidence, source, cfar_snr, yolo_score, class_name,
        tile_index
    )
    SELECT
        unnest($1::uuid[]),
        ST_SetSRID(ST_MakePoint(unnest($2::double precision[]), unnest($3::double precision[])), 4326),
        ST_GeomFromGeoJSON(unnest($4::text[])),
        unnest($5::real[][]),
        unnest($6::real[]),
        unnest($7::text[]),
        unnest($8::real[]),
        unnest($9::real[]),
        unnest($10::text[]),
        unnest($11::integer[])
"""

SELECT_DETECTIONS = """
    SELECT
        d.*,
        ST_X(d.center_geo) AS longitude,
        ST_Y(d.center_geo) AS latitude,
        ST_AsGeoJSON(d.center_geo) AS center_geojson,
        ST_AsGeoJSON(d.bbox_geo) AS bbox_geojson,
        e.constraint_profile,
        e.model_name,
        e.model_version,
        e.image_id
    FROM detections d
    JOIN execution_log e ON d.execution_id = e.id
    WHERE e.status = 'success'
      AND ($1::text IS NULL OR e.constraint_profile = $1)
      AND ($2::text IS NULL OR e.model_name = $2)
      AND ($3::real IS NULL OR d.confidence >= $3)
      AND ($4::timestamptz IS NULL OR d.created_at >= $4)
      AND ($5::timestamptz IS NULL OR d.created_at <= $5)
      AND ($6::geometry IS NULL OR ST_Intersects(d.center_geo, $6))
      AND ($9::boolean IS NULL OR d.on_land = $9)
      AND ($10::boolean IS NULL OR d.cluster_anomaly = $10)
      AND ($11::text IS NULL OR d.quality_verdict = $11)
    ORDER BY d.confidence DESC
    LIMIT $7 OFFSET $8
"""

SELECT_DETECTION_BY_ID = """
    SELECT
        d.id AS detection_id,
        d.execution_id,
        d.created_at AS detection_created_at,
        d.bbox_pixel,
        d.confidence,
        d.source,
        d.cfar_snr,
        d.yolo_score,
        d.class_name,
        d.on_land,
        d.cluster_anomaly,
        d.quality_verdict,
        d.thumbnail_path,
        d.tile_index,
        d.tile_row_offset,
        d.tile_col_offset,
        ST_X(d.center_geo) AS longitude,
        ST_Y(d.center_geo) AS latitude,
        ST_AsGeoJSON(d.center_geo) AS center_geojson,
        ST_AsGeoJSON(d.bbox_geo) AS bbox_geojson,
        e.id AS execution_log_id,
        e.created_at AS execution_created_at,
        e.image_id,
        e.image_title,
        e.image_hash,
        ST_AsGeoJSON(e.image_bbox) AS image_bbox_geojson,
        e.image_sensing_date,
        e.image_size_mb,
        e.search_zone,
        e.model_name,
        e.model_version,
        e.model_hash,
        e.model_size_mb,
        e.model_format,
        e.compression_technique,
        e.confidence_threshold,
        e.iou_threshold,
        e.constraint_profile,
        e.cpu_limit,
        e.memory_limit_mb,
        e.tile_size,
        e.tile_overlap,
        e.num_detections,
        e.avg_confidence,
        e.max_confidence,
        e.min_confidence,
        e.total_duration_ms,
        e.download_ms,
        e.preprocessing_ms,
        e.inference_ms,
        e.postprocessing_ms,
        e.peak_ram_mb,
        e.avg_ram_mb,
        e.cpu_usage_pct,
        e.num_tiles,
        e.output_hash,
        e.input_params_hash,
        e.status,
        e.error_message,
        e.trigger_type,
        e.triggered_by,
        e.pipeline_version,
        e.hostname,
        e.notes
    FROM detections d
    JOIN execution_log e ON d.execution_id = e.id
    WHERE d.id = $1
"""

# ====================================================================
# benchmarks (queries agregadas)
# ====================================================================

SELECT_BENCHMARKS_BY_MODEL = """
    SELECT
        model_name,
        model_version,
        model_size_mb,
        compression_technique,
        constraint_profile,
        COUNT(*) AS runs,
        AVG(inference_ms) AS avg_inference_ms,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY inference_ms) AS p50_inference_ms,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY inference_ms) AS p95_inference_ms,
        AVG(peak_ram_mb) AS avg_peak_ram_mb,
        AVG(cpu_usage_pct) AS avg_cpu_pct,
        AVG(num_detections) AS avg_detections,
        AVG(avg_confidence) AS avg_confidence
    FROM execution_log
    WHERE status = 'success'
      AND ($1::text IS NULL OR model_name = $1)
      AND ($2::text IS NULL OR constraint_profile = $2)
    GROUP BY model_name, model_version, model_size_mb,
             compression_technique, constraint_profile
    ORDER BY model_size_mb, constraint_profile
"""

SELECT_PROFILE_COMPARISON = """
    SELECT
        constraint_profile,
        model_name,
        model_version,
        AVG(inference_ms) AS avg_latency_ms,
        AVG(peak_ram_mb) AS avg_ram_mb,
        AVG(cpu_usage_pct) AS avg_cpu_pct,
        AVG(num_detections) AS avg_detections,
        AVG(avg_confidence) AS avg_confidence,
        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successes,
        SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) AS failures
    FROM execution_log
    WHERE image_id = $1
      AND model_name = $2
    GROUP BY constraint_profile, model_name, model_version
    ORDER BY CASE constraint_profile
        WHEN 'ground' THEN 1
        WHEN 'sat-high' THEN 2
        WHEN 'sat-mid' THEN 3
        WHEN 'sat-low' THEN 4
        WHEN 'sat-extreme' THEN 5
    END
"""

# ====================================================================
# tasking_queue
# ====================================================================

INSERT_CUE = """
    INSERT INTO tasking_queue (
        triggered_by, triggering_detections,
        target_bbox, target_zone, priority, reason
    ) VALUES (
        $1, $2,
        ST_GeomFromGeoJSON($3), $4, $5, $6
    )
    RETURNING id
"""

SELECT_PENDING_CUES = """
    WITH picked AS (
        SELECT id
        FROM tasking_queue
        WHERE status = 'pending'
          AND (cooldown_until IS NULL OR cooldown_until < NOW())
          AND attempts < max_attempts
        ORDER BY priority DESC, created_at
        LIMIT $1
        FOR UPDATE SKIP LOCKED
    )
    UPDATE tasking_queue AS q
    SET status = 'processing',
        scheduled_at = NOW(),
        attempts = attempts + 1,
        last_error = NULL
    FROM picked
    WHERE q.id = picked.id
    RETURNING q.*, ST_AsGeoJSON(q.target_bbox) AS target_bbox_geojson
"""

UPDATE_CUE_STATUS = """
    UPDATE tasking_queue
    SET status = $2,
        executed_at = CASE WHEN $2 = 'completed' THEN NOW() ELSE executed_at END,
        execution_id = $3,
        result_status = $4,
        confirmed_detections = $5
    WHERE id = $1
"""

# Same as UPDATE_CUE_STATUS but the status transitions automatically to
# 'failed' once attempts+1 reaches max_attempts, so cues that exhausted
# their retries don't sit in 'pending' forever (the dashboard would otherwise
# show them as still-queued indefinitely).
UPDATE_CUE_AFTER_ERROR = """
    UPDATE tasking_queue
    SET status = CASE
            WHEN attempts >= max_attempts THEN 'failed'
            ELSE 'pending'
        END,
        execution_id = $2,
        result_status = $3,
        confirmed_detections = $4,
        last_error = $5,
        cooldown_until = CASE
            WHEN attempts >= max_attempts THEN cooldown_until
            ELSE NOW() + INTERVAL '15 minutes'
        END
    WHERE id = $1
"""

# ====================================================================
# models_registry
# ====================================================================

UPSERT_MODEL = """
    INSERT INTO models_registry (
        name, version, format, file_path, file_hash, size_mb,
        base_model, compression_technique, compression_params,
        num_params, num_layers, input_size, classes, metadata
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
    ON CONFLICT (name, version) DO UPDATE SET
        file_hash = EXCLUDED.file_hash,
        size_mb = EXCLUDED.size_mb,
        metadata = EXCLUDED.metadata
"""

SELECT_ALL_MODELS = """
    SELECT * FROM models_registry ORDER BY name, version
"""

# ====================================================================
# validation_runs (mAP / Pd / FAR persistence — migration 011)
# ====================================================================

INSERT_VALIDATION_RUN = """
    INSERT INTO validation_runs (
        execution_id, model_name, model_version, model_hash,
        compression_technique, dataset, dataset_split,
        match_mode, iou_threshold, center_tolerance_px,
        confidence_threshold,
        num_scenes, num_ground_truth, num_predictions,
        true_positives, false_positives, false_negatives,
        total_area_km2,
        map_at_iou, pd_recall, far_per_km2, precision,
        pr_curve_json, notes
    ) VALUES (
        $1, $2, $3, $4,
        $5, $6, $7,
        $8, $9, $10,
        $11,
        $12, $13, $14,
        $15, $16, $17,
        $18,
        $19, $20, $21, $22,
        $23, $24
    )
    RETURNING id
"""

SELECT_VALIDATION_RUNS = """
    SELECT
        id::text AS id,
        execution_id::text AS execution_id,
        model_name, model_version, model_hash, compression_technique,
        dataset, dataset_split,
        match_mode, iou_threshold, center_tolerance_px, confidence_threshold,
        num_scenes, num_ground_truth, num_predictions,
        true_positives, false_positives, false_negatives,
        total_area_km2,
        map_at_iou, pd_recall, far_per_km2, precision,
        notes, created_at
    FROM validation_runs
    WHERE ($1::text IS NULL OR model_name = $1)
      AND ($2::text IS NULL OR compression_technique = $2)
      AND ($3::text IS NULL OR dataset = $3)
    ORDER BY created_at DESC
    LIMIT $4
"""

# ====================================================================
# resilience_runs (orbital resilience: bitflip, orbit-sim, drift)
# ====================================================================

INSERT_BITFLIP_RUN = """
    INSERT INTO bitflip_runs (
        sweep_id, model_variant, model_size_bytes, num_flips,
        avg_detections, avg_confidence, std_detections, degradation_pct,
        baseline_detections, baseline_confidence, critical_threshold
    ) VALUES (
        $1, $2, $3, $4,
        $5, $6, $7, $8,
        $9, $10, $11
    )
"""

INSERT_ORBIT_SIM_RUN = """
    INSERT INTO orbit_sim_runs (
        satellite, total_images, processed_images, skipped_images,
        cfar_fallback_count, process_count, fallback_cfar_count, skip_count,
        models_used, battery_timeline, final_battery_wh, energy_efficiency
    ) VALUES (
        $1, $2, $3, $4,
        $5, $6, $7, $8,
        $9::jsonb, $10, $11, $12
    )
    RETURNING id
"""

INSERT_DRIFT_ALERT = """
    INSERT INTO drift_alerts (
        is_drifting, metric, z_score, recent_mean,
        historical_mean, recommendation, window_size
    ) VALUES (
        $1, $2, $3, $4,
        $5, $6, $7
    )
"""
