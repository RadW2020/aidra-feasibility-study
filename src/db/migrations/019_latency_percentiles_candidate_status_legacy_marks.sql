-- 019: (a) latencia p50/p95 por tesela en execution_log (I-MOD-2);
--      (b) estado 'candidate' para variantes comprimidas sin terna (I-MOD-3);
--      (c) marca metodologica R9 para runs anteriores al fix del throttle.
--
-- (a) total_duration_ms / num_tiles esconde la cola de latencia que importa
--     en un procesador embarcado. DetectionEngine mide cada tesela
--     (CFAR + YOLO) y el recorder persiste los percentiles.
--
-- (b) Hasta ahora una variante recien escaneada nacia 'active'. I-MOD-3 exige
--     que solo pase a activa tras la terna {baseline, variante, perfil}
--     con la degradacion maxima declarada. ModelManager registra ahora las
--     variantes con compression_technique <> 'none' como 'candidate'; el
--     upsert no pisa un status existente.
--
-- (c) R9: los runs anteriores al fix dc8a40d (2026-05-04 18:37Z, "drive
--     CPUThrottle from wall-time of work, not process CPU time") midieron
--     el throttle con time.process_time(). Se anota la fila con el marcador
--     que los dashboards 03/04 ya excluyen ('methodology:pre-321de6b').
--     Corte 2026-05-04T21:00Z (commit + margen de despliegue); idempotente.
--     Marcar, no borrar (CLAUDE.md §8).

ALTER TABLE execution_log
    ADD COLUMN IF NOT EXISTS inference_p50_ms REAL,
    ADD COLUMN IF NOT EXISTS inference_p95_ms REAL;

COMMENT ON COLUMN execution_log.inference_p50_ms IS
    'Mediana de la latencia de inferencia por tesela (CFAR + YOLO), ms (I-MOD-2, migracion 019).';
COMMENT ON COLUMN execution_log.inference_p95_ms IS
    'Percentil 95 de la latencia de inferencia por tesela, ms (I-MOD-2, migracion 019).';

ALTER TABLE models_registry
    DROP CONSTRAINT IF EXISTS models_registry_status_check;
ALTER TABLE models_registry
    ADD CONSTRAINT models_registry_status_check
    CHECK (status IN ('candidate', 'active', 'rejected', 'retired'));

COMMENT ON COLUMN models_registry.status IS
    'candidate (variante sin terna evaluada) | active | rejected (I-MOD-3, requiere rejection_reason) | retired';

UPDATE execution_log
SET notes = COALESCE(NULLIF(notes, '') || ' ', '') || 'methodology:pre-321de6b'
WHERE created_at < '2026-05-04T21:00:00Z'::timestamptz
  AND constraint_profile <> 'ground'
  AND (notes IS NULL OR notes NOT LIKE '%methodology:pre-321de6b%');
