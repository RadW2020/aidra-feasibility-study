-- 020: vesseltracker-sar-yolov8 int8-static pasa de 'candidate' a 'active' (I-MOD-3).
--
-- Terna I-MOD-1 completada el 2026-08-29:
--   * calidad: mismas 11 escenas xView3 que el baseline FP32 ->
--     ΔAP +0.6 pts, ΔPd +0.4 pts, ΔFAR -0.0004/km2 (umbral declarado ΔmAP <= 5 pts)
--     reports/validation_xview3_adriatic_full_vessels_int8_static_*.json
--   * hardware: OCI ARM A1, misma imagen 4b5dfec3..., perfil ground ->
--     inferencia 52.1 min -> 7.1 min, p50 por tesela 2329 -> 263 ms,
--     detecciones 468 -> 460, RSS 4964 -> 4835 MB
--     (executions 58c7afdd-2a74-43a9-b50e-84da5c7292a0 / 9a5cf2ce-7344-4623-afc2-ebb37daebc88).
-- Solo cambia el status; el baseline FP32 sigue siendo el modelo por defecto.
-- Idempotente: solo actua sobre la fila si sigue en 'candidate'.

UPDATE models_registry
SET status = 'active',
    rejection_reason = NULL
WHERE name = 'vesseltracker-sar-yolov8'
  AND version = 'int8-static'
  AND status = 'candidate';
