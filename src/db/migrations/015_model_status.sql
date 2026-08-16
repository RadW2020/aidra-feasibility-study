-- 015: estado de variante en models_registry (I-MOD-3).
--
-- I-MOD-3 exige que una variante comprimida que supere la degradacion
-- maxima tolerable quede marcada `rejected` con justificacion, sin
-- borrarse. El esquema no tenia donde persistir esa marca: el invariante
-- era incumplible. Se añade status + rejection_reason.
--
-- Ademas se aplica la marca retroactiva a la variante int8-dynamic:
-- auditoria 2026-08-16 sobre execution_log encontro que produce 9490
-- detecciones sobre la escena image_hash=0bec6fb4... frente a ~2100 del
-- baseline FP32 (x4.5, reproducido en dos runs independientes,
-- executions 173bbdb5... y 77541bb2..., commits 92b2515 y 3bafeab).
-- Degradacion masiva por falsos positivos: supera cualquier umbral
-- razonable de I-MOD-3. La evidencia se conserva; solo se marca.

ALTER TABLE models_registry
    ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active',
    ADD COLUMN IF NOT EXISTS rejection_reason TEXT;

ALTER TABLE models_registry
    DROP CONSTRAINT IF EXISTS models_registry_status_check;
ALTER TABLE models_registry
    ADD CONSTRAINT models_registry_status_check
    CHECK (status IN ('active', 'rejected', 'retired'));

-- Una variante rejected debe explicar por que (I-MOD-3: "con justificacion").
ALTER TABLE models_registry
    DROP CONSTRAINT IF EXISTS models_registry_rejection_reason_check;
ALTER TABLE models_registry
    ADD CONSTRAINT models_registry_rejection_reason_check
    CHECK (status <> 'rejected' OR rejection_reason IS NOT NULL);

COMMENT ON COLUMN models_registry.status IS
    'active | rejected (I-MOD-3, requiere rejection_reason) | retired';
COMMENT ON COLUMN models_registry.rejection_reason IS
    'Justificacion obligatoria cuando status=rejected (I-MOD-3)';

UPDATE models_registry
SET status = 'rejected',
    rejection_reason = 'I-MOD-3: x4.5 detecciones vs baseline FP32 en la '
        'misma escena (9490 vs ~2100, image_hash 0bec6fb4..., reproducido '
        'en 2 runs: 173bbdb5/77541bb2, commits 92b2515/3bafeab). '
        'Degradacion por falsos positivos masivos de la cuantizacion '
        'dynamic_int8. Auditoria 2026-08-16.'
WHERE name = 'vesseltracker-sar-yolov8'
  AND version = 'int8-dynamic'
  AND status = 'active';
