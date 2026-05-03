---
description: Ejecuta una terna de benchmark de compresión {baseline, variante, perfil}
argument-hint: <baseline_model_id> <variant_model_id> <profile:ground|sat-mid|sat-low>
---

Ejecuta el benchmark completo de la terna: `$ARGUMENTS`.

Reglas no negociables (ver `CLAUDE.md` §5.3 + agente `compression-benchmarker`):
- Sin los **3 elementos** (baseline, variante, perfil) → abortar.
- Sin `MODEL_CARD.md` para la variante → abortar (delegar a `ai-act-compliance`).
- Mismo split de validación que el baseline.
- Seed fija (default 42 si no se indica).
- Métricas obligatorias: `mAP@0.5`, `Pd`, `FAR/km²`, `latency_p50/p95`, `peak_ram_mb`, `disk_size_mb` (energía si el perfil lo permite).

Pasos:

1. **Pre-flight** (lanzar `compression-benchmarker` modo plan):
   - Verificar que existen pesos del baseline y de la variante.
   - Verificar `MODEL_CARD.md` para la variante (delegar a `ai-act-compliance` si falta).
   - Validar que el perfil existe en `src/profiles/definitions.py`.
2. **Confirmación**: mostrar al usuario el plan (datasets, splits, seed, perfiles) y **pedir luz verde** antes de ejecutar runs largos.
3. **Ejecución**: correr inference de baseline + variante bajo el perfil. Recolectar métricas con `src/profiles/metrics_collector.py`.
4. **Validación de la terna**: verificar que todas las métricas obligatorias están presentes.
5. **Decisión**: aceptar/rechazar según ΔmAP ≤ 5 pts. Anotar justificación.
6. **Persistencia**: registrar en `execution_log` con `compression_technique` correspondiente; calcular SHA256 de artefactos (delegar a `traceability-curator`).
7. **Reporte**: tabla comparativa lista para D3 (formato del agente `compression-benchmarker`).

No borrar variantes rechazadas — se conservan como evidencia negativa.
