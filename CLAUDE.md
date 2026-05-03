# AIDRA — Instrucciones para LLM operando este repositorio

> **Lee esto antes de tocar nada.** Es el contrato de trabajo del agente sobre AIDRA.
> Para detalle técnico ver `TECHNICAL_SPEC.md` y `mvp_oci.md`. Para el pliego ver `analisis_completo.md`.

## 1. Norte del proyecto

AIDRA = **Artificial Intelligence In-orbit Data pRocessing Assessment**. Proof-of-concept inspirado en la licitación SatCen `SATCEN/2026/OP/0003` (deadline 2026-05-04). **No** es un producto operativo: es un **estudio de evaluación** sobre si la IA on-board para detección de barcos en SAR es viable bajo restricciones de hardware espacial.

**Lo que se evalúa (Q3 = 40 pts del pliego):** metodología, integración GEOINT, trazabilidad, documentación IA. **Lo que se entrega (alto valor):** D3 = paquete de evidencia + D4 = informe final con interpretabilidad.

> **Regla de oro:** cada cambio de código debe responder a *¿esto produce/mantiene evidencia trazable y respeta los invariantes del dominio?* Si la respuesta es no, no es un cambio terminado.

## 2. Decisiones congeladas (no replantear sin instrucción explícita)

| Decisión | Valor | Razón |
|---|---|---|
| Sensor | **SAR (Sentinel-1)** | Madurez en vessel detection, todo-tiempo, gratuito |
| Detector primario | YOLOv8 + CFAR (ensemble) | YOLO para textura, CFAR para reflectores puntuales |
| Datasets | xView3-SAR, HRSID, OpenSARShip | Verificados, gratuitos, licencia compatible |
| Hardware target | **OCI ARM A1 Free Tier** (4 OCPU, 24 GB) | Simula restricción espacial; dentro UE (Frankfurt) |
| Lenguaje | **100% Python ≥ 3.11** | Coherencia stack (FastAPI, PyTorch, rasterio) |
| Persistencia | PostgreSQL 16 + PostGIS 3.4 | Geometría nativa + auditabilidad |
| Region almacenamiento | **UE únicamente** | Requisito contractual |
| Co-author commits | **Nunca** Co-Authored-By | Preferencia del usuario |

## 3. Definition of Done (todo cambio cumple esto)

1. **Tests verdes** — `pytest` pasa, sin saltarse tests existentes.
2. **Lint limpio** — `ruff check .` sin nuevos avisos.
3. **Invariantes intactos** — los listados en §5 siguen sosteniéndose.
4. **Trazabilidad mantenida** — si el cambio toca pipeline/modelo, el `execution_log` sigue capturando `image_hash`, `model_hash`, `output_hash`, `input_params_hash`.
5. **Evidencia para D3/D4** — si el cambio afecta resultados, queda reproducible (seed fija, configs versionadas).
6. **Sin Co-Authored-By** en commits.

## 4. Matriz: criterio del pliego → módulo del repo → test que lo cubre

| Criterio pliego | Módulo principal | Test que lo cubre | Nota |
|---|---|---|---|
| Q3 — Metodología SAR | `src/pipeline/preprocessing.py` | `tests/test_pipeline/test_preprocessing.py` | Cadena calib→speckle→TC |
| Q3 — Detección | `src/pipeline/detection.py`, `src/models/{cfar,yolo}.py` | `tests/test_pipeline/test_detection.py`, `test_models/test_cfar.py` | Métricas Pd/FAR |
| Q3 — Trazabilidad | `src/traceability/`, `src/db/migrations/001_init.sql` | `tests/test_traceability/test_hasher.py` | SHA256 + linaje |
| Q3 — Compresión modelos | `src/models/compression/` | *pendiente* | Quant/prune/KD |
| Q3 — Perfiles de restricción | `src/profiles/` | *pendiente* | ground/sat-mid/sat-low |
| Q3 — GEOINT integración | `src/api/`, `src/db/queries.py` | *pendiente* | Export OGC |
| Tip & Cue (extra) | `src/tipcue/` | *pendiente* | Re-tasking autónomo |
| AI Act (D1, D4) | `models/<nombre>/MODEL_CARD.md` | *pendiente* | Ficha por modelo |
| Observabilidad | `src/observability/` | — | Prometheus + Loki |

> Cuando aparezca *pendiente*, crear el test al tocar ese módulo. No dejar nuevos *pendientes*.

## 5. Invariantes de dominio (el LLM los vigila siempre)

### 5.1 SAR / Sentinel-1
- **I-SAR-1**: Toda escena que llegue a `detection.py` ha pasado por `preprocessing.preprocess_full()` (orbit → calibración σ⁰ → speckle filter → terrain correction). Si falta un paso, la escena se marca `quality=invalid` y NO se infiere sobre ella.
- **I-SAR-2**: `edge swath filter` activo. Detecciones a < `EDGE_BUFFER_PX` del borde de swath se descartan (commit `5e880eb` introdujo el filtro robusto por clústeres de longitud).
- **I-SAR-3**: Footprint clipping contra geometría real, no bbox (commit `734a591`). `global-land-mask` deshabilitado por no fiable.
- **I-SAR-4**: EPSG de salida = 4326 para `bbox_geom` en BD; reproyecciones documentadas.

### 5.2 Detección y persistencia
- **I-DET-1**: Cada detección persistida lleva como mínimo: `scene_id`, `model_id`, `model_hash`, `confidence`, `bbox_geom (4326)`, `pixel_bbox`, `incidence_angle` (si disponible), `timestamp_utc`.
- **I-DET-2**: Detección sobre tierra (según mask de footprint) → flag `on_land=true`. Se conserva, pero se excluye de métricas de mar.
- **I-DET-3**: Densidad anómala (> umbral por km²) → flag `cluster_anomaly`. Probable artefacto borde/speckle.
- **I-DET-4**: `confidence_threshold` y `iou_threshold` proceden de `Settings` o config explícita; nunca hardcoded en lógica.

### 5.3 Modelos y compresión
- **I-MOD-1**: Ningún benchmark de compresión sin **terna** `{baseline FP32, variante comprimida, perfil de hardware}`. Una variante sin baseline NO produce evidencia válida.
- **I-MOD-2**: Métricas obligatorias por terna: `mAP@0.5`, `Pd`, `FAR/km²`, `latencia p50/p95`, `RAM peak`, `tamaño_disco_MB`. Energía si el perfil lo permite.
- **I-MOD-3**: Degradación máxima tolerable declarada *antes* del run (default ΔmAP ≤ 5 pts). Si se supera, variante marcada `rejected` con justificación; no se borra.
- **I-MOD-4**: Cada modelo en uso tiene `MODEL_CARD.md` (propósito, dataset, métricas, sesgos, limitaciones, fecha). Sin ficha → no se ejecuta en evaluación.

### 5.4 Trazabilidad (núcleo del Q3)
- **I-TRACE-1**: Todo artefacto persistido (imagen, modelo, GeoJSON resultado) tiene SHA256 calculado por `src/traceability/hasher.py`.
- **I-TRACE-2**: `execution_log` se inserta en estado `pending` antes del run y se actualiza al final; nunca se inserta solo al éxito.
- **I-TRACE-3**: `run_id` (UUID) se propaga end-to-end a logs estructurados (Loki) y métricas (Prometheus labels).
- **I-TRACE-4**: Configs versionadas en git; cada run referencia commit SHA + hash del Settings serializado.

### 5.5 Soberanía de datos
- **I-EU-1**: Cualquier path de almacenamiento o región de servicio debe estar en UE. PR que añada región fuera UE → bloquear y avisar.

### 5.6 AI Act (Regulation EU 2024/1689)
- **I-AIA-1**: Cada modelo registrado lleva ficha. Sin ficha → no entra al pipeline de evaluación.
- **I-AIA-2**: Resultados con interpretabilidad disponible (Grad-CAM/SHAP sobre detección SAR) si el modelo lo permite — al menos sobre muestreo del dataset de evaluación para D4.

## 6. Quality Gates (cómo se chequean los invariantes)

Estos gates son **obligatorios** antes de declarar un cambio terminado. Pueden ejecutarse manualmente; los marcados `(hook)` deberían automatizarse en `settings.json` cuando el usuario lo apruebe.

| Gate | Comando | Cuándo | Bloquea si... |
|---|---|---|---|
| `gate:lint` (hook PostToolUse) | `ruff check .` | Tras editar `*.py` | Hay nuevos errores |
| `gate:tests-touched` (hook) | `pytest tests/test_<modulo>/ -x` | Tras editar `src/<modulo>/` | Falla test del módulo |
| `gate:invariants` | `pytest -k "invariant" -x` | Antes de commit en `src/` | Falla algún invariante |
| `gate:schema` | inspect migrations + queries | Tras editar `db/` o modelos persistidos | Falta columna trazabilidad |
| `gate:reproducibility` | `pytest -k "reproducibility" --seed=42` | Antes de generar evidencia | Mismo input → distinto output_hash |
| `gate:compression-triplet` | `/run-triplet` valida estructura | Al añadir variante comprimida | Falta baseline o perfil |
| `gate:ai-act-card` | `/check-ai-act <model>` | Al registrar modelo | Falta `MODEL_CARD.md` o campos |
| `gate:eu-region` | grep configs por regiones | En cualquier config nueva | Región ∉ UE |

## 7. Cómo trabajar

### 7.1 Subagentes (delegar trabajo especializado)
En `.claude/agents/`:
- `sar-preproc-auditor` — auditoría de cadena Sentinel-1.
- `detection-quality-reviewer` — métricas Pd/FAR/mAP, leakage, balance.
- `compression-benchmarker` — orquestación de tripletas {baseline, variante, perfil}.
- `traceability-curator` — verifica SHA256, linaje y compone bundle D3.
- `ai-act-compliance` — fichas modelo, interpretabilidad, checklist Reg. 2024/1689.
- `geoint-integrator` — formatos OGC, exportación a sistemas SatCen-like.
- `tip-and-cue-evaluator` — escenarios de re-tasking, ganancia vs. pasada estándar.

### 7.2 Slash-commands (en `.claude/commands/`)
- `/eval-scene <scene_id>` — corre auditoría completa sobre una escena.
- `/run-triplet <baseline> <variant> <profile>` — ejecuta benchmark de compresión.
- `/build-evidence-bundle` — empaqueta D3 (logs + configs + métricas + muestras).
- `/score-against-rubric` — autoevaluación contra rúbrica del pliego (70 pts Q1+Q2+Q3).
- `/check-ai-act <model>` — verifica conformidad modelo.
- `/diff-vs-myriad` — compara enfoque AIDRA con MYRIAD (proyecto referente).

### 7.3 Comandos cotidianos
```bash
# Tests
pytest                                    # todos
pytest tests/test_pipeline/ -x            # módulo pipeline, fail fast

# Lint
ruff check .                              # auditar
ruff check . --fix                        # corregir auto

# Pipeline local
docker compose up -d                      # levantar PostGIS + Grafana + Prom + Loki
python -m src.main                        # API FastAPI
```

### 7.4 Antes de commit
1. `ruff check .` → 0 nuevos avisos.
2. `pytest -x` del módulo tocado.
3. Si tocaste `src/pipeline/`, `src/models/` o `src/traceability/`: revisar invariantes §5.
4. Mensaje de commit en imperativo, sin Co-Authored-By, sin emojis.

## 8. Anti-patrones prohibidos

- ❌ Mockear BD en tests de pipeline/persistencia (preferencia validada del usuario: integración real).
- ❌ Borrar evidencia, runs fallidos o variantes rechazadas. Marcar, no borrar.
- ❌ Hardcodear thresholds en lógica (usar `Settings`).
- ❌ Crear documentación nueva (`*.md`) sin instrucción explícita.
- ❌ Refactor masivo "de paso". Cambios mínimos al alcance pedido.
- ❌ Bypass de hooks (`--no-verify`) sin permiso explícito.
- ❌ `git push --force` o `reset --hard` sin confirmación.

## 9. Ante la duda

- ¿Esto suma evidencia para D3/D4? Si no, repensar.
- ¿Rompe algún invariante §5? Si sí, no avanzar.
- ¿Es reproducible? Si no, fijar seed/config.
- ¿Es soberanía UE? Si no, bloquear.
- ¿Tiene ficha AI Act? Si no, exigirla.

Si nada de lo anterior aplica claramente, preguntar al usuario antes de tomar decisiones que afecten evidencia o invariantes.
