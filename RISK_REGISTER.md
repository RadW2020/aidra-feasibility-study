# AIDRA — Risk Register

Registro vivo de riesgos del proof-of-concept y plan de contingencia. Se actualiza cuando un riesgo cambia de estado, no en cada commit.

| Severidad | Probabilidad | Impacto |
|---|---|---|
| Alta (A) | Frecuente / casi seguro | Bloquea entregable |
| Media (M) | Posible | Degrada calidad / retrasa |
| Baja (B) | Poco probable | Inconveniencia operativa |

## R1 — Cuota Copernicus Data Space agotada

- **Severidad:** A · **Probabilidad:** M · **Estado:** abierto
- **Descripcion:** El plan gratuito de Copernicus Data Space tiene cuota mensual de descargas y rate-limit por IP. En picos del MVP se puede agotar a mitad de mes.
- **Impacto:** sin imagenes nuevas, los runs en `execution_log` solo cubren un subconjunto reducido — riesgo de evidencia D3 escasa.
- **Mitigacion:**
  1. Cache local de productos S1 ya descargados (`/data/images/`), reutilizable entre runs (mismo `image_hash`).
  2. Limitar busquedas a las 4 zonas oficiales (`gibraltar`, `mar-rojo`, `canal-suez`, `english-channel`).
  3. Plan B: muestreo offline de xView3-SAR + HRSID si Copernicus cae.
- **Trigger de escalado:** > 2 fallos consecutivos `IngestionError` con HTTP 429.

## R2 — OCI ARM A1 Free Tier reclamado

- **Severidad:** A · **Probabilidad:** B · **Estado:** abierto
- **Descripcion:** Oracle puede reclamar instancias Free Tier ARM con 7 dias de aviso si necesita capacidad para clientes de pago.
- **Impacto:** la suite `docker-compose` (PostGIS + Grafana + Loki + Prometheus + AIDRA) deja de estar accesible — perdida del entorno de evidencia.
- **Mitigacion:**
  1. Backup diario de `aidra` DB (pg_dump) a almacenamiento externo UE (S3-compatible).
  2. `docker-compose.yml` portable: cualquier maquina ARM/x86 64-bit con Docker reproduce el entorno (~15 min).
  3. Plan B: migrar a Hetzner CAX11 (~3.5 EUR/mes, region UE) si OCI reclama.
- **Trigger:** correo de Oracle "Always Free reclaim notice".

## R3 — Datasets de entrenamiento cambian licencia o desaparecen

- **Severidad:** M · **Probabilidad:** B · **Estado:** abierto
- **Descripcion:** xView3-SAR, HRSID y OpenSARShip son gratuitos hoy, pero la disponibilidad y los terminos pueden cambiar (mirror caido, restriccion academica solamente).
- **Impacto:** no se puede re-entrenar / fine-tunear; los pesos actuales se mantienen pero no son auditables si el dataset original desaparece.
- **Mitigacion:**
  1. Mirror local de los splits usados en una particion privada UE (cifrado en reposo).
  2. Hash SHA256 de cada archivo del dataset registrado en `MODEL_CARD.md` del modelo correspondiente.
  3. Plan B: anadir SARFish (Maxar) o S1Ships si alguno se cae.
- **Trigger:** 404 en URL canonica + no hay mirror UE.

## R4 — Auditoria AI Act exige reclasificacion

- **Severidad:** M · **Probabilidad:** B · **Estado:** mitigado
- **Descripcion:** Si el sistema se desplegase en uso operativo (no en POC), un auditor podria clasificarlo como alto riesgo bajo Anexo III si el caso de uso lo lleva a vigilancia maritima estatal.
- **Impacto:** obligacion de cumplir requisitos completos del Capitulo III (gestion de riesgos formalizada, gobernanza de datos, registro de eventos auditable, marcado CE, etc.).
- **Mitigacion:**
  1. Adopcion voluntaria de los principios del Capitulo II ya hecha (ver `AI_ACT_DECLARATION.md`).
  2. Trazabilidad reforzada con `execution_log` + `MODEL_CARD.md` + `MANIFEST.json` del bundle D3 — documentacion ya conforme a Art. 11.
  3. Supervision humana documentada (Art. 14).
- **Trigger:** decision de pasar de POC a despliegue operativo.

## R5 — Drift de modelo no detectado

- **Severidad:** M · **Probabilidad:** A · **Estado:** abierto
- **Descripcion:** El MVP no reentrenamiento online; los modelos congelados pueden degradarse silenciosamente sobre escenas nuevas (estaciones, polarizaciones distintas, nuevas zonas).
- **Impacto:** falsos negativos en operativa real. Para el MVP solo afecta a calidad de la evidencia.
- **Mitigacion:**
  1. `flag_cluster_anomaly` (I-DET-3) marca densidades anomalas — heuristica low-cost.
  2. `confidence` por deteccion + `avg_confidence` por execution permiten detectar drops.
  3. Plan B: panel Grafana de tendencia de `avg_confidence` por modelo + zona.
- **Trigger:** drop de `avg_confidence` > 15% en una semana sobre la misma zona.

## R6 — Reproducibilidad rota: `output_hash` no estable + `input_params_hash NULL`

- **Severidad:** A · **Probabilidad:** A · **Estado:** reabierto parcial (2026-05-10)
- **Descripcion:** Auditoria contra prod (2026-05-08) detecto que `output_hash` difiere en TODAS las re-ejecuciones de la misma escena con mismo modelo y mismo perfil — 12 runs muestrados, 12 hashes distintos. Adicionalmente, `input_params_hash` y `commit_sha` aparecian NULL en runs disparados via `trigger-all-profiles`.
- **Impacto:** El gate `gate:reproducibility` declarado en CLAUDE.md §6 ("mismo input → mismo output_hash") NO se cumplia en produccion. El bundle D3 verificaba presencia de hash, no reproducibilidad real.
- **Mitigaciones aplicadas:**
  1. **`compute_result_hash` purga campos no-de-contenido** antes de hashear (commit `8214b44`): id (UUID por-deteccion), thumbnail_path (incluye execution_id), timestamps. Cubierto por `tests/test_traceability/test_hasher.py::test_result_hash_*`.
  2. **`input_params_hash` y `commit_sha` poblados en `run_all_profiles`** (commit pendiente): ambos kwargs ahora se calculan por iteracion de perfil (clonando el request con el profile correcto) y se pasan a `recorder.create_pending`.
  3. `commit_sha` resuelto via `SOURCE_COMMIT` (Coolify) → `AIDRA_COMMIT_SHA` → `git rev-parse HEAD` (I-TRACE-4, migration 004). ✓
  4. `MODEL_CARD.md` incluye SHA256 del peso. ✓
  5. CI bloquea PRs que muevan pesos sin actualizar la ficha. ✓
- **Evidencia corregida (2026-05-10):** la terna que se habia descrito como FP32 sobre image_id `76f82d1c` era en realidad `dynamic_int8` (`model_name='vesseltracker-sar-yolov8-int8-dynamic'`) por resolucion ambigua de modelo sin version explicita. El `output_hash=2c62f00608a38147…` identico en 4 perfiles prueba la canonicalizacion del hash para ese batch concreto, pero **no cierra reproducibilidad FP32**.
- **Mitigacion adicional en repo (2026-05-10):** `PipelineRequest` y `PipelineTriggerRequest` aceptan `model_version`; `Settings.default_model_version='v1.0'`; `ModelManager.get_model()` rechaza nombres ambiguos con multiples variantes si no se pasa version; `input_params_hash` incluye la version solicitada.
- **Caveats residuales:**
  1. Runs antiguos persistidos antes del fix conservan sus hashes "rotos" — no se backfill-ean (cada run es snapshot a su tiempo); el determinismo es propiedad de runs nuevos.
  2. Para `INT8 dinamico` se observo no-determinismo en `num_detections` (1127 vs 9490 sobre la misma escena con mismos params); ver R12 mas abajo.
  3. Falta ejecutar y registrar nueva ronda FP32 explicita (`model_version='v1.0'`) en prod post-fix.
- **Trigger de re-apertura:** verifier `--bundle` reporta hash mismatch sin justificacion, o nueva ronda de runs identicos produce > 1 `output_hash` distinct.

## R12 — INT8 dinamico no determinista en numero de detecciones

- **Severidad:** M · **Probabilidad:** A · **Estado:** abierto (registrado 2026-05-08)
- **Descripcion:** El modelo `vesseltracker-sar-yolov8-int8-dynamic` produjo 1127 vs 9490 detecciones en runs con misma escena, mismo modelo, mismo perfil y mismo `input_params_hash`. La quantizacion INT8 dinamica de PyTorch reusa caches por-sample y depende del orden de threading, lo que introduce variabilidad sub-confidence-threshold que pasa el cut.
- **Impacto:** Las comparaciones de compresion FP32-vs-INT8 son ruidosas; el panel `03-compression-bench` agrega muestras que no son reproducibles individualmente, aunque la media-poblacion siga siendo informativa. La ficha `MODEL_CARD.md` del INT8 debe reflejar este sesgo.
- **Mitigacion:**
  1. Documentar el sesgo en `models/cards/vesseltracker-sar-yolov8-int8-dynamic.MODEL_CARD.md`.
  2. Considerar migrar a **INT8 estatico** con calibracion (post-quantization sin runtime caches), que devuelve outputs identicos run-a-run a costa de un step de calibracion offline.
  3. Pinning de seeds + `torch.use_deterministic_algorithms(True)` en `models/yolo.py`.
- **Trigger:** > 5% de variacion en `num_detections` sobre el mismo input dentro de una semana.

## R7 — Cobertura geografica sesgada

- **Severidad:** B · **Probabilidad:** A · **Estado:** abierto
- **Descripcion:** Las 4 zonas operativas son dominadas por trafico mediterraneo / canal de Suez. Modelos pueden sobre-aprender ese contexto.
- **Impacto:** generalizacion limitada cuando se usen sobre Atlantico Norte u oceanos abiertos.
- **Mitigacion:** declarado como limitacion en cada `MODEL_CARD.md`. Plan post-MVP: ampliar a 2 zonas mas (Mar del Norte, Atlantico SW).

## R8 — Range-Doppler Terrain Correction excluido del alcance MVP

- **Severidad:** B · **Probabilidad:** A (se asume) · **Estado:** mitigado por exclusion formal (autoaudit 2026-04-26)
- **Descripcion:** El pipeline AIDRA NO ejecuta Range-Doppler TC contra DEM SRTM. El geocoding lo realiza ``preprocessing._build_pixel_to_geo_transform`` mediante ajuste afin de 6 parametros sobre GCPs Sentinel-1. ``src/pipeline/terrain_correction.py`` queda como andamio re-activable pero **no integrado**.
- **Impacto:** detecciones sobre relieve costero (> 200 m) pueden mostrar desplazamiento azimutal de hasta ~30 m. Sobre mar abierto el RMSE GCP-linear es < 1 px de S1 GRD (≈10 m), suficiente para deteccion de barcos.
- **Mitigacion:**
  1. Dominio de evaluacion limitado a AOIs maritimas plano-mar (Gibraltar, Mar Rojo, Canal de Suez, English Channel) declaradas en mvp_oci.md.
  2. Las 5 MODEL_CARDs declaran la limitacion en su seccion *Limitaciones* (Anexo IV AI Act).
  3. ``execution_log`` persiste ``geocoding_backend`` para que cualquier run con TC real pueda diferenciarse en el bundle D3.
  4. La exclusion esta documentada de forma irreversible en el docstring de ``src/pipeline/terrain_correction.py``.
- **Trigger de re-activacion:** (a) ampliacion a zonas con relieve costero (fjordos, sw africano), o (b) requerimiento explicito del evaluador SatCen.
- **Nota palanca L5:** decision tomada para cerrar la auditoria de cadena SAR sin dejar la cadena formalmente incompleta. La metodologia declara mar-abierto como unico dominio cubierto.

## R9 — Benchmarks contaminados por bugs de metodologia

- **Severidad:** A · **Probabilidad:** M · **Estado:** abierto (registrado 2026-05-08)
- **Descripcion:** Auditoria contra prod detecta dos `error_message` de runs auto-marcados como invalidos por el operador: *"throttle over-compensation bug — sat-low needed 4.17h vs expected 1.1h"* y *"Methodology bug: ran with time.process_time() throttle (commit 321de6b)"*. Los runs de benchmark anteriores al fix de throttle estan contaminados — comparaciones de perfil entre commits previos y posteriores a `321de6b` mezclan dos metodologias distintas.
- **Impacto:** Las graficas de `04-constraint-profiles` y `03-compression-bench` que agregan a lo largo del tiempo pueden mostrar latencias y RAM no comparables. Riesgo de conclusiones falsas sobre la viabilidad de cada perfil.
- **Mitigacion:**
  1. Marcar runs contaminados con `notes='methodology:pre-321de6b'` o filtrarlos en las queries de Grafana.
  2. Re-ejecutar la terna FP32 + INT8 × 5 perfiles desde `321de6b` para producir un baseline limpio.
  3. Reaper de orphans (job `orphan_reaper`, ver `src/pipeline/scheduler_jobs.py`) ya activo — evita acumulacion futura de runs `pending`/`running` huerfanos por crashes durante la investigacion.
- **Trigger:** detection_quality_reviewer subagent reporta divergencia entre `inference_ms` antes y despues de un bugfix metodologico.

## R10 — Exposicion de credenciales / Grafana anonimo

- **Severidad:** A · **Probabilidad:** B · **Estado:** parcialmente mitigado (2026-05-08)
- **Descripcion:** (a) `aidra.uliber.com` tenia `GF_AUTH_ANONYMOUS_ENABLED=true` por defecto en `docker-compose.coolify.yml`, lo que permitia ejecutar SQL ad-hoc contra Postgres via `/api/ds/query` sin autenticacion. (b) El fichero `.env` local contiene secretos reales (`AIDRA_API_TOKEN`, `COOLIFY_ROOT_API_TOKEN`, `COPERNICUS_PASSWORD`, `DB_PASSWORD`, `CHECKLY_API_KEY`) que han pasado por el contexto del LLM durante la auditoria del 2026-05-08.
- **Impacto:** (a) lectura no autorizada del schema operacional, error_messages internos, hashes de modelo y rutas de imagenes. (b) si los secretos se filtran fuera del workspace, riesgo de manipulacion del orquestador (Coolify root token), descarga de imagenes Sentinel-1 con la cuota gratuita del usuario, o emision de POST autenticado a la API.
- **Mitigacion:**
  1. ✓ Default de `GRAFANA_ANONYMOUS_ENABLED` cambiado a `false` (compose). Tras el siguiente deploy CI, anonymous queda desactivado salvo override explicito.
  2. ⚠️ Pendiente: rotar `AIDRA_API_TOKEN`, `COOLIFY_ROOT_API_TOKEN`, `GRAFANA_PASSWORD`, `DB_PASSWORD`, `COPERNICUS_PASSWORD`. La rotacion requiere accion del operador (Coolify dashboard, Copernicus account, Postgres ALTER USER) y por tanto NO se ejecuta automaticamente desde el pipeline de codigo.
  3. ⚠️ Pendiente: crear usuario `evaluator` con rol Viewer en Grafana para mantener acceso de evaluacion sin SQL ad-hoc.
- **Trigger:** cualquier reporte de acceso no autorizado a `aidra.uliber.com` o `aidra-api.uliber.com`.

## R11 — Inflacion de detecciones por land artifacts (88% del total)

- **Severidad:** M · **Probabilidad:** A · **Estado:** abierto (registrado 2026-05-08)
- **Descripcion:** Auditoria contra prod muestra que 88% (59,059 de 66,985) de las detecciones persistidas son `land_artifact`. CFAR genera ruido masivo sobre tierra (mascara coarse 32×32 ≈ 200 m), todas se persisten para auditoria pero los dashboards `02-pipeline-metrics` (panel "Detection Counts") y `06-obdp-value` (panel "Compression Ratio") usan `num_detections` raw de `execution_log` — inflando metricas operacionales 40×.
- **Impacto:** Un evaluador externo lee "5391 vesseles detectados" y "INT8 detecta el doble que FP32" cuando la realidad es ruido CFAR sobre tierra. Las graficas de "ahorro de bandwidth" del dashboard `06-obdp-value` quedan infladas y no representan operacion real.
- **Mitigacion:**
  1. Filtrar dashboards operacionales por `quality_verdict='valid_sea_target'`.
  2. Anadir columna `num_valid_targets` en `execution_log` poblada por `_save_detections()`, separada del raw `num_detections`.
  3. Documentar el ratio land/total como sesgo conocido en `MODEL_CARD.md` de cada modelo SAR.
- **Trigger:** ratio `valid_sea_target / total > 0.05` durante > 24h (deteccion de mejora) o caida de `valid_sea_target` absoluto en una zona conocida.

## Plan de contingencia consolidado

1. **Backup diario** de `aidra` DB + `models/` + bundles D3 a S3-compatible UE.
2. **Mirror local** de datasets en almacenamiento cifrado.
3. **`docker-compose.yml` portable** validado en 2 hosts distintos (OCI ARM + Hetzner CAX11).
4. **Documentacion congelada** por release: `git tag` + bundle D3 asociado al tag.
5. **Decision de escalado** a despliegue operativo no se toma sin auditoria AI Act formal previa.

— Ultima revision: 2026-05-08 (auditoria externa contra prod, anadidos R9/R10/R11, R6 reabierto)
