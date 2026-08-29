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

- **Severidad:** M · **Probabilidad:** A · **Estado:** **superado 2026-08-29** por la variante estatica `vesseltracker-sar-yolov8-int8-static` (Conv-only, Percentile sobre 32 teselas de barcos): determinista, y sobre las mismas 11 escenas xView3 que el baseline FP32 da ΔAP +0.6 pts / ΔPd +0.4 pts / ΔFAR −0.0004 (I-MOD-3 cumplido en calidad); 26.5 MB (−49 %). Pata de hardware en OCI (misma imagen, `ground`): 7.4× mas rapido por escena, p50 263 vs 2329 ms por tesela, mismas detecciones → **variante `active` (migracion 020)**. La dinamica sigue `rejected`. Perfiles `sat-*`: ver R17 (registrado 2026-05-08)
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

- **Severidad:** M · **Probabilidad:** A · **Estado:** mitigado parcialmente (2026-08-28: columna `num_valid_targets` + dashboards 02/06 sobre ella; pendiente verificar en prod tras deploy de migracion 017)
- **Descripcion:** Auditoria contra prod muestra que 88% (59,059 de 66,985) de las detecciones persistidas son `land_artifact`. CFAR genera ruido masivo sobre tierra (mascara coarse 32×32 ≈ 200 m), todas se persisten para auditoria pero los dashboards `02-pipeline-metrics` (panel "Detection Counts") y `06-obdp-value` (panel "Compression Ratio") usan `num_detections` raw de `execution_log` — inflando metricas operacionales 40×.
- **Impacto:** Un evaluador externo lee "5391 vesseles detectados" y "INT8 detecta el doble que FP32" cuando la realidad es ruido CFAR sobre tierra. Las graficas de "ahorro de bandwidth" del dashboard `06-obdp-value` quedan infladas y no representan operacion real.
- **Mitigacion:**
  1. Filtrar dashboards operacionales por `quality_verdict='valid_sea_target'`.
  2. ✓ (2026-08-28) Columna `num_valid_targets` en `execution_log` (migracion 017, backfill idempotente desde `detections.quality_verdict`), poblada por `_save_detections()` -> `ExecutionRecorder.update`; exportada en el bundle D3. Dashboards `02` (columna `valid_targets`) y `06` (`COALESCE(num_valid_targets, num_detections)` en las 5 consultas) la usan.
  3. Documentar el ratio land/total como sesgo conocido en `MODEL_CARD.md` de cada modelo SAR.
- **Trigger:** ratio `valid_sea_target / total > 0.05` durante > 24h (deteccion de mejora) o caida de `valid_sea_target` absoluto en una zona conocida.

## R13 — Validacion D2 desacoplada del pipeline y no comparable entre detectores

- **Severidad:** M · **Probabilidad:** A · **Estado:** **cerrado 2026-08-28** — `reports/validation_xview3_adriatic_full_vessels_{cfar,yolo,aidra,fused_only}.json` (ruta `full`, mismos parametros para los 4 conjuntos, anclas de trazabilidad completas, dos corridas con conteos identicos). Los hallazgos que destapo viven ahora en R14 y R15.
- **Descripcion:** `scripts/run_validation.py::_run_inference` lee los rasters `VH_dB.tif` ya preprocesados por xView3 y ejecuta el detector directamente: sin `preprocess_full()` (Lee, sea mask, edge filter), sin flags `on_land`. Ademas CFAR y YOLO se validaron con parametros distintos (conf 0.10 / tiles 1024 vs conf 0.25 / tiles 640) y la fusion CFAR ∩ YOLO —modo operativo real— no se ha medido nunca contra GT. Los JSON de `reports/validation_*.json` no llevan `model_hash`, `image_hash`, `commit_sha` ni seed. `map_at_iou` no aplica envolvente monotona y con `match_mode=center` no es mAP@IoU0.5 (I-MOD-2).
- **Impacto:** Las cifras publicadas describen los detectores aislados, no el sistema AIDRA (I-SAR-1 no ejercitado en la evidencia). Los ratios "3× mas vessels / 25× mas detecciones" no son una comparacion controlada. La decision arquitectural central (fusion) carece de evidencia. README y card lo declaran desde 2026-08-28.
- **Mitigacion:**
  1. Harness homogeneo: mismos `tile_size` y `confidence_threshold` (desde `Settings`) para CFAR, YOLO y fusion; `--match-mode` e `--iou` explicitos.
  2. Ruta `--pipeline-path full` que aplique `preprocess_full()` (o Lee + edge filter + sea mask sobre el raster xView3) y reporte metricas raw y solo-mar.
  3. Validar `--model fused` y publicar `validation_xview3_med_fused.json`; sustituir el parrafo de hipotesis de la card por la fila medida.
  4. `map_at_iou` con envolvente monotona; reportar `mAP@0.5(IoU)` y `F1@center-20px` por separado.
  5. Anclas de trazabilidad en cada reporte (`model_hash`, `image_hashes[]`, `commit_sha`, `settings_hash`, `seed`); `POST /api/validation/import` las exige.
- **Trigger de cierre:** existe `validation_xview3_med_fused.json` con `pipeline_path=full`, parametros identicos a los reportes CFAR/YOLO regenerados y anclas de trazabilidad completas.

## R14 — Fusion CFAR ∩ YOLO por IoU inoperante (0 fusiones en 1 997 barcos)

- **Severidad:** A · **Probabilidad:** A (medido) · **Estado:** **mitigado y medido 2026-08-29** (`Settings.fusion_mode='center'`, 20 px; commit e2c8de1). Re-validacion sobre las mismas 11 escenas (`reports/validation_xview3_adriatic_full_vessels_r14r15_*.json`): `fused_only` pasa de 0 a **763 detecciones, 238 TP, precision 0.312** (2× la de la union), Pd 0.119 a FAR 0.0011/km2; la union apenas cambia (Pd 0.356 → 0.361, FAR 0.0080 → 0.0083, F1 0.220 → 0.218). La fusion no mejora el agregado: aporta un **nivel de alta precision** (`source='fused'`) que dashboards/API deberian exponer como tier. Atribucion (mismas escenas): R14 solo → 489 fusionadas (163 TP, precision 0.333), union F1 0.220 → 0.225 y FAR 0.0080 → 0.0076; R15 solo → YOLO Pd 0.143 pero union F1 0.214 (mas FP) y solo 28 fusiones por IoU; juntas → 763 fusionadas. Decision 2026-08-29: `fused` se expone como **tier de alta precision** (`?tier=high` / `?source=fused` en GeoJSON y OGC, propiedad `tier`, selector y panel en el dashboard del mapa), nunca como filtro por defecto (Pd caeria de 0.36 a 0.12).
- **Descripcion:** `DetectionEngine._fuse_detections` empareja CFAR y YOLO por IoU ≥ `Settings.fusion_iou_threshold` (0.3). Sobre las 11 escenas xView3 del Adriatico, `source='fused'` = **0** de 4 460 detecciones. Causa geometrica, no de desacuerdo: los clusteres CFAR miden ~4×5 px y las cajas YOLO ~39×42 px; el 42 % de las cajas YOLO tiene un CFAR a ≤ 20 px, pero la IoU entre ambos es mediana 0.07 / maxima 0.30 (`reports/analysis_xview3_adriatic_full_vessels.md`).
- **Impacto:** El "ensemble CFAR + YOLO" de la decision congelada (CLAUDE.md §2) es en produccion una union simple. Anadir YOLO a CFAR aporta +14 TP por +642 FP (F1 0.240 → 0.220). El README y la ficha lo vendian como fusion; corregido 2026-08-28.
- **Mitigacion propuesta (requiere decision — cambia el comportamiento de produccion):**
  1. Fusion por distancia de centros ≤ N px (N desde `Settings`, p.ej. 20 px como el matching xView3) en lugar de IoU, conservando la caja YOLO y sumando la confianza ponderada. El 42 % de co-localizacion medido es el techo de fusiones esperables.
  2. Re-validar con la terna {antes, despues} sobre las mismas 11 escenas y publicar `fused_only` medido.
  3. Actualizar `fusion_iou_threshold` → `fusion_center_tolerance_px` en `input_params_hash`.
- **Trigger de cierre:** `fused_only` > 0 con Pd_union ≥ Pd_cfar y FAR_union ≤ FAR_cfar en la validacion xView3.

## R15 — El filtro Lee degrada YOLO (Pd 0.143 → 0.093, 11/11 escenas)

- **Severidad:** M · **Probabilidad:** A (medido) · **Estado:** **cerrado 2026-08-29** (`Settings.yolo_input='unfiltered'`, tile `yolo_input` uint8 desde el σ⁰ sin filtrar; commit e2c8de1). Re-validacion: YOLO en pipeline Pd 0.093 → **0.143**, FAR 0.0027 → 0.0040, F1 0.108 → 0.137 — identico al detector aislado (2 191 vs 2 178 predicciones), es decir, el pipeline ya no le cuesta recall a YOLO. Trigger de cierre cumplido (Pd_pipeline ≥ 0.95 × Pd_detector).
- **Descripcion:** `preprocess_full` aplica Lee 7×7 sobre sigma0 lineal a TODAS las teselas antes de ambos detectores. `vesseltracker-sar-yolov8` sobre el raster dB sin filtrar recupera 286/1 997 barcos (Pd 0.143); tras Lee, 186 (Pd 0.093), en las 11 escenas sin excepcion (`validation_xview3_adriatic_detector_yolo_vessels.json` vs `..._full_vessels_yolo.json`). La FAR tambien baja (0.0041 → 0.0027) pero el F1 cae de 0.137 a 0.108.
- **Impacto:** El preprocesado esta optimizado para el modelo de ruido multiplicativo de CFAR, no para una CNN entrenada con chips sin filtrar. Mientras la fusion no funcione (R14) el coste es pequeno porque la salida la domina CFAR; en cuanto la fusion aporte, YOLO debe ver la tesela sin Lee.
- **Mitigacion propuesta:** alimentar a YOLO con la tesela calibrada sin filtrar y a CFAR con la filtrada (ambas derivan de la misma lectura; coste de RAM +1 tesela). Medir con la terna {antes, despues}.
- **Trigger de cierre:** Pd_yolo en pipeline ≥ 0.95 × Pd_yolo detector-only sobre las mismas escenas.

## R16 — Grad-CAM del D4 no localizaba los barcos (capa P5 + objetivo global)

- **Severidad:** A · **Probabilidad:** A (medido) · **Estado:** **cerrado 2026-08-29** — D4 de produccion regenerado (`60982517…_interp_d8920869`, `gradcam_layer=model.model.15`, `sampling.strategy=stratified_confidence_quantiles`, 20/20 + 20/20); trigger de cierre cumplido
- **Descripcion:** `gradcam_yolov8` hookeaba `model.model.21` (P5, stride 32) y retropropagaba la media de |salida| global. Chequeo con ground truth xView3 (`scripts/run_interpretability_xview3.py`, escena 264ed833, 20 muestras estratificadas TP-alta/TP-baja/FP/FN): pointing-game **0/20**, ≤ 0.7 % de la masa del heatmap dentro de la caja del barco (peor que uniforme; chips de 320 px). El anexo D4 entregado (20 heatmaps) es, por tanto, decorativo: no explica las detecciones (I-AIA-2).
- **Impacto:** El entregable D4 afirmaba interpretabilidad que no existia. Detectado solo porque el muestreo con GT incluye FN/FP y una metrica de fidelidad; el muestreo top-confianza de produccion no podia verlo.
- **Mitigacion aplicada:** capa P3 (`model.model.15`, stride 8, la cabeza que asigna objetos de 4-15 px) + objetivo = puntuacion de clase del ancla mas cercana al centro de la deteccion (`target_xy`). Resultado (320 px): TP-alta 3/5 (57 % de la masa en la caja), FP 2/5, TP-baja 1/5, FN 1/5; el mapa CFAR 4/5 en TP-alta y 4/5 en FN. Queda calor residual en los bordes del chip (padding) — siguiente paso: mascara de borde o Grad-CAM++. Produccion (`run_interpretability_for_execution`) usa ya P3 + centro del thumbnail y muestreo estratificado por cuantiles de confianza y fuente; el manifest registra `gradcam_layer` / `gradcam_target` / `stratum`.
- **Pendiente:** regenerar el D4 de produccion tras el deploy (`POST /api/interpretability/run`) y actualizar `EVIDENCE.md` § D4 con el nuevo run_id; el anexo (§6) ya documenta antes/despues.
- **Trigger de cierre:** manifest de produccion con `gradcam_layer=model.model.15` y `sampling.strategy=stratified_confidence_quantiles`.

## R17 — Ningun perfil sat-* cabe en RAM: el pipeline carga toda la escena en memoria

- **Severidad:** A · **Probabilidad:** A (medido) · **Estado:** abierto (registrado 2026-08-29)
- **Descripcion:** Con `profile_memory_enforcement=abort` (defecto desde 0d48f59), la terna en OCI sobre la imagen `4b5dfec3` aborta los cuatro perfiles `sat-*` en el primer check por tesela: RSS 4964 MB (FP32) / 4835 MB (INT8) frente a presupuestos de 4 096 / 2 048 / 1 024 / 512 MB. El pico se alcanza **antes de inferir**: `preprocess_full` devuelve todas las teselas float32 de la escena (~2.5 GB) mas los tensores de PyTorch; cambiar de modelo solo mueve ~130 MB.
- **Impacto:** Hasta ahora esos runs "completaban" con una nota de exceso; ahora fallan de forma honesta. La evidencia de viabilidad OBDP bajo `sat-*` no existe con el pipeline actual, con independencia de la compresion del modelo. En `ground` la terna si es concluyente: INT8 7.4× mas rapido con las mismas detecciones.
- **Mitigacion propuesta:** procesar la escena por bandas de teselas (el harness `src/validation/harness.py` ya lo hace con `band_tile_rows`), liberando cada banda tras la deteccion; opcionalmente teselas uint8/float16 para YOLO. Objetivo: RSS pico < 2 GB en `sat-mid` con FP32 y < 1 GB con INT8. Re-ejecutar la terna de perfiles despues.
- **Trigger de cierre:** `trigger-all-profiles` completa `sat-mid` con `status=success` bajo enforcement `abort`.

## Plan de contingencia consolidado

1. **Backup diario** de `aidra` DB + `models/` + bundles D3 a S3-compatible UE.
2. **Mirror local** de datasets en almacenamiento cifrado.
3. **`docker-compose.yml` portable** validado en 2 hosts distintos (OCI ARM + Hetzner CAX11).
4. **Documentacion congelada** por release: `git tag` + bundle D3 asociado al tag.
5. **Decision de escalado** a despliegue operativo no se toma sin auditoria AI Act formal previa.

— Ultima revision: 2026-05-08 (auditoria externa contra prod, anadidos R9/R10/R11, R6 reabierto)
