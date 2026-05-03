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

## R6 — Reproducibilidad rota por cambios no versionados

- **Severidad:** A · **Probabilidad:** M · **Estado:** mitigado
- **Descripcion:** Si un Settings o un peso cambia sin reflejarse en `commit_sha` / `model_hash`, los runs antiguos no son reproducibles.
- **Impacto:** invalida criterio Q3 de trazabilidad. El bundle D3 deja de ser auditable.
- **Mitigacion:**
  1. `commit_sha` capturado en `execution_log` al inicio del run (I-TRACE-4, ver migration 004).
  2. `input_params_hash` (Settings + request) tambien capturado.
  3. `MODEL_CARD.md` incluye SHA256 del peso.
  4. CI bloquea PRs que muevan pesos sin actualizar la ficha.
- **Trigger:** verifier `--bundle` reporta hash mismatch sin justificacion.

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

## Plan de contingencia consolidado

1. **Backup diario** de `aidra` DB + `models/` + bundles D3 a S3-compatible UE.
2. **Mirror local** de datasets en almacenamiento cifrado.
3. **`docker-compose.yml` portable** validado en 2 hosts distintos (OCI ARM + Hetzner CAX11).
4. **Documentacion congelada** por release: `git tag` + bundle D3 asociado al tag.
5. **Decision de escalado** a despliegue operativo no se toma sin auditoria AI Act formal previa.

— Ultima revision: 2026-04-25
