# AIDRA MVP — Deteccion de barcos con IA en OCI Free Tier

Basado en los pliegos tecnicos reales (Appendix I.1 + Annex I, SATCEN/2026/OP/0003).

---

## Que pide AIDRA realmente (resumen de los pliegos)

### Caso de uso concreto
**Deteccion de barcos (vessel detection) en areas maritimas** usando imagenes satelitales procesadas con IA, simulando que el procesamiento ocurre a bordo del satelite.

### Lo que NO pide
- No pide entrenar un modelo nuevo
- No pide desplegar nada operativo
- No pide precisión específica del modelo de IA — dice textualmente que "no incluirá una evaluación detallada del rendimiento o precisión de modelos de IA individuales"

### Lo que SI pide
1. **Cadena end-to-end funcional**: imagen satelital entra → modelo IA detecta barcos → resultado sale
2. **Simulacion de restricciones espaciales**: ejecutar en entorno que simule hardware de satelite (CPU/RAM/energia limitados). Aceptan simulacion en tierra si TRL >= 6
3. **Compresion de modelos**: evaluar tecnicas de quantizacion, pruning y knowledge distillation para que el modelo quepa en hardware limitado
4. **Traceability**: documentar todo — que imagen entro, que modelo se uso, que parametros tenia, que resultado salio
5. **Metricas de rendimiento**: latencia, uso de memoria, uso de CPU, tamano del modelo
6. **Opcional (puntos extra)**: Tip & Cue — el satelite decide autonomamente que area fotografiar basandose en alertas previas

### Entregables del contrato real
| Ref | Que es | Cuando |
|---|---|---|
| D1 | Plan de demostracion + definicion del escenario | T0 + 2 meses |
| D2 | Informe de progreso intermedio | T0 + 4 meses |
| D3 | **Paquete de evidencia** (logs, imagenes, benchmarks de compresion) | T0 + 9 meses |
| D4 | **Informe final de analisis y recomendaciones** | T0 + 11 meses |
| D5 | Informe de cierre contractual | T0 + 12 meses |

### Modalidad del sensor
Libre: **optico, SAR o multi-sensor**. El licitador elige y justifica.

---

## Tu MVP en OCI Free Tier

### Concepto
Replicar el D3 (Demonstration Evidence Package) de AIDRA: un pipeline end-to-end de deteccion de barcos que corra con recursos limitados (OCI ARM simula el hardware espacial), con metricas de observabilidad y benchmarks de compresion de modelos.

### Arquitectura

**Stack 100% Python.** El ecosistema EO/satelital (rasterio, GDAL, PyTorch, pystac-client, ultralytics) es nativo Python. FastAPI para la API, APScheduler para el scheduler, todo en un solo contenedor de aplicacion.

```
┌──────────────────────────────────────────────────────────┐
│                 OCI ARM A1 (4 OCPU, 24 GB)               │
│                                                          │
│  ┌──────────────────────────────┐      ┌────────────┐    │
│  │ Python App (FastAPI)         │      │ Grafana    │    │
│  │  :8000                       │      │  :3000     │    │
│  │                              │      │            │    │
│  │  ┌─────────┐ ┌────────────┐  │      │ - Mapa     │    │
│  │  │ API     │ │ Pipeline   │  │      │   barcos   │    │
│  │  │ REST    │ │            │  │      │ - Metricas │    │
│  │  │         │ │ 1.Descarga │  │      │   pipeline │    │
│  │  │ /detect │ │ 2.Preproceso│ │      │ - Perfiles │    │
│  │  │ /trace  │ │ 3.Inferencia│ │      │   restricc.│    │
│  │  │ /bench  │ │ 4.Guarda   │  │      │ - Benchmarks│   │
│  │  │ /task   │ │ 5.Borra img│  │      │ - Traceab. │    │
│  │  └─────────┘ │ 6.Metricas │  │      └────────────┘    │
│  │              └────────────┘  │                         │
│  │  ┌───────────────┐           │      ┌────────────┐    │
│  │  │ APScheduler   │           │      │ PostgreSQL │    │
│  │  │ (cron + cue)  │           │──────│ + PostGIS  │    │
│  │  └───────────────┘           │      │  :5432     │    │
│  └──────────────────────────────┘      └────────────┘    │
│                                                          │
│  ┌─────────────┐    ┌─────────────┐                      │
│  │ Prometheus  │    │ Loki        │                      │
│  │  :9090      │    │  :3100      │                      │
│  └─────────────┘    └─────────────┘                      │
└──────────────────────────────────────────────────────────┘
```

### Modulo 1: Ingesta de imagenes (Python)

**Fuente de datos**: Sentinel-1 (SAR) via Copernicus OData API
- SAR es preferible a optico para deteccion maritima: funciona de noche y con nubes
- Los barcos aparecen como puntos brillantes en imagen SAR sobre fondo oscuro del mar
- Sentinel-1 GRD (Ground Range Detected) es el producto mas usado para deteccion de barcos

**Alternativa**: Sentinel-2 (optico) si quieres imagenes mas intuitivas visualmente, pero solo funciona de dia y sin nubes.

**Flujo**:
1. Consultar Copernicus STAC/OData: buscar imagenes Sentinel-1 GRD sobre un area maritima (ej: Estrecho de Gibraltar, Mar Mediterraneo)
2. Filtrar por fecha reciente y cobertura
3. Descargar producto (~500 MB - 1 GB)
4. Preprocesar con Python (calibracion, correccion geometrica)

### Modulo 2: Deteccion de barcos con IA (Python)

**Enfoque recomendado**: Deteccion por umbral CFAR + modelo ML para clasificacion

**Opcion A — Clasica (mas facil, funciona bien en CPU)**:
- CFAR (Constant False Alarm Rate): algoritmo estandar para detectar barcos en SAR
- No requiere modelo de IA — es procesamiento de señal
- Muy ligero en CPU
- Desventaja: mas falsos positivos

**Opcion B — IA con modelo preentrenado (mas cercano a AIDRA)**:
- Usar un modelo de deteccion de objetos preentrenado en imagenes satelitales
- Datasets publicos para vessel detection:
  - **xView** (Defense Innovation Unit): 1M+ objetos anotados en imagenes satelitales, incluye barcos
  - **DOTA**: 188K instancias anotadas en imagenes aereas
  - **Airbus Ship Detection** (Kaggle): 192K imagenes con mascaras de barcos
- Modelos preentrenados posibles:
  - YOLOv8/YOLOv11 fine-tuned en datos maritimos (ligero, rapido en CPU)
  - EfficientDet
  - Faster R-CNN (mas pesado)

**Opcion recomendada para OCI Free Tier**: YOLOv8-nano o YOLOv8-small
- Tamano: 6-22 MB
- Inferencia CPU: 50-200ms por imagen recortada
- Facil de exportar a ONNX para benchmarks de compresion

### Modulo 3: Compresion de modelos (lo que diferencia a AIDRA)

AIDRA pide explicitamente evaluar estas tecnicas:

| Tecnica | Que hace | Herramienta |
|---|---|---|
| **Quantizacion** | Reduce precision de pesos (FP32 → INT8) | PyTorch quantization / ONNX Runtime |
| **Pruning** | Elimina conexiones/neuronas poco importantes | torch.nn.utils.prune |
| **Knowledge Distillation** | Entrena modelo pequeno imitando a uno grande | Implementacion custom |

**Metricas a comparar por cada tecnica**:
- Tamano del modelo (MB)
- Tiempo de inferencia (ms)
- Uso de RAM pico (MB)
- Uso de CPU (%)
- Precision (mAP si hay ground truth, o comparacion cualitativa)

**Esto es el nucleo del valor del proyecto**: no solo detectar barcos, sino demostrar que puedes comprimir el modelo y que siga funcionando en hardware limitado. OCI Free Tier con sus 4 OCPU ARM es una simulacion razonable de hardware espacial restringido.

### Modulo 4: Traceability — Cadena de proveniencia (requerido por AIDRA)

AIDRA menciona traceability 6 veces en los pliegos. No basta con logs genericos — cada ejecucion del pipeline debe generar un **registro de proveniencia** inmutable que responda:

- ¿Que imagen entro? (ID del producto Copernicus, hash SHA256 del archivo)
- ¿Que modelo se uso? (nombre, version, hash SHA256 de los pesos)
- ¿Con que parametros? (umbral de confianza, perfil de restriccion)
- ¿Que resultado salio? (numero de detecciones, bounding boxes, confianza media)
- ¿Es reproducible? (con los mismos inputs + modelo + parametros, ¿sale lo mismo?)

**Implementacion**:

Tabla `execution_log` en PostGIS:

```sql
CREATE TABLE execution_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Input
    image_id        TEXT NOT NULL,          -- ID producto Copernicus
    image_hash      TEXT NOT NULL,          -- SHA256 de la imagen descargada
    image_bbox      GEOMETRY(POLYGON,4326), -- area geografica de la imagen
    -- Modelo
    model_name      TEXT NOT NULL,          -- ej: "yolov8n-sar-xview3"
    model_version   TEXT NOT NULL,          -- ej: "v1.2-quantized-int8"
    model_hash      TEXT NOT NULL,          -- SHA256 de los pesos
    model_size_mb   REAL NOT NULL,          -- tamano del modelo en MB
    -- Parametros
    confidence_thr  REAL NOT NULL,          -- umbral de confianza
    constraint_profile TEXT NOT NULL,       -- ej: "ground", "sat-high", "sat-low"
    cpu_limit       REAL,                   -- CPUs asignadas
    memory_limit_mb INTEGER,               -- RAM asignada en MB
    -- Resultados
    num_detections  INTEGER NOT NULL,
    avg_confidence  REAL,
    inference_ms    REAL NOT NULL,          -- tiempo de inferencia
    peak_ram_mb     REAL NOT NULL,          -- RAM pico durante inferencia
    cpu_usage_pct   REAL NOT NULL,          -- uso medio de CPU
    -- Trazabilidad
    output_hash     TEXT NOT NULL,          -- SHA256 del archivo de resultados
    status          TEXT NOT NULL           -- "success", "error", "timeout"
);
```

Cada fila es un registro inmutable. Se puede consultar via API y visualizar en Grafana.

### Modulo 5: Perfiles de restriccion — Simulacion de hardware espacial

AIDRA pide evaluar rendimiento "under computational, power and memory constraints of space-grade hardware". No basta con correr el pipeline una vez — hay que ejecutar el **mismo escenario** bajo diferentes perfiles de restriccion y comparar.

**Perfiles definidos**:

| Perfil | CPU | RAM | Simula | Docker flags |
|---|---|---|---|---|
| `ground` | 4 OCPU | 24 GB | Estacion terrena (sin limites) | (sin limites) |
| `sat-high` | 2 OCPU | 4 GB | Satelite gama alta (ej: procesador Xilinx Zynq) | `--cpus=2 --memory=4g` |
| `sat-mid` | 1 OCPU | 2 GB | Satelite gama media | `--cpus=1 --memory=2g` |
| `sat-low` | 0.5 OCPU | 1 GB | Satelite gama baja / CubeSat | `--cpus=0.5 --memory=1g` |
| `sat-extreme` | 0.25 OCPU | 512 MB | Limite inferior: ¿donde se rompe? | `--cpus=0.25 --memory=512m` |

**Flujo**:
1. FastAPI lanza el pipeline como subprocess dentro de un contenedor Docker con los flags del perfil (via Docker SDK para Python)
2. El pipeline se ejecuta, registra metricas y escribe en `execution_log` con el perfil usado
3. Se repite para cada perfil con la misma imagen y modelo
4. Grafana muestra la comparativa

**Metricas a comparar por perfil**:
- ¿Completa la inferencia o hace timeout/OOM?
- Tiempo de inferencia (ms)
- RAM pico (MB)
- CPU medio (%)
- Numero de detecciones (¿pierde barcos al limitar recursos?)
- Confianza media de las detecciones

**Esto es el nucleo de la evaluacion AIDRA** y lo que ningun paper incluye: no solo "funciona en hardware limitado" sino "como degrada progresivamente y donde esta el limite".

### Modulo 6: Tip & Cue — Tasking inteligente (opcional, puntos extra)

Tip & Cue = si el pipeline detecta algo interesante, reprograma automaticamente una nueva observacion de esa zona.

**Simulacion en el MVP**:

```
Pasada 1 (programada):
  Descarga imagen Sentinel-1 de zona amplia (ej: Estrecho de Gibraltar)
  → Pipeline detecta 3 barcos en subzona X

Tip (alerta):
  FastAPI recibe las detecciones
  → Evalua: ¿hay detecciones en zonas de interes? ¿confianza > umbral?
  → Si: genera un "tasking request" (entrada en BD)

Cue (reprogramacion):
  APScheduler ajusta la proxima ejecucion:
  → En vez de zona amplia, descarga solo la subzona X
  → Con mayor prioridad / frecuencia
  → Registra en execution_log que fue un "cue" (no programado)

Pasada 2 (triggered por Cue):
  Descarga imagen de subzona X (mas reciente)
  → Confirma o descarta la deteccion
  → Registra el resultado vinculado al tip original
```

**Implementacion**:

Tabla `tasking_queue` en PostGIS:

```sql
CREATE TABLE tasking_queue (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trigger_type    TEXT NOT NULL,         -- "scheduled" | "cue"
    triggered_by    UUID,                  -- execution_log.id que genero el cue
    target_bbox     GEOMETRY(POLYGON,4326),-- area a observar
    priority        INTEGER DEFAULT 0,     -- 0=normal, 1=alta, 2=urgente
    status          TEXT DEFAULT 'pending',-- "pending", "executing", "completed"
    executed_at     TIMESTAMPTZ,
    execution_id    UUID                   -- execution_log.id del resultado
);
```

**Dashboard Grafana**: Mapa con dos capas — detecciones originales (tips) y detecciones de seguimiento (cues), vinculadas por lineas.

### Modulo 7: Observabilidad (Grafana + Prometheus + Loki)

**Dashboard 1 — Mapa de detecciones** (GeoMap):
- Puntos con barcos detectados, color por confianza
- Capa de tips (naranja) y cues (rojo) si Tip & Cue activo
- Filtro temporal y por perfil de restriccion

**Dashboard 2 — Metricas del pipeline**:
- Imagenes procesadas / hora
- Tiempo de inferencia por imagen
- Uso de CPU/RAM durante inferencia
- Detecciones por imagen
- Errores del pipeline

**Dashboard 3 — Benchmarks de compresion**:
- Comparativa: modelo original vs quantizado vs pruned
- Grafico de barras: tamano, latencia, RAM
- Trade-off curve: precision vs latencia

**Dashboard 4 — Perfiles de restriccion** (nuevo):
- Comparativa de metricas por perfil (ground vs sat-high vs sat-low...)
- Heatmap: perfil x metrica
- Grafico: "donde se rompe" — punto de inflexion de degradacion

**Dashboard 5 — Traceability**:
- Tabla de execution_log con filtros
- Detalle de cada ejecucion: inputs, modelo, parametros, resultados, hashes
- Verificacion de reproducibilidad

### Modulo 8: API REST (FastAPI)

```
GET  /api/health                         → estado del sistema
GET  /api/detections                     → lista de detecciones (filtros geo/temporal/perfil)
GET  /api/detections/{id}                → detalle + proveniencia completa
POST /api/pipeline/trigger               → lanzar pipeline (params: imagen, modelo, perfil)
POST /api/pipeline/trigger-all-profiles  → lanzar misma imagen con todos los perfiles
GET  /api/pipeline/status                → estado del pipeline actual
GET  /api/benchmarks                     → resultados de compresion de modelos
GET  /api/benchmarks/compare             → comparativa entre variantes y perfiles
GET  /api/traceability/{execution_id}    → cadena de proveniencia completa
GET  /api/tasking/queue                  → cola de Tip & Cue
POST /api/tasking/cue                    → crear cue manual
GET  /api/metrics                        → metricas Prometheus (via prometheus_client)
GET  /docs                               → Swagger UI (auto-generado por FastAPI)
```

### Stack definitivo

**Todo Python.** Un solo lenguaje, un solo contenedor de aplicacion, alineado con el ecosistema EO/satelital.

| Componente | Tecnologia | RAM estimada |
|---|---|---|
| API REST | FastAPI + uvicorn | ~100 MB |
| Scheduler | APScheduler (in-process) | ~50 MB |
| Pipeline IA | PyTorch + ultralytics (YOLO) | ~2-4 GB (picos) |
| Deteccion SAR clasica | scipy (CFAR) | ~500 MB |
| Ingesta satelital | pystac-client + requests + rasterio | ~200 MB |
| Procesamiento geoespacial | rasterio + GDAL + shapely + numpy | (incluido arriba) |
| Metricas app | prometheus_client | ~10 MB |
| Logs app | python-logging-loki | ~10 MB |
| Base de datos | PostgreSQL + PostGIS (contenedor separado) | ~1-2 GB |
| Dashboards | Grafana (contenedor separado) | ~500 MB |
| Metricas sistema | Prometheus (contenedor separado) | ~500 MB |
| Logs sistema | Loki + Promtail (contenedor separado) | ~300 MB |
| Perfiles restriccion | Docker SDK for Python (lanza contenedores con limites) | ~50 MB |
| **TOTAL** | | **~5-8 GB** |

Sobran ~16 GB para los picos de procesamiento de imagenes.

---

## Datasets publicos para vessel detection (verificados 23/04/2026)

| Dataset | Contenido | Sensor | Acceso | Estado |
|---|---|---|---|---|
| **xView3-SAR** | 991 imagenes Sentinel-1, 243K objetos maritimos, 43M km2 | SAR (Sentinel-1) | https://iuu.xview.us/ | **OK** — el mas completo y relevante |
| **HRSID** | 5.604 imagenes SAR, 16.951 barcos, 3 resoluciones | SAR | GitHub + Google Drive | **OK** — GPL-3.0 |
| **SSDD** | ~1.160 imagenes SAR con barcos | SAR | GitHub + Google Drive | **OK** — Apache-2.0 |
| **OpenSARShip** | 11.346 chips de barcos en Sentinel-1 | SAR (Sentinel-1) | https://opensar.sjtu.edu.cn/ | **OK** — descarga directa |
| **SARFish** | Dataset SAR en HuggingFace | SAR | https://huggingface.co/datasets/ConnorLuckettDSTG/SARFish | **OK** — HuggingFace |
| Airbus Ship Detection | 192K imagenes con mascaras | Optico | Kaggle | OK — requiere registro Kaggle |
| xView (original) | 1M+ objetos aereos, 60 clases, 0.3m | Optico | https://xviewdataset.org/ | OK — web de 2018, puede estar legacy |

### Recomendacion de dataset

**xView3-SAR es el mejor para este proyecto** porque:
- Usa exactamente Sentinel-1 (lo mismo que descargaras de Copernicus)
- 243.018 objetos anotados (barcos, infraestructura, pesca ilegal)
- Incluye datos de batimetria y estado del viento
- Paper publicado en NeurIPS 2022
- Codigo disponible en GitHub (DIUx-xView)
- Gratuito y abierto

### Modelo preentrenado recomendado

Hay multiples papers recientes (2024-2025) de YOLOv8 adaptado a SAR vessel detection:

| Paper | Que aporta | Enlace | Relevancia para tu MVP |
|---|---|---|---|
| **ADV-YOLO** | YOLOv8 mejorado con space-to-depth blocks + dilation-wise residual para deteccion multi-escala en SAR | https://link.springer.com/article/10.1007/s11227-024-06527-6 | Alta — mejora deteccion de barcos pequenos |
| **SAR-LtYOLOv8** | Version lightweight de YOLOv8 para objetos pequenos en SAR | https://www.techscience.com/csse/v48n6/58696/html | Alta — optimizado para recursos limitados |
| **ECF-YOLO** | Modulo C2f-EMSCP que reduce parametros y coste computacional manteniendo precision | https://www.aimspress.com/article/doi/10.3934/era.2025150 | Alta — menos parametros = mejor para edge |
| **YOLOv8 para FPGA** | Solo 2-3% peor que GPU pero **50-2500x mas eficiente** computacionalmente. Probado en xView3-SAR | https://arxiv.org/html/2507.04842 | **Muy alta** — es exactamente lo que AIDRA quiere evaluar: rendimiento vs restricciones hardware |
| **YOLOShipTracker** | YOLOv8 lightweight + tracking de barcos en secuencias SAR | https://www.sciencedirect.com/science/article/pii/S1569843224004916 | Media — tracking es un plus |
| **YOLOv8-ResAttNet** | Backbone con atencion residual, probado en HRSID (5604 imgs, 16951 barcos) | https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.70085 | Media — referencia de benchmarks |

### Estrategia recomendada para el MVP

1. **Empezar con YOLOv8-nano de ultralytics** (6 MB, rapido en CPU). Base funcional.
2. **Fine-tune en local** (con GPU si tienes) usando xView3-SAR o HRSID.
3. **Desplegar solo inferencia** en OCI ARM.
4. **En Fase 3 (compresion)**: exportar a ONNX, aplicar quantizacion INT8, comparar con los benchmarks de los papers de arriba (especialmente el de FPGA como referencia de "que es posible").
5. Si quieres ir mas alla: implementar las ideas de SAR-LtYOLOv8 o ECF-YOLO para reducir parametros antes de comprimir.

---

## Fases de desarrollo

### Fase 1 — Pipeline basico + traceability (35-45h)
- [ ] Setup OCI ARM A1 con Docker Compose (Python/FastAPI + PostGIS + Grafana + Prometheus + Loki)
- [ ] Ingesta de imagenes Sentinel-1 GRD de zona maritima via Copernicus OData API
- [ ] Deteccion basica con CFAR (sin IA, validar que el pipeline funciona)
- [ ] Tabla `execution_log` en PostGIS con cadena de proveniencia completa (hashes, modelo, params)
- [ ] Guardar detecciones geolocalizadas en PostGIS
- [ ] API FastAPI basica (health, detections, traceability, /docs Swagger)
- [ ] Grafana: Dashboard mapa GeoMap + Dashboard traceability

### Fase 2 — Modelo IA + perfiles de restriccion (35-45h)
- [ ] Integrar YOLOv8-nano preentrenado (o fine-tuned con xView3-SAR/HRSID)
- [ ] Implementar perfiles de restriccion via Docker SDK for Python (ground → sat-extreme)
- [ ] Endpoint `POST /api/pipeline/trigger-all-profiles` que ejecuta misma imagen con todos los perfiles
- [ ] Registrar metricas por perfil en execution_log
- [ ] prometheus_client para metricas + python-logging-loki para logs
- [ ] Grafana: Dashboard metricas pipeline + Dashboard perfiles de restriccion (comparativa)
- [ ] Scheduler automatico con APScheduler (in-process)

### Fase 3 — Compresion de modelos (25-35h)
- [ ] Exportar modelo a ONNX
- [ ] Quantizacion INT8 (PyTorch / ONNX Runtime) y medir rendimiento
- [ ] Pruning (torch.nn.utils.prune) y medir rendimiento
- [ ] Ejecutar cada variante del modelo con todos los perfiles de restriccion
- [ ] Grafana: Dashboard benchmarks de compresion (tamano vs latencia vs precision por perfil)
- [ ] Documentar trade-offs (equivalente al D4 de AIDRA)

### Fase 4 — Tip & Cue + pulido (20-30h)
- [ ] Tabla `tasking_queue` en PostGIS
- [ ] Logica de Tip: si deteccion con confianza > umbral en zona de interes → generar cue
- [ ] Logica de Cue: APScheduler prioriza descargas de zonas con cues pendientes
- [ ] Vincular tips con cues en execution_log (triggered_by)
- [ ] Grafana: Mapa con capa de tips/cues vinculados
- [ ] API endpoints de tasking (queue, cue manual)
- [ ] README completo, Docker Compose un solo comando, documentacion de arquitectura

**Total estimado: ~115-155 horas**

---

## Plan de proyecto formal — WBS, dependencias, hitos absolutos (Q2)

Esta seccion cierra la palanca **L13** del autoaudit y formaliza el
plan que el evaluador SatCen espera bajo el criterio Q2 "Plan de
proyecto" (10 pts). Las horas son orientativas para un solo
desarrollador sobre proyecto personal; un consorcio real las dividiria
entre roles (PM / SAR engineer / ML / GEOINT integrator).

### Hitos contractuales (deadlines absolutos del pliego)

Tomando como **T0** la fecha de firma del contrato (estimada para el
proceso real ~ 2026-09-15, post-evaluacion):

| Hito | Fecha relativa | Fecha absoluta | Entregable |
|---|---|---|---|
| **M0**  | T0 + 0       | 2026-09-15 | Kick-off + plan congelado |
| **M1**  | T0 + 2 meses | 2026-11-15 | **D1** — Plan de demostracion + escenario |
| **M2**  | T0 + 4 meses | 2027-01-15 | **D2** — Informe progreso intermedio |
| **M3**  | T0 + 9 meses | 2027-06-15 | **D3** — Evidence Package (logs + benchmarks + samples) |
| **M4**  | T0 + 11 meses| 2027-08-15 | **D4** — Informe final + recomendaciones |
| **M5**  | T0 + 12 meses| 2027-09-15 | **D5** — Cierre contractual |

> **Nota POC personal**: el repositorio AIDRA arranca en 2026-04 y
> alcanza paridad con el contenido tecnico de D3+D4 antes de la
> deadline del pliego (2026-05-04) — ver `RISK_REGISTER.md` y la
> autoevaluacion de palancas.

### Estructura de descomposicion del trabajo (WBS)

```
AIDRA POC
├── WP1 — Project Management
│   ├── WP1.1 Kick-off + alineamiento stakeholders
│   ├── WP1.2 Risk register vivo (RISK_REGISTER.md)
│   ├── WP1.3 Sprint cadence + commits firmados
│   └── WP1.4 Reporting hacia D2 / D5
│
├── WP2 — Data Layer (Q3 metodologia)
│   ├── WP2.1 Ingesta Sentinel-1 GRD (Copernicus Data Space)
│   ├── WP2.2 Preprocesado SAR (calib sigma0 + Lee + tiling + footprint)
│   ├── WP2.3 Esquema PostGIS + migrations 001..005
│   └── WP2.4 Datasets de validacion (xView3-SAR / HRSID / OpenSARShip)
│
├── WP3 — Detection (Q3 metodologia)
│   ├── WP3.1 CFAR (Rayleigh CA-CFAR + DBSCAN clustering)
│   ├── WP3.2 YOLOv8 SAR-fine-tuned
│   ├── WP3.3 Fusion CFAR x YOLO
│   └── WP3.4 Edge swath filter (I-SAR-2) + footprint clip (I-SAR-3)
│
├── WP4 — Compression & Profiles (Q3 metodologia + simulacion)
│   ├── WP4.1 Perfiles de restriccion (ground / sat-high / sat-mid / sat-low / sat-extreme)
│   ├── WP4.2 ResourceCollector (RAM/CPU/p95/energia)
│   ├── WP4.3 Quantizacion INT8 + Pruning + Knowledge Distillation
│   └── WP4.4 Triplet runner (baseline / variante / perfil)
│
├── WP5 — Traceability & D3 (Q3 trazabilidad — nucleo)
│   ├── WP5.1 SHA256 hasher (image / model / output / input_params)
│   ├── WP5.2 ExecutionRecorder (pending → running → success/error)
│   ├── WP5.3 EvidenceBundler (tar.gz + MANIFEST + signature)
│   ├── WP5.4 ReproducibilityVerifier (re-run + IoU equivalence)
│   └── WP5.5 Tests gate:reproducibility (e2e synthetic, palanca L15)
│
├── WP6 — GEOINT Integration (Q3 integracion GEOINT)
│   ├── WP6.1 GeoJSON RFC 7946 + STAC 1.0.0
│   ├── WP6.2 OGC API Features Part 1 (palanca L2)
│   ├── WP6.3 STAC Item Search (POST /search)
│   └── WP6.4 Symbology + Grafana GeoMap dashboards
│
├── WP7 — AI Act Compliance (Q3 documentacion IA)
│   ├── WP7.1 MODEL_CARD.md por modelo (gate I-AIA-1)
│   ├── WP7.2 AI_ACT_DECLARATION.md sustantivo
│   ├── WP7.3 Interpretabilidad (Grad-CAM YOLO + heatmap CFAR)
│   └── WP7.4 D4 anexo + sample gallery
│
├── WP8 — Validation & Metrics (D2 + D4)
│   ├── WP8.1 Validation harness (scripts/run_validation.py, palanca L3)
│   ├── WP8.2 Synthetic baseline (palanca L14)
│   ├── WP8.3 Real labelled manifest (xView3-SAR Med / AIS overlay) [POST-MVP]
│   └── WP8.4 mAP / Pd / FAR formal por modelo
│
├── WP9 — Tip & Cue (puntos extra del pliego)
│   ├── WP9.1 Tasking queue + cooldown
│   ├── WP9.2 Tip evaluator (deteccion → cue)
│   ├── WP9.3 Re-tasking simulator (orbital window)
│   └── WP9.4 Replay endpoint
│
└── WP10 — Observability & Operations
    ├── WP10.1 Prometheus metrics + OpenMetrics exemplars (run_id, palanca L11)
    ├── WP10.2 Loki structured logs (run_id end-to-end)
    ├── WP10.3 Grafana dashboards (mapa + traceability + benchmarks)
    └── WP10.4 docker-compose.yml portable + setup-oci.sh
```

### Dependencias criticas (DAG)

```
WP2 (data)        →  WP3 (detection)   →  WP4 (compression)
                   ↘  WP6 (GEOINT export)
WP3 + WP5 (trace) →  WP8 (validation)  →  WP7 (AI Act compliance)
WP3               →  WP9 (tip&cue)
WP10 (observability) corre transversal a todas las demas.
WP1 (PM) corre transversal a todas las demas.
```

- **WP2 bloquea WP3 y WP6** (sin esquema + tiles preprocesados, no hay
  deteccion ni export geojson).
- **WP5 bloquea WP8** (sin SHA256 / execution_log no hay manifest
  reproducible para evaluacion).
- **WP7 bloquea cualquier inferencia** (gate I-AIA-1 en
  `ModelManager._require_model_card`).
- **WP4 puede correr en paralelo con WP6** una vez WP3 estable.
- **WP9 (tip&cue) es opcional** — bonus del pliego; no en ruta critica.

### Gantt simplificado (12 meses contractuales)

```
Mes:        1     2     3     4     5     6     7     8     9    10    11    12
            │     │     │     │     │     │     │     │     │     │     │     │
WP1 PM      ████████████████████████████████████████████████████████████████████
WP2 data    ██████████░░░
WP3 detect       ██████████████░░░
WP4 compr                       ██████████████░░░
WP5 trace        ██████████████████████░░░
WP6 geoint                            ████████████░░░
WP7 ai-act              █████░░░               ██████████░░░
WP8 valid                              ████████████░░░██████░░░
WP9 tipcue                                          ████████░░░
WP10 obs    ███████████████████████████████████░░░
            │     │     │     │     │     │     │     │     │     │     │     │
            M0    │     │   M1=D1  │     │     │     │     │     │  M3=D3  │
                  │     │          │     │     │     │     │     │         │
                  │     │       M2=D2 (mes 4)  │     │     │     │      M4=D4 (mes 11)
                  │     │                                                      │
                                                                         M5=D5 (mes 12)
```

Leyenda: `█` trabajo activo, `░` slack / consolidacion.

### Paquetes de carga vs entregables

| WP | Carga (h personal) | Hito principal |
|---|---:|---|
| WP1 PM            |  20 | Continuo |
| WP2 data          |  30 | M1 |
| WP3 detection     |  35 | M2 |
| WP4 compression   |  25 | M2-M3 |
| WP5 traceability  |  25 | M3 (nucleo D3) |
| WP6 GEOINT        |  15 | M3 |
| WP7 AI Act        |  15 | M3-M4 |
| WP8 validation    |  20 | M4 (D2 dataset real + nucleo D4) |
| WP9 tip & cue     |  20 | M4 (puntos extra) |
| WP10 observability|  15 | Continuo (M3 dashboards congelados) |
| **TOTAL**         | **~220** | — |

> Carga estimada > rango POC (115-155 h): la diferencia es la fase
> contractual real (D1 plan formal, D2 dataset etiquetado, validacion
> AIS, redaccion D4). El POC personal cierra ~140 h y deja la cola
> documentada como riesgos POST-MVP en `RISK_REGISTER.md`.

### Gestion de cambios

- Cualquier desviacion > 10 % en horas o > 5 dias en un hito se
  registra en `RISK_REGISTER.md` con trigger + mitigacion.
- Las decisiones tecnicas que afecten a un invariante (CLAUDE.md §5)
  requieren entrada explicita y justificacion (ej. R8 Terrain
  Correction excluido).
- Las palancas (L1..L15) del autoaudit son la evidencia auditable de
  que el plan se ejecuta y se mide contra la rubrica SatCen.

---

## Relacion con la licitacion original

| Requisito AIDRA | Tu MVP | Modulo |
|---|---|---|
| Cadena end-to-end vessel detection | Pipeline Copernicus → YOLO → PostGIS → Grafana | M1+M2 |
| Traceability y verificacion de outputs | Tabla `execution_log` con hashes SHA256 de inputs, modelo y outputs. Registro inmutable consultable via API | M4 |
| Simulacion espacio-representativa | 5 perfiles de restriccion Docker (ground → sat-extreme), comparativa en Grafana | M5 |
| Compresion de modelos (quantizacion, pruning) | ONNX + PyTorch quantization + pruning, cada variante con todos los perfiles | M3+M5 |
| Metricas de rendimiento | Prometheus: latencia, CPU, RAM, tamano modelo, detecciones — por perfil y variante | M7 |
| Tip & Cue (opcional, puntos extra) | Bucle: deteccion → tasking_queue → descarga priorizada subzona → confirmacion | M6 |
| D3 Evidence Package | Dashboards Grafana + execution_log exportable + hashes de artefactos | M4+M7 |
| D4 Analysis Report | Documentacion trade-offs: compresion x perfil x precision | M3 |
| Interpretabilidad de outputs IA | Confianza, bounding boxes, comparacion ground vs restricted por ejecucion | M4+M5 |

---

## Lo que NO puedes replicar (y esta bien)

1. **Ejecucion real en orbita** — necesitas un satelite (~imposible)
2. **Hardware flight-proven** — usas ARM generico, no procesadores resistentes a radiacion
3. **Clasificacion bajo AI Act** — es un requisito legal del contrato, no aplica a proyecto personal
4. **Consultas con EDA/Estados Miembros** — es la parte institucional del contrato
5. **Seguridad y marcado de clasificacion** — los datos de Copernicus son publicos, no clasificados

---

## Clarificaciones del Q&A oficial (verificadas 23/04/2026)

Preguntas y respuestas publicadas en el portal Funding & Tenders:

1. **No se requiere entregar codigo fuente** (Q89225): "No standalone delivery of software, including source code or executables, is required." Esto confirma que AIDRA es evaluacion, no desarrollo. Tu MVP es 100% tuyo.

2. **"Raw data" = imagenes Level 1 o equivalente justificado** (Q89818): "The reference to 'raw data' may be understood as including Level 1 imagery or equivalent data provided in a standard format." Para no desalinear la evaluacion end-to-end, lo mas directo es usar Sentinel-1 GRD y, en optico, Sentinel-2 L1C (L2A solo si queda tecnicamente justificado).

3. **Datos de terceros con licencia abierta son validos** (Q89885): Los datos de entrada pueden ser pre-existing rights del contratista. Copernicus tiene licencia abierta de la UE — encaja.

4. **Deadline extendido 1 semana** (Q89150): Nueva fecha: 04/05/2026.

---

## Verificacion final de viabilidad (23/04/2026)

| Componente | Verificado | Resultado |
|---|---|---|
| Copernicus STAC API (catalogo) | SI | Funciona sin auth |
| Copernicus OData API (catalogo) | SI | Devuelve Sentinel-1/2 reales |
| Descarga Copernicus | SI | Requiere registro gratuito + token |
| Cuota gratuita | SI | 12 TB/mes |
| xView3-SAR dataset | SI | Web activa, descarga gratuita |
| HRSID dataset | SI | GitHub activo, Google Drive OK |
| SSDD dataset | SI | GitHub activo, Google Drive OK |
| OpenSARShip dataset | SI | Web activa, descarga directa |
| SARFish (HuggingFace) | SI | Activo |
| Ultralytics YOLOv8 | SI | GitHub activo, pip install |
| PostGIS Docker | SI | 270M+ pulls |
| Grafana GeoMap | SI | Built-in desde Grafana 8 |
| OCI ARM A1 Free Tier | SI | Disponible y operativo |
