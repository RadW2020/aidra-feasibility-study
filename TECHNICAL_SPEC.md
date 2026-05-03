# AIDRA — Especificaciones Tecnicas para Desarrollo por Enjambre de Agentes LLM

**Version**: 2.0
**Fecha**: 2026-04-23
**Proyecto**: AIDRA MVP — Deteccion de barcos con IA en OCI Free Tier
**Basado en**: Licitacion SATCEN/2026/OP/0003 (Appendix I.1 + Annex I Version 2)

---

## INDICE

### Parte I — Arquitectura y Diseno
1. [Vision General del Proyecto](#1-vision-general-del-proyecto)
2. [Arquitectura del Sistema](#2-arquitectura-del-sistema)
3. [Estructura de Directorios](#3-estructura-de-directorios)

### Parte II — Modulos (Especificaciones de Implementacion)
4. [Modulo 0: Infraestructura OCI + Docker](#4-modulo-0-infraestructura-oci--docker)
5. [Modulo 1: Ingesta de Imagenes Satelitales](#5-modulo-1-ingesta-de-imagenes-satelitales)
6. [Modulo 2: Deteccion de Barcos con IA](#6-modulo-2-deteccion-de-barcos-con-ia)
7. [Modulo 3: Compresion de Modelos](#7-modulo-3-compresion-de-modelos)
8. [Modulo 4: Traceability — Cadena de Proveniencia](#8-modulo-4-traceability--cadena-de-proveniencia)
9. [Modulo 5: Perfiles de Restriccion (Simulacion Espacial)](#9-modulo-5-perfiles-de-restriccion-simulacion-espacial)
10. [Modulo 6: Tip & Cue — Tasking Inteligente](#10-modulo-6-tip--cue--tasking-inteligente)
11. [Modulo 7: Observabilidad (Grafana + Prometheus + Loki)](#11-modulo-7-observabilidad-grafana--prometheus--loki)
12. [Modulo 8: API REST (FastAPI)](#12-modulo-8-api-rest-fastapi)

### Parte III — Base de Datos y Configuracion
13. [Base de Datos: Esquema SQL Completo](#13-base-de-datos-esquema-sql-completo)
14. [Configuracion y Variables de Entorno](#14-configuracion-y-variables-de-entorno)
15. [Pipeline de CI/CD y Testing](#15-pipeline-de-cicd-y-testing)
16. [Dependencias y Versiones](#16-dependencias-y-versiones)

### Parte IV — Planificacion y Ejecucion
17. [Plan de Ejecucion por Fases](#17-plan-de-ejecucion-por-fases)
18. [Criterios de Aceptacion por Modulo](#18-criterios-de-aceptacion-por-modulo)
19. [Glosario y Acronimos](#19-glosario-y-acronimos)
20. [Asignacion de Agentes](#20-asignacion-de-agentes)

### Parte V — Modulos de Valor Orbital (Diferenciadores)
21. [Modulo 9: Perfil Energetico](#21-modulo-9-perfil-energetico)
22. [Modulo 10: Analisis de Downlink](#22-modulo-10-analisis-de-downlink)
23. [Modulo 11: Latencia Orbital](#23-modulo-11-latencia-orbital)
24. [Modulo 12: Resiliencia y Autonomia](#24-modulo-12-resiliencia-y-autonomia)

### Parte VI — Detalle Tecnico Adicional
25. [Detalle de Endpoints API (Handlers)](#205-detalle-de-endpoints-api-handlers)
26. [Loki Logger Estructurado](#206-loki-logger-estructurado)
27. [Orquestador del Pipeline (Engine)](#21-orquestador-del-pipeline-engine)
28. [APScheduler — Jobs Programados](#22-apscheduler--jobs-programados)
29. [Manejo de Errores y Resiliencia](#23-manejo-de-errores-y-resiliencia)
30. [Seguridad y Buenas Practicas](#24-seguridad-y-buenas-practicas)
31. [Datos de Referencia para Validacion](#25-datos-de-referencia-para-validacion)
32. [Reglas del Enjambre](#26-reglas-del-enjambre)
33. [Checklist Pre-Despliegue](#27-checklist-pre-despliegue)
34. [Archivos de Configuracion del Proyecto](#28-archivos-de-configuracion-del-proyecto)
35. [Diagrama de Flujo de Datos Completo](#29-diagrama-de-flujo-de-datos-completo)

---

## 1. Vision General del Proyecto

### 1.1 Que es AIDRA

AIDRA (Artificial Intelligence In-orbit Data pRocessing Assessment) es un proof-of-concept independiente inspirado en el pliego SatCen/EDA para evaluar si el procesamiento de datos con IA a bordo de satelites (On-Board Data Processing / OBDP) es viable para flujos de trabajo GEOINT.

### 1.2 Que construimos

Un **MVP funcional** que replica el D3 (Demonstration Evidence Package) de AIDRA: un pipeline end-to-end de deteccion de barcos en imagenes SAR que ejecuta en OCI Free Tier ARM, con:

- Ingesta automatizada de imagenes Sentinel-1 desde Copernicus
- Deteccion de barcos con YOLOv8 (modelo preentrenado + fine-tuned)
- Benchmarks de compresion de modelos (quantizacion, pruning, knowledge distillation)
- Simulacion de restricciones de hardware espacial via perfiles Docker
- Trazabilidad completa (hashes SHA256 de inputs, modelos y outputs)
- Tip & Cue: re-tasking autonomo basado en detecciones
- Observabilidad completa con Grafana, Prometheus y Loki

### 1.3 Restricciones de la plataforma

| Recurso | Limite OCI Free Tier |
|---|---|
| CPU | 4 OCPU ARM Ampere A1 |
| RAM | 24 GB |
| Disco boot | 200 GB |
| Disco block | 200 GB adicionales (2x 50 GB gratis + upgrade) |
| Red | 10 Gbps, bandwidth gratuito |
| SO | Oracle Linux 8 / Ubuntu 22.04 ARM |
| Arquitectura | aarch64 (ARM64) |

### 1.4 Principios de diseno

1. **Todo Python**: un solo lenguaje, alineado con el ecosistema EO/satelital
2. **Contenedores Docker**: reproducibilidad total, despliegue con `docker compose up`
3. **Observabilidad nativa**: cada operacion emite metricas y logs estructurados
4. **Trazabilidad inmutable**: cada ejecucion genera un registro de proveniencia con hashes
5. **Perfiles de restriccion**: el mismo pipeline se ejecuta bajo multiples limites de recursos
6. **Datos reales**: Sentinel-1 GRD de Copernicus, no datos sinteticos

### 1.5 Lo que NO incluye el MVP

- Entrenamiento de modelos desde cero (se usa fine-tuning de modelo preentrenado)
- Ejecucion real en orbita
- Hardware flight-proven resistente a radiacion
- Seguridad de nivel clasificado — datos de Copernicus son publicos

> **AI Act:** AIDRA *si* se evalua bajo el Reglamento (UE) 2024/1689. Ver
> `AI_ACT_DECLARATION.md` (clasificacion, base legal, supervision humana).

---

## 2. Arquitectura del Sistema

### 2.1 Diagrama de componentes

```
┌──────────────────────────────────────────────────────────────┐
│                   OCI ARM A1 (4 OCPU, 24 GB)                 │
│                                                              │
│  ┌───────────────────────────────┐      ┌────────────────┐   │
│  │ aidra-app (Python/FastAPI)    │      │ aidra-grafana   │   │
│  │  :8000                        │      │  :3000          │   │
│  │                               │      │                 │   │
│  │  ┌──────────┐ ┌────────────┐  │      │ 5 Dashboards:  │   │
│  │  │ API REST │ │ Pipeline   │  │      │ - Mapa barcos  │   │
│  │  │          │ │ Engine     │  │      │ - Metricas     │   │
│  │  │ FastAPI  │ │            │  │      │ - Benchmarks   │   │
│  │  │ +Uvicorn │ │ Ingesta    │  │      │ - Perfiles     │   │
│  │  │          │ │ Preproceso │  │      │ - Traceability │   │
│  │  │ /api/*   │ │ Inferencia │  │      └────────────────┘   │
│  │  │ /docs    │ │ Postproc.  │  │                            │
│  │  └──────────┘ │ Metricas   │  │      ┌────────────────┐   │
│  │               └────────────┘  │      │ aidra-db       │   │
│  │  ┌───────────────┐            │      │ PostgreSQL 16  │   │
│  │  │ APScheduler   │            │──────│ + PostGIS 3.4  │   │
│  │  │ (in-process)  │            │      │  :5432         │   │
│  │  └───────────────┘            │      └────────────────┘   │
│  │  ┌───────────────┐            │                            │
│  │  │ Docker SDK     │            │      ┌────────────────┐   │
│  │  │ (perfiles)     │            │      │ aidra-prom     │   │
│  │  └───────────────┘            │      │ Prometheus     │   │
│  └───────────────────────────────┘      │  :9090         │   │
│                                          └────────────────┘   │
│  ┌────────────────┐  ┌────────────────┐                       │
│  │ aidra-loki     │  │ aidra-promtail │                       │
│  │ Loki :3100     │  │ Promtail       │                       │
│  └────────────────┘  └────────────────┘                       │
└──────────────────────────────────────────────────────────────┘

Externos:
  ┌─────────────────┐     ┌─────────────────┐
  │ Copernicus API  │     │ xView3-SAR      │
  │ (OData + STAC)  │     │ (dataset)       │
  └─────────────────┘     └─────────────────┘
```

### 2.2 Contenedores Docker

| Contenedor | Imagen base | Puerto | RAM estimada | Funcion |
|---|---|---|---|---|
| `aidra-app` | `python:3.11-slim-bookworm` (ARM) | 8000 | 2-4 GB (picos 6 GB) | API + Pipeline + Scheduler |
| `aidra-db` | `postgis/postgis:16-3.4-alpine` (ARM) | 5432 | 1-2 GB | Base de datos geoespacial |
| `aidra-grafana` | `grafana/grafana-oss:11.0.0` (ARM) | 3000 | 300-500 MB | Dashboards |
| `aidra-prom` | `prom/prometheus:v2.53.0` (ARM) | 9090 | 300-500 MB | Metricas de sistema |
| `aidra-loki` | `grafana/loki:3.1.0` (ARM) | 3100 | 200-300 MB | Almacen de logs |
| `aidra-promtail` | `grafana/promtail:3.1.0` (ARM) | — | 50-100 MB | Recolector de logs |

**Total estimado: ~5-8 GB RAM en reposo. Picos hasta ~12 GB durante inferencia con imagen grande.**

### 2.3 Red Docker

```yaml
networks:
  aidra-net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

Todos los contenedores en la misma red. Comunicacion interna por nombre de contenedor.

### 2.4 Volumenes Docker

```yaml
volumes:
  aidra-db-data:        # Persistencia PostgreSQL
  aidra-grafana-data:   # Dashboards y configuracion Grafana
  aidra-prom-data:      # Datos historicos Prometheus
  aidra-loki-data:      # Logs historicos
  aidra-models:         # Modelos IA (pesos .pt, .onnx)
  aidra-images:         # Cache temporal de imagenes satelitales
```

---

## 3. Estructura de Directorios

```
AIDRA/
├── docker-compose.yml              # Orquestacion de todos los servicios
├── .env                            # Variables de entorno (NO commitear)
├── .env.example                    # Plantilla de variables de entorno
├── Dockerfile                      # Imagen de la app Python
├── pyproject.toml                  # Dependencias Python (uv/pip)
├── README.md                       # Documentacion del proyecto
├── TECHNICAL_SPEC.md               # Este documento
│
├── src/                            # Codigo fuente principal
│   ├── __init__.py
│   ├── main.py                     # Entrypoint FastAPI + lifespan
│   ├── config.py                   # Configuracion centralizada (pydantic-settings)
│   │
│   ├── api/                        # Endpoints REST
│   │   ├── __init__.py
│   │   ├── router.py               # Router principal
│   │   ├── health.py               # GET /api/health
│   │   ├── detections.py           # GET /api/detections, GET /api/detections/{id}
│   │   ├── pipeline.py             # POST /api/pipeline/trigger, trigger-all-profiles
│   │   ├── benchmarks.py           # GET /api/benchmarks, /api/benchmarks/compare
│   │   ├── traceability.py         # GET /api/traceability/{execution_id}
│   │   ├── tasking.py              # GET /api/tasking/queue, POST /api/tasking/cue
│   │   └── metrics.py              # GET /api/metrics (Prometheus)
│   │
│   ├── pipeline/                   # Motor del pipeline de procesamiento
│   │   ├── __init__.py
│   │   ├── engine.py               # Orquestador del pipeline completo
│   │   ├── ingestion.py            # Descarga de imagenes Copernicus
│   │   ├── preprocessing.py        # Calibracion, correccion geometrica SAR
│   │   ├── detection.py            # Inferencia: CFAR + YOLO
│   │   ├── postprocessing.py       # NMS, filtrado, geolocalizacion de detecciones
│   │   └── cleanup.py              # Limpieza de imagenes temporales
│   │
│   ├── models/                     # Gestion de modelos IA
│   │   ├── __init__.py
│   │   ├── manager.py              # Carga, versionado, hashing de modelos
│   │   ├── yolo.py                 # Wrapper para ultralytics YOLO
│   │   ├── cfar.py                 # Implementacion CFAR (Constant False Alarm Rate)
│   │   └── compression/            # Tecnicas de compresion
│   │       ├── __init__.py
│   │       ├── quantization.py     # Quantizacion FP32 → INT8 (PyTorch + ONNX)
│   │       ├── pruning.py          # Pruning estructurado y no estructurado
│   │       └── distillation.py     # Knowledge distillation
│   │
│   ├── profiles/                   # Perfiles de restriccion de hardware
│   │   ├── __init__.py
│   │   ├── manager.py              # Gestion de perfiles, lanzamiento Docker
│   │   ├── definitions.py          # Definicion de perfiles (ground, sat-high, etc.)
│   │   └── metrics_collector.py    # Recoleccion de metricas bajo perfil
│   │
│   ├── tipcue/                     # Logica Tip & Cue
│   │   ├── __init__.py
│   │   ├── evaluator.py            # Evalua detecciones → genera tips
│   │   ├── scheduler.py            # Gestiona la cola de cues
│   │   └── zones.py                # Definicion de zonas de interes
│   │
│   ├── orbital/                    # Modulos de valor orbital (diferenciadores)
│   │   ├── __init__.py
│   │   ├── energy.py               # M9: Perfil energetico (joules/inferencia, TOPS/W)
│   │   ├── downlink.py             # M10: Analisis de downlink (ratio compresion, ahorro BW)
│   │   ├── latency.py              # M11: Latencia orbital (sensor → resultado)
│   │   ├── resilience.py           # M12: Resiliencia (bit-flips, fallback, drift)
│   │   ├── decision_engine.py      # M12: Motor de decision autonomo
│   │   └── orbit_params.py         # Parametros orbitales (LEO, SSO, altitudes)
│   │
│   ├── traceability/               # Sistema de trazabilidad
│   │   ├── __init__.py
│   │   ├── hasher.py               # Calculo SHA256 de archivos
│   │   ├── recorder.py             # Escritura de registros en execution_log
│   │   └── verifier.py             # Verificacion de reproducibilidad
│   │
│   ├── observability/              # Metricas y logs
│   │   ├── __init__.py
│   │   ├── prometheus_metrics.py   # Definicion de metricas Prometheus
│   │   └── loki_logger.py          # Logger estructurado para Loki
│   │
│   └── db/                         # Capa de base de datos
│       ├── __init__.py
│       ├── connection.py           # Pool de conexiones asyncpg
│       ├── models.py               # Modelos Pydantic para la API
│       ├── queries.py              # Consultas SQL parametrizadas
│       └── migrations/             # Migraciones SQL
│           ├── 001_init.sql        # Esquema inicial
│           ├── 002_indexes.sql     # Indices y optimizaciones
│           └── 003_tipcue.sql      # Tablas Tip & Cue
│
├── models/                         # Pesos de modelos (gitignored, volumen Docker)
│   ├── .gitkeep
│   └── README.md                   # Instrucciones para descargar modelos
│
├── grafana/                        # Configuracion Grafana
│   ├── provisioning/
│   │   ├── datasources/
│   │   │   └── datasources.yml     # PostgreSQL + Prometheus + Loki
│   │   └── dashboards/
│   │       └── dashboards.yml      # Provider de dashboards
│   └── dashboards/
│       ├── 01-map-detections.json  # Dashboard mapa GeoMap
│       ├── 02-pipeline-metrics.json # Dashboard metricas pipeline
│       ├── 03-compression-bench.json # Dashboard benchmarks compresion
│       ├── 04-constraint-profiles.json # Dashboard perfiles restriccion
│       ├── 05-traceability.json    # Dashboard trazabilidad
│       ├── 06-obdp-value.json      # Dashboard valor OBDP (downlink)
│       ├── 07-orbital-latency.json # Dashboard latencia orbital
│       └── 08-orbital-resilience.json # Dashboard resiliencia (bit-flips, decisiones)
│
├── prometheus/
│   └── prometheus.yml              # Configuracion scrape
│
├── loki/
│   └── loki-config.yml             # Configuracion Loki
│
├── promtail/
│   └── promtail-config.yml         # Configuracion Promtail
│
├── scripts/                        # Scripts de utilidad
│   ├── setup-oci.sh                # Provisioning OCI ARM A1
│   ├── download-models.sh          # Descarga de modelos preentrenados
│   ├── download-dataset.sh         # Descarga de dataset xView3-SAR
│   ├── fine-tune.py                # Script de fine-tuning YOLOv8
│   └── seed-db.sh                  # Seed datos de prueba
│
└── tests/                          # Tests
    ├── __init__.py
    ├── conftest.py                 # Fixtures compartidas
    ├── test_api/                   # Tests de endpoints
    │   ├── test_health.py
    │   ├── test_detections.py
    │   ├── test_pipeline.py
    │   └── test_traceability.py
    ├── test_pipeline/              # Tests del motor de pipeline
    │   ├── test_ingestion.py
    │   ├── test_detection.py
    │   └── test_postprocessing.py
    ├── test_models/                # Tests de modelos
    │   ├── test_yolo.py
    │   ├── test_cfar.py
    │   └── test_compression.py
    └── test_traceability/          # Tests de trazabilidad
        ├── test_hasher.py
        └── test_recorder.py
```

---

## 4. Modulo 0: Infraestructura OCI + Docker

### 4.1 Agente responsable: `AGENT-INFRA`

### 4.2 Provisioning OCI ARM A1

**Prerequisito**: cuenta OCI Free Tier con instancia ARM A1 (4 OCPU, 24 GB).

**Script `scripts/setup-oci.sh`**:

```bash
#!/bin/bash
set -euo pipefail

# 1. Actualizar sistema
sudo dnf update -y  # Oracle Linux 8
# o sudo apt update && sudo apt upgrade -y  # Ubuntu 22.04

# 2. Instalar Docker
sudo dnf install -y dnf-utils
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER

# 3. Configurar firewall
sudo firewall-cmd --permanent --add-port=8000/tcp  # API
sudo firewall-cmd --permanent --add-port=3000/tcp  # Grafana
sudo firewall-cmd --reload

# 4. Crear directorios de datos
sudo mkdir -p /opt/aidra/{models,images,data}
sudo chown -R $USER:$USER /opt/aidra

# 5. Configurar swap (seguridad para picos de RAM)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab

# 6. Limites de open files para PostgreSQL
echo 'fs.file-max = 65536' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### 4.3 Dockerfile

```dockerfile
# ---- Dockerfile ----
FROM python:3.11-slim-bookworm AS base

# Dependencias del sistema para GDAL, rasterio, OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    libffi-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Variable para GDAL
ENV GDAL_CONFIG=/usr/bin/gdal-config

WORKDIR /app

# Instalar dependencias Python
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[all]"

# Copiar codigo fuente
COPY src/ ./src/
COPY models/ ./models/

# Puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Entrypoint
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### 4.4 docker-compose.yml

```yaml
# ---- docker-compose.yml ----
version: "3.9"

services:
  # ====== APP PRINCIPAL ======
  aidra-app:
    build: .
    container_name: aidra-app
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://aidra:${DB_PASSWORD}@aidra-db:5432/aidra
      - COPERNICUS_USER=${COPERNICUS_USER}
      - COPERNICUS_PASSWORD=${COPERNICUS_PASSWORD}
      - MODELS_DIR=/app/models
      - IMAGES_DIR=/data/images
      - LOG_LEVEL=INFO
      - PROMETHEUS_ENABLED=true
      - LOKI_URL=http://aidra-loki:3100
    volumes:
      - aidra-models:/app/models
      - aidra-images:/data/images
      - /var/run/docker.sock:/var/run/docker.sock  # Para perfiles de restriccion
    depends_on:
      aidra-db:
        condition: service_healthy
    networks:
      - aidra-net

  # ====== BASE DE DATOS ======
  aidra-db:
    image: postgis/postgis:16-3.4-alpine
    container_name: aidra-db
    restart: unless-stopped
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=aidra
      - POSTGRES_USER=aidra
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - aidra-db-data:/var/lib/postgresql/data
      - ./src/db/migrations:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U aidra -d aidra"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - aidra-net

  # ====== GRAFANA ======
  aidra-grafana:
    image: grafana/grafana-oss:11.0.0
    container_name: aidra-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_INSTALL_PLUGINS=grafana-worldmap-panel
    volumes:
      - aidra-grafana-data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    depends_on:
      - aidra-db
      - aidra-prom
    networks:
      - aidra-net

  # ====== PROMETHEUS ======
  aidra-prom:
    image: prom/prometheus:v2.53.0
    container_name: aidra-prom
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - aidra-prom-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'
      - '--storage.tsdb.retention.size=5GB'
    networks:
      - aidra-net

  # ====== LOKI ======
  aidra-loki:
    image: grafana/loki:3.1.0
    container_name: aidra-loki
    restart: unless-stopped
    ports:
      - "3100:3100"
    volumes:
      - ./loki/loki-config.yml:/etc/loki/local-config.yaml
      - aidra-loki-data:/loki
    command: -config.file=/etc/loki/local-config.yaml
    networks:
      - aidra-net

  # ====== PROMTAIL ======
  aidra-promtail:
    image: grafana/promtail:3.1.0
    container_name: aidra-promtail
    restart: unless-stopped
    volumes:
      - ./promtail/promtail-config.yml:/etc/promtail/config.yml
      - /var/log:/var/log:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    command: -config.file=/etc/promtail/config.yml
    networks:
      - aidra-net

volumes:
  aidra-db-data:
  aidra-grafana-data:
  aidra-prom-data:
  aidra-loki-data:
  aidra-models:
  aidra-images:

networks:
  aidra-net:
    driver: bridge
```

### 4.5 Archivos de configuracion auxiliares

**`prometheus/prometheus.yml`**:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'aidra-app'
    static_configs:
      - targets: ['aidra-app:8000']
    metrics_path: '/api/metrics'
```

**`loki/loki-config.yml`**:
```yaml
auth_enabled: false
server:
  http_listen_port: 3100
common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    instance_addr: 127.0.0.1
    kvstore:
      store: inmemory
schema_config:
  configs:
    - from: 2024-01-01
      store: tsdb
      object_store: filesystem
      schema: v13
      index:
        prefix: index_
        period: 24h
limits_config:
  reject_old_samples: true
  reject_old_samples_max_age: 168h
```

**`promtail/promtail-config.yml`**:
```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://aidra-loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: docker
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: ['__meta_docker_container_name']
        regex: '/(.*)'
        target_label: 'container'
```

**`grafana/provisioning/datasources/datasources.yml`**:
```yaml
apiVersion: 1
datasources:
  - name: PostgreSQL
    type: postgres
    url: aidra-db:5432
    database: aidra
    user: aidra
    secureJsonData:
      password: ${DB_PASSWORD}
    jsonData:
      sslmode: disable
      postgresVersion: 1600
      timescaledb: false

  - name: Prometheus
    type: prometheus
    url: http://aidra-prom:9090
    isDefault: true
    jsonData:
      timeInterval: 15s

  - name: Loki
    type: loki
    url: http://aidra-loki:3100
```

**`grafana/provisioning/dashboards/dashboards.yml`**:
```yaml
apiVersion: 1
providers:
  - name: 'AIDRA'
    orgId: 1
    folder: 'AIDRA'
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /var/lib/grafana/dashboards
      foldersFromFilesStructure: false
```

---

## 5. Modulo 1: Ingesta de Imagenes Satelitales

### 5.1 Agente responsable: `AGENT-INGESTION`

### 5.2 Fuente de datos

**Sentinel-1 GRD (Ground Range Detected)** via Copernicus Data Space Ecosystem.

Justificacion:
- SAR funciona de noche y con nubes (optico no)
- Los barcos aparecen como puntos brillantes sobre fondo oscuro del mar
- Sentinel-1 GRD es el producto mas usado para vessel detection en la literatura
- Copernicus es gratuito (12 TB/mes), licencia abierta UE

### 5.3 APIs de Copernicus

**API primaria: OData API** (catalogo + descarga)
- Base URL: `https://catalogue.dataspace.copernicus.eu/odata/v1/`
- Autenticacion: OAuth2 via `https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token`

**API secundaria: STAC API** (solo catalogo, sin descarga directa)
- Base URL: `https://catalogue.dataspace.copernicus.eu/stac/`

### 5.4 Archivo `src/pipeline/ingestion.py`

```python
"""
Modulo de ingesta de imagenes satelitales Sentinel-1 desde Copernicus.

Responsabilidades:
1. Autenticacion OAuth2 con Copernicus Data Space
2. Busqueda de productos Sentinel-1 GRD por area y fecha
3. Descarga del producto (archivo .zip, ~500 MB - 1 GB)
4. Extraccion y validacion del producto descargado
5. Calculo de hash SHA256 del archivo descargado
6. Limpieza de archivos temporales

Dependencias externas:
- requests (HTTP)
- pystac-client (busqueda STAC, opcional)

Notas:
- Los tokens OAuth2 de Copernicus expiran en 600 segundos (10 minutos)
- La cuota gratuita permite 12 TB/mes de descarga
- Los productos Sentinel-1 GRD tienen ~500 MB - 1 GB por escena
- El area de busqueda se define como un bounding box [lon_min, lat_min, lon_max, lat_max]
"""

# --- Clases y funciones a implementar ---

class CopernicusAuth:
    """
    Gestiona autenticacion OAuth2 con Copernicus Data Space.

    Metodos:
    - get_token() -> str: Obtiene token de acceso. Refresca si esta expirado.
    - _refresh_token() -> str: Solicita nuevo token al endpoint OAuth2.

    Atributos:
    - username: str (de config/env)
    - password: str (de config/env)
    - _token: str | None
    - _token_expiry: datetime | None

    Endpoint token:
    POST https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token
    Body (form-urlencoded):
        grant_type=password
        username={username}
        password={password}
        client_id=cdse-public
    """

class CopernicusSearchResult:
    """
    Pydantic model para resultado de busqueda.

    Campos:
    - product_id: str          # ID del producto en Copernicus
    - title: str               # Nombre del producto
    - sensing_date: datetime   # Fecha de adquisicion
    - footprint: dict          # GeoJSON del area cubierta
    - size_mb: float           # Tamano estimado en MB
    - download_url: str        # URL de descarga directa
    - online: bool             # Si esta disponible para descarga inmediata
    """

class ImageIngester:
    """
    Orquesta la busqueda y descarga de imagenes Sentinel-1.

    Constructor:
    - auth: CopernicusAuth
    - images_dir: Path (directorio donde guardar las imagenes)

    Metodos:
    - search(bbox, start_date, end_date, max_results=5) -> list[CopernicusSearchResult]:
        Busca productos Sentinel-1 GRD en el area y rango de fechas.
        Query OData:
        GET /odata/v1/Products
        ?$filter=Collection/Name eq 'SENTINEL-1'
            and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType'
                and att/OData.CSC.StringAttribute/Value eq 'GRD')
            and OData.CSC.Intersects(area=geography'SRID=4326;POLYGON((
                {lon_min} {lat_min},
                {lon_max} {lat_min},
                {lon_max} {lat_max},
                {lon_min} {lat_max},
                {lon_min} {lat_min}
            ))')
            and ContentDate/Start gt {start_date}T00:00:00.000Z
            and ContentDate/Start lt {end_date}T23:59:59.999Z
        &$orderby=ContentDate/Start desc
        &$top={max_results}

    - download(product: CopernicusSearchResult) -> Path:
        Descarga el producto. Retorna la ruta al archivo .zip descargado.
        URL descarga: GET /odata/v1/Products({product_id})/$value
        Header: Authorization: Bearer {token}
        Implementar descarga con streaming (chunks de 8 MB).
        Mostrar progreso.

    - extract(zip_path: Path) -> Path:
        Extrae el .zip. Retorna ruta al directorio del producto (contiene .tiff).

    - compute_hash(file_path: Path) -> str:
        Calcula SHA256 del archivo. Lee en chunks de 64 KB.
"""
```

### 5.5 Zonas de busqueda predefinidas

```python
# src/pipeline/ingestion.py (constantes)

SEARCH_ZONES = {
    "gibraltar": {
        "name": "Estrecho de Gibraltar",
        "bbox": [-5.8, 35.7, -5.2, 36.2],
        "description": "Alto trafico maritimo, estrecho natural"
    },
    "mediterranean_west": {
        "name": "Mediterraneo Occidental",
        "bbox": [-1.0, 36.5, 4.0, 39.5],
        "description": "Ruta comercial principal, costas Espana-Argelia"
    },
    "suez_approach": {
        "name": "Aproximacion Canal de Suez",
        "bbox": [32.0, 29.5, 34.0, 31.5],
        "description": "Zona de espera, alta densidad de barcos"
    },
    "english_channel": {
        "name": "Canal de la Mancha",
        "bbox": [-2.0, 49.5, 2.0, 51.5],
        "description": "Ruta comercial Europa del Norte"
    },
    "north_adriatic": {
        "name": "Norte del Adriatico",
        "bbox": [12.0, 44.5, 14.0, 45.8],
        "description": "Zona portuaria, Venecia-Trieste"
    }
}
```

### 5.6 Preprocesamiento SAR

**Archivo `src/pipeline/preprocessing.py`**:

```python
"""
Preprocesamiento de imagenes SAR Sentinel-1 GRD.

Responsabilidades:
1. Calibracion radiometrica: DN (digital number) → sigma0 (backscatter en dB)
2. Correccion geometrica / geocodificacion
3. Filtrado de speckle (ruido inherente en SAR)
4. Recorte al area de interes (AOI)
5. Tiling: dividir imagen grande en tiles manejables para inferencia

Dependencias:
- rasterio (lectura/escritura GeoTIFF)
- numpy (operaciones matriciales)
- scipy.ndimage (filtro de speckle Lee)

Notas:
- Las imagenes Sentinel-1 GRD vienen en formato TIFF con metadatos XML
- La calibracion usa los coeficientes del archivo annotation XML del producto
- El filtro de speckle Lee con ventana 7x7 es el estandar para vessel detection
- Los tiles deben tener tamano fijo (ej: 640x640 px) con overlap (ej: 64 px)
  para evitar perder detecciones en los bordes
"""

# --- Funciones a implementar ---

def calibrate_sigma0(tiff_path: Path, annotation_xml: Path) -> np.ndarray:
    """
    Calibracion radiometrica: convierte DN a sigma0 (dB).

    Formula: sigma0 = 10 * log10((DN^2 + offset) / calibration_constant)

    Los coeficientes estan en el archivo annotation XML del producto S1:
    <calibrationVectorList>
      <calibrationVector>
        <sigmaNought>...</sigmaNought>
      </calibrationVector>
    </calibrationVectorList>

    Args:
        tiff_path: Ruta al archivo .tiff del producto S1
        annotation_xml: Ruta al archivo de calibracion

    Returns:
        np.ndarray: Imagen calibrada en sigma0 (dB), dtype float32
    """

def apply_lee_filter(image: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    Filtro de speckle Lee.
    Reduce el ruido multiplicativo inherente en SAR preservando bordes.

    Algoritmo:
    1. Calcular media local en ventana
    2. Calcular varianza local en ventana
    3. Calcular coeficiente de variacion (CV)
    4. Aplicar: filtered = mean + k * (original - mean)
       donde k = var_local / (var_local + var_ruido)
    """

def create_tiles(
    image: np.ndarray,
    tile_size: int = 640,
    overlap: int = 64,
    geo_transform: tuple = None
) -> list[dict]:
    """
    Divide imagen grande en tiles para inferencia.

    Args:
        image: Imagen SAR calibrada
        tile_size: Tamano del tile (px)
        overlap: Solapamiento entre tiles (px)
        geo_transform: Transformacion afin para geolocalizacion

    Returns:
        Lista de dicts con:
        - "array": np.ndarray (tile)
        - "row_offset": int
        - "col_offset": int
        - "geo_bounds": dict con lat_min, lat_max, lon_min, lon_max
    """

def preprocess_full(
    product_dir: Path,
    aoi_bbox: list[float] | None = None,
    tile_size: int = 640,
    overlap: int = 64
) -> dict:
    """
    Pipeline completo de preprocesamiento.

    1. Localizar .tiff y annotation XML en product_dir
    2. Calibrar a sigma0
    3. Recortar a AOI si se proporciona
    4. Aplicar filtro Lee
    5. Crear tiles
    6. Retornar dict con tiles + metadatos

    Returns:
        {
            "tiles": list[dict],
            "metadata": {
                "product_dir": str,
                "original_shape": tuple,
                "calibration": "sigma0_db",
                "filter": "lee_7x7",
                "tile_size": int,
                "overlap": int,
                "num_tiles": int,
                "crs": str,
                "geo_transform": tuple
            }
        }
    """
```

---

## 6. Modulo 2: Deteccion de Barcos con IA

### 6.1 Agente responsable: `AGENT-DETECTION`

### 6.2 Estrategia dual: CFAR + YOLO

El pipeline implementa dos detectores complementarios:

1. **CFAR (Constant False Alarm Rate)**: algoritmo clasico de procesamiento de senal SAR, muy ligero en CPU, alta recall pero mas falsos positivos
2. **YOLOv8**: red neuronal de deteccion de objetos, mas precisa pero mas costosa computacionalmente

Ambos detectores se ejecutan en cada tile y sus resultados se fusionan en el postprocesamiento.

### 6.3 Archivo `src/models/cfar.py`

```python
"""
Detector CFAR (Constant False Alarm Rate) para imagenes SAR.

El CFAR detecta pixeles cuyo valor de backscatter es significativamente
superior al fondo local. Es el algoritmo estandar para detectar barcos
en SAR porque los barcos producen reflexiones metalicas muy fuertes
(corner reflector effect).

Algoritmo:
1. Para cada pixel "bajo test" (CUT):
   a. Definir ventana de guarda (guard cells): excluye pixeles adyacentes al CUT
   b. Definir ventana de entrenamiento (training cells): estima el fondo local
   c. Calcular media y varianza del fondo en training cells
   d. Calcular umbral adaptativo: threshold = mean + k * std
      (k depende de la probabilidad de falsa alarma deseada)
   e. Si CUT > threshold: deteccion

Parametros:
- guard_size: Radio de la ventana de guarda (tipico: 2-4 px)
- training_size: Radio de la ventana de entrenamiento (tipico: 10-20 px)
- pfa: Probabilidad de falsa alarma deseada (tipico: 1e-6 a 1e-4)
  k se calcula como: k = sqrt(2 * ln(num_training_cells / pfa))

Variantes implementadas:
- CA-CFAR (Cell-Averaging): media simple del fondo
- OS-CFAR (Ordered Statistics): usa percentil del fondo (mas robusto a clutter)
"""

class CFARDetector:
    """
    Constructor:
    - guard_size: int = 3
    - training_size: int = 15
    - pfa: float = 1e-5
    - method: str = "ca"  # "ca" o "os"
    - os_percentile: float = 0.75  # Solo para OS-CFAR

    Metodos:
    - detect(image: np.ndarray) -> list[dict]:
        Ejecuta CFAR sobre la imagen SAR calibrada.
        Returns: lista de detecciones, cada una con:
        {
            "x": int,           # columna del centro
            "y": int,           # fila del centro
            "intensity": float, # valor sigma0 del pixel
            "snr": float,       # signal-to-noise ratio vs fondo local
            "method": str       # "ca-cfar" o "os-cfar"
        }

    - detect_with_clustering(image: np.ndarray, min_cluster_size: int = 3) -> list[dict]:
        Ejecuta CFAR + clustering DBSCAN para agrupar pixeles de deteccion
        adyacentes (un barco genera multiples pixeles detectados).
        Returns: lista de detecciones agrupadas con bounding box:
        {
            "bbox": [x_min, y_min, x_max, y_max],
            "center": [x, y],
            "num_pixels": int,
            "mean_intensity": float,
            "max_intensity": float,
            "mean_snr": float,
            "method": str
        }
    """
```

### 6.4 Archivo `src/models/yolo.py`

```python
"""
Wrapper para modelo YOLOv8 de deteccion de barcos.

Modelos soportados:
- YOLOv8n (nano): 3.2M params, 6.2 MB, ~50ms/imagen en CPU ARM
- YOLOv8s (small): 11.2M params, 22.5 MB, ~100ms/imagen en CPU ARM
- YOLOv8m (medium): 25.9M params, 52 MB, ~200ms/imagen en CPU ARM

Formatos de exportacion soportados:
- PyTorch (.pt): formato nativo, mas flexible
- ONNX (.onnx): para quantizacion y deployment
- OpenVINO: optimizado para Intel (referencia)

Dependencias:
- ultralytics (pip install ultralytics)
- torch (viene con ultralytics)
- onnxruntime (para inferencia ONNX)
"""

class YOLODetector:
    """
    Constructor:
    - model_path: Path  # Ruta al archivo de pesos (.pt o .onnx)
    - confidence_threshold: float = 0.25
    - iou_threshold: float = 0.45  # Para NMS
    - device: str = "cpu"  # Siempre CPU en OCI ARM

    Atributos calculados en __init__:
    - model_name: str (extraido del nombre de archivo)
    - model_hash: str (SHA256 del archivo de pesos)
    - model_size_mb: float (tamano del archivo en MB)
    - model_format: str ("pytorch" o "onnx")

    Metodos:
    - predict(image: np.ndarray) -> list[dict]:
        Ejecuta inferencia en una imagen (tile).
        Returns: lista de detecciones:
        {
            "bbox": [x_min, y_min, x_max, y_max],  # en pixeles del tile
            "confidence": float,                     # 0-1
            "class_id": int,                         # 0 = vessel
            "class_name": str                        # "vessel"
        }

    - predict_batch(tiles: list[np.ndarray]) -> list[list[dict]]:
        Inferencia por lotes. Mas eficiente si hay multiples tiles.

    - export_onnx(output_path: Path, opset: int = 13) -> Path:
        Exporta modelo PyTorch a ONNX. Retorna ruta del archivo .onnx.

    - get_model_info() -> dict:
        Retorna metadatos del modelo:
        {
            "name": str,
            "format": str,
            "size_mb": float,
            "hash": str,
            "num_params": int,
            "num_layers": int,
            "input_size": [int, int],
            "classes": list[str]
        }

    - benchmark(image: np.ndarray, num_runs: int = 100) -> dict:
        Ejecuta num_runs inferencias y retorna estadisticas:
        {
            "mean_ms": float,
            "std_ms": float,
            "min_ms": float,
            "max_ms": float,
            "p50_ms": float,
            "p95_ms": float,
            "p99_ms": float,
            "peak_ram_mb": float,
            "cpu_percent": float
        }
    """
```

### 6.5 Archivo `src/pipeline/detection.py`

```python
"""
Orquestador de deteccion. Combina CFAR y YOLO.

Flujo:
1. Recibe tiles preprocesados
2. Ejecuta CFAR en cada tile (rapido, alta recall)
3. Ejecuta YOLO en cada tile (preciso, costoso)
4. Fusiona resultados
5. Registra metricas de rendimiento

La fusion usa la siguiente logica:
- Si CFAR y YOLO detectan en la misma zona (IoU > 0.3): alta confianza
- Si solo YOLO detecta: confianza media (depende del score YOLO)
- Si solo CFAR detecta: confianza baja (posible falso positivo)
"""

class DetectionEngine:
    """
    Constructor:
    - cfar: CFARDetector
    - yolo: YOLODetector
    - fusion_iou_threshold: float = 0.3

    Metodos:
    - run(tiles: list[dict], constraint_profile: str = "ground") -> DetectionResult:
        Ejecuta pipeline de deteccion completo.

        Args:
            tiles: Lista de tiles del preprocesamiento
            constraint_profile: Perfil de restriccion activo

        Returns: DetectionResult con:
            - detections: list[Detection]  # Detecciones fusionadas
            - metrics: DetectionMetrics    # Tiempos, RAM, CPU
            - cfar_raw: list[dict]         # Detecciones CFAR crudas
            - yolo_raw: list[dict]         # Detecciones YOLO crudas

    - _fuse_detections(cfar_dets, yolo_dets) -> list[Detection]:
        Fusiona detecciones de ambos detectores.
        Aplica NMS final con IoU threshold = 0.5.

    - _geolocate(detection: dict, tile_info: dict) -> Detection:
        Convierte coordenadas de pixel a coordenadas geograficas (lat/lon)
        usando la geo_transform del tile.
    """

class Detection:
    """
    Pydantic model para una deteccion individual.

    Campos:
    - id: UUID
    - bbox_pixel: list[float]        # [x_min, y_min, x_max, y_max] en pixeles
    - bbox_geo: list[float]          # [lon_min, lat_min, lon_max, lat_max] en WGS84
    - center_geo: list[float]        # [lon, lat] del centro
    - confidence: float              # 0-1, confianza fusionada
    - source: str                    # "cfar", "yolo", "fused"
    - cfar_snr: float | None        # SNR del detector CFAR
    - yolo_score: float | None      # Score del detector YOLO
    - tile_index: int                # Indice del tile donde se detecto
    - geometry: dict                 # GeoJSON Point para PostGIS
    """

class DetectionMetrics:
    """
    Pydantic model para metricas de rendimiento de la deteccion.

    Campos:
    - total_inference_ms: float      # Tiempo total de inferencia
    - cfar_ms: float                 # Tiempo CFAR
    - yolo_ms: float                 # Tiempo YOLO
    - fusion_ms: float              # Tiempo de fusion
    - peak_ram_mb: float            # RAM pico durante inferencia
    - cpu_percent: float            # CPU medio durante inferencia
    - num_tiles: int                 # Numero de tiles procesados
    - num_detections_cfar: int      # Detecciones CFAR antes de fusion
    - num_detections_yolo: int      # Detecciones YOLO antes de fusion
    - num_detections_fused: int     # Detecciones despues de fusion
    """
```

### 6.6 Modelo preentrenado y fine-tuning

**Estrategia**:
1. Comenzar con `yolov8n.pt` (nano, 6 MB) de ultralytics
2. Fine-tune con dataset xView3-SAR (243K objetos maritimos en Sentinel-1)
3. Exportar a ONNX para benchmarks de compresion

**Script `scripts/fine-tune.py`** (ejecutar en maquina con GPU, no en OCI ARM):

```python
"""
Fine-tuning de YOLOv8 con dataset xView3-SAR.

Prerequisitos:
- Descargar xView3-SAR de https://iuu.xview.us/
- GPU con >= 8 GB VRAM (o Google Colab)
- Dataset convertido a formato YOLO (images/ + labels/)

El script:
1. Carga YOLOv8n preentrenado
2. Fine-tune con xView3-SAR (vessel class)
3. Exporta modelo fine-tuned (.pt)
4. Exporta a ONNX (.onnx)
5. Copia modelos al directorio models/

Parametros de entrenamiento:
- epochs: 50
- imgsz: 640
- batch: 16
- optimizer: AdamW
- lr0: 0.001
- lrf: 0.01
- warmup_epochs: 3
- augment: True (mosaico, flip horizontal, escala)

Uso:
    python scripts/fine-tune.py --data /path/to/xview3-sar/data.yaml --epochs 50
"""
```

### 6.7 Archivo `src/pipeline/postprocessing.py`

```python
"""
Postprocesamiento de detecciones.

Responsabilidades:
1. Non-Maximum Suppression (NMS) final sobre detecciones fusionadas
2. Conversion de coordenadas pixel → geo (lat/lon WGS84)
3. Filtrado por confianza minima
4. Generacion de GeoJSON de salida
5. Calculo de estadisticas agregadas
"""

def apply_nms(
    detections: list[dict],
    iou_threshold: float = 0.5
) -> list[dict]:
    """
    Non-Maximum Suppression para eliminar detecciones duplicadas.

    Algoritmo:
    1. Ordenar por confianza (descendente)
    2. Para cada deteccion (empezando por la de mayor confianza):
       a. Calcular IoU con todas las de menor confianza
       b. Eliminar las que tengan IoU > threshold
    3. Retornar detecciones supervivientes
    """

def pixel_to_geo(
    bbox_pixel: list[float],
    tile_row_offset: int,
    tile_col_offset: int,
    geo_transform: tuple,
    crs: str = "EPSG:4326"
) -> dict:
    """
    Convierte bounding box en coordenadas de pixel a coordenadas geograficas.

    Args:
        bbox_pixel: [x_min, y_min, x_max, y_max] en pixeles del tile
        tile_row_offset: Offset de fila del tile en la imagen original
        tile_col_offset: Offset de columna del tile
        geo_transform: Tupla (origin_x, pixel_size_x, 0, origin_y, 0, -pixel_size_y)
        crs: Sistema de referencia de la imagen original

    Returns:
        {
            "bbox_geo": [lon_min, lat_min, lon_max, lat_max],
            "center_geo": [lon, lat],
            "geometry_point": {"type": "Point", "coordinates": [lon, lat]},
            "geometry_polygon": {"type": "Polygon", "coordinates": [...]},
        }
    """

def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Calcula Intersection over Union entre dos bounding boxes."""

def merge_tile_detections(
    tile_detections: list[list[dict]],
    tile_infos: list[dict],
    overlap: int = 64,
    iou_threshold: float = 0.5
) -> list[dict]:
    """
    Fusiona detecciones de todos los tiles, eliminando duplicados
    en las zonas de solapamiento.

    1. Convertir coordenadas de cada deteccion de tile-local a imagen-global
    2. Aplicar NMS global
    3. Retornar detecciones unicas
    """

def detections_to_geojson(detections: list[dict]) -> dict:
    """
    Genera FeatureCollection GeoJSON con todas las detecciones.
    Util para exportar y para el hash del resultado.
    """

def compute_detection_stats(detections: list[dict]) -> dict:
    """
    Estadisticas agregadas de las detecciones.
    Returns:
        {
            "total": int,
            "avg_confidence": float,
            "max_confidence": float,
            "min_confidence": float,
            "by_source": {"cfar": int, "yolo": int, "fused": int},
            "spatial_extent": [lon_min, lat_min, lon_max, lat_max],
        }
    """
```

### 6.8 Archivo `src/pipeline/cleanup.py`

```python
"""
Limpieza de archivos temporales.

Las imagenes Sentinel-1 ocupan ~500 MB - 1 GB cada una.
Con 24 GB de RAM y 200 GB de disco, es critico limpiar.
"""

async def cleanup_product(product_dir: Path) -> None:
    """Elimina el directorio completo del producto descargado."""

async def cleanup_old_products(
    images_dir: Path,
    max_age_hours: int = 24
) -> int:
    """
    Elimina productos descargados hace mas de max_age_hours.
    Retorna numero de productos eliminados.
    """

def get_disk_usage(path: Path) -> dict:
    """
    Retorna uso de disco del directorio.
    Returns: {"total_gb": float, "used_gb": float, "free_gb": float, "percent": float}
    """
```

---

## 7. Modulo 3: Compresion de Modelos

### 7.1 Agente responsable: `AGENT-COMPRESSION`

### 7.2 Contexto AIDRA

Los pliegos mencionan explicitamente: *"assessment of model compression and optimisation approaches (e.g. quantisation, pruning, knowledge distillation) and the trade-offs observed between detection performance and the computational, power and memory constraints of on-board hardware."*

Este modulo es el **nucleo diferenciador** del proyecto.

### 7.3 Archivo `src/models/compression/quantization.py`

```python
"""
Quantizacion de modelos: reduce la precision de los pesos.

Tecnicas implementadas:
1. Dynamic Quantization (PyTorch): FP32 → INT8, post-entrenamiento
2. Static Quantization (ONNX Runtime): FP32 → INT8, con calibracion
3. FP16 Half-precision: FP32 → FP16 (intermedio)

Cada tecnica produce un nuevo archivo de modelo (.onnx o .pt) que
se registra en el sistema con su propio hash SHA256.
"""

class ModelQuantizer:
    """
    Constructor:
    - model_path: Path  # Modelo original (.pt o .onnx)

    Metodos:
    - quantize_dynamic_pytorch(output_path: Path) -> QuantizationResult:
        Quantizacion dinamica con PyTorch.
        Convierte capas Linear y Conv2d a INT8.
        No requiere datos de calibracion.

    - quantize_static_onnx(
          output_path: Path,
          calibration_data: list[np.ndarray],  # 50-100 imagenes representativas
          quant_format: str = "QDQ"  # "QDQ" o "QOperator"
      ) -> QuantizationResult:
        Quantizacion estatica con ONNX Runtime.
        Requiere datos de calibracion para determinar rangos de activacion.
        Produce el modelo mas pequeno y rapido.

    - quantize_fp16(output_path: Path) -> QuantizationResult:
        Conversion a FP16.
        Reduce tamano a la mitad sin perdida significativa.

    Cada metodo retorna:
    QuantizationResult:
        - original_path: Path
        - quantized_path: Path
        - original_size_mb: float
        - quantized_size_mb: float
        - compression_ratio: float  # original / quantized
        - technique: str
        - original_hash: str
        - quantized_hash: str
    """
```

### 7.4 Archivo `src/models/compression/pruning.py`

```python
"""
Pruning de modelos: elimina conexiones/neuronas poco importantes.

Tecnicas implementadas:
1. Unstructured Pruning (L1): elimina pesos individuales con menor magnitud
2. Structured Pruning: elimina canales/filtros completos (mas eficiente en hardware)

El pruning se aplica al modelo PyTorch antes de exportar a ONNX.
Despues del pruning se puede hacer fine-tuning corto (5-10 epochs)
para recuperar precision perdida.
"""

class ModelPruner:
    """
    Constructor:
    - model_path: Path  # Modelo PyTorch (.pt)

    Metodos:
    - prune_unstructured(
          sparsity: float = 0.3,  # 0-1, porcentaje de pesos a eliminar
          output_path: Path = None
      ) -> PruningResult:
        Pruning no estructurado L1.
        Usa torch.nn.utils.prune.l1_unstructured en todas las capas Conv2d.

    - prune_structured(
          amount: float = 0.2,  # 0-1, porcentaje de canales a eliminar
          output_path: Path = None
      ) -> PruningResult:
        Pruning estructurado por canal.
        Usa torch.nn.utils.prune.ln_structured con norma L2.
        Elimina canales completos → reduccion real de computo.

    - prune_and_finetune(
          sparsity: float = 0.3,
          finetune_data: Path,  # Ruta a dataset YOLO
          finetune_epochs: int = 5
      ) -> PruningResult:
        Pruning + fine-tuning corto para recuperar precision.

    Cada metodo retorna:
    PruningResult:
        - original_path: Path
        - pruned_path: Path
        - original_size_mb: float
        - pruned_size_mb: float
        - sparsity_achieved: float  # Sparsity real del modelo pruned
        - technique: str
        - original_hash: str
        - pruned_hash: str
        - num_params_original: int
        - num_params_pruned: int     # Params no-cero
        - num_params_removed: int
    """
```

### 7.5 Archivo `src/models/compression/distillation.py`

```python
"""
Knowledge Distillation: entrena un modelo pequeno (student) para
imitar a uno grande (teacher).

Implementacion:
- Teacher: YOLOv8m (medium, 52 MB) — se asume preentrenado
- Student: YOLOv8n (nano, 6 MB)
- Loss: alpha * CE_loss(student, labels) + (1-alpha) * KL_div(student_logits, teacher_logits)
- Temperature: T (tipico 3-5), suaviza las distribuciones

NOTA: Knowledge distillation requiere entrenamiento, por lo que
este modulo es opcional y se ejecuta en maquina con GPU, no en OCI ARM.
En OCI ARM solo se usa el modelo student ya destilado para inferencia.
"""

class KnowledgeDistiller:
    """
    Constructor:
    - teacher_path: Path  # Modelo teacher (.pt)
    - student_path: Path  # Modelo student (.pt)

    Metodos:
    - distill(
          train_data: Path,   # Dataset YOLO
          epochs: int = 20,
          alpha: float = 0.5,
          temperature: float = 4.0,
          output_path: Path = None
      ) -> DistillationResult:
        Ejecuta knowledge distillation.
        NOTA: Requiere GPU. En OCI ARM, solo cargar resultado.

    DistillationResult:
        - teacher_path: Path
        - student_path: Path
        - distilled_path: Path
        - teacher_size_mb: float
        - student_size_mb: float
        - distilled_size_mb: float
        - technique: str = "knowledge_distillation"
        - teacher_hash: str
        - distilled_hash: str
        - epochs_trained: int
        - final_loss: float
    """
```

### 7.6 Archivo `src/models/manager.py`

```python
"""
Gestion centralizada de variantes de modelo.

Responsabilidades:
1. Descubrir modelos disponibles en el directorio models/
2. Registrar modelos en la tabla models_registry
3. Cargar/descargar modelos bajo demanda
4. Calcular y verificar hashes SHA256
5. Proporcionar interfaz unica para obtener cualquier variante
"""

class ModelManager:
    """
    Constructor:
    - models_dir: Path
    - db: Database

    Metodos:
    - async scan_and_register() -> list[ModelInfo]:
        Escanea models_dir, calcula hashes, registra en models_registry.
        Llamar en startup de la app.
        Soporta .pt (PyTorch) y .onnx (ONNX).
        Extrae metadatos: num_params, input_size, classes.

    - async get_model(name: str, version: str | None = None) -> YOLODetector:
        Retorna un YOLODetector configurado con el modelo solicitado.
        Si version es None, retorna la version mas reciente.
        Cachea modelos cargados en memoria.

    - async list_models() -> list[ModelInfo]:
        Lista todos los modelos registrados.

    - async get_model_by_hash(file_hash: str) -> ModelInfo | None:
        Busca modelo por hash SHA256.

    - _parse_model_name(filename: str) -> tuple[str, str, str]:
        Extrae (name, version, format) del nombre del archivo.
        Convencion:
            yolov8n-sar.pt         → ("yolov8n-sar", "v1.0", "pytorch")
            yolov8n-sar-int8.onnx  → ("yolov8n-sar", "int8", "onnx")
            yolov8n-sar-pruned30.pt → ("yolov8n-sar", "pruned30", "pytorch")

    Atributos:
    - _cache: dict[str, YOLODetector]  # Cache de modelos cargados
    - models_dir: Path
    """
```

### 7.7 Matriz de variantes de modelo

El sistema gestiona multiples variantes del modelo base. Cada variante se registra con su hash SHA256 y se puede ejecutar con cualquier perfil de restriccion.

| Variante | Tecnica | Tamano esperado | Formato |
|---|---|---|---|
| `yolov8n-base` | Ninguna (baseline) | ~6 MB | .pt |
| `yolov8n-sar` | Fine-tuned xView3 | ~6 MB | .pt |
| `yolov8n-sar-onnx` | Exportado ONNX | ~6 MB | .onnx |
| `yolov8n-sar-fp16` | Quantizacion FP16 | ~3 MB | .onnx |
| `yolov8n-sar-int8-dynamic` | Quant. dinamica INT8 | ~1.5 MB | .pt |
| `yolov8n-sar-int8-static` | Quant. estatica INT8 | ~1.5 MB | .onnx |
| `yolov8n-sar-pruned30` | Pruning 30% | ~4.2 MB | .pt |
| `yolov8n-sar-pruned50` | Pruning 50% | ~3 MB | .pt |
| `yolov8n-sar-pruned30-int8` | Pruning 30% + INT8 | ~1 MB | .onnx |
| `yolov8n-distilled` | Knowledge distillation | ~6 MB | .pt |

---

## 8. Modulo 4: Traceability — Cadena de Proveniencia

### 8.1 Agente responsable: `AGENT-TRACE`

### 8.2 Contexto AIDRA

Los pliegos mencionan traceability 6+ veces. Requisito clave del D3 (Evidence Package): *"documentation of test conditions, outputs and observed behaviour"*, *"ensuring traceability and verification of on-board outputs"*.

Cada ejecucion del pipeline genera un **registro de proveniencia inmutable** que responde:
- **Que imagen entro?** (ID Copernicus, hash SHA256)
- **Que modelo se uso?** (nombre, version, hash SHA256 de los pesos)
- **Con que parametros?** (umbral, perfil de restriccion, limites CPU/RAM)
- **Que resultado salio?** (numero de detecciones, confianza media)
- **Es reproducible?** (mismos inputs + modelo + params → mismo resultado)

### 8.3 Archivo `src/traceability/hasher.py`

```python
"""
Calculo de hashes SHA256 para garantizar integridad.

Se hashean:
- Imagenes satelitales descargadas (input)
- Archivos de pesos del modelo (.pt, .onnx)
- Archivos de resultados (GeoJSON de detecciones)

El hash se calcula en streaming (chunks de 64 KB) para no cargar
archivos grandes en memoria.
"""

def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Calcula SHA256 de un archivo. Retorna hex digest."""

def compute_array_hash(array: np.ndarray) -> str:
    """Calcula SHA256 de un numpy array (para resultados en memoria)."""

def compute_result_hash(detections: list[dict]) -> str:
    """
    Calcula hash determinista de las detecciones.
    Serializa a JSON ordenado, luego SHA256.
    Garantiza que el mismo conjunto de detecciones produce el mismo hash.
    """
```

### 8.4 Archivo `src/traceability/recorder.py`

```python
"""
Grabacion de registros de proveniencia en la tabla execution_log.

Cada ejecucion del pipeline llama a recorder.record() con todos
los datos de la ejecucion. El registro es inmutable — nunca se
actualiza ni borra.
"""

class ExecutionRecorder:
    """
    Constructor:
    - db_pool: asyncpg.Pool

    Metodos:
    - record(execution: ExecutionRecord) -> UUID:
        Inserta un registro en execution_log.
        Retorna el UUID generado.

    - get(execution_id: UUID) -> ExecutionRecord | None:
        Recupera un registro por ID.

    - list(
          limit: int = 50,
          offset: int = 0,
          profile: str | None = None,
          model_name: str | None = None,
          status: str | None = None,
          date_from: datetime | None = None,
          date_to: datetime | None = None
      ) -> list[ExecutionRecord]:
        Lista registros con filtros.

    - verify_reproducibility(execution_id: UUID) -> ReproducibilityResult:
        Re-ejecuta el pipeline con los mismos inputs y parametros,
        compara el output_hash. Si coincide → reproducible.
    """

class ExecutionRecord:
    """
    Pydantic model del registro de proveniencia.
    Mapea 1:1 con la tabla execution_log.

    Campos: (ver esquema SQL en seccion 13)
    """
```

### 8.5 Archivo `src/traceability/verifier.py`

```python
"""
Verificacion de reproducibilidad.

Dado un execution_id, re-ejecuta el pipeline con los mismos inputs
y parametros, y compara el output_hash.

Si coincide → resultado reproducible (determinista).
Si no coincide → resultado no determinista (documentar por que).

NOTA: Los modelos de deep learning no son 100% deterministas en CPU
(diferencias de precision aritmetica, orden de operaciones).
Diferencias menores en confianza (< 0.01) son aceptables.
"""

class ReproducibilityVerifier:
    """
    Constructor:
    - recorder: ExecutionRecorder
    - engine: PipelineEngine

    Metodos:
    - async verify(execution_id: UUID) -> ReproducibilityResult:
        1. Obtener execution_record original
        2. Re-ejecutar pipeline con mismos parametros:
           - Misma imagen (image_id)
           - Mismo modelo (model_name + model_version)
           - Mismo perfil (constraint_profile)
           - Mismos umbrales (confidence_threshold, iou_threshold)
        3. Comparar output_hash del original vs re-ejecucion
        4. Si output_hash coincide: reproducible=True
        5. Si no: comparar detecciones individualmente
           (permitir tolerancia de 0.01 en confianza)

    ReproducibilityResult:
        - execution_id: UUID        # Original
        - verification_id: UUID     # Re-ejecucion
        - reproducible: bool
        - original_hash: str
        - verification_hash: str
        - original_detections: int
        - verification_detections: int
        - matching_detections: int   # Detecciones con IoU > 0.9 y confianza similar
        - confidence_diff_mean: float  # Diferencia media en confianza
        - notes: str
    """
```

---

## 9. Modulo 5: Perfiles de Restriccion (Simulacion Espacial)

### 9.1 Agente responsable: `AGENT-PROFILES`

### 9.2 Contexto AIDRA

*"maintaining model performance within the computational, power and memory constraints of space-grade hardware"*

No basta con ejecutar el pipeline una vez. Hay que ejecutar el **mismo escenario** bajo diferentes perfiles de restriccion y comparar como degrada.

### 9.3 Definicion de perfiles

```python
# src/profiles/definitions.py

from dataclasses import dataclass

@dataclass
class ConstraintProfile:
    name: str
    display_name: str
    description: str
    cpu_limit: float      # Numero de CPUs (puede ser fraccion)
    memory_limit_mb: int  # RAM en MB
    docker_cpus: str      # Flag para Docker --cpus
    docker_memory: str    # Flag para Docker --memory
    simulates: str        # Que hardware simula

PROFILES = {
    "ground": ConstraintProfile(
        name="ground",
        display_name="Ground Station",
        description="Sin restricciones, estacion terrena",
        cpu_limit=4.0,
        memory_limit_mb=24576,
        docker_cpus="4",
        docker_memory="24g",
        simulates="Ground processing station (baseline)"
    ),
    "sat-high": ConstraintProfile(
        name="sat-high",
        display_name="Satellite High-End",
        description="Satelite gama alta (Xilinx Zynq / Unibap iX10)",
        cpu_limit=2.0,
        memory_limit_mb=4096,
        docker_cpus="2",
        docker_memory="4g",
        simulates="High-end satellite processor (e.g. Xilinx Zynq UltraScale+)"
    ),
    "sat-mid": ConstraintProfile(
        name="sat-mid",
        display_name="Satellite Mid-Range",
        description="Satelite gama media",
        cpu_limit=1.0,
        memory_limit_mb=2048,
        docker_cpus="1",
        docker_memory="2g",
        simulates="Mid-range satellite processor"
    ),
    "sat-low": ConstraintProfile(
        name="sat-low",
        display_name="Satellite Low-End / CubeSat",
        description="Satelite gama baja o CubeSat",
        cpu_limit=0.5,
        memory_limit_mb=1024,
        docker_cpus="0.5",
        docker_memory="1g",
        simulates="Low-end processor / CubeSat (e.g. Raspberry Pi class)"
    ),
    "sat-extreme": ConstraintProfile(
        name="sat-extreme",
        display_name="Extreme Constraint",
        description="Limite inferior: donde se rompe el pipeline",
        cpu_limit=0.25,
        memory_limit_mb=512,
        docker_cpus="0.25",
        docker_memory="512m",
        simulates="Extreme constraint — find breaking point"
    ),
}
```

### 9.4 Archivo `src/profiles/manager.py`

```python
"""
Gestion de ejecucion con perfiles de restriccion.

Implementacion:
- Usa Docker SDK for Python para lanzar el pipeline dentro de un
  contenedor con limites de CPU y RAM.
- El contenedor "restricted" usa la misma imagen Docker de la app
  pero con flags --cpus y --memory.
- El pipeline se ejecuta como subprocess, capturando metricas.

Alternativa sin Docker-in-Docker:
- Si la app ya corre en Docker, usar cgroups directamente
  via /sys/fs/cgroup para limitar el proceso del pipeline.
  Esto evita el problema de Docker-in-Docker.

Metodo preferido: resource.setrlimit + psutil para limitar y medir
el proceso de inferencia directamente, sin lanzar otro contenedor.
"""

class ProfileManager:
    """
    Constructor:
    - profiles: dict[str, ConstraintProfile] = PROFILES

    Metodos:
    - run_with_profile(
          profile_name: str,
          pipeline_fn: Callable,
          *args, **kwargs
      ) -> ProfiledResult:
        Ejecuta pipeline_fn bajo las restricciones del perfil.

        Implementacion:
        1. Leer perfil de PROFILES[profile_name]
        2. Configurar limites de CPU y RAM via resource/cgroups
        3. Iniciar monitorizacion con psutil (thread separado)
        4. Ejecutar pipeline_fn(*args, **kwargs)
        5. Recoger metricas: tiempo, RAM pico, CPU medio
        6. Retornar resultado + metricas

    - run_all_profiles(
          pipeline_fn: Callable,
          *args, **kwargs
      ) -> dict[str, ProfiledResult]:
        Ejecuta el pipeline con TODOS los perfiles secuencialmente.
        Retorna dict con resultados por perfil.

    - compare_profiles(results: dict[str, ProfiledResult]) -> ComparisonReport:
        Genera tabla comparativa y detecta:
        - Punto de inflexion de degradacion
        - Perfiles donde el pipeline falla (OOM, timeout)
        - Trade-off precision vs recursos

    ProfiledResult:
        - profile: ConstraintProfile
        - success: bool
        - error: str | None  # Si fallo: "OOM", "timeout", "error"
        - inference_ms: float | None
        - peak_ram_mb: float | None
        - cpu_percent: float | None
        - num_detections: int | None
        - avg_confidence: float | None
        - detections: list[Detection] | None
    """
```

### 9.5 Medicion de recursos con psutil

```python
# src/profiles/metrics_collector.py

"""
Recolector de metricas de recursos durante la inferencia.

Usa psutil para monitorizar el proceso del pipeline en un thread
separado, muestreando cada 100ms.
"""

class ResourceCollector:
    """
    Constructor:
    - sample_interval_ms: int = 100
    - pid: int | None = None  # Si None, usa el proceso actual

    Metodos:
    - start() -> None: Inicia recoleccion en thread separado
    - stop() -> ResourceMetrics: Para recoleccion, retorna metricas

    ResourceMetrics:
        - duration_ms: float
        - peak_ram_mb: float
        - avg_ram_mb: float
        - peak_cpu_percent: float
        - avg_cpu_percent: float
        - samples: int  # Numero de muestras tomadas
        - ram_timeline: list[float]  # Serie temporal de RAM
        - cpu_timeline: list[float]  # Serie temporal de CPU
    """
```

---

## 10. Modulo 6: Tip & Cue — Tasking Inteligente

### 10.1 Agente responsable: `AGENT-TIPCUE`

### 10.2 Contexto AIDRA

*"intelligence-driven satellite tasking capabilities (e.g. Tip & Cue). Proposals incorporating such capabilities will be positively evaluated but this is not a mandatory requirement."*

Tip & Cue = si el pipeline detecta algo interesante, reprograma automaticamente una nueva observacion de esa zona.

### 10.3 Logica del flujo

```
PASADA 1 (programada por APScheduler):
  → Descarga imagen Sentinel-1 de zona amplia (ej: Estrecho de Gibraltar)
  → Pipeline detecta N barcos

TIP (evaluacion):
  → TipEvaluator recibe detecciones
  → Evalua reglas:
     a) Deteccion en zona de interes (definida en zones.py)?
     b) Confianza > umbral (ej: 0.7)?
     c) Numero de barcos > minimo (ej: 2)?
  → Si cumple reglas: crea entrada en tasking_queue con status='pending'

CUE (reprogramacion):
  → CueScheduler consulta tasking_queue cada N minutos
  → Para cada cue pendiente:
     a) Calcula subzona centrada en las detecciones (bbox reducido)
     b) Busca imagen Sentinel-1 mas reciente de esa subzona
     c) Programa ejecucion del pipeline con prioridad alta
  → Registra en execution_log con trigger_type='cue' y triggered_by=ID del tip

PASADA 2 (triggered por Cue):
  → Descarga imagen de subzona (mas reciente)
  → Confirma o descarta las detecciones del tip
  → Vincula resultado al tip original via execution_id
```

### 10.4 Archivo `src/tipcue/scheduler.py`

```python
"""
CueScheduler: gestiona la cola de cues y su ejecucion.

Este modulo es complementario al APScheduler definido en la seccion 22.
El CueScheduler se encarga de la logica de negocio de los cues,
mientras que APScheduler se encarga de la programacion temporal.
"""

class CueScheduler:
    """
    Constructor:
    - db: Database
    - engine: PipelineEngine (referencia circular, inyectar despues de crear engine)

    Metodos:
    - async process_pending(max_cues: int = 5) -> list[CueResult]:
        1. Consultar tasking_queue WHERE status='pending' ORDER BY priority DESC
        2. Para cada cue:
           a. Buscar imagen mas reciente en target_bbox via Copernicus
           b. Crear PipelineRequest con trigger_type='cue', triggered_by=cue.triggered_by
           c. Ejecutar pipeline
           d. Comparar detecciones con las del tip original
           e. Actualizar cue con resultado
        3. Retornar resultados

    - async create_cue(
          triggered_by: UUID,
          target_bbox: list[float],
          priority: int = 1,
          reason: str = "auto",
          zone: str | None = None
      ) -> UUID:
        Inserta cue en tasking_queue. Retorna ID del cue.
        Verifica cooldown: no crear cue si ya hay uno pendiente
        en la misma zona hace menos de cooldown_minutes.

    - async get_queue(status: str | None = None, limit: int = 50) -> list[TaskingEntry]:
        Consulta la cola de cues.

    CueResult:
        - cue_id: UUID
        - execution_id: UUID | None
        - status: str  # "completed", "no_image_found", "error"
        - confirmed_detections: int | None
        - original_detections: int | None
    """
```

### 10.5 Archivo `src/tipcue/evaluator.py`

```python
"""
Evaluador de Tips: decide si una deteccion genera un Cue.
"""

class TipEvaluator:
    """
    Constructor:
    - zones_of_interest: list[Zone]  # Zonas definidas en zones.py
    - min_confidence: float = 0.7
    - min_detections: int = 2
    - cooldown_minutes: int = 60  # No generar cue si ya hay uno reciente en la misma zona

    Metodos:
    - evaluate(
          detections: list[Detection],
          execution_id: UUID
      ) -> list[TipResult]:
        Evalua las detecciones y genera tips si cumplen las reglas.

    TipResult:
        - should_cue: bool
        - reason: str  # "high_confidence_in_interest_zone", "cluster_detected", etc.
        - target_bbox: list[float]  # Subzona para el cue [lon_min, lat_min, lon_max, lat_max]
        - priority: int  # 0=normal, 1=alta, 2=urgente
        - triggering_detections: list[UUID]  # IDs de las detecciones que generaron el tip
        - execution_id: UUID  # Ejecucion original
    """
```

### 10.5 Archivo `src/tipcue/zones.py`

```python
"""
Definicion de zonas de interes para Tip & Cue.

Las zonas de interes son areas geograficas donde las detecciones
tienen mayor relevancia y pueden generar cues automaticos.
"""

class Zone:
    """
    Pydantic model.

    Campos:
    - id: str
    - name: str
    - bbox: list[float]  # [lon_min, lat_min, lon_max, lat_max]
    - geometry: dict  # GeoJSON Polygon
    - priority: int  # 0=normal, 1=alta
    - description: str
    - active: bool = True
    """

DEFAULT_ZONES = [
    Zone(
        id="gibraltar_strait",
        name="Estrecho de Gibraltar - Zona de transito",
        bbox=[-5.6, 35.8, -5.3, 36.1],
        priority=1,
        description="Punto de paso obligatorio entre Atlantico y Mediterraneo"
    ),
    Zone(
        id="algeciras_port",
        name="Puerto de Algeciras - Zona de fondeo",
        bbox=[-5.5, 36.05, -5.35, 36.15],
        priority=1,
        description="Zona de fondeo del puerto de Algeciras, alta densidad"
    ),
    Zone(
        id="med_patrol",
        name="Patrulla Mediterraneo Central",
        bbox=[10.0, 33.0, 16.0, 38.0],
        priority=0,
        description="Ruta migratoria y de trafico, Tunez-Sicilia"
    ),
]
```

---

## 11A. Modulo 9: Perfil Energetico

### 11A.1 Agente responsable: `AGENT-ORBITAL`

### 11A.2 Contexto

En orbita, la energia es el recurso mas escaso. Un satelite tipico asigna 5-50W al payload de procesamiento. Una bateria de CubeSat 3U tiene ~20-40 Wh. Si la inferencia consume demasiada energia, no queda para el downlink o los sistemas de control.

Ningun MVP terrestre mide esto. Hacerlo nos diferencia de cualquier otro licitador.

### 11A.3 Que medimos

No podemos medir vatios reales en OCI (no tenemos acceso al hardware), pero podemos **estimar** con alta precision:

1. **Energia por inferencia (Joules)**: `CPU_time(s) × TDP_estimado(W)` por perfil
2. **TOPS/W**: operaciones del modelo (FLOPs) / energia estimada
3. **Energia por imagen completa**: incluyendo descarga, preproceso, inferencia, postproceso
4. **Presupuesto energetico orbital**: "con una bateria de X Wh, puedes procesar Y imagenes por orbita"
5. **Comparacion con procesadores de vuelo**: tabla de referencia con hardware real publicado

### 11A.4 Archivo `src/orbital/energy.py`

```python
"""
Perfil energetico del pipeline.

Estimacion de consumo energetico basada en tiempo de CPU y
TDP (Thermal Design Power) de referencia para procesadores
de vuelo conocidos.

La estimacion no es exacta (no medimos vatios reales), pero
permite comparar variantes de modelo y perfiles, y extrapolar
a hardware de vuelo especifico.
"""

# TDP de referencia para procesadores usados en espacio / edge
PROCESSOR_TDP_WATTS = {
    "oci_arm_a1": 3.0,           # OCI Ampere A1: ~3W por core (estimado)
    "xilinx_zynq_ultrascale": 5.0,  # Xilinx Zynq UltraScale+ ZU9EG (flight-proven)
    "intel_myriad_x": 1.5,      # Intel Movidius Myriad X (usado en PhiSat-1)
    "google_coral_tpu": 2.0,    # Google Coral Edge TPU
    "nvidia_jetson_nano": 5.0,  # NVIDIA Jetson Nano (no flight-proven, referencia)
    "raspberry_pi4_arm": 3.5,   # RPi4 ARM Cortex-A72 (referencia CubeSat)
    "leon3_gr740": 1.5,         # LEON3 GR740 (ESA radiation-hardened)
    "unibap_ix10": 15.0,        # Unibap iX10 (flight-proven, potente)
}

# Presupuestos energeticos tipicos por tipo de satelite
SATELLITE_POWER_BUDGETS = {
    "cubesat_3u": {
        "total_w": 6.0,
        "payload_w": 2.0,
        "battery_wh": 30.0,
        "orbit_period_min": 95,
        "sunlit_fraction": 0.6,
        "description": "CubeSat 3U (ej: PhiSat-1)"
    },
    "cubesat_6u": {
        "total_w": 15.0,
        "payload_w": 5.0,
        "battery_wh": 60.0,
        "orbit_period_min": 95,
        "sunlit_fraction": 0.6,
        "description": "CubeSat 6U"
    },
    "small_sat": {
        "total_w": 80.0,
        "payload_w": 30.0,
        "battery_wh": 300.0,
        "orbit_period_min": 100,
        "sunlit_fraction": 0.6,
        "description": "Small satellite (100-500 kg)"
    },
    "medium_sat": {
        "total_w": 300.0,
        "payload_w": 100.0,
        "battery_wh": 2000.0,
        "orbit_period_min": 100,
        "sunlit_fraction": 0.6,
        "description": "Medium satellite (ej: Sentinel-1 class)"
    },
}

class EnergyProfiler:
    """
    Constructor:
    - reference_processor: str = "oci_arm_a1"

    Metodos:
    - estimate_inference_energy(
          cpu_time_seconds: float,
          cpu_cores_used: float,
          processor: str = "oci_arm_a1"
      ) -> EnergyEstimate:
        Estima energia de inferencia.
        Formula: energy_joules = cpu_time_s * cpu_cores * tdp_watts
        Ajuste: multiplicar por cpu_utilization (si CPU al 50%, usa 50% del TDP)

    - estimate_pipeline_energy(
          execution_record: ExecutionRecord,
          processor: str = "oci_arm_a1"
      ) -> PipelineEnergyEstimate:
        Estima energia de todo el pipeline (download + preprocess + inference + postprocess).
        Download se asume negligible (en orbita no hay download, el sensor esta a bordo).
        Preprocessing es mayormente CPU-bound.
        Postprocessing es ligero.

    - calculate_tops_per_watt(
          model_flops: int,        # FLOPs del modelo (ultralytics los reporta)
          inference_seconds: float,
          processor: str = "oci_arm_a1"
      ) -> float:
        TOPS/W = (FLOPs / inference_time) / TDP
        Metrica estandar de eficiencia energetica en edge AI.

    - calculate_orbital_budget(
          energy_per_image_joules: float,
          satellite_type: str = "cubesat_6u",
          images_per_orbit: int = 10
      ) -> OrbitalBudgetResult:
        Calcula si el presupuesto energetico permite procesar N imagenes por orbita.

    - extrapolate_to_processor(
          measured_on: str,         # Procesador donde se midio (ej: "oci_arm_a1")
          target_processor: str,    # Procesador objetivo (ej: "xilinx_zynq_ultrascale")
          measured_time_s: float,
          measured_cpu_cores: float
      ) -> EnergyEstimate:
        Extrapola a otro procesador escalando por TDP.
        NOTA: Es una estimacion grosera — el rendimiento real depende
        de la arquitectura. Pero da un orden de magnitud.

    - compare_all_processors(
          cpu_time_seconds: float,
          cpu_cores_used: float
      ) -> list[EnergyEstimate]:
        Genera tabla comparativa para todos los procesadores de referencia.
    """

class EnergyEstimate:
    """
    Pydantic model.
    - processor: str
    - tdp_watts: float
    - cpu_time_seconds: float
    - cpu_cores: float
    - energy_joules: float
    - energy_wh: float               # joules / 3600
    - equivalent_battery_percent: float  # % de bateria tipica del sat
    """

class PipelineEnergyEstimate:
    """
    - preprocessing_joules: float
    - inference_joules: float
    - postprocessing_joules: float
    - total_joules: float
    - total_wh: float
    - breakdown_percent: dict         # {"preprocessing": 15, "inference": 80, "post": 5}
    """

class OrbitalBudgetResult:
    """
    - satellite_type: str
    - battery_wh: float
    - payload_power_w: float
    - energy_per_image_wh: float
    - max_images_per_orbit: int       # Floor(available_wh / energy_per_image)
    - available_energy_wh: float      # payload_w * orbit_period_hours * sunlit_fraction
    - utilization_percent: float      # (images * energy) / available * 100
    - feasible: bool                  # True si max_images >= 1
    - notes: str
    """
```

### 11A.5 Metricas Prometheus adicionales

```python
# Anadir a src/observability/prometheus_metrics.py

ENERGY_JOULES = Gauge(
    'aidra_energy_joules',
    'Estimated energy consumption per inference',
    ['profile', 'model_variant', 'processor']
)

TOPS_PER_WATT = Gauge(
    'aidra_tops_per_watt',
    'Tera operations per second per watt',
    ['model_variant', 'processor']
)

IMAGES_PER_ORBIT = Gauge(
    'aidra_images_per_orbit',
    'Max images processable per orbit with current model/profile',
    ['model_variant', 'profile', 'satellite_type']
)
```

---

## 11B. Modulo 10: Analisis de Downlink

### 11B.1 Contexto

La razon economica de OBDP: un satelite en LEO tiene una ventana de contacto con la estacion terrena de 5-10 minutos por pasada. El ancho de banda tipico es 100-800 Mbps. Una imagen SAR pesa 500 MB - 1 GB.

**Sin OBDP**: el satelite baja la imagen cruda completa. Con las ventanas disponibles, puede bajar pocas imagenes por dia.

**Con OBDP**: el satelite procesa a bordo, baja solo las detecciones (JSON, ~10 KB) + metadatos + thumbnails de las zonas con barcos. El ratio de reduccion de datos puede ser muy alto (ordenes de magnitud), pero debe calcularse y justificarse con resultados medidos del MVP.

Esto transforma el argumento de "podemos hacer IA en hardware limitado" en "OBDP puede multiplicar significativamente la capacidad efectiva del satelite", sujeto a validacion empirica.

### 11B.2 Archivo `src/orbital/downlink.py`

```python
"""
Analisis de downlink: cuantifica el ahorro de ancho de banda
que OBDP proporciona respecto al downlink de imagenes crudas.
"""

# Parametros de downlink tipicos
DOWNLINK_PROFILES = {
    "cubesat_uhf": {
        "name": "CubeSat UHF",
        "bandwidth_mbps": 0.009,  # 9.6 kbps
        "window_minutes": 8,
        "passes_per_day": 4,
        "description": "CubeSat con radio UHF basica"
    },
    "cubesat_sband": {
        "name": "CubeSat S-Band",
        "bandwidth_mbps": 2.0,
        "window_minutes": 8,
        "passes_per_day": 4,
        "description": "CubeSat con S-Band"
    },
    "smallsat_xband": {
        "name": "SmallSat X-Band",
        "bandwidth_mbps": 100.0,
        "window_minutes": 10,
        "passes_per_day": 6,
        "description": "SmallSat con X-Band (ej: Sentinel-1 class)"
    },
    "highcap_ka": {
        "name": "High-Capacity Ka-Band",
        "bandwidth_mbps": 800.0,
        "window_minutes": 10,
        "passes_per_day": 8,
        "description": "Satelite grande con Ka-Band + red EDRS"
    },
}

class DownlinkAnalyzer:
    """
    Constructor: ninguno (stateless, solo calculos)

    Metodos:
    - analyze_single_image(
          image_size_mb: float,       # Tamano de la imagen SAR cruda
          result_size_kb: float,      # Tamano del resultado procesado (JSON detecciones)
          thumbnail_size_kb: float = 50.0,  # Thumbnail opcional de la zona con barcos
          metadata_size_kb: float = 5.0,    # Metadatos de ejecucion
          downlink_profile: str = "smallsat_xband"
      ) -> DownlinkAnalysis:
        Compara downlink con vs sin OBDP para una imagen.

    - analyze_daily_capacity(
          images_per_day: int,
          image_size_mb: float,
          result_size_kb: float,
          downlink_profile: str = "smallsat_xband"
      ) -> DailyCapacityAnalysis:
        Cuantas imagenes puede bajar el satelite por dia con vs sin OBDP.

    - analyze_all_profiles(
          image_size_mb: float,
          result_size_kb: float
      ) -> list[DownlinkAnalysis]:
        Genera tabla comparativa para todos los perfiles de downlink.

    - generate_obdp_value_report(
          execution_records: list[ExecutionRecord]
      ) -> OBDPValueReport:
        Genera informe de valor usando datos reales de ejecuciones pasadas.
        Calcula: ratio medio de compresion, ahorro de BW, capacidad extra.
    """

class DownlinkAnalysis:
    """
    - downlink_profile: str
    - image_size_mb: float
    - result_size_kb: float

    # Sin OBDP
    - raw_downlink_seconds: float     # Tiempo para bajar imagen cruda
    - raw_images_per_window: float    # Imagenes crudas por ventana de contacto
    - raw_images_per_day: float       # Imagenes crudas por dia

    # Con OBDP
    - obdp_downlink_seconds: float    # Tiempo para bajar resultado procesado
    - obdp_results_per_window: float  # Resultados por ventana
    - obdp_results_per_day: float     # Resultados por dia

    # Ratios
    - compression_ratio: float        # image_size / result_size
    - bandwidth_saving_percent: float  # (1 - result/image) * 100
    - capacity_multiplier: float       # obdp_per_day / raw_per_day
    - time_saving_percent: float       # (1 - obdp_time/raw_time) * 100
    """

class OBDPValueReport:
    """
    Informe ejecutivo de valor OBDP.
    - avg_compression_ratio: float
    - avg_image_size_mb: float
    - avg_result_size_kb: float
    - total_images_analyzed: int
    - total_bandwidth_saved_gb: float  # Si todas se hubieran bajado crudas
    - equivalent_extra_capacity: str   # "Con OBDP, un CubeSat S-Band equivale a un SmallSat X-Band"
    - recommendations: list[str]
    """
```

### 11B.3 Dashboard Grafana adicional

**Dashboard 6: Valor OBDP** (`06-obdp-value.json`)
- Gauge grande: "Ratio de compresion de datos: 80.000:1"
- Comparativa barras: downlink time sin vs con OBDP por tipo de satelite
- Tabla: capacidad diaria (imagenes/dia) por perfil de downlink
- Linea temporal: ratio de compresion por ejecucion
- KPI: "Total bandwidth saved: X GB"

Query SQL:
```sql
SELECT
    e.image_size_mb,
    (LENGTH(
        json_build_object(
            'detections', e.num_detections,
            'avg_confidence', e.avg_confidence,
            'execution_id', e.id
        )::text
    ) / 1024.0) AS result_size_kb,
    e.image_size_mb * 1024 / NULLIF(
        LENGTH(json_build_object(
            'detections', e.num_detections
        )::text) / 1024.0, 0
    ) AS compression_ratio,
    e.created_at
FROM execution_log e
WHERE e.status = 'success'
    AND e.image_size_mb IS NOT NULL
ORDER BY e.created_at DESC
```

---

## 11C. Modulo 11: Latencia Orbital

### 11C.1 Contexto

La latencia total "sensor captura → resultado disponible en tierra" determina el valor operativo de OBDP para escenarios de seguridad/defensa. En vigilancia maritima, un barco se mueve ~20 nudos; en 1 hora recorre ~37 km. La latencia determina si la deteccion es accionable.

### 11C.2 Archivo `src/orbital/latency.py`

```python
"""
Simulacion de latencia orbital end-to-end.

Modela el tiempo total desde que el sensor SAR captura la imagen
hasta que el resultado esta disponible en tierra, bajo diferentes
escenarios (con/sin OBDP, diferentes orbitas, diferentes estaciones).
"""

# Parametros orbitales comunes
ORBIT_PARAMS = {
    "leo_500": {
        "altitude_km": 500,
        "period_min": 94.6,
        "velocity_km_s": 7.6,
        "ground_track_km_s": 6.9,
        "max_contact_min": 10,
        "avg_revisit_hours": 12,
        "description": "LEO 500 km (tipica EO)"
    },
    "sso_700": {
        "altitude_km": 700,
        "period_min": 98.8,
        "velocity_km_s": 7.5,
        "ground_track_km_s": 6.8,
        "max_contact_min": 12,
        "avg_revisit_hours": 6,
        "description": "SSO 700 km (Sentinel-1)"
    },
    "leo_350_isstyle": {
        "altitude_km": 350,
        "period_min": 91.4,
        "velocity_km_s": 7.7,
        "ground_track_km_s": 7.1,
        "max_contact_min": 7,
        "avg_revisit_hours": 24,
        "description": "LEO baja 350 km (ISS-like)"
    },
}

# Escenarios de cadena de tierra
GROUND_PROCESSING = {
    "fast_automated": {
        "ingest_minutes": 5,
        "processing_minutes": 10,
        "dissemination_minutes": 2,
        "description": "Cadena automatizada rapida"
    },
    "standard_nrt": {
        "ingest_minutes": 15,
        "processing_minutes": 30,
        "dissemination_minutes": 10,
        "description": "Near Real-Time estandar (ESA NRT)"
    },
    "manual_analysis": {
        "ingest_minutes": 30,
        "processing_minutes": 120,
        "dissemination_minutes": 30,
        "description": "Analisis con intervencion humana"
    },
}

class OrbitalLatencySimulator:
    """
    Metodos:
    - simulate_without_obdp(
          orbit: str = "sso_700",
          ground_chain: str = "standard_nrt",
          image_size_mb: float = 800,
          downlink_profile: str = "smallsat_xband"
      ) -> LatencyBreakdown:
        Calcula latencia sin OBDP:
        1. Captura del sensor: 0 (referencia temporal)
        2. Almacenamiento a bordo: ~0s
        3. Espera hasta ventana de contacto: 0 - max(period/2) minutos
           (promedio: period / 4 para LEO con estaciones globales)
        4. Downlink de imagen cruda: image_size / bandwidth
        5. Ingestion en tierra: ground_chain.ingest_minutes
        6. Procesamiento en tierra: ground_chain.processing_minutes
        7. Diseminacion del resultado: ground_chain.dissemination_minutes
        TOTAL = sum(2..7)

    - simulate_with_obdp(
          orbit: str = "sso_700",
          inference_ms: float = 150,
          result_size_kb: float = 10,
          downlink_profile: str = "smallsat_xband"
      ) -> LatencyBreakdown:
        Calcula latencia con OBDP:
        1. Captura del sensor: 0
        2. Procesamiento a bordo: inference_ms (< 1 segundo)
        3. Espera hasta ventana de contacto: mismo que arriba
        4. Downlink del resultado: result_size / bandwidth (~instantaneo)
        5. Ingestion en tierra: ~1 min (resultado pequeno)
        6. Diseminacion: ~1 min
        TOTAL = sum(2..6)

        NOTA CLAVE: con Tip & Cue, el paso 3 puede ser 0 si el satelite
        tiene enlace inter-satelital (ISL) o relay (EDRS).

    - compare_scenarios(
          inference_ms: float,
          image_size_mb: float,
          result_size_kb: float
      ) -> list[LatencyComparison]:
        Genera tabla comparativa: todas las combinaciones de
        orbita x ground_chain x downlink, con vs sin OBDP.

    - calculate_actionability(
          latency_minutes: float,
          vessel_speed_knots: float = 20
      ) -> ActionabilityResult:
        Calcula la "utilidad" de la deteccion:
        - Distance_moved_km = speed * latency_hours * 1.852
        - Si el barco se ha movido > 50 km: deteccion poco accionable
        - Si < 10 km: altamente accionable
        Metrica clave para defensa/seguridad maritima.
    """

class LatencyBreakdown:
    """
    - scenario: str                    # "with_obdp" o "without_obdp"
    - orbit: str
    - capture_s: float = 0
    - onboard_processing_s: float      # 0 sin OBDP, inference_ms con OBDP
    - wait_for_contact_s: float        # Mayor componente de latencia
    - downlink_s: float                # Drasticamente diferente con/sin OBDP
    - ground_ingest_s: float
    - ground_processing_s: float       # 0 con OBDP (ya procesado a bordo)
    - dissemination_s: float
    - total_seconds: float
    - total_minutes: float
    """

class LatencyComparison:
    """
    - orbit: str
    - downlink_profile: str
    - without_obdp_minutes: float
    - with_obdp_minutes: float
    - speedup_factor: float            # without / with
    - time_saved_minutes: float
    - actionability_without: str       # "low", "medium", "high"
    - actionability_with: str
    """

class ActionabilityResult:
    """
    - latency_minutes: float
    - vessel_speed_knots: float
    - distance_moved_km: float
    - actionability: str               # "high" (<10km), "medium" (10-50km), "low" (>50km)
    - search_radius_km: float          # Radio de busqueda necesario para re-encontrar el barco
    - notes: str
    """
```

### 11C.3 Dashboard Grafana adicional

**Dashboard 7: Latencia Orbital** (`07-orbital-latency.json`)
- Barra apilada: desglose de latencia (onboard, wait, downlink, ground) con vs sin OBDP
- Tabla: comparativa por orbita x downlink x escenario
- Gauge: "Speedup factor: 8.5x"
- Indicador de accionabilidad: "Con OBDP: barco se mueve 2 km. Sin OBDP: 37 km"

---

## 11D. Modulo 12: Resiliencia y Autonomia

### 11D.1 Contexto

En orbita no hay operador humano. El sistema debe funcionar autonomamente durante semanas/meses, tolerar fallos de hardware (radiacion), y degradar gracefully cuando los recursos son insuficientes.

Los pliegos mencionan: *"aligning autonomous decision chains with standard workflows"* y *"balancing technical flexibility with reproducibility and security requirements"*.

### 11D.2 Archivo `src/orbital/resilience.py`

```python
"""
Simulacion de resiliencia: que pasa cuando las cosas van mal en orbita.

Tres dimensiones:
1. Bit-flips (SEU): corrupcion de pesos del modelo por radiacion
2. Fallback graceful: cambio automatico de modelo cuando los recursos no alcanzan
3. Drift detection: detectar si el modelo produce resultados anomalos
"""

import numpy as np
from copy import deepcopy

class BitFlipSimulator:
    """
    Simula Single Event Upsets (SEU) en los pesos del modelo.

    En orbita LEO, la tasa de SEU es ~1e-7 a 1e-5 bit-flips por bit por dia
    dependiendo de la orbita y blindaje.

    Constructor:
    - model_weights: dict[str, np.ndarray]  # Pesos del modelo (state_dict)

    Metodos:
    - inject_bitflips(
          num_flips: int = 1,
          target_layers: list[str] | None = None,  # None = aleatorio
          bit_position: str = "random"  # "random", "msb" (mas destructivo), "lsb" (menos)
      ) -> tuple[dict, list[BitFlipRecord]]:
        Inyecta num_flips bit-flips aleatorios en los pesos.
        Retorna (pesos_corruptos, registro_de_flips).

        Implementacion:
        1. Seleccionar tensor aleatorio (o del layer especificado)
        2. Seleccionar indice aleatorio en el tensor
        3. Obtener representacion binaria del peso (float32 = 32 bits)
        4. Flipear 1 bit (aleatorio o especifico)
        5. Reconstruir float32

    - sweep_bitflips(
          image: np.ndarray,
          model,
          flip_counts: list[int] = [0, 1, 5, 10, 50, 100, 500, 1000],
          runs_per_count: int = 5
      ) -> BitFlipSweepResult:
        Para cada numero de flips, inyecta y ejecuta inferencia N veces.
        Mide: num_detections, avg_confidence, precision_vs_baseline.
        Encuentra el punto de inflexion: "con >X bit-flips, las detecciones
        se degradan un >20%".

    - estimate_mtbf(
          orbit: str = "leo_500",
          model_size_bytes: int = 6_000_000,
          shielding_mm_al: float = 1.0  # mm de aluminio de blindaje
      ) -> MTBFEstimate:
        Estima Mean Time Between Failures (bit-flip que causa degradacion >20%).
        Basado en tasa SEU tipica para la orbita y blindaje.
    """

class BitFlipRecord:
    """
    - layer_name: str
    - tensor_index: tuple
    - original_value: float
    - corrupted_value: float
    - bit_position: int  # 0-31 para float32
    - bit_significance: str  # "sign", "exponent", "mantissa"
    """

class BitFlipSweepResult:
    """
    - baseline_detections: int
    - baseline_confidence: float
    - results: list[dict]  # {num_flips, avg_detections, avg_confidence, std_detections, degradation_pct}
    - critical_threshold: int  # Numero de flips donde degradacion > 20%
    - model_name: str
    - model_size_bytes: int
    """

class MTBFEstimate:
    """
    - orbit: str
    - model_size_bits: int
    - seu_rate_per_bit_per_day: float
    - expected_flips_per_day: float
    - expected_flips_per_orbit: float
    - critical_threshold: int  # De BitFlipSweepResult
    - estimated_mtbf_days: float  # Dias hasta alcanzar critical_threshold
    - mitigation_recommendations: list[str]
    """
```

### 11D.3 Archivo `src/orbital/decision_engine.py`

```python
"""
Motor de decision autonomo.

En orbita, el sistema debe decidir SOLO:
1. Que modelo usar (segun recursos disponibles)
2. Si procesar o no (segun energia/bateria)
3. Que hacer si falla (fallback)
4. Si la salida es sospechosa (drift detection)

Este modulo simula ese comportamiento de decision.
"""

class DecisionEngine:
    """
    Constructor:
    - models: list[ModelInfo]         # Modelos disponibles, ordenados por tamano
    - profiles: dict[str, ConstraintProfile]
    - energy_profiler: EnergyProfiler
    - config: DecisionConfig

    Metodos:
    - decide_model(
          available_cpu: float,       # CPUs disponibles
          available_ram_mb: int,      # RAM disponible
          available_energy_wh: float, # Energia restante en bateria
          priority: int = 0           # 0=normal, 1=alta, 2=urgente
      ) -> DecisionResult:
        Logica de decision:
        1. Filtrar modelos que caben en RAM disponible
        2. De esos, filtrar por energia (descartar si no alcanza para 1 inferencia)
        3. Si priority >= 2 (urgente): elegir el mas rapido (menor latencia)
        4. Si priority == 1 (alta): elegir el mas preciso que quepa
        5. Si priority == 0 (normal): elegir el que maximice precision/energia
        6. Si ningun modelo cabe: retornar fallback="cfar" (sin IA, solo procesamiento senal)
        7. Si ni CFAR cabe: retornar action="skip" (no procesar, guardar para downlink)

    - detect_drift(
          recent_executions: list[ExecutionRecord],
          window_size: int = 10
      ) -> DriftResult:
        Detecta anomalias en las ultimas ejecuciones:
        - Numero de detecciones muy diferente de la media historica
        - Confianza media anormalmente baja
        - Detecciones en zonas donde nunca hubo barcos (potencial corrupcion)

        Metodo: Z-score sobre las ultimas `window_size` ejecuciones vs historico.
        Si Z > 3: alerta de drift.

    - simulate_orbit_sequence(
          num_images: int = 20,
          initial_battery_wh: float = 60,
          solar_recharge_w: float = 5.0,
          orbit_period_min: float = 95,
          image_interval_min: float = 10
      ) -> OrbitSimulationResult:
        Simula una secuencia de decisiones durante una orbita completa:
        - Cada 10 min: el sensor captura una imagen
        - El Decision Engine decide: procesar (y con que modelo) o saltar
        - La bateria se drena con cada procesamiento y se recarga con el sol
        - Resultado: cuantas imagenes proceso, cuantas salto, estado de bateria
    """

class DecisionResult:
    """
    - action: str                    # "process", "fallback_cfar", "skip"
    - selected_model: str | None     # Nombre del modelo elegido
    - selected_profile: str | None   # Perfil de restriccion
    - reason: str                    # Explicacion de la decision
    - estimated_energy_wh: float     # Energia estimada para esta decision
    - estimated_latency_ms: float    # Latencia estimada
    - confidence_estimate: str       # "high", "medium", "low"
    """

class DriftResult:
    """
    - is_drifting: bool
    - metric: str                    # "num_detections", "avg_confidence", "spatial"
    - z_score: float
    - recent_mean: float
    - historical_mean: float
    - recommendation: str            # "continue", "recalibrate", "switch_model", "alert_ground"
    """

class OrbitSimulationResult:
    """
    - total_images: int
    - processed_images: int
    - skipped_images: int
    - cfar_fallback_count: int
    - models_used: dict[str, int]    # Conteo por modelo
    - battery_timeline: list[float]  # Bateria en cada paso (Wh)
    - decisions: list[DecisionResult]
    - final_battery_wh: float
    - energy_efficiency: float       # images_processed / total_energy_consumed
    """

class DecisionConfig:
    """
    Pydantic model.
    - min_battery_reserve_pct: float = 20.0   # No procesar si bateria < 20%
    - prefer_precision_over_speed: bool = True
    - enable_cfar_fallback: bool = True
    - drift_detection_enabled: bool = True
    - drift_z_threshold: float = 3.0
    - max_consecutive_skips: int = 5  # Alertar si se saltan >5 imagenes seguidas
    """
```

### 11D.4 Dashboard Grafana adicional

**Dashboard 8: Resiliencia Orbital** (`08-orbital-resilience.json`)

- Grafico de linea: degradacion por bit-flips (X=num_flips, Y=detecciones/confianza)
- Marcador: "Critical threshold: X bit-flips"
- Simulacion de orbita: timeline de bateria + decisiones (grafico apilado)
- Indicadores de drift: Z-score de las ultimas ejecuciones
- Tabla de decisiones: modelo elegido, razon, energia consumida

### 11D.5 Metricas Prometheus adicionales

```python
# Anadir a src/observability/prometheus_metrics.py

DECISION_ACTION = Counter(
    'aidra_decision_action_total',
    'Autonomous decisions made',
    ['action']  # "process", "fallback_cfar", "skip"
)

DRIFT_ALERTS = Counter(
    'aidra_drift_alerts_total',
    'Drift detection alerts',
    ['metric']  # "num_detections", "avg_confidence", "spatial"
)

BITFLIP_DEGRADATION = Gauge(
    'aidra_bitflip_degradation_pct',
    'Detection degradation under simulated bit-flips',
    ['num_flips', 'model_variant']
)

BATTERY_LEVEL_WH = Gauge(
    'aidra_sim_battery_wh',
    'Simulated battery level during orbit simulation'
)
```

---

## 11. Modulo 7: Observabilidad (Grafana + Prometheus + Loki)

### 11.1 Agente responsable: `AGENT-OBSERVABILITY`

### 11.2 Metricas Prometheus

```python
# src/observability/prometheus_metrics.py

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    generate_latest, CONTENT_TYPE_LATEST
)

# ---- Contadores ----
PIPELINE_RUNS_TOTAL = Counter(
    'aidra_pipeline_runs_total',
    'Total pipeline executions',
    ['profile', 'model_variant', 'status']  # status: success, error, timeout, oom
)

DETECTIONS_TOTAL = Counter(
    'aidra_detections_total',
    'Total detections across all runs',
    ['source', 'profile']  # source: cfar, yolo, fused
)

IMAGES_DOWNLOADED_TOTAL = Counter(
    'aidra_images_downloaded_total',
    'Total satellite images downloaded'
)

TIPS_GENERATED_TOTAL = Counter(
    'aidra_tips_generated_total',
    'Total tips generated by Tip & Cue evaluator'
)

CUES_EXECUTED_TOTAL = Counter(
    'aidra_cues_executed_total',
    'Total cues executed',
    ['status']  # confirmed, discarded
)

# ---- Histogramas ----
INFERENCE_DURATION = Histogram(
    'aidra_inference_duration_seconds',
    'Inference time per pipeline run',
    ['profile', 'model_variant'],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

DOWNLOAD_DURATION = Histogram(
    'aidra_download_duration_seconds',
    'Satellite image download time',
    buckets=[10, 30, 60, 120, 300, 600]
)

# ---- Gauges ----
PEAK_RAM_MB = Gauge(
    'aidra_peak_ram_mb',
    'Peak RAM usage during last inference',
    ['profile', 'model_variant']
)

CPU_USAGE_PERCENT = Gauge(
    'aidra_cpu_usage_percent',
    'Average CPU usage during last inference',
    ['profile', 'model_variant']
)

MODEL_SIZE_MB = Gauge(
    'aidra_model_size_mb',
    'Model file size in MB',
    ['model_variant']
)

ACTIVE_CUES = Gauge(
    'aidra_active_cues',
    'Number of pending cues in tasking queue'
)

# ---- Info ----
SYSTEM_INFO = Info(
    'aidra_system',
    'System information'
)
```

### 11.3 Dashboards Grafana

Cada dashboard se define como archivo JSON en `grafana/dashboards/`. A continuacion las especificaciones de cada uno.

**Dashboard 1: Mapa de Detecciones** (`01-map-detections.json`)
- Panel principal: GeoMap con puntos de detecciones
- Puntos coloreados por confianza (verde > 0.8, amarillo > 0.5, rojo < 0.5)
- Capa adicional para tips (naranja) y cues (rojo) si Tip & Cue activo
- Filtros: rango temporal, perfil de restriccion, variante de modelo
- Query SQL:
```sql
SELECT
    ST_X(center_geo) AS longitude,
    ST_Y(center_geo) AS latitude,
    confidence,
    source,
    model_name,
    constraint_profile,
    timestamp
FROM detections d
JOIN execution_log e ON d.execution_id = e.id
WHERE e.timestamp BETWEEN $__timeFrom() AND $__timeTo()
    AND ($profile = 'all' OR e.constraint_profile = $profile)
```

**Dashboard 2: Metricas del Pipeline** (`02-pipeline-metrics.json`)
- Rate de ejecuciones por hora (Prometheus: `rate(aidra_pipeline_runs_total[1h])`)
- Tiempo de inferencia P50/P95/P99 (Prometheus: `histogram_quantile`)
- RAM pico por ejecucion (Prometheus: `aidra_peak_ram_mb`)
- Detecciones por ejecucion (Prometheus: `aidra_detections_total`)
- Errores del pipeline (Prometheus: `aidra_pipeline_runs_total{status="error"}`)
- Logs del pipeline (Loki: `{container="aidra-app"} |= "pipeline"`)

**Dashboard 3: Benchmarks de Compresion** (`03-compression-bench.json`)
- Tabla comparativa: modelo × tamano × latencia × RAM × precision
- Grafico de barras agrupado: tamano del modelo por variante
- Grafico de barras agrupado: latencia por variante
- Scatter plot: precision vs latencia (trade-off curve)
- Query SQL:
```sql
SELECT
    model_name,
    model_version,
    model_size_mb,
    AVG(inference_ms) AS avg_inference_ms,
    AVG(peak_ram_mb) AS avg_peak_ram,
    AVG(num_detections) AS avg_detections,
    AVG(avg_confidence) AS avg_confidence
FROM execution_log
WHERE constraint_profile = 'ground'
    AND status = 'success'
GROUP BY model_name, model_version, model_size_mb
ORDER BY model_size_mb
```

**Dashboard 4: Perfiles de Restriccion** (`04-constraint-profiles.json`)
- Heatmap: perfil × metrica (latencia, RAM, CPU, detecciones)
- Grafico lineal: degradacion progresiva por perfil (ground → sat-extreme)
- Semaforo: verde (OK), amarillo (degradado), rojo (fallo/OOM) por perfil
- Tabla: detalle por perfil con misma imagen/modelo
- Query SQL:
```sql
SELECT
    constraint_profile,
    AVG(inference_ms) AS avg_latency,
    AVG(peak_ram_mb) AS avg_ram,
    AVG(cpu_usage_pct) AS avg_cpu,
    AVG(num_detections) AS avg_detections,
    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successes,
    SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) AS failures
FROM execution_log
WHERE image_id = $image_id
    AND model_name = $model_name
GROUP BY constraint_profile
ORDER BY CASE constraint_profile
    WHEN 'ground' THEN 1
    WHEN 'sat-high' THEN 2
    WHEN 'sat-mid' THEN 3
    WHEN 'sat-low' THEN 4
    WHEN 'sat-extreme' THEN 5
END
```

**Dashboard 5: Traceability** (`05-traceability.json`)
- Tabla interactiva con execution_log completo
- Detalle expandible de cada ejecucion (click en fila)
- Columnas: timestamp, image_id, model_name, model_version, profile, detections, latency, status
- Panel de hashes: image_hash, model_hash, output_hash
- Filtros: fecha, modelo, perfil, status

---

## 12. Modulo 8: API REST (FastAPI)

### 12.1 Agente responsable: `AGENT-API`

### 12.2 Archivo `src/main.py`

```python
"""
Entrypoint de la aplicacion FastAPI.

Lifespan:
- startup: conectar a DB, cargar modelos, iniciar APScheduler
- shutdown: cerrar conexiones, parar scheduler

Middleware:
- CORS (origenes permitidos: localhost, IP de la instancia OCI)
- Request timing (header X-Process-Time)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    # 1. Crear pool de conexiones asyncpg
    # 2. Ejecutar migraciones pendientes
    # 3. Cargar modelos YOLO disponibles en models/
    # 4. Iniciar APScheduler con jobs configurados
    # 5. Registrar metricas de sistema en Prometheus
    yield
    # Shutdown
    # 1. Parar APScheduler
    # 2. Cerrar pool DB

app = FastAPI(
    title="AIDRA — AI-Enabled On-Board Data Processing Assessment",
    description="Pipeline de deteccion de barcos con IA en imagenes SAR",
    version="1.0.0",
    lifespan=lifespan
)

# Incluir routers
# app.include_router(health.router, prefix="/api")
# app.include_router(detections.router, prefix="/api")
# app.include_router(pipeline.router, prefix="/api")
# app.include_router(benchmarks.router, prefix="/api")
# app.include_router(traceability.router, prefix="/api")
# app.include_router(tasking.router, prefix="/api")
# app.include_router(metrics.router, prefix="/api")
```

### 12.3 Endpoints completos

| Metodo | Ruta | Descripcion | Request Body | Response |
|---|---|---|---|---|
| `GET` | `/api/health` | Estado del sistema | — | `{"status":"ok","db":"connected","models_loaded":3,"scheduler":"running"}` |
| `GET` | `/api/detections` | Lista detecciones | Query: `?limit=50&offset=0&profile=&model=&bbox=&date_from=&date_to=` | `{"items":[Detection], "total":int}` |
| `GET` | `/api/detections/{id}` | Detalle + proveniencia | — | `Detection + ExecutionRecord` |
| `POST` | `/api/pipeline/trigger` | Lanzar pipeline | `{"zone":"gibraltar","model":"yolov8n-sar","profile":"ground","image_id":null}` | `{"execution_id":"uuid","status":"started"}` |
| `POST` | `/api/pipeline/trigger-all-profiles` | Misma imagen, todos los perfiles | `{"zone":"gibraltar","model":"yolov8n-sar","image_id":null}` | `{"executions":{"ground":"uuid1","sat-high":"uuid2",...}}` |
| `GET` | `/api/pipeline/status` | Estado del pipeline activo | — | `{"running":bool,"current_profile":"str","progress":0.75}` |
| `GET` | `/api/benchmarks` | Resultados de compresion | Query: `?model=&profile=` | `[BenchmarkResult]` |
| `GET` | `/api/benchmarks/compare` | Comparativa variantes x perfiles | Query: `?models=a,b,c&profiles=ground,sat-high` | `ComparisonMatrix` |
| `GET` | `/api/traceability/{execution_id}` | Cadena de proveniencia completa | — | `ExecutionRecord` (todos los campos) |
| `GET` | `/api/tasking/queue` | Cola de Tip & Cue | Query: `?status=pending` | `[TaskingEntry]` |
| `POST` | `/api/tasking/cue` | Crear cue manual | `{"bbox":[...], "priority":1, "reason":"manual"}` | `{"cue_id":"uuid"}` |
| `GET` | `/api/metrics` | Metricas Prometheus | — | `text/plain` (formato Prometheus) |
| `GET` | `/docs` | Swagger UI | — | HTML (auto-generado) |
| `GET` | `/api/models` | Lista de modelos disponibles | — | `[ModelInfo]` |
| `GET` | `/api/profiles` | Lista de perfiles de restriccion | — | `[ConstraintProfile]` |
| `GET` | `/api/zones` | Zonas de busqueda | — | `[SearchZone]` |
| `GET` | `/api/orbital/energy` | Perfil energetico por modelo/perfil | Query: `?model=&profile=&processor=` | `[EnergyEstimate]` |
| `GET` | `/api/orbital/energy/budget` | Presupuesto orbital | Query: `?model=&profile=&satellite=cubesat_6u` | `OrbitalBudgetResult` |
| `GET` | `/api/orbital/downlink` | Analisis de downlink | Query: `?downlink_profile=&image_id=` | `DownlinkAnalysis` |
| `GET` | `/api/orbital/downlink/value` | Informe de valor OBDP | — | `OBDPValueReport` |
| `GET` | `/api/orbital/latency` | Comparativa de latencia | Query: `?orbit=&downlink=&inference_ms=` | `[LatencyComparison]` |
| `POST` | `/api/orbital/resilience/bitflip` | Ejecutar sweep de bit-flips | `{"model":"yolov8n-sar","flip_counts":[0,1,5,10,50]}` | `BitFlipSweepResult` |
| `POST` | `/api/orbital/resilience/simulate-orbit` | Simular orbita completa | `{"num_images":20,"satellite":"cubesat_6u"}` | `OrbitSimulationResult` |
| `GET` | `/api/orbital/resilience/drift` | Estado de drift detection | — | `DriftResult` |
| `POST` | `/api/orbital/decision` | Consultar decision engine | `{"available_cpu":1,"available_ram_mb":1024,"available_energy_wh":5}` | `DecisionResult` |

### 12.4 Archivo `src/config.py`

```python
"""
Configuracion centralizada con pydantic-settings.
Lee variables de entorno y .env.
"""

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Base de datos
    database_url: str = "postgresql+asyncpg://aidra:changeme@localhost:5432/aidra"

    # Copernicus
    copernicus_user: str = ""
    copernicus_password: str = ""

    # Directorios
    models_dir: str = "/app/models"
    images_dir: str = "/data/images"

    # Pipeline defaults
    default_zone: str = "gibraltar"
    default_model: str = "yolov8n-sar"
    default_profile: str = "ground"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45

    # CFAR defaults
    cfar_guard_size: int = 3
    cfar_training_size: int = 15
    cfar_pfa: float = 1e-5

    # Tile defaults
    tile_size: int = 640
    tile_overlap: int = 64

    # Tip & Cue
    tipcue_enabled: bool = True
    tipcue_min_confidence: float = 0.7
    tipcue_min_detections: int = 2
    tipcue_cooldown_minutes: int = 60

    # Scheduler
    scheduler_enabled: bool = True
    scheduler_interval_hours: int = 6  # Cada cuantas horas buscar nuevas imagenes

    # Observabilidad
    prometheus_enabled: bool = True
    loki_url: str = "http://aidra-loki:3100"
    log_level: str = "INFO"

    # Limites
    max_image_size_gb: float = 2.0  # No descargar imagenes mayores
    max_concurrent_pipelines: int = 1  # Solo un pipeline a la vez en OCI Free Tier
    pipeline_timeout_seconds: int = 600  # 10 minutos max por ejecucion

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

### 12.5 Archivo `.env.example`

```bash
# ---- AIDRA Configuration ----

# Base de datos
DB_PASSWORD=changeme_strong_password_here

# Copernicus Data Space (registrarse en https://dataspace.copernicus.eu)
COPERNICUS_USER=your_email@example.com
COPERNICUS_PASSWORD=your_copernicus_password

# Grafana
GRAFANA_PASSWORD=changeme_grafana_admin

# Pipeline
DEFAULT_ZONE=gibraltar
DEFAULT_MODEL=yolov8n-sar
CONFIDENCE_THRESHOLD=0.25

# Tip & Cue
TIPCUE_ENABLED=true
TIPCUE_MIN_CONFIDENCE=0.7

# Scheduler
SCHEDULER_ENABLED=true
SCHEDULER_INTERVAL_HOURS=6

# Logging
LOG_LEVEL=INFO
```

---

## 13. Base de Datos: Esquema SQL Completo

### 13.1 Agente responsable: `AGENT-DB`

### 13.2 Migracion inicial `src/db/migrations/001_init.sql`

```sql
-- ====================================================
-- AIDRA Database Schema
-- PostgreSQL 16 + PostGIS 3.4
-- ====================================================

-- Extensiones
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- Para busquedas de texto

-- ====================================================
-- Tabla: execution_log
-- Registro inmutable de cada ejecucion del pipeline.
-- Nucleo de la trazabilidad AIDRA.
-- ====================================================
CREATE TABLE execution_log (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Input
    image_id            TEXT NOT NULL,
    image_title         TEXT,
    image_hash          TEXT NOT NULL,
    image_bbox          GEOMETRY(POLYGON, 4326),
    image_sensing_date  TIMESTAMPTZ,
    image_size_mb       REAL,
    search_zone         TEXT,

    -- Modelo
    model_name          TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    model_hash          TEXT NOT NULL,
    model_size_mb       REAL NOT NULL,
    model_format        TEXT NOT NULL DEFAULT 'pytorch',
    compression_technique TEXT DEFAULT 'none',

    -- Parametros
    confidence_threshold REAL NOT NULL DEFAULT 0.25,
    iou_threshold       REAL NOT NULL DEFAULT 0.45,
    constraint_profile  TEXT NOT NULL DEFAULT 'ground',
    cpu_limit           REAL,
    memory_limit_mb     INTEGER,
    tile_size           INTEGER DEFAULT 640,
    tile_overlap        INTEGER DEFAULT 64,

    -- Resultados
    num_detections      INTEGER NOT NULL DEFAULT 0,
    avg_confidence      REAL,
    max_confidence      REAL,
    min_confidence      REAL,

    -- Metricas de rendimiento
    total_duration_ms   REAL,
    download_ms         REAL,
    preprocessing_ms    REAL,
    inference_ms        REAL,
    postprocessing_ms   REAL,
    peak_ram_mb         REAL,
    avg_ram_mb          REAL,
    cpu_usage_pct       REAL,
    num_tiles           INTEGER,

    -- Trazabilidad
    output_hash         TEXT NOT NULL,
    input_params_hash   TEXT,

    -- Estado
    status              TEXT NOT NULL DEFAULT 'pending',
    error_message       TEXT,
    trigger_type        TEXT NOT NULL DEFAULT 'manual',
    triggered_by        UUID REFERENCES execution_log(id),

    -- Metadatos
    pipeline_version    TEXT DEFAULT '1.0.0',
    hostname            TEXT,
    notes               TEXT
);

-- ====================================================
-- Tabla: detections
-- Detecciones individuales de barcos.
-- Cada deteccion pertenece a una ejecucion del pipeline.
-- ====================================================
CREATE TABLE detections (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id        UUID NOT NULL REFERENCES execution_log(id) ON DELETE CASCADE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Ubicacion
    center_geo          GEOMETRY(POINT, 4326) NOT NULL,
    bbox_geo            GEOMETRY(POLYGON, 4326),
    bbox_pixel          REAL[] NOT NULL,

    -- Deteccion
    confidence          REAL NOT NULL,
    source              TEXT NOT NULL,
    cfar_snr            REAL,
    yolo_score          REAL,
    class_name          TEXT DEFAULT 'vessel',

    -- Tile
    tile_index          INTEGER NOT NULL,
    tile_row_offset     INTEGER,
    tile_col_offset     INTEGER
);

-- ====================================================
-- Tabla: models_registry
-- Registro de variantes de modelo disponibles.
-- ====================================================
CREATE TABLE models_registry (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    registered_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    name                TEXT NOT NULL,
    version             TEXT NOT NULL,
    format              TEXT NOT NULL,
    file_path           TEXT NOT NULL,
    file_hash           TEXT NOT NULL UNIQUE,
    size_mb             REAL NOT NULL,

    base_model          TEXT,
    compression_technique TEXT DEFAULT 'none',
    compression_params  JSONB,

    num_params          BIGINT,
    num_layers          INTEGER,
    input_size          INTEGER[] DEFAULT '{640,640}',
    classes             TEXT[] DEFAULT '{"vessel"}',

    metadata            JSONB,

    UNIQUE(name, version)
);
```

### 13.3 Migracion de indices `src/db/migrations/002_indexes.sql`

```sql
-- Indices para execution_log
CREATE INDEX idx_execution_log_created_at ON execution_log(created_at DESC);
CREATE INDEX idx_execution_log_profile ON execution_log(constraint_profile);
CREATE INDEX idx_execution_log_model ON execution_log(model_name, model_version);
CREATE INDEX idx_execution_log_status ON execution_log(status);
CREATE INDEX idx_execution_log_image_id ON execution_log(image_id);
CREATE INDEX idx_execution_log_trigger ON execution_log(trigger_type);
CREATE INDEX idx_execution_log_bbox ON execution_log USING GIST(image_bbox);

-- Indices para detections
CREATE INDEX idx_detections_execution ON detections(execution_id);
CREATE INDEX idx_detections_confidence ON detections(confidence DESC);
CREATE INDEX idx_detections_source ON detections(source);
CREATE INDEX idx_detections_geo ON detections USING GIST(center_geo);
CREATE INDEX idx_detections_bbox ON detections USING GIST(bbox_geo);
CREATE INDEX idx_detections_created ON detections(created_at DESC);

-- Indices para models_registry
CREATE INDEX idx_models_name ON models_registry(name);
CREATE INDEX idx_models_technique ON models_registry(compression_technique);
```

### 13.4 Migracion Tip & Cue `src/db/migrations/003_tipcue.sql`

```sql
-- ====================================================
-- Tabla: tasking_queue
-- Cola de Tip & Cue.
-- ====================================================
CREATE TABLE tasking_queue (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    trigger_type        TEXT NOT NULL DEFAULT 'cue',
    triggered_by        UUID REFERENCES execution_log(id),
    triggering_detections UUID[],

    target_bbox         GEOMETRY(POLYGON, 4326) NOT NULL,
    target_zone         TEXT,
    priority            INTEGER NOT NULL DEFAULT 0,
    reason              TEXT,

    status              TEXT NOT NULL DEFAULT 'pending',
    scheduled_at        TIMESTAMPTZ,
    executed_at         TIMESTAMPTZ,
    execution_id        UUID REFERENCES execution_log(id),

    result_status       TEXT,
    confirmed_detections INTEGER,

    cooldown_until      TIMESTAMPTZ,
    attempts            INTEGER DEFAULT 0,
    max_attempts        INTEGER DEFAULT 3,
    last_error          TEXT
);

-- Indices
CREATE INDEX idx_tasking_status ON tasking_queue(status);
CREATE INDEX idx_tasking_priority ON tasking_queue(priority DESC, created_at);
CREATE INDEX idx_tasking_bbox ON tasking_queue USING GIST(target_bbox);
CREATE INDEX idx_tasking_triggered_by ON tasking_queue(triggered_by);
```

### 13.5 Archivo `src/db/connection.py`

```python
"""
Pool de conexiones asyncpg para PostgreSQL.

Usa asyncpg directamente (sin ORM) para maximo rendimiento
y control sobre las queries SQL.
"""

import asyncpg
from src.config import Settings

class Database:
    """
    Singleton para el pool de conexiones.

    Metodos:
    - async connect(settings: Settings) -> None:
        Crea pool de conexiones.
        Pool config: min_size=2, max_size=10, command_timeout=60

    - async disconnect() -> None:
        Cierra pool.

    - async execute(query: str, *args) -> str:
        Ejecuta query sin retorno (INSERT, UPDATE, DELETE).

    - async fetch(query: str, *args) -> list[asyncpg.Record]:
        Ejecuta query con retorno de filas.

    - async fetchrow(query: str, *args) -> asyncpg.Record | None:
        Ejecuta query con retorno de una fila.

    - async fetchval(query: str, *args) -> Any:
        Ejecuta query con retorno de un valor.

    - async run_migrations(migrations_dir: Path) -> None:
        Ejecuta archivos .sql de migraciones en orden.
        Controla cuales ya se aplicaron via tabla _migrations.

    Pool config:
        dsn = settings.database_url.replace('+asyncpg', '')
        min_size = 2
        max_size = 10
        command_timeout = 60
        server_settings = {
            'application_name': 'aidra',
            'jit': 'off',  # JIT off para ARM — mas rapido en queries simples
        }
    """
```

### 13.6 Archivo `src/db/queries.py`

```python
"""
Consultas SQL parametrizadas.

Todas las queries usan $1, $2... (parametros asyncpg, no f-strings).
Nunca concatenar strings SQL.
"""

# ---- execution_log ----

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
        pipeline_version, hostname
    ) VALUES (
        $1, $2, $3, $4, ST_GeomFromGeoJSON($5),
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
        $41, $42
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

# ---- detections ----

INSERT_DETECTION = """
    INSERT INTO detections (
        id, execution_id,
        center_geo, bbox_geo, bbox_pixel,
        confidence, source, cfar_snr, yolo_score, class_name,
        tile_index, tile_row_offset, tile_col_offset
    ) VALUES (
        $1, $2,
        ST_SetSRID(ST_MakePoint($3, $4), 4326),
        ST_GeomFromGeoJSON($5),
        $6,
        $7, $8, $9, $10, $11,
        $12, $13, $14
    )
"""

INSERT_DETECTIONS_BATCH = """
    INSERT INTO detections (
        execution_id, center_geo, bbox_geo, bbox_pixel,
        confidence, source, cfar_snr, yolo_score, class_name,
        tile_index
    )
    SELECT * FROM unnest(
        $1::uuid[], -- execution_ids
        -- ... (batch via unnest para rendimiento)
    )
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
    WHERE ($1::text IS NULL OR e.constraint_profile = $1)
      AND ($2::text IS NULL OR e.model_name = $2)
      AND ($3::real IS NULL OR d.confidence >= $3)
      AND ($4::timestamptz IS NULL OR d.created_at >= $4)
      AND ($5::timestamptz IS NULL OR d.created_at <= $5)
      AND ($6::geometry IS NULL OR ST_Intersects(d.center_geo, $6))
    ORDER BY d.confidence DESC
    LIMIT $7 OFFSET $8
"""

SELECT_DETECTION_BY_ID = """
    SELECT
        d.*,
        ST_X(d.center_geo) AS longitude,
        ST_Y(d.center_geo) AS latitude,
        ST_AsGeoJSON(d.center_geo) AS center_geojson,
        ST_AsGeoJSON(d.bbox_geo) AS bbox_geojson,
        e.*
    FROM detections d
    JOIN execution_log e ON d.execution_id = e.id
    WHERE d.id = $1
"""

# ---- benchmarks (queries agregadas) ----

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

# ---- tasking_queue ----

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
    SELECT *, ST_AsGeoJSON(target_bbox) AS target_bbox_geojson
    FROM tasking_queue
    WHERE status = 'pending'
      AND (cooldown_until IS NULL OR cooldown_until < NOW())
      AND attempts < max_attempts
    ORDER BY priority DESC, created_at
    LIMIT $1
"""

UPDATE_CUE_STATUS = """
    UPDATE tasking_queue
    SET status = $2,
        executed_at = CASE WHEN $2 = 'completed' THEN NOW() ELSE executed_at END,
        execution_id = $3,
        result_status = $4,
        confirmed_detections = $5,
        attempts = attempts + 1
    WHERE id = $1
"""

# ---- models_registry ----

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
```

### 13.7 Archivo `src/db/models.py`

```python
"""
Pydantic models para la API. Mapean los datos de la DB a objetos Python tipados.
Estos modelos los usan TODOS los agentes como contratos de interfaz.
"""

from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field

class ExecutionRecord(BaseModel):
    id: UUID
    created_at: datetime
    image_id: str
    image_title: str | None = None
    image_hash: str
    image_sensing_date: datetime | None = None
    image_size_mb: float | None = None
    search_zone: str | None = None
    model_name: str
    model_version: str
    model_hash: str
    model_size_mb: float
    model_format: str = "pytorch"
    compression_technique: str = "none"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    constraint_profile: str = "ground"
    cpu_limit: float | None = None
    memory_limit_mb: int | None = None
    tile_size: int = 640
    tile_overlap: int = 64
    num_detections: int = 0
    avg_confidence: float | None = None
    max_confidence: float | None = None
    min_confidence: float | None = None
    total_duration_ms: float | None = None
    download_ms: float | None = None
    preprocessing_ms: float | None = None
    inference_ms: float | None = None
    postprocessing_ms: float | None = None
    peak_ram_mb: float | None = None
    avg_ram_mb: float | None = None
    cpu_usage_pct: float | None = None
    num_tiles: int | None = None
    output_hash: str
    input_params_hash: str | None = None
    status: str = "pending"
    error_message: str | None = None
    trigger_type: str = "manual"
    triggered_by: UUID | None = None
    pipeline_version: str = "1.0.0"
    hostname: str | None = None
    notes: str | None = None

class DetectionRecord(BaseModel):
    id: UUID
    execution_id: UUID
    created_at: datetime
    longitude: float
    latitude: float
    bbox_pixel: list[float]
    confidence: float
    source: str  # "cfar", "yolo", "fused"
    cfar_snr: float | None = None
    yolo_score: float | None = None
    class_name: str = "vessel"
    tile_index: int
    # Joined fields
    constraint_profile: str | None = None
    model_name: str | None = None
    image_id: str | None = None

class ModelInfo(BaseModel):
    id: UUID
    name: str
    version: str
    format: str
    file_hash: str
    size_mb: float
    base_model: str | None = None
    compression_technique: str = "none"
    num_params: int | None = None
    input_size: list[int] = [640, 640]
    classes: list[str] = ["vessel"]

class TaskingEntry(BaseModel):
    id: UUID
    created_at: datetime
    trigger_type: str
    triggered_by: UUID | None = None
    target_bbox_geojson: dict | None = None
    target_zone: str | None = None
    priority: int = 0
    reason: str | None = None
    status: str = "pending"
    execution_id: UUID | None = None
    result_status: str | None = None
    confirmed_detections: int | None = None
    attempts: int = 0

class BenchmarkResult(BaseModel):
    model_name: str
    model_version: str
    model_size_mb: float
    compression_technique: str
    constraint_profile: str
    runs: int
    avg_inference_ms: float
    p50_inference_ms: float | None = None
    p95_inference_ms: float | None = None
    avg_peak_ram_mb: float
    avg_cpu_pct: float
    avg_detections: float
    avg_confidence: float | None = None

# ---- Request/Response models para la API ----

class PaginatedResponse(BaseModel):
    items: list
    total: int
    limit: int
    offset: int

class PipelineTriggerRequest(BaseModel):
    zone: str = "gibraltar"
    model: str = "yolov8n-sar"
    profile: str = "ground"
    image_id: str | None = None
    aoi_bbox: list[float] | None = None
    confidence_threshold: float = 0.25

class PipelineTriggerResponse(BaseModel):
    execution_id: UUID
    status: str = "started"

class PipelineStatusResponse(BaseModel):
    running: bool
    current_profile: str | None = None
    progress: float | None = None
    current_execution_id: UUID | None = None

class HealthResponse(BaseModel):
    status: str
    db: str
    models_loaded: int
    scheduler: str
    version: str = "1.0.0"
    uptime_seconds: float | None = None

class CueCreateRequest(BaseModel):
    bbox: list[float]
    priority: int = 1
    reason: str = "manual"
    zone: str | None = None

class ComparisonRequest(BaseModel):
    models: list[str] | None = None
    profiles: list[str] | None = None
    image_id: str | None = None
```

---

## 14. Configuracion y Variables de Entorno

### 14.1 Variables requeridas

| Variable | Requerida | Descripcion | Ejemplo |
|---|---|---|---|
| `DB_PASSWORD` | Si | Password PostgreSQL | `AiDrA_2026!secure` |
| `COPERNICUS_USER` | Si | Email de Copernicus Data Space | `user@example.com` |
| `COPERNICUS_PASSWORD` | Si | Password de Copernicus | `MyC0pern1cus!` |
| `GRAFANA_PASSWORD` | Si | Password admin Grafana | `grafana_admin_2026` |
| `LOG_LEVEL` | No | Nivel de logging | `INFO` |
| `DEFAULT_ZONE` | No | Zona de busqueda por defecto | `gibraltar` |
| `DEFAULT_MODEL` | No | Modelo por defecto | `yolov8n-sar` |
| `CONFIDENCE_THRESHOLD` | No | Umbral de confianza | `0.25` |
| `TIPCUE_ENABLED` | No | Activar Tip & Cue | `true` |
| `SCHEDULER_ENABLED` | No | Activar scheduler | `true` |
| `SCHEDULER_INTERVAL_HOURS` | No | Intervalo del scheduler | `6` |

---

## 15. Pipeline de CI/CD y Testing

### 15.1 Agente responsable: `AGENT-TEST`

### 15.2 Testing strategy

```
tests/
├── test_api/            # Tests de integracion de endpoints
│   Tests con httpx.AsyncClient contra la app FastAPI.
│   Requieren DB PostgreSQL de test (Docker).
│
├── test_pipeline/       # Tests unitarios del pipeline
│   Tests con imagenes SAR de prueba (tiles pequenos, 64x64).
│   Mock de Copernicus API.
│
├── test_models/         # Tests de modelos
│   Tests con modelo YOLOv8n preentrenado (sin fine-tune).
│   Verificar que la inferencia produce resultados validos.
│   Tests de compresion: verificar que el modelo comprimido funciona.
│
└── test_traceability/   # Tests de trazabilidad
    Tests de hashing: verificar determinismo.
    Tests de recorder: verificar insercion en DB.
```

### 15.3 Fixtures compartidas `tests/conftest.py`

```python
"""
Fixtures para tests.

- db_pool: Pool de conexiones a DB de test
- test_client: httpx.AsyncClient configurado
- sample_tile: numpy array 640x640 con patron de prueba
- sample_detections: Lista de detecciones de prueba
- yolo_model: Modelo YOLOv8n cargado
"""
```

### 15.4 Tests detallados por modulo

**`tests/conftest.py`**:
```python
import pytest
import asyncio
import numpy as np
from uuid import uuid4
from pathlib import Path

@pytest.fixture
def sample_sar_tile():
    """Tile SAR sintetico 640x640 con 5 barcos simulados."""
    from src.pipeline.preprocessing import generate_synthetic_sar_tile  # ver seccion 25
    image, ground_truth = generate_synthetic_sar_tile(size=640, num_vessels=5, seed=42)
    return image, ground_truth

@pytest.fixture
def sample_detections():
    """Lista de detecciones de prueba."""
    return [
        {
            "bbox": [100, 200, 120, 220],
            "center": [110, 210],
            "confidence": 0.85,
            "source": "fused",
            "cfar_snr": 12.5,
            "yolo_score": 0.85,
            "tile_index": 0,
        },
        {
            "bbox": [300, 400, 310, 415],
            "center": [305, 407],
            "confidence": 0.72,
            "source": "yolo",
            "yolo_score": 0.72,
            "tile_index": 1,
        },
    ]

@pytest.fixture
def mock_execution_record():
    """Registro de ejecucion de prueba."""
    return {
        "id": uuid4(),
        "image_id": "S1A_IW_GRDH_TEST_001",
        "image_hash": "abc123" * 10 + "abcd",
        "model_name": "yolov8n-sar",
        "model_version": "v1.0",
        "model_hash": "def456" * 10 + "defg",
        "model_size_mb": 6.2,
        "confidence_threshold": 0.25,
        "constraint_profile": "ground",
        "num_detections": 5,
        "inference_ms": 150.0,
        "peak_ram_mb": 512.0,
        "cpu_usage_pct": 45.0,
        "output_hash": "ghi789" * 10 + "ghij",
        "status": "success",
    }
```

**`tests/test_models/test_cfar.py`** — Tests clave:
```python
def test_cfar_detects_bright_points(sample_sar_tile):
    """CFAR debe detectar los 5 barcos simulados en el tile sintetico."""
    image, ground_truth = sample_sar_tile
    detector = CFARDetector(guard_size=3, training_size=15, pfa=1e-5)
    detections = detector.detect_with_clustering(image)
    assert len(detections) >= 3  # Al menos 3 de 5 (tolerancia por clustering)

def test_cfar_no_false_positives_on_noise():
    """CFAR no debe detectar nada en imagen de solo ruido."""
    rng = np.random.default_rng(42)
    noise = rng.rayleigh(scale=0.3, size=(640, 640)).astype(np.float32)
    detector = CFARDetector(guard_size=3, training_size=15, pfa=1e-6)
    detections = detector.detect_with_clustering(noise)
    assert len(detections) <= 1  # Maximo 1 falso positivo
```

**`tests/test_pipeline/test_ingestion.py`** — Tests clave:
```python
@pytest.mark.asyncio
async def test_copernicus_auth(monkeypatch):
    """Autenticacion OAuth2 retorna token valido."""
    # Mock HTTP response con token
    auth = CopernicusAuth(username="test", password="test")
    token = await auth.get_token()
    assert token is not None
    assert len(token) > 10

@pytest.mark.asyncio
async def test_search_returns_results(monkeypatch):
    """Busqueda en zona de Gibraltar retorna al menos 1 resultado."""
    # Mock OData response
    ingester = ImageIngester(auth=mock_auth, images_dir=tmp_path)
    results = await ingester.search(
        bbox=[-5.8, 35.7, -5.2, 36.2],
        start_date=datetime(2026, 4, 1),
        end_date=datetime(2026, 4, 23),
    )
    assert len(results) >= 1
    assert results[0].product_id is not None

def test_sha256_deterministic(tmp_path):
    """SHA256 del mismo archivo produce el mismo hash."""
    file = tmp_path / "test.bin"
    file.write_bytes(b"test data" * 1000)
    hash1 = compute_sha256(file)
    hash2 = compute_sha256(file)
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex digest
```

**`tests/test_traceability/test_recorder.py`** — Tests clave:
```python
@pytest.mark.asyncio
async def test_record_creates_immutable_entry(db_pool, mock_execution_record):
    """Un registro se inserta y no se puede borrar."""
    recorder = ExecutionRecorder(db_pool)
    execution_id = await recorder.record(mock_execution_record)
    retrieved = await recorder.get(execution_id)
    assert retrieved is not None
    assert retrieved.image_hash == mock_execution_record["image_hash"]
    assert retrieved.output_hash == mock_execution_record["output_hash"]

def test_result_hash_deterministic(sample_detections):
    """El hash de las mismas detecciones siempre es igual."""
    hash1 = compute_result_hash(sample_detections)
    hash2 = compute_result_hash(sample_detections)
    assert hash1 == hash2

    # Orden diferente -> mismo hash (serializado con sort)
    reversed_dets = list(reversed(sample_detections))
    hash3 = compute_result_hash(reversed_dets)
    assert hash1 == hash3
```

### 15.5 Comando de ejecucion

```bash
# Ejecutar todos los tests
pytest tests/ -v --tb=short

# Solo tests unitarios (no requieren Docker)
pytest tests/test_models/ tests/test_traceability/ -v

# Tests de integracion (requieren Docker Compose)
docker compose -f docker-compose.test.yml up -d
pytest tests/test_api/ tests/test_pipeline/ -v
docker compose -f docker-compose.test.yml down
```

---

## 16. Dependencias y Versiones

### 16.1 `pyproject.toml`

```toml
[project]
name = "aidra"
version = "1.0.0"
description = "AI-Enabled On-Board Data Processing Assessment — Vessel Detection MVP"
requires-python = ">=3.11"

dependencies = [
    # Web framework
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.30.0",

    # Base de datos
    "asyncpg>=0.29.0",

    # Geoespacial
    "rasterio>=1.3.10",
    "shapely>=2.0.4",
    "pyproj>=3.6.1",

    # IA / ML
    "ultralytics>=8.2.0",
    "torch>=2.3.0",
    "onnx>=1.16.0",
    "onnxruntime>=1.18.0",

    # Ciencia de datos
    "numpy>=1.26.0",
    "scipy>=1.13.0",
    "scikit-learn>=1.5.0",

    # Procesamiento de imagenes
    "Pillow>=10.3.0",
    "opencv-python-headless>=4.10.0",

    # Copernicus
    "requests>=2.32.0",
    "pystac-client>=0.8.0",

    # Scheduler
    "APScheduler>=3.10.4",

    # Observabilidad
    "prometheus-client>=0.20.0",
    "python-logging-loki>=0.3.1",

    # Configuracion
    "pydantic>=2.7.0",
    "pydantic-settings>=2.3.0",

    # Utilidades
    "psutil>=5.9.8",
    "python-multipart>=0.0.9",

    # Docker SDK (para perfiles de restriccion)
    "docker>=7.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.27.0",
    "ruff>=0.4.0",
]
all = ["aidra[dev]"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

---

## 17. Plan de Ejecucion por Fases

### Fase 1 — Infraestructura + Pipeline basico + Traceability (35-45h)

| # | Tarea | Agente | Dependencia | Criterio de done |
|---|---|---|---|---|
| 1.1 | Setup OCI ARM A1 (script setup-oci.sh) | AGENT-INFRA | — | SSH funcional, Docker instalado |
| 1.2 | Docker Compose completo (todos los servicios) | AGENT-INFRA | 1.1 | `docker compose up` levanta todo |
| 1.3 | Esquema SQL + migraciones | AGENT-DB | 1.2 | Tablas creadas, indices aplicados |
| 1.4 | Configuracion Grafana (datasources + provisioning) | AGENT-OBSERVABILITY | 1.2 | Grafana conecta a PostgreSQL y Prometheus |
| 1.5 | Modulo de ingesta Copernicus | AGENT-INGESTION | 1.3 | Descarga una imagen S1 GRD real |
| 1.6 | Preprocesamiento SAR (calibracion + tiles) | AGENT-INGESTION | 1.5 | Genera tiles 640x640 de imagen calibrada |
| 1.7 | Deteccion CFAR basica | AGENT-DETECTION | 1.6 | Detecta puntos brillantes en tile SAR |
| 1.8 | Tabla execution_log + recorder | AGENT-TRACE | 1.3 | Registro con hashes en DB |
| 1.9 | Guardar detecciones en PostGIS | AGENT-DB | 1.7, 1.8 | Detecciones con geometria en DB |
| 1.10 | API FastAPI basica (health, detections, traceability) | AGENT-API | 1.9 | Endpoints responden con datos reales |
| 1.11 | Dashboard Grafana: mapa + traceability | AGENT-OBSERVABILITY | 1.9 | Mapa muestra detecciones, tabla muestra logs |

### Fase 2 — Modelo IA + Perfiles de restriccion (35-45h)

| # | Tarea | Agente | Dependencia | Criterio de done |
|---|---|---|---|---|
| 2.1 | Integrar YOLOv8n preentrenado | AGENT-DETECTION | Fase 1 | Inferencia funciona en CPU ARM |
| 2.2 | Fine-tune YOLOv8n con xView3-SAR (GPU externa) | AGENT-DETECTION | 2.1 | Modelo .pt fine-tuned disponible |
| 2.3 | Fusion CFAR + YOLO | AGENT-DETECTION | 2.1, 1.7 | Detecciones fusionadas con confianza combinada |
| 2.4 | Perfiles de restriccion (definitions + manager) | AGENT-PROFILES | 2.3 | Pipeline ejecuta con limites CPU/RAM |
| 2.5 | Endpoint trigger-all-profiles | AGENT-API | 2.4 | Ejecuta misma imagen con 5 perfiles |
| 2.6 | Metricas Prometheus (definiciones + endpoint) | AGENT-OBSERVABILITY | 2.5 | /api/metrics retorna metricas validas |
| 2.7 | Loki logger estructurado | AGENT-OBSERVABILITY | Fase 1 | Logs consultables en Grafana |
| 2.8 | APScheduler integrado | AGENT-API | 2.3 | Pipeline se ejecuta automaticamente cada N horas |
| 2.9 | Dashboard metricas pipeline | AGENT-OBSERVABILITY | 2.6 | Graficos de latencia, RAM, CPU |
| 2.10 | Dashboard perfiles restriccion | AGENT-OBSERVABILITY | 2.5 | Comparativa entre perfiles |

### Fase 3 — Compresion de modelos (25-35h)

| # | Tarea | Agente | Dependencia | Criterio de done |
|---|---|---|---|---|
| 3.1 | Exportar modelo a ONNX | AGENT-COMPRESSION | 2.2 | Archivo .onnx funcional |
| 3.2 | Quantizacion FP16 | AGENT-COMPRESSION | 3.1 | Modelo FP16, benchmark |
| 3.3 | Quantizacion INT8 dinamica (PyTorch) | AGENT-COMPRESSION | 2.2 | Modelo INT8, benchmark |
| 3.4 | Quantizacion INT8 estatica (ONNX Runtime) | AGENT-COMPRESSION | 3.1 | Modelo INT8 calibrado, benchmark |
| 3.5 | Pruning 30% + 50% | AGENT-COMPRESSION | 2.2 | Modelos pruned, benchmarks |
| 3.6 | Pruning + INT8 combinado | AGENT-COMPRESSION | 3.3, 3.5 | Modelo combinado, benchmark |
| 3.7 | Knowledge distillation (GPU externa) | AGENT-COMPRESSION | 2.2 | Modelo destilado, benchmark |
| 3.8 | Ejecutar todas las variantes x todos los perfiles | AGENT-PROFILES | 3.1-3.7 | Matriz completa de resultados |
| 3.9 | Dashboard benchmarks compresion | AGENT-OBSERVABILITY | 3.8 | Graficos comparativos |
| 3.10 | Documentar trade-offs (equiv. D4 AIDRA) | AGENT-API | 3.8 | Endpoint /benchmarks/compare funcional |

### Fase 4 — Tip & Cue + Pulido (20-30h)

| # | Tarea | Agente | Dependencia | Criterio de done |
|---|---|---|---|---|
| 4.1 | Tabla tasking_queue | AGENT-DB | Fase 1 | Migracion 003 aplicada |
| 4.2 | TipEvaluator | AGENT-TIPCUE | 4.1 | Genera tips a partir de detecciones |
| 4.3 | CueScheduler | AGENT-TIPCUE | 4.2 | Programa ejecuciones de cue |
| 4.4 | Vincular tips → cues en execution_log | AGENT-TRACE | 4.3 | Cadena tip→cue trazable |
| 4.5 | Endpoints tasking (queue, cue manual) | AGENT-API | 4.3 | API de Tip & Cue funcional |
| 4.6 | Dashboard mapa con tips/cues | AGENT-OBSERVABILITY | 4.5 | Mapa con capas tip/cue vinculadas |
| 4.7 | Models registry endpoint | AGENT-API | Fase 3 | /api/models lista variantes |
| 4.8 | Tests completos | AGENT-TEST | Todo | pytest pasa al 100% |
| 4.9 | README final + documentacion | AGENT-API | Todo | README con instrucciones completas |
| 4.10 | Docker Compose: `docker compose up` y todo funciona | AGENT-INFRA | Todo | Despliegue one-command |

### Fase 5 — Modulos Orbitales: Diferenciadores (25-35h)

| # | Tarea | Agente | Dependencia | Criterio de done |
|---|---|---|---|---|
| 5.1 | Perfil energetico: procesadores, TDP, calculo joules/inferencia | AGENT-ORBITAL | Fase 2 | Energia estimada por variante y perfil |
| 5.2 | Presupuesto orbital: imagenes/orbita por tipo de satelite | AGENT-ORBITAL | 5.1 | Tabla cubesat→medium_sat con feasibility |
| 5.3 | Analisis de downlink: ratio compresion, capacidad diaria | AGENT-ORBITAL | Fase 1 | Ratio >10.000:1 calculado con datos reales |
| 5.4 | Latencia orbital: simulacion con/sin OBDP | AGENT-ORBITAL | 5.3 | Tabla comparativa por orbita x downlink |
| 5.5 | Accionabilidad: distancia movida del barco vs latencia | AGENT-ORBITAL | 5.4 | Indicador high/medium/low por escenario |
| 5.6 | Bit-flip simulator: inyeccion en pesos, sweep | AGENT-ORBITAL | Fase 2 | Curva de degradacion, threshold critico |
| 5.7 | Decision Engine: seleccion autonoma de modelo | AGENT-ORBITAL | 5.1, Fase 3 | Secuencia de decisiones coherente |
| 5.8 | Simulacion de orbita completa: bateria + decisiones | AGENT-ORBITAL | 5.7 | Timeline de 1 orbita con >10 imagenes |
| 5.9 | Drift detection: Z-score sobre ejecuciones | AGENT-ORBITAL | Fase 2 | Detecta anomalias inyectadas |
| 5.10 | 3 Dashboards Grafana orbitales (6, 7, 8) | AGENT-OBSERVABILITY | 5.1-5.9 | Dashboards con datos reales |
| 5.11 | Endpoints API orbitales | AGENT-API | 5.1-5.9 | /api/orbital/* funcional |
| 5.12 | Tests modulos orbitales | AGENT-TEST | 5.1-5.9 | pytest pasa |

### Fase 6 — Validacion en Hardware Real (Nice-to-have)

| # | Tarea | Dependencia | Criterio de done |
|---|---|---|---|
| 6.1 | Integrar Orion CubeSat Testbed (https://github.com/omega-space-group/orion-cubesat-testbed) como plataforma de validacion en hardware real (FPGA/GPU/neuromorphic) | Fase 3 | YOLOv8n-sar ejecuta en flatsat con metricas reales |
| 6.2 | Ejecutar benchmarks de compresion en hardware Orion (quantizado INT8, pruned, distilled) | 6.1 | Tabla comparativa OCI ARM vs hardware real |
| 6.3 | Documentar resultados como evidencia TRL 5-6 | 6.2 | Seccion en D3 Evidence Package |
| 6.4 | (Nice-to-have) Aplicar a ESA Phi-Lab Visiting Researcher para validacion en PhiSat-2 en orbita (https://cin.philab.esa.int/schemes/visiting-researchers-in-onboard-ai-for-sat-2-mission) | Fase 5 | Solicitud enviada / aceptada |
| 6.5 | (Nice-to-have) Si aceptado: adaptar modelo a PhiSat-2 API (Jupyter Notebook + NanoSat MO Framework) y ejecutar vessel detection en orbita real | 6.4 | Resultados de orbita documentados como TRL 7 |

**Total estimado actualizado: ~140-190 horas (Fases 1-5) + Fase 6 variable**

---

## 18. Criterios de Aceptacion por Modulo

### Modulo 0 (Infraestructura)
- [ ] `docker compose up -d` levanta todos los servicios sin errores
- [ ] Grafana accesible en :3000
- [ ] API accesible en :8000/docs (Swagger)
- [ ] PostgreSQL+PostGIS responde queries geoespaciales
- [ ] Prometheus scrapes metricas de la app
- [ ] Loki recibe logs de todos los contenedores

### Modulo 1 (Ingesta)
- [ ] Autenticacion OAuth2 con Copernicus funciona
- [ ] Busqueda OData retorna productos Sentinel-1 GRD reales
- [ ] Descarga completa de un producto (~500 MB)
- [ ] Hash SHA256 calculado y almacenado
- [ ] Preprocesamiento genera tiles 640x640 calibrados en sigma0

### Modulo 2 (Deteccion)
- [ ] CFAR detecta puntos brillantes en imagen SAR real
- [ ] YOLO ejecuta inferencia en CPU ARM en < 500ms por tile
- [ ] Fusion CFAR+YOLO produce detecciones con confianza combinada
- [ ] Detecciones geolocalizadas (lat/lon) correctamente
- [ ] Detecciones almacenadas en PostGIS como geometrias Point

### Modulo 3 (Compresion)
- [ ] Al menos 5 variantes de modelo generadas (ver tabla seccion 7.6)
- [ ] Cada variante tiene hash SHA256 unico registrado
- [ ] Cada variante ejecuta inferencia sin errores
- [ ] Benchmarks miden: tamano, latencia, RAM, CPU, detecciones
- [ ] Datos de benchmark almacenados en execution_log

### Modulo 4 (Traceability)
- [ ] Cada ejecucion tiene registro inmutable con todos los hashes
- [ ] API /traceability/{id} retorna cadena completa
- [ ] Verificacion de reproducibilidad funciona (mismos inputs → mismo hash)

### Modulo 5 (Perfiles)
- [ ] 5 perfiles definidos y funcionales (ground → sat-extreme)
- [ ] Misma imagen ejecutada con todos los perfiles
- [ ] Metricas de rendimiento por perfil almacenadas
- [ ] Se identifica punto de degradacion/fallo

### Modulo 6 (Tip & Cue)
- [ ] Detecciones en zona de interes generan tip automatico
- [ ] Tip genera cue en tasking_queue
- [ ] Cue programa nueva ejecucion del pipeline
- [ ] Resultado del cue vinculado al tip original
- [ ] Dashboard muestra tips y cues en mapa

### Modulo 7 (Observabilidad)
- [ ] 8 dashboards Grafana funcionales
- [ ] Metricas Prometheus con datos reales (no vacios)
- [ ] Logs estructurados en Loki consultables
- [ ] GeoMap muestra detecciones con coordenadas correctas

### Modulo 8 (API)
- [ ] Todos los endpoints documentados en Swagger
- [ ] Responses con pagination (limit/offset)
- [ ] Manejo de errores consistente (HTTPException con codigos correctos)
- [ ] CORS configurado
- [ ] Health check funcional

### Modulo 9 (Energia)
- [ ] Energia estimada (joules) por variante de modelo y perfil de restriccion
- [ ] TOPS/W calculado para cada variante
- [ ] Tabla comparativa con procesadores de vuelo reales (Zynq, Myriad X, etc.)
- [ ] Presupuesto orbital: imagenes/orbita para cubesat_3u, 6u, small_sat, medium_sat
- [ ] Dashboard Grafana con metricas energeticas

### Modulo 10 (Downlink)
- [ ] Ratio de compresion de datos calculado con ejecuciones reales
- [ ] Comparativa con/sin OBDP para 4 perfiles de downlink
- [ ] Capacidad diaria (imagenes/dia) calculada
- [ ] KPI "total bandwidth saved" acumulado

### Modulo 11 (Latencia Orbital)
- [ ] Desglose de latencia: onboard, wait, downlink, ground
- [ ] Comparativa con/sin OBDP para 3 orbitas x 3 escenarios tierra
- [ ] Speedup factor calculado
- [ ] Indicador de accionabilidad (distancia movida del barco)

### Modulo 12 (Resiliencia)
- [ ] Bit-flip sweep ejecutado con al menos 8 niveles (0 a 1000 flips)
- [ ] Threshold critico identificado (donde degradacion > 20%)
- [ ] MTBF estimado para al menos 2 orbitas diferentes
- [ ] Decision Engine funcional: elige modelo segun recursos disponibles
- [ ] Fallback a CFAR funciona cuando YOLO no cabe en RAM
- [ ] Drift detection alerta cuando Z-score > 3
- [ ] Simulacion de orbita completa con timeline de bateria

---

## 19. Glosario y Acronimos

| Termino | Definicion |
|---|---|
| AIDRA | Artificial Intelligence In-orbit Data pRocessing Assessment |
| AI-OBDP | AI-enabled On-Board Data Processing |
| AOI | Area Of Interest — zona geografica de interes |
| CFAR | Constant False Alarm Rate — algoritmo de deteccion en SAR |
| CUT | Cell Under Test — pixel evaluado en CFAR |
| D1-D5 | Entregables del contrato AIDRA (Demonstration Plan → Final Report) |
| EDA | European Defence Agency |
| EO | Earth Observation |
| GEOINT | Geospatial Intelligence |
| GRD | Ground Range Detected — producto Sentinel-1 preprocesado |
| IoU | Intersection over Union — metrica de solapamiento de bounding boxes |
| KD | Knowledge Distillation |
| mAP | mean Average Precision |
| NMS | Non-Maximum Suppression — elimina detecciones duplicadas |
| OBDP | On-Board Data Processing |
| OCI | Oracle Cloud Infrastructure |
| ONNX | Open Neural Network Exchange — formato portable de modelos |
| SAR | Synthetic Aperture Radar |
| SatCen | European Union Satellite Centre |
| Sigma0 | Coeficiente de backscatter calibrado en SAR |
| STAC | SpatioTemporal Asset Catalog |
| Tile | Recorte de imagen grande para procesamiento por partes |
| TRL | Technology Readiness Level |
| Tip | Alerta generada por una deteccion que cumple criterios |
| Cue | Accion de reprogramacion de observacion tras un tip |
| SEU | Single Event Upset — bit-flip causado por radiacion cosmica |
| TDP | Thermal Design Power — potencia maxima de un procesador |
| TOPS/W | Tera Operations Per Second Per Watt — eficiencia energetica |
| FLOPs | Floating Point Operations — operaciones del modelo |
| MTBF | Mean Time Between Failures — tiempo medio entre fallos |
| LEO | Low Earth Orbit — orbita terrestre baja (200-2000 km) |
| SSO | Sun-Synchronous Orbit — orbita heliosincrona |
| ISL | Inter-Satellite Link — enlace entre satelites |
| EDRS | European Data Relay System — sistema de relay via GEO |
| Downlink | Transmision de datos del satelite a la estacion terrena |
| PhiSat-1 | Primer satelite de la ESA con IA a bordo (Intel Myriad X, 2020) |

---

## 20. Asignacion de Agentes

### Modelo de trabajo del enjambre

Cada agente LLM trabaja de forma autonoma en su modulo, siguiendo estas reglas:

1. **Lee solo las secciones que le aplican** de este documento
2. **Respeta las interfaces** definidas (clases, metodos, tipos de retorno)
3. **No modifica archivos de otros agentes** sin coordinacion
4. **Escribe tests** para su modulo
5. **Emite logs estructurados** usando el modulo de observabilidad
6. **Registra hashes** de cualquier artefacto que genere

### Tabla de asignacion

| Agente | Modulos | Archivos principales | Dependencias |
|---|---|---|---|
| `AGENT-INFRA` | M0 | Dockerfile, docker-compose.yml, scripts/setup-oci.sh, configs de Prometheus/Loki/Promtail | Ninguna |
| `AGENT-DB` | M13 | src/db/*, src/db/migrations/* | AGENT-INFRA (DB running) |
| `AGENT-INGESTION` | M1 | src/pipeline/ingestion.py, src/pipeline/preprocessing.py | AGENT-DB (tablas creadas) |
| `AGENT-DETECTION` | M2 | src/models/yolo.py, src/models/cfar.py, src/pipeline/detection.py, src/pipeline/postprocessing.py, scripts/fine-tune.py | AGENT-INGESTION (tiles) |
| `AGENT-COMPRESSION` | M3 | src/models/compression/* , src/models/manager.py | AGENT-DETECTION (modelo base) |
| `AGENT-TRACE` | M4 | src/traceability/* | AGENT-DB, AGENT-DETECTION |
| `AGENT-PROFILES` | M5 | src/profiles/* | AGENT-DETECTION |
| `AGENT-TIPCUE` | M6 | src/tipcue/* | AGENT-DETECTION, AGENT-DB |
| `AGENT-OBSERVABILITY` | M7 | src/observability/*, grafana/dashboards/* | AGENT-INFRA (Grafana running) |
| `AGENT-ORBITAL` | M9+M10+M11+M12 | src/orbital/* | AGENT-DETECTION, AGENT-PROFILES |
| `AGENT-API` | M8 | src/api/*, src/main.py, src/config.py | Todos los demas |
| `AGENT-TEST` | M15 | tests/* | Todos los demas |

### Orden de ejecucion

```
Paralelo 1 (sin dependencias):
  AGENT-INFRA  → Dockerfile, docker-compose, configs
  AGENT-DB     → Esquema SQL, migraciones (tras AGENT-INFRA DB ready)

Paralelo 2 (requiere infra + DB):
  AGENT-INGESTION      → Copernicus auth, search, download, preprocessing
  AGENT-TRACE          → Hasher, recorder
  AGENT-OBSERVABILITY  → Metricas Prometheus, dashboards base

Paralelo 3 (requiere ingestion):
  AGENT-DETECTION   → CFAR, YOLO, fusion, postprocessing
  AGENT-API         → Endpoints basicos (health, detections)

Paralelo 4 (requiere detection):
  AGENT-COMPRESSION → Quantizacion, pruning, distillation
  AGENT-PROFILES    → Perfiles, manager, metrics collector
  AGENT-TIPCUE      → Evaluator, scheduler, zones

Paralelo 5 (requiere profiles + compression):
  AGENT-ORBITAL     → Energy, downlink, latency, resilience, decision engine

Secuencial final:
  AGENT-API           → Endpoints completos (incluyendo orbital)
  AGENT-OBSERVABILITY → Dashboards completos (8 dashboards)
  AGENT-TEST          → Tests completos

Validacion final:
  AGENT-INFRA → docker compose up → todo funciona
```

### Interfaces de comunicacion entre agentes

Los agentes se comunican a traves de:
1. **Archivos en disco** (modelos, imagenes, configs)
2. **Base de datos** (execution_log, detections, tasking_queue)
3. **Clases Python** (imports entre modulos)

Cada agente expone sus interfaces como clases/funciones con type hints estrictos (Pydantic models para datos, dataclasses para configuracion).

---

## Relacion con la licitacion original AIDRA

### Mapeo de requisitos de los pliegos a modulos del MVP

| Requisito AIDRA (pliegos) | Implementacion MVP | Modulo(s) | Seccion Spec |
|---|---|---|---|
| End-to-end vessel detection chain | Copernicus → CFAR+YOLO → PostGIS → Grafana | M1+M2+M8+M7 | 5, 6, 12, 11 |
| Traceability and verification of outputs | execution_log con SHA256 de inputs, modelo, outputs | M4 | 8 |
| Space-representative simulation environment | 5 perfiles Docker (ground → sat-extreme) + perfil energetico + latencia orbital | M5+M9+M11 | 9, 11A, 11C |
| Model compression (quantisation, pruning, KD) | 10 variantes de modelo, cada una con todos los perfiles | M3+M5 | 7, 9 |
| Performance metrics (latency, CPU, RAM, model size) | Prometheus + execution_log + Grafana dashboards + **energia (joules, TOPS/W)** | M7+M9 | 11, 11A |
| Tip & Cue (optional, bonus points) | TipEvaluator → tasking_queue → CueScheduler → re-detection | M6 | 10 |
| D3 Evidence Package | Dashboards + execution_log exportable + hashes (nucleo contractual). **Analisis downlink** como extension opcional | M4+M7 (+M10 opcional) | 8, 11, 11B |
| D4 Analysis Report | Trade-offs: compression × profile × precision (nucleo contractual). **Energia/latencia/resiliencia** como extension opcional | M3 (+M9+M10+M11+M12 opcional) | 7, 11A-11D |
| Interpretability of AI outputs | Confidence scores, bbox, fusion source, profile comparison | M2+M4+M5 | 6, 8, 9 |
| AI Act compliance documentation | Declaracion en D1: clasificacion AI Act (cuando aplique), arquitectura/rationale, provenance de modelos y base legal de datasets. `models_registry` aporta trazabilidad tecnica de apoyo | M4 | 8, 13.2 |
| Documentation of test conditions | execution_log con todos los parametros de cada ejecucion | M4 | 8, 13 |
| **Operational implications of AI-OBDP** | **Analisis downlink y latencia orbital con metricas medidas en el MVP (sin fijar ratios/speedups a priori)** | **M10+M11** | **11B, 11C** |
| **Robustness and reliability** | **Bit-flip tolerance, fallback YOLO→CFAR, drift detection, autonomous decision engine** | **M12** | **11D** |
| **Model optimisation for on-board execution** | **Perfil energetico (joules/inferencia), TOPS/W, presupuesto orbital por tipo de satelite** | **M9** | **11A** |

### Mapeo de tareas del contrato (Appendix I.1) a modulos

| Tarea | Descripcion | Modulo(s) MVP | Como se cubre |
|---|---|---|---|
| T0 | Gestion de proyecto | — | docker-compose.yml + TECHNICAL_SPEC.md + README |
| T1 | Planificacion demo + escenario | M1 | Zonas de busqueda, datasets, modelos seleccionados |
| T2 | Ejecucion de la demo | M1+M2+M3+M5 | Pipeline end-to-end con perfiles y compresion |
| T3 | Analisis y recomendaciones | M7 | Dashboards Grafana + /benchmarks/compare endpoint |

### Mapeo de entregables a artefactos del MVP

| Entregable | Artefactos MVP |
|---|---|
| D1 — Plan de demo | `TECHNICAL_SPEC.md` + `mvp_oci.md` + zonas + modelos + perfiles orbitales |
| D2 — Informe intermedio | Dashboard metricas pipeline (Grafana) + dashboard energia |
| D3 — Evidence Package | execution_log (exportable), dashboards, hashes, config Docker (nucleo contractual). Analisis downlink como anexo opcional |
| D4 — Analisis final | Benchmarks compresion + perfiles restriccion (nucleo contractual). Energia/latencia/resiliencia como anexos opcionales |
| D5 — Cierre contractual | README + docker compose + documentacion completa |

### Mapeo de criterios de evaluacion (Annex I, Table 6) a valor demostrable

| Criterio | Puntos | Que demuestra el MVP |
|---|---|---|
| Q1 — Equipo (15 pts) | — | N/A (proyecto personal, no aplica) |
| Q2 — Gestion (15 pts) | Plan proyecto (10), riesgos (5) | TECHNICAL_SPEC con fases, dependencias, criterios de aceptacion |
| Q3 — Propuesta tecnica (40 pts) | Metodologia (10), GEOINT (10), Demo+traceability (20) | Pipeline funcional, trazabilidad SHA256, perfiles de restriccion, Tip & Cue, dashboards |

---

---

## 20.5 Detalle de Endpoints API (Handlers)

### `src/api/router.py`

```python
"""
Router principal que agrupa todos los sub-routers.
"""

from fastapi import APIRouter
from src.api import (
    health, detections, pipeline, benchmarks,
    traceability, tasking, metrics, models_api
)

router = APIRouter(prefix="/api")

router.include_router(health.router)
router.include_router(detections.router)
router.include_router(pipeline.router)
router.include_router(benchmarks.router)
router.include_router(traceability.router)
router.include_router(tasking.router)
router.include_router(metrics.router)
router.include_router(models_api.router)
```

### `src/api/health.py`

```python
from fastapi import APIRouter, Depends
from src.db.models import HealthResponse
from src.db.connection import Database

router = APIRouter(tags=["system"])

@router.get("/health", response_model=HealthResponse)
async def health_check(db: Database = Depends(get_db)):
    """
    Verifica el estado de todos los componentes.

    Checks:
    1. Conexion a PostgreSQL (simple SELECT 1)
    2. Numero de modelos cargados en models/
    3. Estado del scheduler (running/stopped)
    4. Uptime del servicio
    """
```

### `src/api/detections.py`

```python
from fastapi import APIRouter, Query, HTTPException
from uuid import UUID

router = APIRouter(tags=["detections"])

@router.get("/detections", response_model=PaginatedResponse)
async def list_detections(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    profile: str | None = Query(None, description="Filtrar por perfil de restriccion"),
    model: str | None = Query(None, description="Filtrar por nombre de modelo"),
    min_confidence: float | None = Query(None, ge=0, le=1),
    bbox: str | None = Query(None, description="Bounding box: lon_min,lat_min,lon_max,lat_max"),
    date_from: str | None = Query(None, description="Fecha inicio ISO 8601"),
    date_to: str | None = Query(None, description="Fecha fin ISO 8601"),
):
    """
    Lista detecciones con filtros opcionales.
    Soporta filtro geoespacial por bounding box.
    Paginacion con limit/offset.
    """

@router.get("/detections/{detection_id}")
async def get_detection(detection_id: UUID):
    """
    Retorna deteccion con cadena de proveniencia completa.
    JOIN con execution_log para incluir todos los datos de trazabilidad.
    """
```

### `src/api/pipeline.py`

```python
from fastapi import APIRouter, BackgroundTasks, HTTPException
from src.db.models import PipelineTriggerRequest, PipelineTriggerResponse

router = APIRouter(tags=["pipeline"])

@router.post("/pipeline/trigger", response_model=PipelineTriggerResponse)
async def trigger_pipeline(
    request: PipelineTriggerRequest,
    background_tasks: BackgroundTasks,
):
    """
    Lanza pipeline de deteccion.

    El pipeline se ejecuta en background (BackgroundTasks de FastAPI).
    Retorna inmediatamente con el execution_id para consultar status.

    Validaciones:
    - Verificar que el modelo existe en models_registry
    - Verificar que el perfil es valido
    - Verificar que no hay otro pipeline en ejecucion (max_concurrent=1)
    """

@router.post("/pipeline/trigger-all-profiles")
async def trigger_all_profiles(
    request: PipelineTriggerRequest,
    background_tasks: BackgroundTasks,
):
    """
    Ejecuta la misma imagen con TODOS los perfiles de restriccion.
    Primero descarga la imagen una vez, luego ejecuta el pipeline
    5 veces con diferentes limites de recursos.
    """

@router.get("/pipeline/status", response_model=PipelineStatusResponse)
async def pipeline_status():
    """Estado del pipeline actualmente en ejecucion (si hay alguno)."""
```

### `src/api/benchmarks.py`

```python
from fastapi import APIRouter, Query

router = APIRouter(tags=["benchmarks"])

@router.get("/benchmarks")
async def list_benchmarks(
    model: str | None = Query(None),
    profile: str | None = Query(None),
):
    """
    Resultados agregados de benchmarks.
    Agrupa por modelo + perfil y calcula estadisticas
    (media, P50, P95 de latencia, RAM, CPU, detecciones).
    """

@router.get("/benchmarks/compare")
async def compare_benchmarks(
    models: str | None = Query(None, description="Modelos separados por coma"),
    profiles: str | None = Query(None, description="Perfiles separados por coma"),
    image_id: str | None = Query(None, description="Comparar en la misma imagen"),
):
    """
    Genera matriz comparativa: modelo x perfil.
    Si se proporciona image_id, solo compara ejecuciones de esa imagen.
    """
```

### `src/api/traceability.py`

```python
from fastapi import APIRouter
from uuid import UUID

router = APIRouter(tags=["traceability"])

@router.get("/traceability/{execution_id}")
async def get_traceability(execution_id: UUID):
    """
    Cadena de proveniencia completa de una ejecucion.

    Retorna:
    - Todos los campos de execution_log
    - Detecciones asociadas
    - Si fue un cue: datos del tip que lo genero
    - Si genero cues: datos de los cues generados
    """
```

### `src/api/tasking.py`

```python
from fastapi import APIRouter, Query
from src.db.models import CueCreateRequest, TaskingEntry

router = APIRouter(tags=["tasking"])

@router.get("/tasking/queue", response_model=list[TaskingEntry])
async def list_tasking_queue(
    status: str | None = Query(None, description="pending, executing, completed"),
    limit: int = Query(50, ge=1, le=200),
):
    """Lista la cola de Tip & Cue con filtro por status."""

@router.post("/tasking/cue")
async def create_manual_cue(request: CueCreateRequest):
    """
    Crear un cue manualmente (sin pasar por el TipEvaluator).
    Util para forzar una observacion en una zona especifica.
    """
```

### `src/api/metrics.py`

```python
from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

router = APIRouter(tags=["monitoring"])

@router.get("/metrics")
async def prometheus_metrics():
    """Endpoint para scraping de Prometheus."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
```

### `src/api/models_api.py`

```python
from fastapi import APIRouter
from src.db.models import ModelInfo

router = APIRouter(tags=["models"])

@router.get("/models", response_model=list[ModelInfo])
async def list_models():
    """Lista todos los modelos registrados y sus variantes de compresion."""

@router.get("/profiles")
async def list_profiles():
    """Lista todos los perfiles de restriccion disponibles."""

@router.get("/zones")
async def list_zones():
    """Lista todas las zonas de busqueda predefinidas."""
```

---

## 20.6 Loki Logger Estructurado

### `src/observability/loki_logger.py`

```python
"""
Logger estructurado que envia logs a Loki.

Cada log incluye:
- timestamp
- level (INFO, WARNING, ERROR)
- message
- module (aidra.pipeline, aidra.detection, etc.)
- extra fields (execution_id, profile, model, etc.)

Formato: JSON para Loki, texto para stdout.
"""

import logging
import json
from logging.handlers import HTTPHandler
from src.config import Settings

def setup_logging(settings: Settings) -> None:
    """
    Configura logging para toda la aplicacion.

    1. Root logger: level segun settings.log_level
    2. Stream handler: formato texto para stdout (Docker logs)
    3. Loki handler: formato JSON para Loki (si loki_url configurada)
    """
    root = logging.getLogger("aidra")
    root.setLevel(getattr(logging, settings.log_level.upper()))

    # Stream handler (stdout → Docker logs → Promtail → Loki)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    root.addHandler(stream_handler)

    # Nota: Promtail recoge los logs de Docker automaticamente,
    # asi que no es necesario un handler HTTP directo a Loki.
    # El formato JSON en el log permite a Loki parsear campos.


class StructuredLogger:
    """
    Logger con contexto extra para el pipeline.

    Uso:
        log = StructuredLogger("aidra.pipeline", execution_id=uuid, profile="ground")
        log.info("Pipeline started", extra={"zone": "gibraltar"})
        log.error("Download failed", extra={"error": str(e)})
    """

    def __init__(self, name: str, **context):
        self.logger = logging.getLogger(name)
        self.context = context

    def info(self, msg: str, extra: dict | None = None):
        self.logger.info(msg, extra={**self.context, **(extra or {})})

    def warning(self, msg: str, extra: dict | None = None):
        self.logger.warning(msg, extra={**self.context, **(extra or {})})

    def error(self, msg: str, extra: dict | None = None, exc_info: bool = False):
        self.logger.error(msg, extra={**self.context, **(extra or {})}, exc_info=exc_info)
```

---

## 21. Orquestador del Pipeline (Engine)

### 21.1 Archivo `src/pipeline/engine.py`

Este es el corazon del sistema. Coordina todos los pasos del pipeline en secuencia.

```python
"""
Orquestador del pipeline completo de deteccion de barcos.

Flujo de ejecucion:
1. Validar parametros de entrada
2. Autenticar con Copernicus (si no hay token vigente)
3. Buscar imagen en Copernicus (o usar image_id proporcionado)
4. Descargar imagen
5. Calcular hash SHA256 de la imagen
6. Preprocesar (calibrar, filtrar, tilear)
7. Ejecutar deteccion (CFAR + YOLO)
8. Postprocesar (fusion, NMS, geolocalizacion)
9. Calcular hash SHA256 del resultado
10. Registrar en execution_log
11. Guardar detecciones en PostGIS
12. Evaluar Tip & Cue (si habilitado)
13. Emitir metricas Prometheus
14. Limpiar archivos temporales
15. Retornar resultado

Manejo de errores:
- Cada paso tiene timeout individual
- Si un paso falla, se registra en execution_log con status="error"
- La imagen descargada se limpia incluso si el pipeline falla
- Los errores se loguean en Loki con contexto completo
"""

import asyncio
import time
import logging
from uuid import UUID
from pathlib import Path

logger = logging.getLogger("aidra.pipeline")

class PipelineEngine:
    """
    Constructor:
    - ingester: ImageIngester
    - preprocessor: module (preprocessing functions)
    - detector: DetectionEngine
    - recorder: ExecutionRecorder
    - profile_manager: ProfileManager
    - tip_evaluator: TipEvaluator | None
    - config: Settings

    Metodos principales:
    - async run(request: PipelineRequest) -> PipelineResult
    - async run_all_profiles(request: PipelineRequest) -> dict[str, PipelineResult]
    """

    async def run(self, request: PipelineRequest) -> PipelineResult:
        """
        Ejecuta pipeline completo.

        Pasos detallados:
        """
        execution_id = None
        image_path = None
        start_time = time.monotonic()

        try:
            # 1. Crear registro preliminar en execution_log (status='running')
            execution_id = await self.recorder.create_pending(request)

            # 2. Obtener imagen
            if request.image_id:
                # Buscar imagen especifica
                search_results = await self.ingester.search_by_id(request.image_id)
            else:
                # Buscar imagen mas reciente en la zona
                zone = SEARCH_ZONES[request.zone]
                search_results = await self.ingester.search(
                    bbox=zone["bbox"],
                    start_date=request.date_from or (datetime.now() - timedelta(days=7)),
                    end_date=request.date_to or datetime.now(),
                    max_results=1
                )

            if not search_results:
                raise PipelineError("No images found for the given criteria")

            product = search_results[0]

            # 3. Descargar imagen
            download_start = time.monotonic()
            image_path = await self.ingester.download(product)
            download_ms = (time.monotonic() - download_start) * 1000

            # 4. Hash de la imagen
            image_hash = compute_sha256(image_path)

            # 5. Preprocesar
            preprocess_start = time.monotonic()
            preprocessed = preprocess_full(
                product_dir=image_path,
                aoi_bbox=request.aoi_bbox,
                tile_size=self.config.tile_size,
                tile_overlap=self.config.tile_overlap
            )
            preprocessing_ms = (time.monotonic() - preprocess_start) * 1000

            # 6. Detectar (bajo perfil de restriccion si aplica)
            if request.profile != "ground":
                result = await self.profile_manager.run_with_profile(
                    profile_name=request.profile,
                    pipeline_fn=self.detector.run,
                    tiles=preprocessed["tiles"],
                    constraint_profile=request.profile
                )
                detection_result = result.result
                profiled_metrics = result
            else:
                detection_result = self.detector.run(
                    tiles=preprocessed["tiles"],
                    constraint_profile="ground"
                )
                profiled_metrics = None

            # 7. Hash del resultado
            output_hash = compute_result_hash(
                [d.model_dump() for d in detection_result.detections]
            )

            # 8. Registrar en execution_log
            execution_record = ExecutionRecord(
                id=execution_id,
                image_id=product.product_id,
                image_title=product.title,
                image_hash=image_hash,
                image_sensing_date=product.sensing_date,
                image_size_mb=product.size_mb,
                search_zone=request.zone,
                model_name=self.detector.yolo.model_name,
                model_version=self.detector.yolo.model_version,
                model_hash=self.detector.yolo.model_hash,
                model_size_mb=self.detector.yolo.model_size_mb,
                model_format=self.detector.yolo.model_format,
                confidence_threshold=request.confidence_threshold,
                constraint_profile=request.profile,
                num_detections=len(detection_result.detections),
                avg_confidence=detection_result.metrics.avg_confidence,
                download_ms=download_ms,
                preprocessing_ms=preprocessing_ms,
                inference_ms=detection_result.metrics.total_inference_ms,
                peak_ram_mb=detection_result.metrics.peak_ram_mb,
                cpu_usage_pct=detection_result.metrics.cpu_percent,
                num_tiles=preprocessed["metadata"]["num_tiles"],
                output_hash=output_hash,
                status="success",
                trigger_type=request.trigger_type,
                triggered_by=request.triggered_by,
            )
            await self.recorder.update(execution_record)

            # 9. Guardar detecciones en PostGIS
            await self._save_detections(execution_id, detection_result.detections)

            # 10. Evaluar Tip & Cue
            if self.tip_evaluator and self.config.tipcue_enabled:
                tips = self.tip_evaluator.evaluate(
                    detections=detection_result.detections,
                    execution_id=execution_id
                )
                for tip in tips:
                    if tip.should_cue:
                        await self._create_cue(tip)

            # 11. Emitir metricas
            self._emit_metrics(execution_record, detection_result)

            total_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                "Pipeline completed",
                extra={
                    "execution_id": str(execution_id),
                    "profile": request.profile,
                    "detections": len(detection_result.detections),
                    "total_ms": total_ms,
                }
            )

            return PipelineResult(
                execution_id=execution_id,
                status="success",
                detections=detection_result.detections,
                metrics=detection_result.metrics,
                execution_record=execution_record,
            )

        except Exception as e:
            # Registrar error
            if execution_id:
                await self.recorder.update_status(
                    execution_id, status="error", error_message=str(e)
                )
            logger.error(
                "Pipeline failed",
                extra={"execution_id": str(execution_id), "error": str(e)},
                exc_info=True
            )
            PIPELINE_RUNS_TOTAL.labels(
                profile=request.profile,
                model_variant=request.model,
                status="error"
            ).inc()
            raise

        finally:
            # Limpiar imagen temporal
            if image_path and image_path.exists():
                await self._cleanup(image_path)


class PipelineRequest:
    """
    Pydantic model para solicitud de ejecucion del pipeline.

    Campos:
    - zone: str = "gibraltar"           # Zona de busqueda
    - model: str = "yolov8n-sar"        # Variante de modelo
    - profile: str = "ground"           # Perfil de restriccion
    - image_id: str | None = None       # ID especifico de imagen (si None, busca la mas reciente)
    - aoi_bbox: list[float] | None      # Sub-area de interes [lon_min, lat_min, lon_max, lat_max]
    - confidence_threshold: float = 0.25
    - iou_threshold: float = 0.45
    - date_from: datetime | None = None
    - date_to: datetime | None = None
    - trigger_type: str = "manual"      # "manual", "scheduled", "cue"
    - triggered_by: UUID | None = None  # Si es cue, ID de la ejecucion que lo genero
    """

class PipelineResult:
    """
    Pydantic model para resultado del pipeline.

    Campos:
    - execution_id: UUID
    - status: str                       # "success", "error"
    - detections: list[Detection]
    - metrics: DetectionMetrics
    - execution_record: ExecutionRecord
    - error: str | None
    """
```

---

## 22. APScheduler — Jobs Programados

### 22.1 Configuracion del scheduler

```python
"""
Jobs programados con APScheduler (in-process, sin broker externo).

Jobs definidos:
1. scheduled_scan: Ejecuta pipeline en zonas predefinidas periodicamente
2. cue_processor: Procesa cues pendientes en tasking_queue
3. cleanup_images: Limpia imagenes temporales antiguas
4. health_probe: Verifica conectividad con Copernicus y DB
"""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

def configure_scheduler(engine: PipelineEngine, config: Settings) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler()

    # Job 1: Escaneo programado de zonas
    # Ejecuta pipeline cada N horas en la zona por defecto
    scheduler.add_job(
        scheduled_scan,
        trigger=IntervalTrigger(hours=config.scheduler_interval_hours),
        kwargs={"engine": engine, "zone": config.default_zone},
        id="scheduled_scan",
        name="Scheduled zone scan",
        max_instances=1,  # No permitir ejecuciones concurrentes
        misfire_grace_time=3600,
    )

    # Job 2: Procesador de cues (Tip & Cue)
    # Revisa tasking_queue cada 15 minutos
    if config.tipcue_enabled:
        scheduler.add_job(
            process_pending_cues,
            trigger=IntervalTrigger(minutes=15),
            kwargs={"engine": engine},
            id="cue_processor",
            name="Tip & Cue processor",
            max_instances=1,
        )

    # Job 3: Limpieza de imagenes temporales
    # Borra imagenes descargadas hace mas de 24h
    scheduler.add_job(
        cleanup_old_images,
        trigger=CronTrigger(hour=3, minute=0),  # 3:00 AM
        kwargs={"images_dir": config.images_dir, "max_age_hours": 24},
        id="cleanup_images",
        name="Image cleanup",
    )

    # Job 4: Health probe
    # Verifica Copernicus + DB cada 30 min
    scheduler.add_job(
        health_probe,
        trigger=IntervalTrigger(minutes=30),
        id="health_probe",
        name="System health probe",
    )

    return scheduler

async def scheduled_scan(engine: PipelineEngine, zone: str):
    """Ejecuta pipeline en la zona especificada."""
    request = PipelineRequest(
        zone=zone,
        model=engine.config.default_model,
        profile="ground",
        trigger_type="scheduled",
    )
    await engine.run(request)

async def process_pending_cues(engine: PipelineEngine):
    """Procesa cues pendientes en tasking_queue, ordenados por prioridad."""
    # 1. Consultar cues con status='pending', ordenados por prioridad DESC
    # 2. Para cada cue:
    #    a. Buscar imagen mas reciente en target_bbox
    #    b. Ejecutar pipeline con trigger_type='cue', triggered_by=cue.triggered_by
    #    c. Actualizar cue con execution_id y status='completed'
    #    d. Comparar detecciones con las del tip original
    pass

async def cleanup_old_images(images_dir: str, max_age_hours: int = 24):
    """Borra imagenes descargadas hace mas de max_age_hours."""
    pass

async def health_probe():
    """Verifica conectividad con servicios externos."""
    pass
```

---

## 23. Manejo de Errores y Resiliencia

### 23.1 Estrategia de errores

```python
# src/pipeline/engine.py (excepciones)

class PipelineError(Exception):
    """Error generico del pipeline."""
    pass

class IngestionError(PipelineError):
    """Error durante la descarga/busqueda de imagenes."""
    pass

class AuthenticationError(IngestionError):
    """Error de autenticacion con Copernicus."""
    pass

class PreprocessingError(PipelineError):
    """Error durante el preprocesamiento SAR."""
    pass

class DetectionError(PipelineError):
    """Error durante la inferencia."""
    pass

class ProfileError(PipelineError):
    """Error al ejecutar bajo perfil de restriccion."""
    pass

class OOMError(ProfileError):
    """Out of memory bajo perfil de restriccion."""
    pass

class TimeoutError(ProfileError):
    """Timeout bajo perfil de restriccion."""
    pass
```

### 23.2 Retry policy

```python
# Reintentos solo para operaciones de red (Copernicus API)
# NO reintentar inferencia (si falla con un perfil, es un dato valido)

RETRY_CONFIG = {
    "copernicus_auth": {"max_retries": 3, "backoff_seconds": [2, 5, 10]},
    "copernicus_search": {"max_retries": 2, "backoff_seconds": [3, 10]},
    "copernicus_download": {"max_retries": 2, "backoff_seconds": [5, 15]},
    "db_write": {"max_retries": 3, "backoff_seconds": [1, 2, 5]},
}
```

### 23.3 Timeouts

```python
TIMEOUTS = {
    "copernicus_auth": 30,        # segundos
    "copernicus_search": 60,
    "copernicus_download": 600,   # 10 min (imagenes grandes)
    "preprocessing": 300,          # 5 min
    "inference_per_tile": 60,      # 1 min por tile
    "inference_total": 600,        # 10 min total
    "pipeline_total": 1800,        # 30 min maximo por ejecucion completa
}
```

---

## 24. Seguridad y Buenas Practicas

### 24.1 Datos sensibles

- **NUNCA** commitear `.env` a git (esta en .gitignore)
- Credenciales de Copernicus solo en variables de entorno
- Passwords de DB y Grafana solo en variables de entorno
- Token OAuth2 de Copernicus se refresca automaticamente y nunca se almacena en disco

### 24.2 Acceso a la API

- La API escucha en 0.0.0.0:8000 (accesible desde fuera)
- CORS configurado para origenes especificos
- No hay autenticacion en el MVP (proyecto personal, no produccion)
- Si se expone a internet: usar Nginx reverse proxy con rate limiting

### 24.3 Docker

- Socket Docker montado read-write para perfiles de restriccion
- Los contenedores de restriccion se ejecutan con `--rm` (limpieza automatica)
- No se ejecuta nada como root dentro de los contenedores de restriccion

### 24.4 Base de datos

- Pool de conexiones asyncpg (max 10 conexiones)
- Queries parametrizadas (sin concatenacion de strings SQL)
- execution_log es append-only (no updates destructivos, no deletes)

---

## 25. Datos de Referencia para Validacion

### 25.1 Imagen de prueba Sentinel-1

Para desarrollo y tests sin descargar imagenes reales de Copernicus:

- **Producto**: S1A_IW_GRDH_1SDV_20260415T174529_20260415T174554_058123_071DE3_E2F4
- **Zona**: Estrecho de Gibraltar
- **Tamano**: ~800 MB
- **Formato**: TIFF (VV + VH polarization)
- **Alternativa para tests**: crear imagen SAR sintetica con numpy:
  - Fondo: ruido Rayleigh (simula mar en SAR)
  - Barcos: puntos brillantes gaussianos (sigma=3, amplitud 10-50x fondo)
  - Tamano: 640x640 px para un tile de test

```python
# Generar imagen SAR sintetica de prueba
import numpy as np

def generate_synthetic_sar_tile(
    size: int = 640,
    num_vessels: int = 5,
    noise_mean: float = 0.3,
    vessel_amplitude: float = 5.0,
    seed: int = 42
) -> tuple[np.ndarray, list[dict]]:
    """
    Genera tile SAR sintetico con barcos simulados.

    Returns:
        (image, ground_truth)
        image: np.ndarray float32 (size x size)
        ground_truth: lista de dicts con bbox y centro de cada barco
    """
    rng = np.random.default_rng(seed)

    # Fondo: distribucion Rayleigh (simula clutter marino en SAR)
    background = rng.rayleigh(scale=noise_mean, size=(size, size)).astype(np.float32)

    ground_truth = []
    for _ in range(num_vessels):
        # Posicion aleatoria (evitar bordes)
        cx = rng.integers(50, size - 50)
        cy = rng.integers(50, size - 50)

        # Tamano aleatorio del barco (3-15 px)
        w = rng.integers(3, 15)
        h = rng.integers(3, 15)

        # Crear barco como gaussiana 2D
        y_grid, x_grid = np.ogrid[
            max(0, cy - h):min(size, cy + h),
            max(0, cx - w):min(size, cx + w)
        ]
        gaussian = np.exp(
            -((x_grid - cx)**2 / (2 * (w/3)**2) + (y_grid - cy)**2 / (2 * (h/3)**2))
        )
        background[
            max(0, cy - h):min(size, cy + h),
            max(0, cx - w):min(size, cx + w)
        ] += vessel_amplitude * gaussian

        ground_truth.append({
            "bbox": [cx - w, cy - h, cx + w, cy + h],
            "center": [cx, cy],
            "width": w * 2,
            "height": h * 2,
        })

    return background, ground_truth
```

### 25.2 Valores esperados de referencia

| Metrica | Ground | Sat-High | Sat-Mid | Sat-Low | Sat-Extreme |
|---|---|---|---|---|---|
| Inferencia YOLO (ms/tile) | 50-150 | 100-300 | 200-600 | 500-2000 | Timeout/OOM |
| Inferencia CFAR (ms/tile) | 10-30 | 15-50 | 30-100 | 50-200 | 100-500 |
| RAM pico (MB) | 500-2000 | 500-2000 | 500-1500 | 500-1000 | OOM |
| Detecciones | N | N | N | N-1 | 0 (OOM) |

Nota: estos son valores aproximados. Los valores reales dependeran del tamano
de la imagen, numero de tiles, y el modelo especifico.

### 25.3 Valores esperados de compresion

| Variante | Tamano | Latencia vs base | RAM vs base | Detecciones vs base |
|---|---|---|---|---|
| Base (FP32) | 6 MB | 1.0x | 1.0x | 100% |
| FP16 | ~3 MB | 0.7-0.9x | 0.6-0.8x | ~99% |
| INT8 dynamic | ~1.5 MB | 0.5-0.7x | 0.4-0.6x | ~95-98% |
| INT8 static | ~1.5 MB | 0.4-0.6x | 0.3-0.5x | ~93-97% |
| Pruned 30% | ~4.2 MB | 0.8-0.9x | 0.8-0.9x | ~97-99% |
| Pruned 50% | ~3 MB | 0.6-0.8x | 0.6-0.8x | ~90-95% |
| Pruned 30% + INT8 | ~1 MB | 0.3-0.5x | 0.3-0.5x | ~88-94% |

---

## 26. Reglas del Enjambre

### 26.1 Reglas generales para todos los agentes

1. **Idioma del codigo**: Ingles (nombres de variables, funciones, clases, comentarios)
2. **Idioma de la documentacion**: Espanol (README, TECHNICAL_SPEC, docstrings descriptivos)
3. **Type hints**: Obligatorios en todas las funciones publicas
4. **Pydantic**: Usar Pydantic v2 models para todos los datos que cruzan fronteras de modulo
5. **Async**: Todas las operaciones de I/O (DB, HTTP, disco) deben ser async
6. **Logging**: Usar `logging.getLogger("aidra.<modulo>")` con structured logging
7. **Metricas**: Cada operacion significativa emite al menos una metrica Prometheus
8. **Hashing**: Todo artefacto (imagen, modelo, resultado) se hashea con SHA256
9. **Tests**: Minimo un test por funcion publica del modulo
10. **No dependencias implicitas**: Si tu modulo necesita algo de otro, importalo explicitamente
11. **Configuracion**: Toda constante configurable va en Settings (pydantic-settings)
12. **No print()**: Usar logger. Nunca print().
13. **No secrets en codigo**: Todo en variables de entorno.
14. **Formato**: Ruff con line-length=100.

### 26.2 Protocolo de integracion entre agentes

Cuando un agente termina su modulo:
1. Verifica que sus tests pasan (`pytest tests/test_<modulo>/ -v`)
2. Verifica que el linter pasa (`ruff check src/<modulo>/`)
3. Documenta las interfaces publicas (clases/funciones que otros agentes usan)
4. Crea un archivo `src/<modulo>/README.md` con:
   - Que hace el modulo
   - Como usarlo (ejemplo de codigo minimo)
   - Que necesita de otros modulos (dependencias)
   - Que expone a otros modulos (interfaces publicas)

### 26.3 Resolucion de conflictos

Si dos agentes necesitan modificar el mismo archivo:
1. Solo `AGENT-API` modifica `src/main.py` y `src/api/router.py`
2. Solo `AGENT-DB` modifica archivos en `src/db/migrations/`
3. Solo `AGENT-INFRA` modifica `docker-compose.yml` y `Dockerfile`
4. Cada agente es dueno de su directorio (ej: `AGENT-DETECTION` es dueno de `src/models/` y `src/pipeline/detection.py`)
5. Los Pydantic models compartidos van en `src/db/models.py` (propiedad de `AGENT-DB`, otros agentes pueden proponer cambios)

### 26.4 Datos compartidos

Los agentes comparten datos a traves de:
1. **Base de datos**: fuente de verdad para execution_log, detections, tasking_queue, models_registry
2. **Filesystem**: `/app/models/` para pesos de modelos, `/data/images/` para imagenes temporales
3. **Imports Python**: clases y funciones expuestas en `__init__.py` de cada modulo

---

## 27. Checklist Pre-Despliegue

Antes de declarar el proyecto listo:

- [ ] `docker compose up -d` levanta todos los servicios en < 5 minutos
- [ ] `curl localhost:8000/api/health` retorna `{"status":"ok"}`
- [ ] Grafana accesible en `localhost:3000` con 8 dashboards
- [ ] Swagger UI accesible en `localhost:8000/docs`
- [ ] Pipeline ejecuta end-to-end con imagen Sentinel-1 real
- [ ] Detecciones visibles en mapa GeoMap de Grafana
- [ ] execution_log tiene al menos 1 registro con todos los hashes
- [ ] Al menos 5 variantes de modelo comprimido generadas
- [ ] Comparativa de perfiles ejecutada (ground → sat-extreme)
- [ ] Tip & Cue genera al menos 1 cue automatico
- [ ] Metricas Prometheus con datos reales
- [ ] Logs en Loki consultables
- [ ] Perfil energetico: joules/inferencia calculado para cada variante
- [ ] Presupuesto orbital: tabla con imagenes/orbita por tipo de satelite
- [ ] Analisis downlink: ratio de compresion >10.000:1 demostrado
- [ ] Latencia orbital: comparativa con/sin OBDP generada
- [ ] Bit-flip sweep: curva de degradacion con threshold critico
- [ ] Decision Engine: simulacion de orbita con >10 decisiones
- [ ] `pytest tests/ -v` pasa al 100%
- [ ] `.env.example` documentado
- [ ] README con instrucciones de despliegue
- [ ] No hay credenciales en el repositorio

---

---

## 28. Archivos de Configuracion del Proyecto

### 28.1 `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
.eggs/
*.egg

# Virtual environments
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Environment
.env
.env.local
.env.production

# Docker
docker-compose.override.yml

# Data (demasiado grandes para git)
models/*.pt
models/*.onnx
models/*.bin
data/
images/
*.tiff
*.tif
*.zip
*.tar.gz

# Datasets
datasets/

# Grafana (generado)
grafana/grafana.db

# Claude Code local settings
.claude/settings.local.json

# OS
Thumbs.db
.DS_Store

# Logs
*.log
logs/

# Test
.coverage
htmlcov/
.pytest_cache/
```

### 28.2 Instrucciones de Primer Arranque

```bash
# 1. Clonar el repositorio
git clone <repo-url> AIDRA && cd AIDRA

# 2. Copiar y configurar variables de entorno
cp .env.example .env
# Editar .env con credenciales reales:
#   - DB_PASSWORD: password segura para PostgreSQL
#   - COPERNICUS_USER: email registrado en https://dataspace.copernicus.eu
#   - COPERNICUS_PASSWORD: password de Copernicus
#   - GRAFANA_PASSWORD: password para admin de Grafana

# 3. Descargar modelos preentrenados
./scripts/download-models.sh
# Descarga:
#   - yolov8n.pt (base, 6 MB)
#   - yolov8n-sar.pt (fine-tuned, 6 MB) -- si disponible
#   Los coloca en models/

# 4. Construir y levantar
docker compose build
docker compose up -d

# 5. Verificar
curl http://localhost:8000/api/health
# Debe retornar: {"status":"ok","db":"connected","models_loaded":1,...}

# 6. Acceder a los servicios
# API + Swagger:  http://localhost:8000/docs
# Grafana:        http://localhost:3000 (admin / <GRAFANA_PASSWORD>)
# Prometheus:     http://localhost:9090

# 7. Ejecutar primer pipeline manualmente
curl -X POST http://localhost:8000/api/pipeline/trigger \
  -H "Content-Type: application/json" \
  -d '{"zone": "gibraltar", "model": "yolov8n-sar", "profile": "ground"}'

# 8. Verificar resultados
curl http://localhost:8000/api/detections?limit=10
# Deberia mostrar detecciones con coordenadas
```

### 28.3 Script `scripts/download-models.sh`

```bash
#!/bin/bash
set -euo pipefail

MODELS_DIR="${1:-models}"
mkdir -p "$MODELS_DIR"

echo "=== Downloading AIDRA models ==="

# YOLOv8 nano (base model from Ultralytics)
if [ ! -f "$MODELS_DIR/yolov8n.pt" ]; then
    echo "Downloading YOLOv8n base model..."
    python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
    mv yolov8n.pt "$MODELS_DIR/"
    echo "  -> $MODELS_DIR/yolov8n.pt ($(du -h "$MODELS_DIR/yolov8n.pt" | cut -f1))"
else
    echo "  YOLOv8n already exists, skipping"
fi

# Calculate SHA256 of all models
echo ""
echo "=== Model checksums ==="
for f in "$MODELS_DIR"/*.pt "$MODELS_DIR"/*.onnx 2>/dev/null; do
    [ -f "$f" ] && echo "  $(sha256sum "$f")"
done

echo ""
echo "=== Done. Models available in $MODELS_DIR/ ==="
ls -lh "$MODELS_DIR/"
```

---

## 29. Diagrama de Flujo de Datos Completo

```
                                    ┌─────────────┐
                                    │   Usuario    │
                                    │  (API/Grafana)│
                                    └──────┬──────┘
                                           │
                               POST /pipeline/trigger
                                           │
                                    ┌──────▼──────┐
                                    │   FastAPI    │
                                    │   Router     │
                                    └──────┬──────┘
                                           │
                                    ┌──────▼──────┐
                                    │  Pipeline    │
                                    │  Engine      │
                                    └──────┬──────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
             ┌──────▼──────┐        ┌──────▼──────┐       ┌──────▼──────┐
             │  Ingestion  │        │   Profile   │       │    Tip &    │
             │  Module     │        │   Manager   │       │    Cue      │
             └──────┬──────┘        └──────┬──────┘       └──────┬──────┘
                    │                      │                      │
         ┌──────────┤                      │               ┌─────┴─────┐
         │          │               ┌──────▼──────┐        │ Tasking   │
  ┌──────▼──────┐   │               │  Resource   │        │ Queue     │
  │ Copernicus  │   │               │  Collector  │        │ (PostGIS) │
  │ API         │   │               └──────┬──────┘        └───────────┘
  └──────┬──────┘   │                      │
         │          │               ┌──────▼──────┐
  ┌──────▼──────┐   │               │  Detection  │
  │ Download    │   │               │  Engine     │
  │ S1 GRD      │   │               └──────┬──────┘
  └──────┬──────┘   │                      │
         │          │          ┌────────────┼────────────┐
         │   ┌──────▼──────┐   │            │            │
         │   │ Preprocess  │   │     ┌──────▼───┐  ┌─────▼──────┐
         │   │ SAR         │   │     │   CFAR   │  │   YOLO     │
         │   │ (calibrate, │   │     │ Detector │  │  Detector  │
         │   │  tile)      │   │     └──────┬───┘  └─────┬──────┘
         │   └──────┬──────┘   │            │            │
         │          │          │     ┌──────▼────────────▼──────┐
         │          │          │     │    Fusion + NMS           │
         │          │          │     │    Geolocalizacion        │
         │          │          │     └──────────┬───────────────┘
         │          │          │                │
         │   ┌──────▼──────────▼────────────────▼──────┐
         │   │              PostGIS                     │
         │   │  execution_log | detections | tasking_q  │
         │   └──────────────────┬───────────────────────┘
         │                      │
         │          ┌───────────┼───────────┐
         │          │           │           │
         │   ┌──────▼───┐ ┌────▼────┐ ┌────▼────┐
         │   │Prometheus│ │  Loki   │ │ Grafana │
         │   │ metrics  │ │  logs   │ │ 8 dash  │
         │   └──────────┘ └─────────┘ └─────────┘
         │
  ┌──────▼──────┐
  │ Hash SHA256 │
  │ (input,     │
  │  model,     │
  │  output)    │
  └─────────────┘
```

---

*Fin del documento de especificaciones tecnicas v1.0*
*Generado para uso por enjambre de agentes LLM.*
*Cada agente debe leer su seccion asignada (ver seccion 20) y respetar las interfaces definidas.*
*Cualquier decision de implementacion no cubierta aqui debe documentarse en el README del modulo correspondiente.*
