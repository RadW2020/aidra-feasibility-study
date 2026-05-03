# AIDRA — Declaracion de conformidad AI Act

**Reglamento (UE) 2024/1689** del Parlamento Europeo y del Consejo, sobre normas armonizadas en materia de inteligencia artificial.

| Campo | Valor |
|---|---|
| Sistema | AIDRA — Artificial Intelligence In-orbit Data pRocessing Assessment |
| Version | 1.0.0-mvp |
| Fecha | 2026-04-25 |
| Responsable | RadW2020 |
| Contexto | Proof-of-concept inspirado en licitacion `SATCEN/2026/OP/0003` (no es un sistema operativo) |

## 1. Clasificacion

AIDRA se clasifica como **Sistema de IA de proposito limitado, no de alto riesgo** (Art. 6 + Anexo III, lectura inversa):

- **No aplica Anexo III** (alto riesgo): la deteccion de barcos en SAR sobre aguas internacionales/UE no figura en infraestructura critica, RRHH, justicia, asilo o aplicacion de la ley en el sentido del Anexo III.
- **No es uso prohibido** (Art. 5): no hay scoring social, identificacion biometrica remota en tiempo real, manipulacion subliminal ni explotacion de vulnerabilidades.
- **No es modelo GPAI** (Art. 51): es un detector especializado YOLOv8 + CFAR, no un foundation model.
- **Aplican obligaciones de transparencia** (Art. 50) cuando el sistema interactue con personas, lo cual *no es el caso* en AIDRA (output a operadores GEOINT cualificados, no al publico general).

> **Implicacion:** AIDRA no requiere conformidad bajo Anexo III, pero adopta voluntariamente los principios del Capitulo II (gestion de riesgos, gobernanza de datos, documentacion, registro de eventos, transparencia, supervision humana, robustez) como **buena practica** orientada a la calidad del entregable D4 del pliego.

## 2. Base legal

- **Tratamiento de datos:** AIDRA procesa imagenes Sentinel-1 publicas de Copernicus Data Space Ecosystem (Reglamento (UE) 377/2014) bajo licencia **CC-BY-SA**. No se procesan datos personales en el sentido del RGPD (Art. 4.1 (UE) 2016/679); las detecciones identifican embarcaciones, no personas fisicas.
- **Soberania de datos:** todo almacenamiento (BD, modelos, logs, evidencia) se restringe a region UE (OCI Frankfurt). Invariante I-EU-1.
- **Datasets de entrenamiento:** xView3-SAR, HRSID, OpenSARShip — todos publicos, gratuitos, con licencia compatible y sin PII.

## 3. Supervision humana (Art. 14, principio adoptado)

- **Operacion supervisada:** el output de AIDRA es una capa de detecciones georreferenciadas con `confidence`, `on_land`, `cluster_anomaly` y enlace al `execution_log`. **No** se toman decisiones automatizadas con efecto sobre personas.
- **Punto de revision humana:** el analista GEOINT consume las detecciones via API/STAC y decide si validan, descartan o escalan cada caso. AIDRA no actua sobre embarcaciones ni dispara acciones operativas por si solo.
- **Override:** todo run queda en `execution_log` con `status` y `error_message`; los runs marcados `quality=invalid` o variantes `rejected` no se borran (solo se marcan).
- **Trazabilidad reforzada:** cada inferencia lleva `image_hash`, `model_hash`, `output_hash`, `input_params_hash`, `commit_sha` (I-TRACE-1..4). Cualquier resultado es auditable end-to-end.

## 4. Gestion de riesgos y limitaciones

- **Falsos positivos sobre tierra:** mitigado por footprint clipping geometrico real (I-SAR-3) + flag `on_land` materializado (I-DET-2).
- **Artefactos de borde de swath:** mitigado por edge filter robusto por clusteres de longitud (I-SAR-2).
- **Speckle / clusters anomalos:** marcado con flag `cluster_anomaly` y excluido de metricas de mar (I-DET-3).
- **Sesgo de dataset:** declarado por modelo en su `MODEL_CARD.md` (zona geografica de entrenamiento, distribucion de tamanos de barco, limitaciones de polarizacion).
- **Drift:** no aplica al MVP (no hay reentrenamiento online); declarado como riesgo abierto en `RISK_REGISTER.md`.

## 5. Documentacion (Art. 11, principio adoptado)

- **Documentacion tecnica:** `TECHNICAL_SPEC.md`, `CLAUDE.md` (contrato del agente), `mvp_oci.md`.
- **Ficha por modelo:** `models/cards/<nombre>.MODEL_CARD.md` — sin ficha, el modelo no entra al pipeline (gate `ai-act-card`, I-AIA-1).
- **Registro de eventos:** tabla `execution_log` en PostgreSQL + logs estructurados Loki + metricas Prometheus, todo correlacionado por `run_id` (UUID).
- **Interpretabilidad:** Grad-CAM sobre cabeza YOLOv8 + heatmap CFAR pre-threshold sobre muestreo del eval set, output como anexo del entregable D4 (I-AIA-2).
- **Bundle de evidencia:** `/build-evidence-bundle` empaqueta logs + configs + metricas + muestras + MANIFEST SHA256 para D3.

## 6. Declaracion final

El responsable declara que AIDRA, en su estado actual de proof-of-concept, **no se encuadra como sistema de alto riesgo bajo el AI Act**, y que adopta voluntariamente los principios de transparencia, trazabilidad y supervision humana del Reglamento como base de calidad para el entregable D4.

Cualquier evolucion hacia un despliegue operativo (ej. integracion en sistemas de vigilancia maritima de un Estado Miembro o de la UE) requeriria una reclasificacion previa contra el Anexo III en el momento del despliegue.

— Firmado: 2026-04-25
