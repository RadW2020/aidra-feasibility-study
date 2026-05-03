# AIDRA — Analisis de la Licitacion Original

Documento de referencia sobre la licitacion real. Para el plan de implementacion del MVP ver [mvp_oci.md](mvp_oci.md).

---

## 1. Datos de la licitacion

| Campo | Valor |
|---|---|
| Numero publicacion | 197981-2026 (corregido por 246340-2026) |
| Expediente | SATCEN/2026/OP/0003 |
| Fecha publicacion | 23/03/2026 (original), 10/04/2026 (correccion) |
| Fecha limite | 04/05/2026 16:00 UTC+02:00 |
| Valor estimado | 210.000 EUR (sin IVA), techo maximo |
| Duracion | 12 meses (no renovable) |
| Procedimiento | Abierto |
| CPV | 73200000, 73210000, 73100000 (Consultoria I+D) |
| Idioma | Ingles (acepta todos los oficiales UE) |
| Lugar de ejecucion | Principalmente en instalaciones del contratista u otras ubicaciones definidas para la demo (incl. orbita o simulacion cualificada) |
| Financiacion | Cofinanciado SatCen + EDA (fondos UE) |

## 2. Organo de contratacion

**European Union Satellite Centre (SatCen)** — Agencia descentralizada de la UE, bajo supervision del Comite Politico y de Seguridad.

| Campo | Valor |
|---|---|
| Direccion | Base Aerea de Torrejon de Ardoz, Av. de Cadiz 457, 28850 Madrid |
| Email | (ver portal F&T / web SatCen) |
| Tel | +34 91 678 60 00 |
| Web | https://www.satcen.europa.eu/ |

Mision: proporcionar productos y servicios de explotacion de activos espaciales (imagenes satelitales) para la politica exterior, seguridad y defensa de la UE.

## 3. Que es AIDRA

**AIDRA = Artificial Intelligence In-orbit Data pRocessing Assessment**

Iniciativa conjunta SatCen + EDA. Proof-of-concept para evaluar si el procesamiento de datos con IA **a bordo de satelites** (On-Board Data Processing / OBDP) es viable para flujos de trabajo GEOINT.

### Caso de uso seleccionado
**Deteccion de barcos (vessel detection) en areas maritimas.** Elegido por madurez tecnica, relevancia operativa y comparabilidad con otras iniciativas.

### Que NO es AIDRA
- No es un producto operativo
- No busca evaluar la precision individual de modelos de IA
- No requiere entrega de codigo fuente ni ejecutables (confirmado Q&A Q89225)
- No es un proyecto de desarrollo de software — es un estudio de evaluacion

### Que SI busca evaluar
- Cadena end-to-end completa: imagen → modelo IA → resultado
- Latencia, priorizacion de datos, trazabilidad, controles de seguridad
- Tecnicas de compresion de modelos (quantizacion, pruning, knowledge distillation)
- Trade-offs entre rendimiento de deteccion y restricciones de hardware espacial (CPU, memoria, energia)
- Integracion con flujos de trabajo GEOINT existentes

### Modalidad del sensor
Libre: el licitador puede proponer **optico, SAR o multi-sensor** y justificar la eleccion.

### Entorno de demostracion
- Preferencia: **en orbita** usando satelites propios o de terceros
- Aceptable: **simulacion en tierra** que reproduzca condiciones de orbita (TRL >= 6, hardware flight-proven o representativo)

### Opcional (puntos extra)
Tip & Cue: capacidad de tasking inteligente del satelite (si detecta algo, reprograma otra pasada).

## 4. Tareas del contrato

| Tarea | Descripcion |
|---|---|
| T0 | Gestion de proyecto: plan, QA, riesgos |
| T1 | Planificacion de la demo + definicion del escenario (vessel detection) |
| T2 | Ejecucion de la demo: integrar, ejecutar, recoger datos. Evaluar compresion de modelos |
| T3 | Analisis de resultados y recomendaciones |

## 5. Entregables

| Ref | Descripcion | Plazo |
|---|---|---|
| D1 | Plan de demostracion + definicion del escenario + declaracion de modelos IA (AI Act) | T0 + 2 meses |
| D2 | Informe intermedio de progreso | T0 + 4 meses |
| D3 | **Paquete de evidencia**: logs, imagenes, benchmarks de compresion, config files | T0 + 9 meses |
| D4 | **Informe final**: analisis, recomendaciones, evaluacion de interpretabilidad IA | T0 + 11 meses |
| D5 | Informe de cierre contractual | T0 + 12 meses |

## 6. Reuniones

| Reunion | Cuando | Que |
|---|---|---|
| KoM | T0 | Enfoque, calendario, borrador de escenario |
| MTR | T0 + 4 meses | Progreso, problemas, acciones correctivas |
| FRM | T0 + 12 meses | Resultados, conclusiones, cierre |

Reuniones por VTC salvo acuerdo. Hasta 3 reuniones presenciales posibles (Torrejon de Ardoz u otro lugar UE).

## 7. Criterios de evaluacion

### Calidad tecnica (max 70 puntos, minimo 30 para pasar)

| Criterio | Puntos | Desglose |
|---|---|---|
| Q1 — Equipo | 15 | Expertos adicionales (5), complementariedad (5), plan backup (5) |
| Q2 — Gestion | 15 | Plan de proyecto (10), gestion de riesgos (5) |
| Q3 — Propuesta tecnica | 40 | Metodologia y simulacion (10), integracion GEOINT (10), entorno demo + trazabilidad + documentacion IA (20) |

### Precio (max 30 puntos)
Formula: FS = (Pmin / Ptender) x 30

### Puntuacion final
FINAL = QS + FS (max 100)

## 8. Requisitos minimos del equipo

### Project Manager
- Master en ingenieria espacial, informatica o ciencias de la tierra
- 7 anos experiencia en EO/espacio
- Experiencia en proyectos UE (FP7, H2020, Horizon Europe)

### AI/OBDP Specialist
- Master en IA, Computer Science o Data Science
- 7 anos en modelos AI/ML aplicados a EO/teledeteccion
- 7 anos en procesamiento a bordo / edge / entornos restringidos
- Experiencia en evaluacion y validacion de sistemas IA

### EO/GEOINT Expert
- Grado en ciencias de la tierra, geoinformatica, teledeteccion
- 5 anos en aplicaciones EO para seguridad/defensa
- Experiencia operativa con datos EO (ej: vigilancia maritima)

Al menos un miembro con experiencia en gobernanza de sistemas IA.

## 9. Equipamiento tecnico minimo

- Recursos de computo para alojar modelos AI/ML y procesar datasets EO
- Entorno de simulacion/test para demo espacio-representativa (TRL >= 6)
- Herramientas de videoconferencia
- Sistemas de backup y recuperacion
- Sistemas de transferencia y almacenamiento de datos **dentro de la UE**

## 10. Condiciones contractuales relevantes

- **IP**: SatCen y EDA adquieren co-propiedad de todos los resultados. Licencia irrevocable, mundial, gratuita. El contratista renuncia a sus derechos IP sobre los resultados.
- **Pre-existing rights** (modelos, plataformas, datasets del contratista): permanecen del titular, pero se otorga licencia no exclusiva a SatCen/EDA.
- **Pagos**: 40% tras D1+D2, 60% tras D3+D4+D5.
- **AI Act**: El contratista declara conformidad con Regulation (EU) 2024/1689. SatCen no actua como evaluador de conformidad.
- **"Raw data"**: Se acepta como Level 1 imagery en formato estandar (confirmado Q&A Q89818).
- **Datos de terceros**: Validos si la licencia permite evaluacion por SatCen/EDA (confirmado Q&A Q89885).

## 11. Contexto: Proyectos relacionados

### MYRIAD (Marzo 2026)
Coordinado por GMV (Madrid), 48 meses, ~5M EUR del European Defence Fund:
- IA para analisis de imagenes satelitales multi-sensor (optico + SAR)
- Fusion multi-sensor, calibracion radiometrica, IA explicable (XAI)
- Integracion con infraestructura SatCen
- 9 socios europeos

### Copernicus
Programa EU de observacion terrestre. Datos abiertos y gratuitos:
- **Sentinel-1**: Radar SAR (relevante para vessel detection)
- **Sentinel-2**: Optico multiespectral
- **Sentinel-3**: Tierra y costas
- **Sentinel-5P**: Atmosfera

## 12. Fuentes y enlaces

| Recurso | URL |
|---|---|
| TED (original) | https://ted.europa.eu/en/notice/-/detail/197981-2026 |
| TED (correccion) | https://ted.europa.eu/en/notice/-/detail/246340-2026 |
| Pliegos (Funding & Tenders) | https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/tender-details/5415d210-77e4-4664-9928-6157aa05992c-CN |
| SatCen | https://www.satcen.europa.eu/ |
| Copernicus Data Space | https://dataspace.copernicus.eu |
| Copernicus APIs | https://documentation.dataspace.copernicus.eu/APIs.html |
| MYRIAD | https://www.gmv.com/en-es/communication/press-room/press-releases/myriad-new-european-defence-fund-initiative-revolutionizing |

## 13. Documentos descargados (carpeta download/)

| Archivo | Que contiene |
|---|---|
| Appendix I.1 - Technical Specifications | **Especificaciones tecnicas**: tareas, entregables, aspectos clave |
| Annex I - Tender Specifications (Version 2) | Condiciones, criterios de evaluacion, requisitos de equipo |
| Annex II - Draft Contract | Borrador de contrato |
| Invitation to tender | Carta de invitacion |
| Appendix I.3 - Tender Submission Form | Formulario de envio |
| Appendix I.4 - Declaration on Honour | Declaracion de honor |
| Appendix I.5 - Average Annual Turnover | Declaracion de facturacion |
| Appendix I.6 - Economic Offer Template | Plantilla de oferta economica |
| FT Portal-Public-QA.xlsx | Preguntas y respuestas oficiales (5 Q&A) |
| FT Portal-FAQs.xlsx | FAQ del portal eSubmission (no relevante tecnicamente) |
