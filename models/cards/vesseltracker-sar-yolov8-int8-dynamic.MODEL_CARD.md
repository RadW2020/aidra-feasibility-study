---
model_id: vesseltracker-sar-yolov8-int8-dynamic
version: int8-dynamic
created_at: 2026-05-04
base_model: vesseltracker-sar-yolov8 v1.0 (FP32 pytorch)
compression_technique: dynamic_int8
format: onnx
license: Apache-2.0
authors: ["AIDRA project — quantization by RadW2020"]
---

# Propósito

Variante INT8 cuantificada dinámicamente del modelo base
`vesseltracker-sar-yolov8 v1.0` (FP32 PyTorch). Destino: detección
de embarcaciones en imágenes Sentinel-1 SAR GRD (modo IW, VV/VH) bajo
perfiles de hardware espacial con presupuesto de memoria y CPU reducido.
Formato ONNX, inferencia vía ONNX Runtime con cuantización dinámica de
pesos a INT8.

El propósito de esta variante es **demostrar la viabilidad de compresión
para OBDP** en el contexto del pliego SATCEN/2026/OP/0003. No sustituye
al modelo base para uso operativo hasta validación adicional.

# Origen de la cuantización

- Proceso: cuantización dinámica (`torch.quantization.quantize_dynamic`)
  sobre el modelo FP32 base, seguida de exportación a ONNX con opset 13.
- Script: `scripts/quantize_to_int8.py` (repo AIDRA, commit traceable).
- Pesos de entrada: `vesseltracker-sar-yolov8.pt`
  (`sha256: 18aec1bb3caf7dd2c5ace8d397e241c485e917c28df248eefe794578c996d671`)
- Parámetros: `quantize_dynamic(dtype=torch.qint8)` → `export(opset=13)`

# Datos de entrenamiento

Idénticos al modelo base — esta variante **no fue re-entrenada**. Ver
`vesseltracker-sar-yolov8.MODEL_CARD.md` para descripción completa del
dataset, sesgos geográficos y domain shift.

# Métricas de rendimiento bajo constraint profiles

Medidas sobre producción AIDRA (escenas Sentinel-1 IW GRD del
Estrecho de Gibraltar y zona oeste mediterránea). Comparativa head-to-
head bajo perfil `ground` para aislar el efecto de la cuantización del
efecto de las restricciones de hardware:

| Métrica | FP32 (3 runs) | INT8 (5 runs) | Δ |
|---|---:|---:|---:|
| Tamaño en disco | 49.6 MB | **25.1 MB** | **−49.4% (1.97×)** |
| Inference time / escena | 52.66 min | **34.01 min** | **−35.4% (1.55×)** |
| Peak RAM (RSS) | 3.42 GB | 4.26 GB | **+24.6%** (ver nota) |
| Detecciones medias | 2 454 | 5 562 | **+126.7%** (ver nota) |
| Avg confidence | 53.9% | 53.6% | −0.3 pp |

Datos extraídos de `execution_log` el 2026-05-07 (commit `42585be`),
queryable directamente en el dashboard `aidra-compression-bench`.

## Comportamiento bajo perfiles más restrictivos

| Profile | FP32 latency | INT8 latency | INT8/FP32 |
|---|---:|---:|---:|
| ground       | 52.66 min | 34.01 min | 0.65× |
| sat-high     | 52.20 min | 33.18 min | 0.64× |
| sat-mid      | 52.26 min | 33.11 min | 0.63× |
| sat-low      | 63.79 min | 66.35 min | **1.04×** |
| sat-extreme  | 64.23 min | **148.53 min** | **2.31×** |

**Hallazgo importante**: la ganancia de latencia del INT8 sólo se
mantiene cuando el CPU no está throttled. Bajo `sat-extreme` (≈0.25
OCPU equivalente), el INT8 es **2.3× MÁS LENTO** que el FP32. Causa
probable: ONNX Runtime spawnea threads internos que el CPUThrottle
penaliza más agresivamente que el lazo de inferencia mono-thread del
FP32 PyTorch. **Para deployment en hardware muy restrictivo, INT8
dinámico de ONNX puede ser contraproducente** — se recomienda repetir
con static INT8 o con un runtime de inferencia mono-thread.

## Notas sobre resultados contraintuitivos

**Peak RAM INT8 > FP32 (+24.6%)**: la cuantización dinámica reduce el
tamaño de los pesos en disco pero ONNX Runtime crea buffers de
dequantización en tiempo de ejecución (FP32 en memoria) para cada
operador cuantificado. Adicionalmente, los perfiles sat-low/sat-extreme
con CPUThrottle intercalan sleeps que permiten que GC libere buffers
en FP32 pero no en INT8 (buffers ONNX Runtime no gestionados por el GC
de Python). Para hardware con restricción de RAM estricta, INT8 dinámico
**no es la técnica adecuada** — considerar static INT8 o pruning.

**Detecciones INT8 > FP32 (+126.7%) — material crítico para D4**: la
INT8 produce más del doble de detecciones que el FP32 con un Δ de
confianza media despreciable (−0.3 pp). Sin ground-truth etiquetado no
podemos atribuir el incremento a:
  (a) **mejor recall** (la cuantización dispara detecciones débiles
       que el FP32 perdía), o
  (b) **falsos positivos** (el ruido de cuantización activa detecciones
       espurias bajo el threshold operativo de 0.25).

Pruebas circunstanciales sugieren (b):
  - El avg_confidence cae 0.3 pp pero el conteo se duplica → consistente
    con detecciones extra de baja confianza (no de alta confianza
    "rescatadas").
  - El dashboard `aidra-map-detections` muestra que ~88% de las
    detecciones CFAR caen sobre tierra (efecto independiente del modelo
    pero que multiplica el coste del incremento INT8).
  - La distribución espacial de las detecciones INT8 vs FP32 sobre las
    mismas escenas muestra mayor densidad cerca de costas y bordes
    de swath, zonas típicas de falsos positivos por speckle.

**Hasta validación con xView3-SAR test split etiquetado, esta variante
NO debe usarse como detector primario en producción**. La diferencia
de +126% en conteo bruto puede invalidar las métricas de Pd/FAR del
modelo base si se asume equivalencia funcional.

## Degradación declarada vs límite AI Act

- ΔmAP@0.5 declarado: **no medido** (requiere split etiquetado xView3).
- Δavg_confidence: −0.3 pp (dentro del umbral operativo AIDRA ≤ 5 pp).
- Δconteo de detecciones: **+126.7%** sobre el mismo set de escenas →
  proxy de confianza ya **no es suficiente** para asegurar equivalencia
  operativa. La duplicación del conteo bruto puede reflejar incremento
  de FAR (false alarm rate), no mejora de Pd.
- Límite tolerable (I-MOD-3): ΔmAP ≤ 5 pts. **Sin validación mAP el
  proxy de confianza es insuficiente**.
- Estado: **CONDITIONALLY REJECTED** para uso operativo — sólo apto
  para evaluación de viabilidad de compresión en el contexto del
  pliego SatCen. El uso operativo requiere validación mAP/Pd/FAR
  sobre split etiquetado.

# Limitaciones

- **+126.7% en conteo de detecciones**: sin ground-truth etiquetado
  no se puede distinguir si es mejora de recall o aumento de FAR.
  Hipótesis predominante: aumento de FAR por ruido de cuantización
  (ver Notas sobre resultados contraintuitivos).
- **+24.6% en peak RAM**: la cuantización dinámica con ONNX Runtime
  crea buffers FP32 de dequantización — desfavorable para hardware
  espacial con RAM < 4 GB.
- **Latencia degradada bajo throttling agresivo**: bajo `sat-extreme`
  (~0.25 OCPU) el INT8 es 2.3× más lento que el FP32. La ganancia de
  speed sólo aplica cuando el CPU no está throttled.
- Sin re-validación sobre ground truth etiquetado: las métricas
  reportadas son operacionales (producción AIDRA), no de validación formal.
- Todas las limitaciones del modelo base aplican (domain shift, sesgo
  geográfico, clase única `ship`).
- No apto para uso clasificado o seguridad nacional sin auditoría externa.

# Sesgos identificados

Los mismos que el modelo base (ver FP32 MODEL_CARD). La cuantización
no introduce sesgos geográficos nuevos documentados, aunque el ruido de
redondeo puede afectar de forma desigual a buques pequeños de baja
retro-dispersión.

# Interpretabilidad

- Grad-CAM no aplicado directamente sobre variante ONNX (ONNX Runtime
  no expone gradientes de la misma forma que PyTorch autograd). Para
  interpretabilidad, usar el modelo FP32 base con los mismos inputs —
  los mapas de atención son transferibles dado que la arquitectura es
  idéntica y la degradación de confianza es < 1 pp.
- Referencia interpretabilidad base: `src/models/interpretability.py`,
  run `fcdf96e2-03ff-4c40-86af-8abffb45fce9_interp_9afa399a` (2026-04-26).

# Trazabilidad

- `file_hash` (SHA256): `ea0ee6dacd5d389ab5a3778362061776cd52307f0978d6c7db0980935f21573b`
- `base_model_hash` (FP32 PT): `18aec1bb3caf7dd2c5ace8d397e241c485e917c28df248eefe794578c996d671`
- Cuantización reproducible: `python scripts/quantize_to_int8.py --model models/vesseltracker-sar-yolov8.pt`
- Registrado en `models_registry` el 2026-05-04 12:27:05 UTC.

# Conformidad AI Act (Reg. EU 2024/1689)

- Categoría riesgo: **limited risk** (idéntica al modelo base).
- Justificación: misma tarea (detección de embarcaciones para análisis
  OSINT/SAR SatCen), misma población afectada, mismo propósito. La
  cuantización no cambia la categoría de riesgo cuando el propósito y el
  output son equivalentes.
- Obligaciones adicionales derivadas de la compresión:
  - La degradación de accuracy debe ser declarada y monitorizada
    (I-MOD-3). Se declara ΔP ≤ 0.4 pp de confianza; mAP pendiente.
  - Si se usa en producción, esta ficha debe actualizarse con métricas
    formales antes del despliegue operativo.
- Documentación base: `vesseltracker-sar-yolov8.MODEL_CARD.md`.
- Pliego referencia: SatCen SATCEN/2026/OP/0003, cláusula 10.
