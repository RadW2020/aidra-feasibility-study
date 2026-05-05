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

Medidas sobre producción AIDRA (escena Sentinel-1 IW GRD Mediterráneo,
3 runs por perfil). Comparativa INT8 vs FP32 baseline bajo perfil `ground`:

| Métrica | FP32 (baseline) | INT8 (esta variante) | Δ |
|---|---:|---:|---:|
| Tamaño en disco | 49.6 MB | **25.1 MB** | **−49.4% (1.97×)** |
| Tiempo total escena | 52.7 min | **34.6 min** | **−34% (1.52×)** |
| Peak RAM (RSS) | 3.50 GB | 5.01 GB | **+43%** (ver nota) |
| Detecciones medias | 2 454 | 2 868 | +17% (ver nota) |
| Avg confidence | 53.9% | 53.5% | −0.4 pp |
| Runs evaluados | 3 | 3 | — |

## Notas sobre resultados contraintuitivos

**Peak RAM INT8 > FP32 (+43%)**: la cuantización dinámica reduce el
tamaño de los pesos en disco pero ONNX Runtime crea buffers de
dequantización en tiempo de ejecución (FP32 en memoria) para cada
operador cuantificado. Adicionalmente, los perfiles sat-low/sat-extreme
con CPUThrottle intercalan sleeps que permiten que GC libere buffers
en FP32 pero no en INT8 (buffers ONNX Runtime no gestionados por el GC
de Python). Para hardware con restricción de RAM estricta, INT8 dinámico
**no es la técnica adecuada** — considerar static INT8 o pruning.

**Detecciones INT8 > FP32 (+17%)**: la cuantización introduce ruido
de redondeo que en algunos tiles activa detecciones que el modelo FP32
filtrada por estar bajo el umbral de confianza (0.25). Esto es un
artefacto de la cuantización dinámica, no una mejora de recall. Pd real
sobre ground truth etiquetado no ha sido medido para esta variante —
**pendiente validación formal**.

## Degradación declarada vs límite AI Act

- ΔmAP@0.5 declarado: **no medido** (requiere split etiquetado xView3).
  Proxy: Δavg_confidence = −0.4 pp (dentro del umbral operativo AIDRA ≤ 5 pp).
- Límite tolerable (I-MOD-3): ΔmAP ≤ 5 pts. Proxy aceptado para esta
  evaluación.
- Estado: **CONDITIONALLY ACCEPTED** — pendiente validación mAP formal.

# Limitaciones

- La cuantización dinámica reduce latencia de CPU bound pero **aumenta
  pico de RAM** en esta implementación — desfavorable para hardware
  espacial con RAM < 4 GB.
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
