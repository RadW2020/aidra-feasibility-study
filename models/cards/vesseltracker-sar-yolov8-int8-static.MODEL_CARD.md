---
model_id: vesseltracker-sar-yolov8-int8-static
version: int8-static
created_at: 2026-08-29
base_model: vesseltracker-sar-yolov8 v1.0 (FP32 pytorch)
compression_technique: static_int8
format: onnx
license: Apache-2.0
authors: ["AIDRA project — quantization by RadW2020"]
status: candidate
---

# Propósito

Variante INT8 **estática** (ONNX Runtime, formato QDQ, pesos QInt8 /
activaciones QUInt8) del modelo base `vesseltracker-sar-yolov8` v1.0. Sustituye a la
variante dinámica, rechazada bajo I-MOD-3 (conteos de detección no
deterministas, +24.6 % RAM). Estado `candidate` hasta completar la terna
{baseline FP32, esta variante, perfil} con la degradación máxima
declarada antes del run (ΔmAP ≤ 5 pts, `Settings`).

# Identidad

| Campo | Valor |
|---|---|
| Fichero | `vesseltracker-sar-yolov8-int8-static.onnx` |
| Tamaño | 26.54 MB (74 % menor que el FP32 ONNX, 49 % menor que el `.pt`) |
| SHA256 | `dfe4c9669c0c19c1f03d71fbae4bc5aaa2119fefa89f78656b665e622e194521` |
| Creado | 2026-08-29 00:11 UTC por `scripts/quantize_static_int8.py` (commit `9156a3651ec7`) |

# Método de compresión

- `onnxruntime.quantization.quantize_static` (QDQ, `weight_type=QInt8`,
  `activation_type=QUInt8`, **`op_types_to_quantize=["Conv"]`**,
  `per_channel=True`, calibración **Percentile**), vía
  `ModelQuantizer.quantize_static_onnx`.
- **Por qué solo Conv:** con la configuración por defecto de ORT (todos los
  tipos de op) el detector quedaba a 0 detecciones incluso a conf 0.01
  (FP32: 100 en las mismas teselas): los Sigmoid / Softmax / Div / Mul de
  la cabeza DFL en UINT8 colapsan las puntuaciones de clase.
- **Por qué Percentile:** MinMax es frágil — una tesela de mar con un
  reflector brillante fija el rango de activación y el acuerdo con FP32
  sobre 44 teselas con barcos osciló entre 15/55 y 45/55 según qué 16
  teselas de mar entraran en la calibración. Barrido 2026-08-29 (44
  teselas con barcos, conf 0.25, FP32 = 55 detecciones): **Percentile con
  32 teselas de barcos, sin mar: 53 recuperadas / 12 extra / 2 perdidas**;
  Percentile-24 barcos 51 / 9 / 4; Percentile 20 barcos + 4 mar 32 / 1 / 23
  (las teselas de mar, sin objetivos, comprimen el rango de puntuaciones);
  MinMax 15 / 1 / 40; MinMax+media móvil 16 / 0 / 39; Entropy(8) ~1.
  60 imágenes con Percentile agotan 16 GB de RAM (ORT guarda todas las
  activaciones para el histograma). El chequeo es in-sample; la prueba
  fuera de muestra es la terna sobre las 11 escenas xView3.
- Acuerdo con FP32 sobre las 32 teselas de calibración con
  barcos (centro ≤ 20 px, conf ≥ 0.25): **38/40**
  detecciones FP32 recuperadas, 7 extra, 2
  perdidas. INT8 determinista (dos `predict` idénticos): True.
- Escalas fijadas offline → salida determinista run a run; mismo FP32 ONNX
  + mismo set de calibración (seed) → mismo fichero INT8.

# Set de calibración (provenance)

| Campo | Valor |
|---|---|
| Escena xView3 | `264ed833a13b7f2av` (split validation, Adriático) |
| Tar SHA256 | `259937fbd52b2e4550bc93182ffb174b1f4b6fa5b48b1a2b4b2e61e727a9470e` |
| Ground truth | `x-view-us-data/validation.csv` (`is_vessel=True`, confianza HIGH/MEDIUM) |
| Teselas | 32 centradas en barcos + 0 de mar abierto aleatorias = 32 × 640×640 px |
| Conversión | σ⁰ dB → lineal → uint8 (stretch −25..0 dB), **sin** filtro Lee (= `Settings.yolo_input="unfiltered"`, R15) |
| Seed | 42 |
| SHA256 del tensor de calibración | `8f4cbb9a664b80b40db8da369f71ffe169cdd13c260e29e9dde95f8a217b22a9` |
| Centros (fila, col) | ver `/Users/rauljm/codeloper/AIDRA/models/vesseltracker-sar-yolov8-int8-static.calibration.json` |

# Cadena de origen

- FP32 PyTorch: `vesseltracker-sar-yolov8.pt` — SHA256 `18aec1bb3caf7dd2c5ace8d397e241c485e917c28df248eefe794578c996d671` — 52.03 MB
- FP32 ONNX intermedio: `vesseltracker-sar-yolov8.onnx` — SHA256 `4e3a6389bcd84ba78cbff5af0d2bfab408da278eb0a34a6249edfd87591343af` — 103.63 MB

# Métricas

Pendientes de la terna sobre las mismas 11 escenas xView3 que el baseline
(`scripts/validate_xview3_serial.py --pipeline-path full --model all
--yolo-model vesseltracker-sar-yolov8-int8-static`): AP, Pd, FAR/km², precisión, latencia p50/p95 por
tesela, RAM pico, tamaño. Se anotan aquí al ejecutarla; hasta entonces esta
variante **no** entra en evaluación (I-MOD-1/3).

# Limitaciones

- Mismo dataset de entrenamiento, sesgos y domain shift que el baseline
  (ver `vesseltracker-sar-yolov8.MODEL_CARD.md`).
- Calibración con una única escena/track del Adriático: las escalas de
  activación pueden no cubrir mares con mayor clutter (Gibraltar, Canal).
- Anexo IV AI Act: esta ficha + `models_registry` (status, hash) +
  `execution_log` constituyen la documentación técnica de la variante.
