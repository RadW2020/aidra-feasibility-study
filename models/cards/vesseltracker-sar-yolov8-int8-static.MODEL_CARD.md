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
| Creado | 2026-08-29 00:21 UTC por `scripts/quantize_static_int8.py` (commit `b81f029cf189`) |

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

Terna I-MOD-1 **pata de calidad** (2026-08-29): baseline FP32 y esta variante
sobre las **mismas 11 escenas xView3-SAR** (Adriático, 1 997 barcos), mismo
pipeline (`fusion_mode=center`, `yolo_input=unfiltered`, CFAR 8/20, conf 0.25,
tiles 640/64), harness `scripts/validate_xview3_serial.py --pipeline-path full
--model all --yolo-model vesseltracker-sar-yolov8-int8-static`.
Reportes: `reports/validation_xview3_adriatic_full_vessels_int8_static_*.json`
vs `..._r14r15_*.json` (provenance completa en cada uno).

| Conjunto | Predicciones | Pd | FAR/km² | Precision | AP | F1 |
|---|---:|---:|---:|---:|---:|---:|
| YOLO FP32 (`vesseltracker-sar-yolov8`) | 2 178 | 0.143 | 0.0040 | 0.131 | 0.026 | 0.137 |
| **YOLO INT8 estático (esta variante)** | 1 985 | 0.147 | 0.0036 | 0.148 | 0.032 | 0.147 |
| Salida AIDRA con YOLO FP32 | 4 610 | 0.360 | 0.0083 | 0.156 | 0.110 | 0.218 |
| Salida AIDRA con YOLO INT8 | 4 395 | 0.355 | 0.0079 | 0.162 | 0.118 | 0.222 |
| Solo fusionadas con FP32 | 763 | 0.119 | 0.0011 | 0.312 | 0.062 | 0.172 |
| Solo fusionadas con INT8 | 823 | 0.128 | 0.0012 | 0.311 | 0.065 | 0.182 |

Δ INT8 − FP32 (conjunto YOLO): AP +0.0061 · Pd +0.0035 · FAR -0.0004/km² · F1 +0.0102.
Tamaño 52.0 MB (`.pt`) → 26.5 MB (−49 %). Tiempo de motor (CFAR + YOLO + fusión, 11 escenas, misma máquina): 3653 s → 1915 s (−48 %; CFAR idéntico en ambas, así que la ganancia es toda de YOLO; medido bajo carga concurrente distinta, orientativo). Salida determinista run a run.

**Veredicto I-MOD-3:** degradación máxima declarada ΔmAP ≤ 5 pts → **no hay
degradación** (ΔAP +0.6 pts, dentro del ruido). La pata de **perfil de
hardware** (latencia p50/p95 por tesela y RAM pico bajo `sat-*`, con
`profile_memory_enforcement=abort`) requiere ejecutarse en el despliegue OCI
(`POST /api/pipeline/trigger-all-profiles` con `model_version=int8-static`);
hasta entonces `status=candidate`.

# Datos de entrenamiento

Idénticos al modelo base — esta variante **no fue re-entrenada**; solo se
fijaron las escalas de activación con el set de calibración descrito
arriba. Ver `vesseltracker-sar-yolov8.MODEL_CARD.md` para dataset, licencia y cobertura
geográfica.

# Sesgos

- Mismos sesgos y domain shift que el baseline (`vesseltracker-sar-yolov8.MODEL_CARD.md`).
- Calibración con una única escena/track del Adriático (VH): las escalas de
  activación pueden no cubrir mares con mayor clutter (Gibraltar, Canal) ni
  otras polarizaciones.

# Limitaciones

- Cuantización de 8 bits en las Conv: las detecciones de confianza media
  son las primeras en perderse (in-sample: 2 de 55 perdidas, 12 extra).
- Reproducibilidad del artefacto condicionada al FP32 ONNX intermedio: dos
  exportaciones de ultralytics del mismo `.pt` no son byte-idénticas (metadatos),
  pero dado el mismo FP32 ONNX + mismo set de calibración el INT8 sí lo es
  (verificado 2026-08-29: SHA256 idéntico en dos cuantizaciones).

# Interpretabilidad

Grad-CAM no es aplicable al grafo ONNX cuantizado (sin autograd). El anexo D4
usa el baseline FP32 `.pt` como *renderer* del mapa de calor y esta variante
como *sujeto* de la explicación (misma convención que la variante dinámica,
`EVIDENCE.md` § D4); el manifest registra ambos hashes por separado.

# Trazabilidad

- `models_registry`: nombre `vesseltracker-sar-yolov8`, versión `int8-static`,
  `compression_technique=static_int8`, `status=candidate`, SHA256 del fichero.
- Cada run persiste `model_hash` (SHA256 de este `.onnx`), `commit_sha`,
  `input_params_hash` e `inference_p50_ms` / `inference_p95_ms` (I-MOD-2).
- Provenance del artefacto: `/Users/rauljm/codeloper/AIDRA/models/vesseltracker-sar-yolov8-int8-static.calibration.json` (config de cuantización,
  hashes del `.pt` / FP32 ONNX / INT8, set de calibración con centros y SHA256).

# Conformidad AI Act

Reglamento (UE) 2024/1689: sistema de propósito limitado, no Anexo III
(`AI_ACT_DECLARATION.md`).
Documentación técnica (Anexo IV) = esta ficha + `models_registry` +
`execution_log` + reportes de validación (`reports/`). Supervisión humana:
las detecciones son una capa georreferenciada con confianza y
`quality_verdict`; ninguna decisión automatizada sobre personas.
