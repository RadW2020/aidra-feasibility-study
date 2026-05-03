---
model_id: yolov8n
version: v8.0.0
created_at: 2023-01-10
authors: ["Ultralytics", "Glenn Jocher et al."]
license: AGPL-3.0
---

# Propósito

Modelo YOLOv8 nano (3.2 M params) entrenado sobre el dataset COCO
2017 — 80 clases genéricas (persona, coche, etc.). En AIDRA se
mantiene como **base model** para operaciones de fine-tuning (ver
`scripts/fine-tune.py`) y como referencia de control para benchmarks
de compresión (`src/models/compression/`). **No está pensado para uso
directo en detección de embarcaciones**: no contiene la clase "ship"
de forma fiable y opera sobre RGB óptico, no SAR.

# Datos de entrenamiento

- Dataset(s): **COCO 2017 train** (Common Objects in Context),
  CC BY 4.0.
  - URL: https://cocodataset.org/
  - Versión: 2017 (Train2017 + Val2017).
- Tamaño: 118 287 imágenes train / 5 000 val, 80 clases. Anotaciones
  de bounding-box.
- Sesgos conocidos:
  - Geográfico: COCO tiene fuerte sesgo hacia escenas urbanas
    occidentales y fotografía de ocio.
  - Clase: "boat" en COCO (clase 9) cubre desde kayaks a yates,
    fotografiados en óptico desde la orilla — ninguna de esas
    distribuciones se parece a SAR satelital.
  - No es maritime-aware.
- Procedencia: pública (CC BY 4.0).

# Métricas de validación

- mAP@0.5 (COCO val): 37.3 (publicado por Ultralytics).
- mAP@0.5:0.95 (COCO val): 28.4.
- Latencia CPU (Ultralytics docs): ~80 ms / imagen 640×640 ARM.
- **Sin métricas medidas en AIDRA** — fuera del dominio de uso.

# Limitaciones

- No detecta clase específica "ship" con precisión: la clase 9 ("boat")
  agrega tipologías muy distintas y casi no responde sobre SAR.
- Aplicado a tiles SAR sin re-entrenamiento → **0 detecciones útiles**
  (verificado durante depuración del bug del 2026-04-25).
- Distribuible solo bajo AGPL-3.0 → cualquier despliegue comercial
  requiere licencia comercial Ultralytics.

# Sesgos identificados

- COCO sub-representa escenas marítimas, vista cenital, condiciones
  nocturnas, lluvia, niebla.
- Inferencia sobre RGB diurno; degrada con baja iluminación.

# Interpretabilidad

- Métodos disponibles: Grad-CAM (Ultralytics expone hooks).
- Ejemplos: ninguno generado en AIDRA.

# Trazabilidad

- weights_sha256: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`
- training_seed: 0 (Ultralytics default)
- training_commit: Ultralytics v8.0.0 release tag
- onnx_sha256: no exportado en este snapshot
- Descarga: `scripts/download-models.sh` via `ultralytics.YOLO("yolov8n.pt")`.

# Conformidad AI Act (Reg. EU 2024/1689)

- Categoría riesgo declarada: **minimal risk**.
- Justificación: modelo de uso general no desplegado en pipeline de
  producción AIDRA. Sólo se utiliza como base para fine-tuning offline
  y como modelo de control en benchmarks de compresión. No emite
  decisiones que afecten a personas. Si en algún momento se desplegase
  directamente sobre imágenes operativas, esta ficha debe re-evaluarse.
- Documentación adicional:
  - https://docs.ultralytics.com/models/yolov8/
  - Roboflow Universe COCO card.
