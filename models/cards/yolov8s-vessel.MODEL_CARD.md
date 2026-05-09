---
model_id: yolov8s-vessel
version: unknown
created_at: 2026-04-26
authors: ["unknown"]
license: AGPL-3.0 (heredada de la arquitectura YOLOv8)
---

# Propósito

> **Estado (2026-04-26):** el binario `yolov8s-vessel.pt` ha sido **eliminado
> del repositorio y de su historial git**. Esta ficha se conserva como
> registro de gobernanza (auditoría L7, AI Act Anexo IV) — *no* como
> referencia a un peso disponible.

Pesos YOLOv8s (small, ~11 M params) etiquetados para detección de
embarcaciones. El fichero estuvo en `models/yolov8s-vessel.pt` (ahora
retirado) y su **procedencia exacta no estaba documentada** en el
repositorio AIDRA.
Tamaño coherente con la arquitectura YOLOv8s (~22 MB), pero no hay
metadatos que confirmen el dataset, hiperparámetros ni evaluación.

> **Aviso de gobernanza (I-AIA-1)**: idéntico al de
> `yolov8n-vessel`. Sin trazabilidad → no debe entrar a producción.

# Datos de entrenamiento

- Dataset(s): desconocido.
- Tamaño: desconocido.
- Sesgos conocidos: imposible declarar.
- Procedencia: desconocida.

# Métricas de validación

- mAP@0.5 / Pd / FAR: **no medidas en AIDRA**.
- Hardware/perfil de validación: desconocido.

# Limitaciones

- **Procedencia no trazable**.
- Modelo "small" → ~3× más pesado que el "nano" en CPU ARM, con
  beneficio incierto si la fuente del fine-tune no era SAR. Mejor
  sustituir por `vesseltracker-sar-yolov8` para producción y por
  `yolov8n-sar` (cuando exista) para perfiles satelitales restringidos.

# Sesgos identificados

- Desconocidos.

# Interpretabilidad

- Métodos disponibles: Grad-CAM (Ultralytics).
- Ejemplos: ninguno generado.

# Trazabilidad

- weights_sha256: `20184de7710e6c7f5b1926fd4b7c15e5e3da30fa9fdca789c36abc77e2eded05`
- training_seed: desconocido
- training_commit: desconocido
- onnx_sha256: no exportado
- Tamaño fichero: 21 MB.

# Conformidad AI Act (Reg. EU 2024/1689)

- Categoría riesgo declarada: **bloqueado pendiente de procedencia**.
- Justificación: misma que `yolov8n-vessel`. Anexo IV requiere
  documentación de dataset y resultados de validación que no están
  disponibles.
- Documentación adicional:
  - **TODO**: si corresponde a un release público (e.g. variante de
    HuggingFace, MMRotate, etc.), recuperar la URL canónica y volcar
    métricas oficiales aquí.
