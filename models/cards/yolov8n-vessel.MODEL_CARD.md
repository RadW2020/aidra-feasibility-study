---
model_id: yolov8n-vessel
version: unknown
created_at: unknown
authors: ["unknown"]
license: AGPL-3.0 (heredada de la arquitectura YOLOv8)
---

# Propósito

> **Estado (2026-04-26):** el binario `yolov8n-vessel.pt` ha sido **eliminado
> del repositorio y de su historial git**. Esta ficha se conserva como
> registro de gobernanza (auditoría L7, AI Act Anexo IV) — *no* como
> referencia a un peso disponible.

Pesos YOLOv8n etiquetados para detección de embarcaciones. El fichero
estuvo en `models/yolov8n-vessel.pt` (ahora retirado) y su **procedencia
exacta no estaba documentada** dentro del repositorio AIDRA (no aparece en
`scripts/download-models.sh`, ni hay referencia en `README.md`,
`TECHNICAL_SPEC.md` o el código). Probablemente generado por un
fine-tuning local previo o copiado desde un release externo no
trazado.

> **Aviso de gobernanza (I-AIA-1)**: esta ficha se publica con campos
> "unknown" en lugar de inventar datos. Mientras la procedencia no se
> reconstruya, este modelo **no debe usarse** en producción por
> defecto. AIDRA mantiene `vesseltracker-sar-yolov8` como modelo
> primario.

# Datos de entrenamiento

- Dataset(s): **desconocido**. No hay archivos de entrenamiento ni
  data.yaml asociados en el repo.
- Tamaño: desconocido.
- Sesgos conocidos: imposible declarar sin conocer el dataset.
- Procedencia: desconocida.

# Métricas de validación

- mAP@0.5 / Pd / FAR: **no medidas**. El modelo no se ha sometido a
  evaluación interna AIDRA.
- Hardware/perfil de validación: desconocido.

# Limitaciones

- **Procedencia no trazable** — riesgo de uso indebido.
- Pesos AGPL-3.0 por la arquitectura → mismas restricciones de
  distribución que `yolov8n`.
- Hasta que se documente o sustituya, sólo debería usarse para
  pruebas offline.

# Sesgos identificados

- Desconocidos. Cualquier despliegue debe asumir el peor caso.

# Interpretabilidad

- Métodos disponibles: los mismos que YOLOv8 (Grad-CAM via hooks).
- Ejemplos: ninguno.

# Trazabilidad

- weights_sha256: `dce1edbb5e110048f2af98d049c89060cbf4a8b7b389f4918971ba4b01748a2d`
- training_seed: desconocido
- training_commit: desconocido
- onnx_sha256: no aplica (no exportado)
- Tamaño fichero: 6.0 MB (consistente con YOLOv8n nano).

# Conformidad AI Act (Reg. EU 2024/1689)

- Categoría riesgo declarada: **bloqueado pendiente de procedencia**.
  No se puede declarar conformidad sin datos de entrenamiento ni
  métricas, así que se marca como **inadecuado para uso operativo**
  hasta nueva orden.
- Justificación: el Anexo IV del Reg. (EU) 2024/1689 exige
  documentación técnica de los datasets de entrenamiento y validación;
  imposible cumplirla con esta ficha.
- Documentación adicional:
  - **TODO**: identificar al autor original del fine-tune. Si se
    confirma que es un derivado de COCO (clase "boat"), aplica la
    misma ficha que `yolov8n` pero re-etiquetado y debe documentarse.
  - Si fue entrenado contra xView3-SAR localmente, hay que recuperar
    el `runs/aidra/.../args.yaml` correspondiente.
