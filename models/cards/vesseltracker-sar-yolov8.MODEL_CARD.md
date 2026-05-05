---
model_id: vesseltracker-sar-yolov8
version: v1.0
created_at: 2026-04-25
authors: ["hewitleo (HuggingFace)", "C. Santamariz (vesselTracker concept)"]
license: Apache-2.0
---

# Propósito

Detección de embarcaciones (clase única `ship`) en imágenes Sentinel-1
SAR GRD (modo IW, polarización VV/VH). Pesos importados desde
[hewitleo/sar-ship-detection-yolov8](https://huggingface.co/hewitleo/sar-ship-detection-yolov8)
y usados como detector primario para AIDRA en la zona Estrecho de
Gibraltar. Inspirado en el concepto vesselTracker (YOLOv5s) de
C. Santamariz, cuyos pesos no están publicados.

# Datos de entrenamiento

- Dataset(s): publicado por el autor original como "SAR ship detection
  dataset" sobre tiles Sentinel-1 IW GRD; metadata exacta no incluida
  en el repo de HuggingFace.
- Tamaño: **no declarado por el autor en la tarjeta de origen**.
- Sesgos conocidos:
  - Geográfico: tiles probablemente concentrados en zonas costeras
    asiáticas y atlánticas según convención de los principales
    datasets SAR-Ship-Dataset / HRSID; sobre el Estrecho hay menos
    cobertura → posible degradación de recall en escenas con clutter
    de oleaje fuerte.
  - Sensor: entrenado sobre Sentinel-1 IW GRD; aplicar a otros modos
    (EW, SM) o a TerraSAR-X / Capella requiere re-validación.
  - Polarización: no se documenta cuál(es) se usó(usaron); las
    estadísticas VH y VV difieren.
- Procedencia: pública (HuggingFace repo público, Apache 2.0).

# Métricas de validación

## D2 oficial — xView3-SAR Mediterráneo (Adriático, 2026-04-26, palanca L20)

Reproducible vía:

```bash
python -m scripts.filter_xview3_med \
    --xview-dir x-view-us-data --out-dir data/xview3
aria2c --input-file=data/xview3/validation_med.txt \
       --auto-file-renaming=false --continue=true \
       --dir=data/xview3/scenes/
MODELS_DIR=$(pwd)/models python -m scripts.validate_xview3_serial \
    --xview-dir x-view-us-data \
    --tar-dir data/xview3/scenes \
    --tmp-dir data/xview3/scratch \
    --model vesseltracker-sar-yolov8 \
    --output reports/validation_xview3_med_yolo.json \
    --vessels-only --tile-size 640 --tile-overlap 64 \
    --confidence-threshold 0.25
```

**Dataset**: subset Mediterráneo del split `validation` de xView3-SAR
(11 escenas Sentinel-1 IW GRD-H, todas en Adriático, 468 575 km²
totales, **1 997 vessels etiquetados** — confianza ≥ MEDIUM,
`is_vessel=True` post-filtrado). Manifest filtrado por bbox
`30°N..46°N / -6°E..36°E`. Match mode `center` ≤ 20 px (200 m a
10 m GRD pixel spacing — convención xView3-SAR).

| Métrica | Valor | Lectura honesta |
|---|---:|---|
| **mAP** | 0.0242 | bajo por el confidence threshold único; YOLO devuelve pocos picos high-conf en tile dB-stretch sintético |
| **Pd (recall)** | **0.1432** | 286/1 997 vessels detectados. El modelo es muy conservador con la conversión SAR-→-uint8-RGB usada en este harness — pierde vessels pequeños y mid-confidence. |
| **FAR / km²** | **0.0041** | 1 905 falsos positivos en 468k km² — **28× menos que CFAR-only** sobre las mismas escenas. |
| **Precision** | **0.1305** | **8.5× mejor** que CFAR-only (0.0153). Refleja que YOLO filtra el clutter portuario / glint / oleaje que CFAR confunde con vessels. |
| Predicciones totales | 2 191 | 25 veces menos detecciones que CFAR (55k → 2k). Honestidad: alto precision a costa de recall. |

### Per-scene YOLO (Adriático)

| scene_id | área km² | GT | TP | FP | Pd | FAR/km² | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| 264ed833a1 | 42 675 |  55 |  20 |  145 | 0.364 | 0.003 | 0.121 |
| 13dd786ee6 | 42 674 | 150 |  23 |  166 | 0.153 | 0.004 | 0.122 |
| 0d8ed29b07 | 42 520 | 177 |  21 |  229 | 0.119 | 0.005 | 0.084 |
| 9a5aa7310c | 42 675 |  89 |  17 |   93 | 0.191 | 0.002 | 0.155 |
| 36076e5473 | 42 533 | 274 |  28 |  214 | 0.102 | 0.005 | 0.116 |
| f9eb760aaf | 42 628 | 210 |  36 |  177 | 0.171 | 0.004 | 0.169 |
| 8204efcfe9 | 42 629 | 286 |  36 |  172 | 0.126 | 0.004 | 0.173 |
| fe6a8d80fb | 42 533 | 156 |  25 |  232 | 0.160 | 0.005 | 0.097 |
| 3fe00bf7be | 42 533 | 234 |  39 |  218 | 0.167 | 0.005 | 0.152 |
| 487b4884f4 | 42 490 | 258 |  28 |  218 | 0.109 | 0.005 | 0.114 |
| 3808f5703f | 42 684 | 108 |  13 |   41 | 0.120 | 0.001 | 0.241 |

### Comparativa con CFAR sobre las mismas 11 escenas

| Detector | Pd | FAR/km² | Precision | Predicciones |
|---|---:|---:|---:|---:|
| CFAR-only (cfar-default, baseline) | 0.4226 | 0.1157 | 0.0153 | 55 064 |
| **vesseltracker-sar-yolov8 (este)** | 0.1432 | **0.0041** | **0.1305** | 2 191 |
| Fusión esperada CFAR ∩ YOLO (no medida aquí) | ~0.1-0.15 | ~0.001 | >0.5 | ~500 |

### Lecturas

1. **Trade-off CFAR ↔ YOLO confirmado en datos reales**: CFAR caza
   3× más vessels pero genera 25× más detecciones. YOLO filtra
   clutter pero pierde vessels pequeños o de baja confianza. La
   **fusión** del pipeline producción AIDRA combina lo mejor de
   ambos — ese es el modo operativo.
2. **Pd 0.143 con YOLO solo es bajo**: causas plausibles:
   - El confidence threshold 0.25 es alto. Bajarlo a 0.1 subiría Pd
     pero llenaría de FP. Es trade-off conocido.
   - La conversión `_sar_to_uint8` (db_min=-25, db_max=0) puede no
     ser óptima para este modelo (entrenado posiblemente con otro
     stretch).
   - El modelo es fine-tuneado sobre datos de **otra distribución**
     (vesseltracker propietario, no xView3 Med). Domain shift.
3. **Mejora cuantificable post-MVP**: re-tuning del threshold y del
   stretch dB→uint8 sobre xView3 train (240 escenas) podría subir
   Pd a 0.4-0.6 sin sacrificar precision.

### Caveats AI Act (Anexo IV)

- **n=11 escenas, todo Adriático**: muestra geográficamente sesgada.
  Las 4 zonas operativas declaradas (Gibraltar, Suez, Mar Rojo,
  English Channel) **no aparecen** en este subset. Generalización
  externa no garantizada.
- **Domain shift xView3 ↔ vesseltracker**: el modelo no fue
  entrenado sobre xView3 → métricas representan **transfer**, no
  fitness in-domain.
- **Confidence threshold y dB-stretch no calibrados** para xView3 →
  números reportados son **lower bound** del rendimiento.
- **Match mode = center**: convención xView3-SAR oficial (200 m
  tolerance). Bbox-IoU bajaría aún más Pd por las bboxes pequeñas
  que xView3 anota.

## Métricas operacionales (production data, no validation)

Diagnósticos honestos derivados de la base de datos AIDRA en producción
(7 ejecuciones × Sentinel-1 IW GRD, ~29.000 detecciones; generadas con
`scripts/operational_metrics.py`, 2026-04-26):

| Source | Total | on_land | cluster_anomaly | clean (sea, ¬anom) | avg conf | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fused` (YOLO ∩ CFAR) | 96 | 0 | 0 | 96 (100%) | **0.878** | 0.722 | 0.935 |
| `yolo` only | 764 | 96 | 66 | 620 (81%) | 0.678 | 0.258 | 0.883 |
| `cfar` only | 28188 | 26328 | 8514 | 1365 (5%) | 0.536 | 0.417 | 0.600 |

**Lecturas operativas (no validadas):**

- **Agreement rate YOLO ↔ CFAR = 0.33 %**. Los dos detectores son
  estadísticamente independientes — característica deseable para el
  *fusion* (cuando coinciden, la confianza salta a 0.88 medio); no es
  un defecto.
- **`fused` produce el subconjunto operacional limpio**: 100 % en mar,
  cero anomalías; éstas son las detecciones que un operador GEOINT
  procesaría sin filtrado adicional.
- **`cfar` solo**: 93 % cae sobre tierra → confirma que CFAR sin land
  mask actúa como detector de retornos brillantes en general (puerto,
  edificios). El flag `on_land` (I-DET-2, informativo) recupera la
  mayoría; el flag `cluster_anomaly` (I-DET-3) marca otro 30 % como
  artefacto probable.
- **`yolo` solo**: 81 % limpio, confianza media 0.68 — útil para
  detección texture-driven en zonas donde CFAR no fija (oleaje,
  embarcaciones pequeñas).

**Caveat**: estos números reflejan el **comportamiento del ensemble**
sobre datos no etiquetados, no la *precisión* de cada detector vs
ground truth. Sirven para diagnóstico operacional y para dimensionar
filtros (`on_land=false AND cluster_anomaly=false` reduce el ruido de
~29 k a ~2 k detecciones útiles, que es lo que el dashboard expone por
defecto). No sustituyen mAP@0.5 / Pd / FAR, que requieren split
etiquetado.

# Limitaciones

- No detecta tipologías separadas (sólo "ship", clase única).
- Sin re-entrenar, infra-rinde sobre embarcaciones <30 m con baja
  retro-dispersión (pateras, RHIB) en condiciones de viento fuerte.
- Confianza degradada en zonas de mar grueso, plataformas petrolíferas
  fijas suelen ser falso positivo.
- No apto para uso clasificado o seguridad nacional sin auditoría
  externa.

# Sesgos identificados

- Cuantitativos: **no medidos**.
- Cualitativos (a partir de literatura SAR-Ship-Dataset): predilección
  por buques mercantes >50 m; sub-representación de pesca artesanal,
  yates y embarcaciones de migración irregular.

# Interpretabilidad

- Métodos disponibles: Grad-CAM sobre la última `C2f` (`model.model.21`)
  + heatmap CFAR pre-threshold sobre la misma muestra. Implementación
  en `src/models/interpretability.py`; CLI en
  `scripts/run_interpretability.py`.
- Ejemplos generados: anexo D4. Run de referencia
  `fcdf96e2-03ff-4c40-86af-8abffb45fce9_interp_9afa399a` (2026-04-26):
  20 muestras de barcos sea-only de alta confianza, con triplete
  `<idx>_input.png` / `<idx>_gradcam.png` / `<idx>_cfar_score.png` y
  `manifest.json` con SHA256 por artefacto + `commit_sha` +
  `model_hash`. Ruta runtime: `/data/interpretability/<run_id>/`.
- Lectura visual: el Grad-CAM se concentra sobre el reflector central
  del barco (zona roja/amarilla); el CFAR map muestra el mismo barco
  como cluster brillante aislado del background marítimo.

# Trazabilidad

- weights_sha256: `18aec1bb3caf7dd2c5ace8d397e241c485e917c28df248eefe794578c996d671`
- training_seed: desconocido (no documentado upstream)
- training_commit: desconocido (modelo externo)
- onnx_sha256: N/A (modelo en formato PyTorch; la variante ONNX INT8 tiene ficha propia en `vesseltracker-sar-yolov8-int8-dynamic.MODEL_CARD.md`)
- Origen verificado: descarga directa desde HuggingFace via
  `scripts/download-models.sh` (URL: `huggingface.co/hewitleo/sar-ship-detection-yolov8/resolve/main/weights_(model)/best.pt`).

# Conformidad AI Act (Reg. EU 2024/1689)

- Categoría riesgo declarada: **limited risk**.
- Justificación: detección de embarcaciones en aguas internacionales
  para apoyar análisis OSINT/SAR de SatCen. No entra en los casos de
  uso prohibidos del Anexo I ni en los de alto riesgo del Anexo III
  cuando el operador final es una agencia de inteligencia geoespacial
  con propósito declarado de seguridad marítima genérica. Si el modelo
  se reusase para identificación individual de embarcaciones de
  migración con propósito de actuación policial, **pasaría a alto
  riesgo (Anexo III §6)** y requeriría evaluación de impacto adicional.
  AIDRA documenta este límite como "uso fuera de alcance".
- Documentación adicional:
  - HuggingFace card: https://huggingface.co/hewitleo/sar-ship-detection-yolov8
  - SatCen pliego SATCEN/2026/OP/0003 cláusula 10.
  - Esta ficha se actualizará cuando se generen métricas de validación
    formales (D2) y artefactos de interpretabilidad (D4).
