---
model_id: cfar-default
version: 2.0.0-rayleigh
created_at: 2026-04-25
authors: ["AIDRA team", "RadW2020"]
license: MIT (mismo que el resto del código AIDRA)
---

# Propósito

Detector CFAR (Constant False Alarm Rate) para imágenes Sentinel-1
SAR GRD. Algoritmo clásico de procesado de señal — **no es un modelo
entrenable**, no tiene pesos. Implementado en
`src/models/cfar.py:CFARDetector`. Funciona como detector primario
sobre potencia lineal de sigma0 y como verificador independiente
para fusión con la salida de YOLO.

Variantes implementadas:
- **CA-CFAR exponencial** (default): umbral multiplicativo
  `T = α · mean` donde
  `α = N · (PFA^(-1/N) − 1) ≈ −ln(PFA)`. Apropiado para potencia
  lineal con clutter Rayleigh.
- **CA-CFAR gaussiano** (legacy): umbral aditivo `T = mean + k·std`,
  pensado para datos normalizados (no usar sobre dB ni sobre potencia
  lineal — históricamente fuente del bug del 2026-04-25).
- **OS-CFAR (ordenado)**: usa percentil del fondo, más robusto a
  clutter heterogéneo.

# Datos de entrenamiento

- Dataset(s): **n/a — algoritmo determinista**.
- Tamaño: n/a.
- Sesgos conocidos: derivados de la asunción de clutter Rayleigh
  homogéneo. En transición tierra-mar, plataformas y zonas de oleaje
  fuerte la suposición se rompe → aumento de FAR local.
- Procedencia: implementación propia, basada en literatura clásica
  (Skolnik, *Radar Handbook*, cap. 8; Robey et al. 1992 para OS-CFAR).

# Métricas de validación

## D2 oficial — xView3-SAR Mediterráneo (Adriático, 2026-04-26, palanca L20)

Reproducible vía:

```bash
python -m scripts.filter_xview3_med \
    --xview-dir x-view-us-data --out-dir data/xview3
aria2c --input-file=data/xview3/validation_med.txt \
       --auto-file-renaming=false --continue=true \
       --dir=data/xview3/scenes/
python -m scripts.validate_xview3_serial \
    --xview-dir x-view-us-data \
    --tar-dir data/xview3/scenes \
    --tmp-dir data/xview3/scratch \
    --model cfar-default \
    --output reports/validation_xview3_med_cfar.json \
    --vessels-only --tile-size 1024
```

**Dataset**: subset Mediterráneo del split `validation` de xView3-SAR
(11 escenas Sentinel-1 IW GRD-H, todas en Adriático, 468 575 km²
totales, **1 997 vessels etiquetados** — confianza ≥ MEDIUM,
`is_vessel=True` post-filtrado). Manifest filtrado por bbox
`30°N..46°N / -6°E..36°E`. Match mode `center` ≤ 20 px (200 m a
10 m GRD pixel spacing — convención xView3-SAR).

| Métrica | Valor | Lectura honesta |
|---|---:|---|
| **mAP** | 0.0104 | bajo por design — CFAR-only no produce bbox vessel-shaped, así que la curva PR es plana |
| **Pd (recall)** | **0.4226** | 844/1 997 vessels detectados. Las 1 153 que CFAR pierde son mayoritariamente embarcaciones < 30 m bajo `min_mean_snr=2.0`. |
| **FAR / km²** | **0.1157** | 54 220 falsos positivos en 468k km² — predominantemente plataformas, glint, oleaje, infraestructura no etiquetada por xView3. |
| **Precision** | 0.0153 | refleja que CFAR-only **no es un detector apto** sin fusión YOLO o land-mask aplicada aguas arriba. |
| Predicciones totales | 55 064 | tras `--confidence-threshold 0.10` y guarda de tiles saturados (>5 % CFAR-pixel hits). |

### Per-scene breakdown (Adriático)

| scene_id (10-char) | área km² | GT | TP | FP | Pd | FAR/km² |
|---|---:|---:|---:|---:|---:|---:|
| 264ed833a1 | 42 675 |  55 |  33 | 11 766 | 0.600 | 0.276 |
| 13dd786ee6 | 42 674 | 150 |  53 |  8 688 | 0.353 | 0.204 |
| 0d8ed29b07 | 42 520 | 177 |  84 |  2 768 | 0.475 | 0.065 |
| 9a5aa7310c | 42 675 |  89 |  33 | 11 329 | 0.371 | 0.265 |
| 36076e5473 | 42 533 | 274 |  96 |  2 381 | 0.350 | 0.056 |
| f9eb760aaf | 42 628 | 210 |  94 |  3 128 | 0.448 | 0.073 |
| 8204efcfe9 | 42 629 | 286 | 118 |  3 181 | 0.413 | 0.075 |
| fe6a8d80fb | 42 533 | 156 |  92 |  2 543 | 0.590 | 0.060 |
| 3fe00bf7be | 42 533 | 234 |  99 |  2 964 | 0.423 | 0.070 |
| 487b4884f4 | 42 490 | 258 | 105 |  1 694 | 0.407 | 0.040 |
| 3808f5703f | 42 684 | 108 |  37 |  3 778 | 0.343 | 0.089 |

### Lecturas

1. **CFAR-only descartado para producción**: este resultado refuerza
   la decisión arquitectural de AIDRA (CFAR + YOLO en fusion mode).
   La precision de 1.5 % confirma lo que la inspección operacional
   ya sugería en el operational data del run
   `0500f7d8-2a70-46ff-be6a-644682a220cd`.
2. **Pd 0.42 con CFAR-only sobre vessels reales**: alineado con la
   literatura para CA-CFAR sin land-mask sobre Sentinel-1 GRD
   completo (~0.3-0.5 dependiendo del clutter). No es competitivo
   con detectores ML modernos (Pd ~0.7-0.9), pero **es un baseline
   defendible** y reproducible.
3. **Sesgo Adriático**: las 11 escenas todas caen en la misma
   sub-cuenca (Italia / Croacia / Eslovenia). El subset xView3-SAR
   validation no incluye escenas en Gibraltar / Suez / Red Sea
   (las otras zonas operativas AIDRA), así que estos números
   sub-representan el comportamiento en mar abierto.
4. **Match mode = center**: CFAR clustered devuelve centroides
   con bboxes pequeños (~3 px) — bbox-IoU bajaría Pd al 0 %
   incluso con detección perfecta. Center matching ≤ 20 px es la
   convención xView3-SAR oficial.

### Caveats AI Act (Anexo IV)

- **n=11 escenas**: muestra geográficamente sesgada (solo Adriático).
  Ampliar a Gibraltar / Suez / Red Sea es trabajo POST-MVP.
- **CFAR no es modelo IA**: ver sección "Conformidad AI Act" — métricas
  reportadas por consistencia con el resto del pipeline, no por
  obligación regulatoria.
- **xView3-SAR ground truth**: incluye verificación AIS + manual.
  Los vessels < 20 m son sub-representados — coherente con el techo
  Pd ~0.4 que observamos.

## Synthetic baseline (D2 — 2026-04-26, palanca L14)

Reproducible vía `scripts/build_synthetic_manifest.py` +
`scripts/run_validation.py`. Manifest sintético: 20 escenas
640×640 px (819.2 km² total) con 57 ground-truth vessels (Rayleigh
clutter + Gaussian bright points; seed 42).

```bash
python -m scripts.build_synthetic_manifest \
    --out data/validation/synthetic --num-scenes 20 --seed 42
python -m scripts.run_validation \
    --manifest data/validation/synthetic/manifest.json \
    --model cfar-default \
    --output reports/validation_synthetic_cfar.json \
    --confidence-threshold 0.10 \
    --match-mode center --center-tolerance-px 20
```

| Métrica | Valor | Interpretación honesta |
|---|---:|---|
| **Pd (recall)** | **0.6491** | 37 de 57 vessels detectados. Los 20 que faltan son los más débiles (gaussian half-width 3-4 px, SNR cluster < 2.0 → bajo el gate de `min_mean_snr` de producción). |
| **FAR / km²** | **0.0000** | 0 falsos positivos en 819 km² — el threshold rayleigh+DBSCAN+SNR gate está calibrado conservadoramente. |
| **Precision** | **1.0000** | Todo lo que CFAR cluster devuelve (37/37) coincide con un GT real. |
| **mAP** | **0.6491** | Igual que Pd al usar single-threshold center matching. |

**Match mode** = `center` (Euclidean ≤ 20 px = 200 m a 10 m GRD pixel
spacing, **convención xView3-SAR**). CFAR-clustered devuelve
centroides — bbox-IoU subestima Pd al 0% porque las cajas predichas
son de ~3 px y las GT de hasta 30 px aunque el centroide sea correcto.

**Caveats explícitos:**
- Datos **sintéticos** (Rayleigh+Gaussian): NO sustituye xView3-SAR /
  HRSID. Sirve como baseline reproducible y sanity check del harness.
- 20 escenas + 57 GT: muestra **ilustrativa**, no estadísticamente
  cerrada (intervalos de confianza no calculados).
- Métricas medidas con misma config que producción
  (`detect_with_clustering(min_cluster_size=5, eps=1.5,
  min_mean_snr=2.0)`).

## Datos operacionales reales

- Comportamiento medido en run
  `0500f7d8-2a70-46ff-be6a-644682a220cd` (S1D IW GRD, 2026-04-23):
  - 1 363 tiles 640×640 sobre el Estrecho.
  - 9 574 píxeles disparan threshold; tras DBSCAN + footprint clip +
    land-mask + dedup geográfico → **465 detecciones cfar-only**.
  - 24 fused cuando coinciden con YOLO (max conf 0.94, mean SNR
    cluster ≈ 100).
  - **Esto es muestra única, no es FAR/Pd estadística**.
- PFA configurable vía variable de entorno `CFAR_PFA` (default
  `1e-5`). Para esa PFA el threshold teórico es α ≈ 11.5 sobre la
  media local de las training cells.

# Limitaciones

- Asume sigma0 lineal y clutter Rayleigh homogéneo. **Aplicar a dB o
  a producto sin calibrar es incorrecto** y produce FAR muy
  desviada (incidente del 2026-04-25 documentado en
  `docs/sar-pipeline-invariants.md` cuando exista).
- Tamaño fijo de guard window: `guard_size=8` (default actual)
  cubre embarcaciones hasta ~170 m al pixel size de S1 IW GRD
  (~10 m/px). Buques mayores filtran píxeles propios a las training
  cells y pueden ser sub-detectados.
- Sin re-tunear PFA, FAR sobre mar fuerte / olas largas puede crecer.
- No discrimina tipología (sólo "punto brillante en el mar" → CFAR
  cluster).

# Sesgos identificados

- **Sesgo físico**: el algoritmo amplifica retro-dispersores
  metálicos (corner reflector). Embarcaciones de fibra/madera
  pequeñas pueden no superar threshold incluso si están presentes
  visualmente.
- **Sin sesgo de entrenamiento** (algoritmo determinista).

# Interpretabilidad

- Métodos disponibles: el propio threshold map es la explicación
  directa. Para cada píxel detectado se exporta `intensity`, `snr`,
  `mean_snr` por cluster (ver `Detection.cfar_snr`), y la anchura
  del cluster en píxeles (`num_pixels`).
- Ejemplos: cualquier imagen procesada genera estos campos en
  `detections.cfar_snr` y `bbox_pixel`.

# Trazabilidad

- weights_sha256: n/a (no hay pesos)
- training_seed: n/a
- training_commit: ver
  `git log -- src/models/cfar.py` — versión actual incorpora threshold
  Rayleigh y SNR gate desde el commit 2026-04-25 (`Rayleigh-correct
  CA-CFAR`).
- algorithm_hash: SHA-256 del fichero
  `src/models/cfar.py` se puede recalcular en el momento; no
  cacheado en una constante.

# Conformidad AI Act (Reg. EU 2024/1689)

- Categoría riesgo declarada: **fuera del alcance del Reg. EU
  2024/1689 — IA no aplicable**.
- Justificación: el detector CFAR es un algoritmo determinista de
  procesado de señal estadística. El AI Act define "sistema de IA"
  como un sistema con autonomía y capacidad de inferir, derivado de
  un proceso de aprendizaje (Art. 3 §1). CFAR no aprende ni infiere
  en ese sentido — es un test de hipótesis estadístico clásico. No
  obstante, AIDRA mantiene esta ficha por consistencia con el resto
  de modelos y para documentar limitaciones operativas.
- Documentación adicional:
  - Skolnik M., *Radar Handbook*, McGraw-Hill, 3ª ed. cap. 8.
  - Robey F. et al., *Stat. theory for adaptive matched filter*,
    IEEE Trans. AES, 28(1), 1992.
  - Implementación de referencia: `src/models/cfar.py`.
