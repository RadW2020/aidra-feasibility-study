## Análisis de volcados — 11 escenas, 1997 GT

### Geometría de las cajas
- CFAR: n=3818, mediana 4.0×5.0 px (área mediana 18.0 px²)
- YOLO: n=2178, mediana 29.3×33.5 px (área mediana 956.0 px²)

### ¿Por qué no dispara la fusión (IoU ≥ 0.3)?
- Cajas YOLO con un CFAR a ≤ 20 px: 976/2178 (44.8 %)
- Cajas YOLO con IoU ≥ umbral frente al CFAR más cercano: 38/2178 (1.74 %)
- IoU mediana / máxima cuando están co-localizadas: 0.1273 / 0.4667

### Complementariedad sobre el GT (matching por centro)
- Ambos: 255 · solo CFAR: 442 · solo YOLO: 31 · ninguno: 1269 · recuperado por alguno: 36.4 %

### Tamaño del pool para muestreo D4 (TP / FP / FN por conjunto)
- `cfar`: TP 697 · FP 3121 · FN 1300
- `yolo`: TP 286 · FP 1892 · FN 1711
- `aidra`: TP 720 · FP 3890 · FN 1277
- `fused_only`: TP 238 · FP 525 · FN 1759
