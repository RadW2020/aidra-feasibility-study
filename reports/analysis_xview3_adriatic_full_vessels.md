## Análisis de volcados — 11 escenas, 1997 GT

### Geometría de las cajas
- CFAR: n=3818, mediana 4.0×5.0 px (área mediana 18.0 px²)
- YOLO: n=1442, mediana 38.8×42.3 px (área mediana 1628.1 px²)

### ¿Por qué no dispara la fusión (IoU ≥ 0.3)?
- Cajas YOLO con un CFAR a ≤ 20 px: 610/1442 (42.3 %)
- Cajas YOLO con IoU ≥ umbral frente al CFAR más cercano: 1/1442 (0.07 %)
- IoU mediana / máxima cuando están co-localizadas: 0.0708 / 0.3032

### Complementariedad sobre el GT (matching por centro)
- Ambos: 169 · solo CFAR: 528 · solo YOLO: 17 · ninguno: 1283 · recuperado por alguno: 35.8 %

### Tamaño del pool para muestreo D4 (TP / FP / FN por conjunto)
- `cfar`: TP 697 · FP 3121 · FN 1300
- `yolo`: TP 186 · FP 1256 · FN 1811
- `aidra`: TP 711 · FP 3749 · FN 1286
- `fused_only`: TP 0 · FP 0 · FN 1997
