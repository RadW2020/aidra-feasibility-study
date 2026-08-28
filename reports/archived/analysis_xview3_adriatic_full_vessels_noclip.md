## Análisis de volcados — 11 escenas, 1997 GT

### Geometría de las cajas
- CFAR: n=21696, mediana 192.0×2.0 px (área mediana 303.0 px²)
- YOLO: n=1450, mediana 38.7×42.2 px (área mediana 1622.1 px²)

### ¿Por qué no dispara la fusión (IoU ≥ 0.3)?
- Cajas YOLO con un CFAR a ≤ 20 px: 614/1450 (42.3 %)
- Cajas YOLO con IoU ≥ umbral frente al CFAR más cercano: 1/1450 (0.07 %)
- IoU mediana / máxima cuando están co-localizadas: 0.0715 / 0.3032

### Complementariedad sobre el GT (matching por centro)
- Ambos: 169 · solo CFAR: 531 · solo YOLO: 17 · ninguno: 1280 · recuperado por alguno: 35.9 %

### Tamaño del pool para muestreo D4 (TP / FP / FN por conjunto)
- `cfar`: TP 700 · FP 20996 · FN 1297
- `yolo`: TP 186 · FP 1264 · FN 1811
- `aidra`: TP 714 · FP 17543 · FN 1283
- `fused_only`: TP 0 · FP 0 · FN 1997
