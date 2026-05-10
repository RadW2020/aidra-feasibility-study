# AIDRA Model Cards

Fichas de modelo conforme al Reg. (EU) 2024/1689 (AI Act) y la
plantilla de `.claude/agents/ai-act-compliance.md`. Cada modelo
registrado en `src/models/manager.py` debe tener una ficha aquí.

| Model ID                       | Estado AI Act         | Riesgo       | Ficha                                                        |
|--------------------------------|-----------------------|--------------|--------------------------------------------------------------|
| `vesseltracker-sar-yolov8`     | apto, validado en xView3-SAR Med/Adriatic | limited risk | [vesseltracker-sar-yolov8.MODEL_CARD.md](./vesseltracker-sar-yolov8.MODEL_CARD.md) |
| `yolov8n`                      | referencia COCO — **no operativo para SAR** | minimal risk | [yolov8n.MODEL_CARD.md](./yolov8n.MODEL_CARD.md)             |
| `yolov8n-vessel`               | **bloqueado** — procedencia | n/a       | [yolov8n-vessel.MODEL_CARD.md](./yolov8n-vessel.MODEL_CARD.md) |
| `yolov8s-vessel`               | **bloqueado** — procedencia | n/a       | [yolov8s-vessel.MODEL_CARD.md](./yolov8s-vessel.MODEL_CARD.md) |
| `cfar-default` (algoritmo)     | baseline validado en xView3-SAR Med/Adriatic | n/a          | [cfar-default.MODEL_CARD.md](./cfar-default.MODEL_CARD.md)   |

## Verificación

Los SHA-256 declarados se calculan con `shasum -a 256 models/*.pt`.
Para validar todas las fichas a la vez:

```bash
for f in models/*.pt; do
  decl=$(grep "^- weights_sha256:" "models/cards/$(basename "$f" .pt).MODEL_CARD.md" | awk '{print $3}' | tr -d '`')
  real=$(shasum -a 256 "$f" | cut -d' ' -f1)
  if [ "$decl" = "$real" ]; then echo "OK  $f"; else echo "MISMATCH $f decl=$decl real=$real"; fi
done
```

## Known limitations

- `yolov8n-vessel` y `yolov8s-vessel` requieren reconstruir
  procedencia o ser sustituidos antes de volver al pipeline activo.
- Grad-CAM y CFAR score maps están implementados para el anexo D4; SHAP
  queda fuera de alcance por coste computacional sobre escenas SAR grandes.
- Las métricas formales mAP/Pd/FAR existen para el subset xView3-SAR
  Mediterráneo / Adriático; falta ampliar cobertura geográfica para convertirlo
  en una evaluación global.
