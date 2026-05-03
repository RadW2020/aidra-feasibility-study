# models/archived/ — pesos retirados de evaluación

> **Nota (2026-04-26):** los binarios `.pt` listados abajo **fueron eliminados
> del historial git** del repo público para evitar redistribuir pesos cuya
> procedencia no se pudo reconstruir y cuya licencia (AGPL-3.0 heredada de
> Ultralytics YOLOv8) imponía copyleft viral sobre el resto del proyecto. Las
> fichas (`MODEL_CARD.md`) se conservan como registro de gobernanza.

Esta carpeta conserva la cadena documental de pesos `.pt` que **no entran al
pipeline activo** de AIDRA. El `ModelManager` los ignora porque están fuera
de `models/`.

## Entradas actuales

| Peso | Motivo de archivado | Decisión |
|---|---|---|
| `yolov8n-vessel.pt` | Procedencia desconocida (no aparece en `scripts/download-models.sh` ni en `README.md`/`TECHNICAL_SPEC.md`). Sin dataset, hiperparámetros ni evaluación documentados. | Retirado de evaluación AIDRA en autoaudit 2026-04-26 (palanca L7). Su `MODEL_CARD.md` declaró el modelo como bloqueado bajo I-AIA-1; mantenerlo en `models/` proyectaba ruido de gobernanza. |
| `yolov8s-vessel.pt` | Idéntico al anterior. | Idem. |

## Política

- Los pesos archivados **no** se cargan en `ModelManager.scan_and_register()`
  porque la búsqueda de `.pt` se limita al directorio raíz `models/`,
  no recursivo.
- El test `tests/test_models/test_ai_act_gate.py::test_real_models_directory_has_cards_for_active_weights`
  excluye explícitamente `models/archived/` del chequeo I-AIA-1.
- Para reactivar un modelo: mover de vuelta a `models/`, reconstruir
  procedencia en `models/cards/<name>.MODEL_CARD.md` (Anexo IV AI Act),
  y validar con `scripts/run_validation.py`.
- Las fichas históricas siguen vivas en `models/cards/` para preservar
  la cadena documental.
