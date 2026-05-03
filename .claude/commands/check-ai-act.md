---
description: Verifica conformidad de un modelo con AI Act (Reg. EU 2024/1689) y ficha MODEL_CARD.md
argument-hint: <model_id>
---

Verifica conformidad y ficha del modelo: `$ARGUMENTS`.

Lanzar al agente `ai-act-compliance` con `model_id=$ARGUMENTS`. Procedimiento:

1. **Modo A — Verificar ficha**: localizar `models/$ARGUMENTS/MODEL_CARD.md`, validar campos mínimos, contrastar `weights_sha256`.
2. **Modo C — Checklist AI Act**: completar el checklist (categoría riesgo, datasets, sesgos, limitaciones, interpretabilidad, versionado).
3. **Modo B — Interpretabilidad** (si el usuario lo pide o si D4 está cerca): plan mínimo viable de Grad-CAM/SHAP.

Veredicto final: **APTO PARA EVALUACIÓN** o **BLOQUEADO** con lista de acciones.

Si el modelo está siendo usado en `execution_log` y la ficha falta o es incompleta, marcar **CRÍTICO** — los runs realizados sin ficha no son evidencia válida para D3.
