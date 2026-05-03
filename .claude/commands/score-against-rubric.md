---
description: Autoevaluación honesta del proyecto contra la rúbrica del pliego SatCen (70 pts Q1+Q2+Q3)
---

Realiza una autoevaluación del estado actual del repositorio AIDRA contra la rúbrica de calidad técnica del pliego SatCen (`analisis_completo.md` §7).

Rúbrica (máx. 70 pts; mínimo 30 para pasar):

| Criterio | Puntos | Desglose |
|---|---|---|
| Q1 — Equipo | 15 | expertos adicionales (5) + complementariedad (5) + plan backup (5) |
| Q2 — Gestión | 15 | plan de proyecto (10) + gestión de riesgos (5) |
| Q3 — Propuesta técnica | 40 | metodología y simulación (10) + integración GEOINT (10) + entorno demo + trazabilidad + documentación IA (20) |

Procedimiento:

1. **Q1 (Equipo)**: este es un proyecto personal — declarar honestamente "no aplica como contratista, evalúa solo dimensiones técnicas". Saltar los 15 pts.
2. **Q2 (Gestión)**: revisar `mvp_oci.md`, README, fases declaradas, gestión de riesgos documentada en código (timeouts, fallbacks). Estimar puntuación con justificación.
3. **Q3 (Propuesta técnica)** — lo importante:
   - **Metodología y simulación (10)**: lanzar `sar-preproc-auditor` + revisar perfiles `src/profiles/`. Evaluar si la cadena SAR es completa y si los perfiles representan hardware espacial creíble.
   - **Integración GEOINT (10)**: lanzar `geoint-integrator`. Evaluar exportación, formatos OGC, consumibilidad por sistemas tipo SatCen.
   - **Entorno demo + trazabilidad + IA (20)**: lanzar `traceability-curator` (auditoría) + `ai-act-compliance`. Evaluar SHA256, linaje, fichas de modelo, interpretabilidad.

Output:

```
AIDRA SELF-SCORE vs SatCen RUBRIC
==================================

Q1 Equipo:                            n/a (proyecto personal)
Q2 Gestión:                           xx/15  — justificación
Q3 Metodología/simulación:            xx/10  — justificación
Q3 Integración GEOINT:                xx/10  — justificación
Q3 Demo + trazabilidad + IA:          xx/20  — justificación
                                      -----
TOTAL Q-técnico (Q2+Q3):              xx/55

PALANCAS DE MEJORA (priorizadas por puntos/esfuerzo):
1. <acción — pts esperados — esfuerzo>
2. ...

RIESGOS DE PUNTUACIÓN:
- <riesgo concreto — qué mitigarlo cuesta>
```

Sé honesto. Penalizaciones por falta de evidencia, módulos sin tests, fichas ausentes, etc., **deben** reflejarse. Puntos sin justificación documentable no se cuentan.
