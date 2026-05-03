---
description: Compara el enfoque AIDRA con el proyecto MYRIAD (referente UE de IA + EO multi-sensor)
---

Compara el enfoque actual de AIDRA con el proyecto **MYRIAD** (European Defence Fund, GMV, marzo 2026, ~5M€, 48 meses). Referencia: `analisis_completo.md` §11.

MYRIAD destaca por: análisis multi-sensor (óptico + SAR), fusión, calibración radiométrica, **IA explicable (XAI)**, integración con infraestructura SatCen.

Procedimiento:

1. **Mapear** capacidades AIDRA actuales (lectura de README, mvp_oci.md, src/) y compararlas con cada eje de MYRIAD.
2. **Tabla comparativa**:

```
Eje                      MYRIAD            AIDRA estado actual         Brecha / Oportunidad
-----------------------  ----------------  --------------------------  ----------------------
Multi-sensor             óptico + SAR      solo SAR (S-1)              Roadmap S-2 si tiempo
Fusión multi-sensor      sí                no                          Fuera scope MVP
Calibración radiométrica sí                σ⁰ en preprocess.py         OK base
XAI / interpretabilidad  sí                pendiente (D4)              CRÍTICO para Q3
GEOINT integration       sí                vía API + GeoJSON           Validar OGC
Compresión modelos       parcial           núcleo AIDRA                Diferenciador
On-board / edge          marginal          núcleo AIDRA                Diferenciador
Trazabilidad / AI Act    estándar UE       núcleo AIDRA                Diferenciador
```

3. **Posicionamiento**: identificar 2–3 ejes donde AIDRA es **complementario** (no competidor) a MYRIAD. Esos son los que se enfatizan en la propuesta:
   - On-board / restricciones de hardware (MYRIAD opera en tierra mayoritariamente).
   - Compresión sistemática de modelos (quant/prune/KD) con métricas reproducibles.
   - Trazabilidad SHA256 + AI Act como núcleo, no como añadido.

4. **Riesgos de solapamiento**: dónde AIDRA podría sonar redundante. Proponer matices.

5. **Argumentario para la oferta**: 3 frases que articulen la diferencia.

Output: tabla comparativa + posicionamiento + argumentario, todo en castellano, conciso.
