# Quality Gates — guía de activación

Los **8 quality gates** del proyecto están definidos en `CLAUDE.md` §6. Este fichero documenta cómo activarlos como hooks automáticos en `settings.local.json`. **No se aplican automáticamente** — el usuario decide cuándo encenderlos para no introducir fricción inesperada.

## Resumen de gates

| ID | Comando manual | Cuándo aplica | Severidad si falla |
|---|---|---|---|
| `gate:lint` | `ruff check .` | Tras editar `*.py` | media |
| `gate:tests-touched` | `pytest tests/test_<modulo>/ -x` | Tras editar `src/<modulo>/` | alta |
| `gate:invariants` | `pytest -k "invariant" -x` | Antes de commit en `src/` | crítica |
| `gate:schema` | revisar migrations + queries | Tras editar `db/` | alta |
| `gate:reproducibility` | `pytest -k "reproducibility" --seed=42` | Antes de generar evidencia | crítica |
| `gate:compression-triplet` | `/run-triplet ...` | Al añadir variante | alta |
| `gate:ai-act-card` | `/check-ai-act <model>` | Al registrar modelo | crítica |
| `gate:eu-region` | grep configs por regiones | En cualquier config nueva | crítica |

## Snippet de hooks propuesto (revisar antes de aplicar)

Añadir a `.claude/settings.local.json` bajo la clave `"hooks"` cuando se quiera activar:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "if echo \"$CLAUDE_TOOL_FILE_PATHS\" | grep -qE '\\.py$'; then cd \"$CLAUDE_PROJECT_DIR\" && ruff check . 2>&1 | tail -20; fi"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "cd \"$CLAUDE_PROJECT_DIR\" && pytest -k 'invariant or reproducibility' -x 2>&1 | tail -30 || echo 'GATE FAIL: invariants/reproducibility'"
          }
        ]
      }
    ]
  }
}
```

> **Aviso**: los hooks ejecutan en cada acción del agente. Pueden ralentizar la sesión. Activar gradualmente: primero `gate:lint`, después tests del módulo, después invariantes.

## Tests de invariantes — pendientes de crear

Los gates dependen de tests etiquetados con marcadores. Plan mínimo de tests a añadir:

| Invariante (CLAUDE.md §5) | Test sugerido | Marker |
|---|---|---|
| I-SAR-1 cadena preprocesado | `tests/test_pipeline/test_preprocessing.py::test_invariant_full_chain` | `invariant` |
| I-SAR-2 edge filter | `test_invariant_edge_swath_filter` | `invariant` |
| I-DET-1 columnas detección | `tests/test_pipeline/test_detection.py::test_invariant_detection_schema` | `invariant` |
| I-DET-2 on_land flag | `test_invariant_on_land_excluded_from_metrics` | `invariant` |
| I-MOD-1 regla terna | `tests/test_models/test_compression.py::test_invariant_triplet_required` | `invariant` |
| I-TRACE-1 SHA256 todos | `tests/test_traceability/test_hasher.py::test_invariant_all_artifacts_hashed` | `invariant` |
| I-TRACE-2 pending→final | `test_invariant_pending_before_run` | `invariant` |
| I-EU-1 región UE | `tests/test_config/test_eu_region.py` | `invariant` |
| Reproducibilidad seed | `tests/test_pipeline/test_reproducibility.py` | `reproducibility` |

Configurar `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "invariant: invariantes de dominio AIDRA",
    "reproducibility: tests de reproducibilidad bit-exact",
]
```

## Procedimiento recomendado

1. Crear los tests de invariantes (uno por uno, conforme se toca cada módulo).
2. Cuando haya cobertura mínima, activar `gate:lint` solo.
3. Tras 1 semana sin falsos positivos, activar `gate:invariants`.
4. Hooks de evidencia (`reproducibility`) solo antes de empaquetar D3.

No activar todo de golpe.
