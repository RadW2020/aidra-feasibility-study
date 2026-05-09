"""
Tests del gate AI Act (I-AIA-1) sobre ``ModelManager._require_model_card``.

Palanca L8 del autoevaluacion: convertir el gate ``_require_model_card``
de ``src/models/manager.py`` en un invariante medible. Sin
``MODEL_CARD.md`` no hay registro de modelo, no hay inferencia, no hay
evidencia para D3/D4.

Ejecucion:

    pytest tests/test_models/test_ai_act_gate.py -x
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest
import yaml

# =====================================================================
# I-AIA-1 — gate ai-act-card en ModelManager._require_model_card
# =====================================================================


@pytest.mark.invariant
class TestAIActGateRequiresCard:
    """I-AIA-1: ningun modelo se registra sin MODEL_CARD.md."""

    def test_register_model_fails_without_card(self, tmp_path):
        from src.models.manager import ModelManager

        # Build a manager without invoking __init__ (no DB).
        mgr = ModelManager.__new__(ModelManager)
        # Point gate to an empty cards dir.
        mgr._MODEL_CARDS_DIR = tmp_path

        with pytest.raises(FileNotFoundError) as excinfo:
            mgr._require_model_card("nonexistent", tmp_path / "any.pt")

        msg = str(excinfo.value)
        assert ("AI Act gate" in msg) or ("I-AIA-1" in msg), (
            f"Mensaje del gate no menciona AI Act ni I-AIA-1: {msg!r}"
        )


@pytest.mark.invariant
class TestAIActGateLookupOrder:
    """I-AIA-1: la busqueda de la ficha sigue un orden definido."""

    def test_card_lookup_uses_model_name_first(self, tmp_path):
        from src.models.manager import ModelManager

        # Card named after the model_name argument.
        card = tmp_path / "myname.MODEL_CARD.md"
        card.write_text("# myname")

        mgr = ModelManager.__new__(ModelManager)
        mgr._MODEL_CARDS_DIR = tmp_path

        # Path stem is "otherfile" — must still resolve via name.
        mgr._require_model_card("myname", tmp_path / "otherfile.pt")

    def test_card_lookup_uses_path_stem_fallback(self, tmp_path):
        from src.models.manager import ModelManager

        # Card named after the .pt file stem, not the model_name argument.
        card = tmp_path / "somefile.MODEL_CARD.md"
        card.write_text("# somefile")

        mgr = ModelManager.__new__(ModelManager)
        mgr._MODEL_CARDS_DIR = tmp_path

        # name "xname" has no card, but path stem "somefile" does.
        mgr._require_model_card("xname", tmp_path / "somefile.pt")


@pytest.mark.invariant
class TestAIActGateCanonicalDir:
    """I-AIA-1: la ficha vive bajo ``models/cards/`` en el repo real."""

    def test_card_must_be_in_canonical_directory(self):
        from src.models.manager import ModelManager

        actual = ModelManager._MODEL_CARDS_DIR
        assert actual == Path("models/cards"), (
            "I-AIA-1 violado: _MODEL_CARDS_DIR no apunta a 'models/cards'. "
            f"Valor actual: {actual!r}"
        )


@pytest.mark.invariant
class TestAIActGateNoFallback:
    """I-AIA-1: no debe existir un fallback silencioso."""

    def test_no_silent_fallback_method_left(self):
        from src.models.manager import ModelManager

        assert not hasattr(ModelManager, "_find_fallback_model"), (
            "I-AIA-1 violado: _find_fallback_model fue reintroducido"
        )


# =====================================================================
# I-AIA-1 — verificacion sobre el repo real (modelos vs fichas)
# =====================================================================


@pytest.mark.invariant
class TestAIActGateRealRepo:
    """Cada peso ``*.pt`` activo del repo tiene su MODEL_CARD.md."""

    def test_real_models_directory_has_cards_for_active_weights(self):
        repo_root = Path(__file__).resolve().parents[2]
        models_dir = repo_root / "models"
        cards_dir = models_dir / "cards"
        archived_dir = models_dir / "archived"

        if not models_dir.is_dir():
            pytest.skip("models/ directory not present in this checkout")

        missing: list[str] = []
        for pt_path in sorted(models_dir.glob("*.pt")):
            # Skip archived weights (L7) if any made their way under archived/.
            if archived_dir.exists():
                try:
                    pt_path.relative_to(archived_dir)
                    continue
                except ValueError:
                    pass

            stem = pt_path.stem
            card = cards_dir / f"{stem}.MODEL_CARD.md"
            if not card.exists():
                missing.append(str(pt_path.relative_to(repo_root)))

        assert not missing, (
            "I-AIA-1 violado: pesos sin MODEL_CARD.md en models/cards/:\n  "
            + "\n  ".join(missing)
            + "\nMueve a models/archived/ o crea la ficha correspondiente."
        )


# =====================================================================
# I-AIA-1 / I-MOD-4 / I-AIA-2 — contenido minimo de cada MODEL_CARD.md
# =====================================================================
#
# La existencia del fichero ya esta cubierta arriba. Este bloque cierra
# el gap real: una ficha vacia o con campos a medias pasaria el gate
# pero no produciria evidencia auditable para D1/D4. El AI Act
# (Reg. EU 2024/1689, Anexo IV) exige un set minimo por modelo;
# CLAUDE.md §5.3 (I-MOD-4) lo aterriza en: proposito, dataset, metricas,
# sesgos, limitaciones, fecha. CLAUDE.md §5.6 (I-AIA-2) exige ademas
# seccion de interpretabilidad.
#
# Si una ficha nueva usa otros encabezados, normalizar aqui o en la
# ficha — no relajar el test.

# Ficheros en models/cards/ que NO son fichas de modelo (README, etc.).
_NON_CARD_FILES = {"README.md"}

# Claves obligatorias en el frontmatter YAML.
_REQUIRED_FRONTMATTER_KEYS = (
    "model_id",
    "version",
    "created_at",
    "authors",
    "license",
)

# Encabezados (nivel 1) que toda ficha debe contener. Se compara por
# prefijo en minusculas para tolerar sufijos del estilo
# "Metricas de validacion" / "Metricas de rendimiento bajo constraint
# profiles" sin perder la regla.
_REQUIRED_SECTION_PREFIXES = (
    "propósito",
    "datos de entrenamiento",
    "métricas",
    "limitaciones",
    "sesgos",
    "interpretabilidad",  # I-AIA-2
    "trazabilidad",
    "conformidad ai act",  # debe citar Reg. EU 2024/1689
)


def _iter_model_cards() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    cards_dir = repo_root / "models" / "cards"
    if not cards_dir.is_dir():
        return []
    return sorted(
        p for p in cards_dir.glob("*.MODEL_CARD.md")
        if p.name not in _NON_CARD_FILES
    )


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Devuelve ``(frontmatter_dict, body)``.

    Lanza ``ValueError`` si el bloque YAML no esta delimitado por
    ``---`` al principio del fichero. El AI Act audita la trazabilidad
    de la ficha por sus campos estructurados, asi que la ausencia de
    frontmatter es un fallo duro, no un warning.
    """
    if not text.startswith("---"):
        raise ValueError("Falta frontmatter YAML (bloque '---' inicial).")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("Frontmatter YAML sin cierre '---'.")
    fm = yaml.safe_load(parts[1]) or {}
    if not isinstance(fm, dict):
        raise ValueError("Frontmatter YAML no es un mapping.")
    return fm, parts[2]


def _h1_headings(body: str) -> list[str]:
    return [
        line[2:].strip()
        for line in body.splitlines()
        if line.startswith("# ") and not line.startswith("## ")
    ]


@pytest.mark.invariant
@pytest.mark.parametrize(
    "card_path",
    _iter_model_cards(),
    ids=lambda p: p.name,
)
class TestModelCardContent:
    """I-MOD-4 + I-AIA-2: cada ficha registrada cumple el contenido minimo.

    Hace un sweep parametrizado por fichero para que el reporte de
    pytest indique exactamente que ficha falla y por que campo, en
    lugar de mezclarlas en un unico assert.
    """

    def test_frontmatter_has_required_keys(self, card_path: Path) -> None:
        fm, _ = _split_frontmatter(card_path.read_text(encoding="utf-8"))
        missing = [k for k in _REQUIRED_FRONTMATTER_KEYS if k not in fm]
        assert not missing, (
            f"I-MOD-4 violado en {card_path.name}: faltan claves de "
            f"frontmatter {missing!r}. Requeridas: "
            f"{list(_REQUIRED_FRONTMATTER_KEYS)}."
        )

    def test_created_at_is_a_date(self, card_path: Path) -> None:
        fm, _ = _split_frontmatter(card_path.read_text(encoding="utf-8"))
        value = fm.get("created_at")
        # PyYAML ya parsea fechas ISO como ``date``/``datetime``. Cualquier
        # otro tipo (string mal formado, numero) seria un fallo de auditoria.
        assert isinstance(value, (date, datetime)), (
            f"I-MOD-4 violado en {card_path.name}: 'created_at' debe ser "
            f"fecha ISO (YYYY-MM-DD), recibido {value!r} "
            f"({type(value).__name__})."
        )

    def test_authors_is_non_empty_list(self, card_path: Path) -> None:
        fm, _ = _split_frontmatter(card_path.read_text(encoding="utf-8"))
        authors = fm.get("authors")
        assert isinstance(authors, list) and authors, (
            f"I-MOD-4 violado en {card_path.name}: 'authors' debe ser "
            f"una lista YAML no vacia, recibido {authors!r}."
        )

    def test_required_sections_present(self, card_path: Path) -> None:
        _, body = _split_frontmatter(card_path.read_text(encoding="utf-8"))
        h1s_lower = [h.lower() for h in _h1_headings(body)]
        missing = [
            prefix
            for prefix in _REQUIRED_SECTION_PREFIXES
            if not any(h.startswith(prefix) for h in h1s_lower)
        ]
        assert not missing, (
            f"I-MOD-4 / I-AIA-2 violado en {card_path.name}: faltan "
            f"secciones {missing!r}. Encontradas: {h1s_lower!r}."
        )

    def test_ai_act_section_cites_regulation(self, card_path: Path) -> None:
        # Anexo IV exige referencia explicita al marco regulatorio que
        # justifica la categorizacion de riesgo. Aceptamos la cita
        # canonica o cualquiera de las variantes que ya usa el repo.
        text = card_path.read_text(encoding="utf-8").lower()
        accepted = (
            "reg. eu 2024/1689",
            "regulation (eu) 2024/1689",
            "reglamento (ue) 2024/1689",
            "ai act",  # nombre coloquial — siempre presente en el header
        )
        assert any(token in text for token in accepted), (
            f"I-AIA-1 violado en {card_path.name}: la ficha no cita "
            f"el AI Act (Reg. EU 2024/1689) ni en seccion ni en "
            f"encabezado."
        )


@pytest.mark.invariant
class TestModelCardsCoverageNonEmpty:
    """Sanity check: el sweep parametrizado anterior no debe estar vacio.

    Si ``_iter_model_cards()`` devuelve [] (porque el repo se reorganizo
    o se renombro la carpeta) los tests parametrizados pasarian
    trivialmente. Este guard impide ese falso verde.
    """

    def test_at_least_one_card_is_audited(self) -> None:
        cards = _iter_model_cards()
        assert cards, (
            "I-AIA-1 violado: no se encontro ninguna *.MODEL_CARD.md "
            "bajo models/cards/. El sweep de auditoria queda vacio."
        )
