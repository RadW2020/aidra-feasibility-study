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

from pathlib import Path

import pytest

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
