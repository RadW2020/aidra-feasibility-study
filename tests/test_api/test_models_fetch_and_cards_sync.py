"""Cards sync at startup and POST /api/models/fetch (tier 1, mock_db)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.models.cards_sync import sync_model_cards

pytestmark = pytest.mark.anyio


def test_sync_adds_updates_with_backup_and_leaves_volume_only_files(tmp_path: Path):
    dist, dest = tmp_path / "dist", tmp_path / "vol"
    dist.mkdir()
    dest.mkdir()
    (dist / "a.MODEL_CARD.md").write_text("A v2")
    (dist / "b.MODEL_CARD.md").write_text("B")
    (dest / "a.MODEL_CARD.md").write_text("A v1")
    (dest / "only-in-volume.MODEL_CARD.md").write_text("keep me")
    r = sync_model_cards(dist, dest)
    assert r["added"] == ["b.MODEL_CARD.md"] and r["updated"] == ["a.MODEL_CARD.md"]
    assert (dest / "a.MODEL_CARD.md").read_text() == "A v2"
    backups = list(dest.glob("a.MODEL_CARD.md.superseded-*"))
    assert len(backups) == 1 and backups[0].read_text() == "A v1"  # marked, not deleted
    assert (dest / "only-in-volume.MODEL_CARD.md").exists()
    again = sync_model_cards(dist, dest)
    assert again["unchanged"] == ["a.MODEL_CARD.md", "b.MODEL_CARD.md"] and not again["backups"]


def test_sync_is_noop_without_dist_or_same_dir(tmp_path: Path):
    assert sync_model_cards(tmp_path / "missing", tmp_path / "vol")["added"] == []
    (tmp_path / "x.MODEL_CARD.md").write_text("x")
    assert sync_model_cards(tmp_path, tmp_path) == {"added": [], "updated": [], "unchanged": [], "backups": []}


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch):
    d = tmp_path / "models"
    (d / "cards").mkdir(parents=True)
    (d / "cards" / "m-int8-static.MODEL_CARD.md").write_text("card")
    monkeypatch.setenv("MODELS_DIR", str(d))
    return d


def _fake_download(payload: bytes):
    def _dl(url: str, dest: Path, chunk: int = 1 << 20) -> str:
        dest.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()
    return _dl


async def test_fetch_downloads_verifies_and_registers(client, mock_db, models_dir, monkeypatch):
    import src.api.models_api as api

    payload = b"onnx-bytes"
    monkeypatch.setattr(api, "_download", _fake_download(payload))
    body = {"filename": "m-int8-static.onnx", "url": "https://example.org/m.onnx",
            "sha256": hashlib.sha256(payload).hexdigest()}
    resp = await client.post("/api/models/fetch", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert (models_dir / "m-int8-static.onnx").read_bytes() == payload
    assert data["sha256"] == body["sha256"] and data["initial_status"] == "candidate"
    # second call without overwrite -> 409
    assert (await client.post("/api/models/fetch", json=body)).status_code == 409


async def test_fetch_rejects_sha_mismatch_and_discards_file(client, mock_db, models_dir, monkeypatch):
    import src.api.models_api as api

    monkeypatch.setattr(api, "_download", _fake_download(b"tampered"))
    body = {"filename": "m-int8-static.onnx", "url": "https://example.org/m.onnx", "sha256": "0" * 64}
    resp = await client.post("/api/models/fetch", json=body)
    assert resp.status_code == 422 and "SHA256 mismatch" in resp.text
    assert not (models_dir / "m-int8-static.onnx").exists()
    assert not list(models_dir.glob(".*.part"))


async def test_fetch_requires_card_before_downloading(client, mock_db, models_dir, monkeypatch):
    import src.api.models_api as api

    called = []
    monkeypatch.setattr(api, "_download", lambda url, dest, chunk=0: called.append(url) or "x" * 64)
    body = {"filename": "no-card.onnx", "url": "https://example.org/n.onnx", "sha256": "0" * 64}
    resp = await client.post("/api/models/fetch", json=body)
    assert resp.status_code == 422 and "MODEL_CARD" in resp.text
    assert called == []  # gate fires before any bytes move


async def test_fetch_validates_inputs(client, mock_db, models_dir):
    bad_name = {"filename": "../etc/passwd", "url": "https://x/y", "sha256": "0" * 64}
    assert (await client.post("/api/models/fetch", json=bad_name)).status_code == 422
    http_url = {"filename": "m.onnx", "url": "http://x/y", "sha256": "0" * 64}
    assert (await client.post("/api/models/fetch", json=http_url)).status_code == 422
