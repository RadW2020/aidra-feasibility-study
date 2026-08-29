"""Read-only interpretability run endpoints (mirror D4 evidence without SSH)."""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.anyio


@pytest.fixture
def runs_root(tmp_path, monkeypatch):
    root = tmp_path / "interp"
    run = root / "exec1_interp_abcd1234"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "run_id": "exec1_interp_abcd1234", "created_at": "2026-08-29T00:00:00Z", "execution_id": "exec1",
        "commit_sha": "deadbeef", "n_samples": 1, "gradcam_layer": "model.model.15",
        "gradcam_target": "detection_centre", "sampling": {"strategy": "stratified_confidence_quantiles"},
        "samples": [],
    }))
    (run / "000_gradcam.png").write_bytes(b"\x89PNG fake")
    (root / "not_a_run").mkdir()
    monkeypatch.setenv("INTERPRETABILITY_DIR", str(root))
    return root


async def test_list_manifest_and_file(client, mock_db, runs_root):
    r = await client.get("/api/interpretability/runs")
    assert r.status_code == 200
    runs = r.json()
    assert [x["run_id"] for x in runs] == ["exec1_interp_abcd1234"]
    assert runs[0]["gradcam_layer"] == "model.model.15" and runs[0]["files"] == 1
    m = await client.get("/api/interpretability/runs/exec1_interp_abcd1234/manifest")
    assert m.status_code == 200 and m.json()["commit_sha"] == "deadbeef"
    f = await client.get("/api/interpretability/runs/exec1_interp_abcd1234/files/000_gradcam.png")
    assert f.status_code == 200 and f.content.startswith(b"\x89PNG")


async def test_rejects_traversal_and_unknown(client, mock_db, runs_root):
    assert (await client.get("/api/interpretability/runs/../etc/manifest")).status_code in (404, 422)
    assert (await client.get("/api/interpretability/runs/missing_run_1234/manifest")).status_code == 404
    assert (await client.get("/api/interpretability/runs/exec1_interp_abcd1234/files/..%2Fmanifest.json")).status_code in (404, 422)
    assert (await client.get("/api/interpretability/runs/exec1_interp_abcd1234/files/nope.png")).status_code == 404


async def test_empty_root(client, mock_db, tmp_path, monkeypatch):
    monkeypatch.setenv("INTERPRETABILITY_DIR", str(tmp_path / "none"))
    assert (await client.get("/api/interpretability/runs")).json() == []
