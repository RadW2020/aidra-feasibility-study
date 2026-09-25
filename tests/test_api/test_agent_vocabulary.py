"""The published vocabulary covers every value the code writes (src/vocabulary.py)."""

from __future__ import annotations

import re
from pathlib import Path

from src.vocabulary import CUE_STATUSES, EXECUTION_STATUSES, QUALITY_VERDICTS, as_document

ROOT = Path(__file__).resolve().parents[2]


def _literals(pattern: str, files: list[str]) -> set[str]:
    found: set[str] = set()
    for f in files:
        found |= set(re.findall(pattern, (ROOT / f).read_text()))
    return found


def test_every_execution_status_written_is_documented():
    written = _literals(
        r"""(?:update_status\([^,]+,\s*|status\s*=\s*)["']([a-z_]+)["']""",
        ["src/pipeline/engine.py", "src/traceability/recorder.py"],
    )
    # 'oom' is only a PIPELINE_RUNS_TOTAL label; the row itself gets 'error'.
    written -= {"oom"}
    # The reaper writes its status in SQL.
    from src.db.queries import REAP_ORPHAN_EXECUTIONS

    written |= set(re.findall(r"SET status = '([a-z_]+)'", REAP_ORPHAN_EXECUTIONS))
    assert {"success", "failed"} <= written, "scan found too little — the patterns are broken"
    assert set(CUE_STATUSES) != set(EXECUTION_STATUSES)
    missing = written - set(EXECUTION_STATUSES)
    assert not missing, f"statuses written without a documented meaning: {missing}"


def test_every_quality_verdict_the_engine_assigns_is_documented():
    assigned = _literals(r'return "([a-z_]+)"', ["src/pipeline/engine.py"]) & {
        "land_artifact", "cluster_artifact", "valid_sea_target", "candidate",
    }
    assert assigned <= set(QUALITY_VERDICTS)


def test_document_shape():
    doc = as_document()
    statuses = {e["value"]: e for e in doc["execution_statuses"]}
    assert statuses["running"]["terminal"] is False
    assert statuses["skipped"]["terminal"] is True
    assert all(e["meaning"] for section in doc.values() for e in section)
