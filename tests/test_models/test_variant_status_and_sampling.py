"""I-MOD-3 candidate status for scanned variants and stratified D4 sampling."""

from __future__ import annotations

from pathlib import Path

from src.models.interpretability import stratified_sample
from src.models.manager import initial_variant_status


class TestInitialVariantStatus:
    def test_baseline_active_variants_candidate(self):
        assert initial_variant_status("none") == "active"
        assert initial_variant_status("") == "active"
        assert initial_variant_status("static_int8") == "candidate"
        assert initial_variant_status("dynamic_int8") == "candidate"
        assert initial_variant_status("l1_unstructured") == "candidate"

    def test_upsert_writes_status_only_on_insert(self):
        from src.db.queries import UPSERT_MODEL

        assert "status" in UPSERT_MODEL.split("ON CONFLICT")[0]
        assert "status" not in UPSERT_MODEL.split("ON CONFLICT")[1]  # never clobbers a decision

    def test_migration_019_allows_candidate(self):
        mig = Path("src/db/migrations/019_latency_percentiles_candidate_status_legacy_marks.sql").read_text()
        assert "CHECK (status IN ('candidate', 'active', 'rejected', 'retired'))" in mig
        assert "methodology:pre-321de6b" in mig and "NOT LIKE" in mig  # idempotent R9 marking


class TestStratifiedSample:
    def _pool(self):
        pool = []
        for i in range(40):
            pool.append({"id": i, "confidence": 0.3 + i * 0.015, "source": "cfar" if i % 3 else "yolo"})
        return pool

    def test_spreads_over_quantiles_and_sources(self):
        picked, info = stratified_sample(self._pool(), 8, seed=1)
        assert len(picked) == 8 and info["picked"] == 8 and info["bins"] == 4
        strata = sorted({p["stratum"] for p in picked})
        assert strata == ["q1/4", "q2/4", "q3/4", "q4/4"]
        confs = [p["confidence"] for p in picked]
        assert min(confs) < 0.45 and max(confs) > 0.75  # low AND high confidence present
        assert {p["source"] for p in picked} == {"cfar", "yolo"}

    def test_deterministic_for_a_seed(self):
        a, _ = stratified_sample(self._pool(), 6, seed=7)
        b, _ = stratified_sample(self._pool(), 6, seed=7)
        assert [p["id"] for p in a] == [p["id"] for p in b]

    def test_small_pool_and_empty(self):
        picked, info = stratified_sample(self._pool()[:3], 8, seed=0)
        assert len(picked) == 3
        assert stratified_sample([], 5, seed=0) == ([], {"strategy": "stratified_confidence_quantiles", "bins": 4, "picked": 0})

    def test_production_sampler_no_longer_top_confidence_only(self):
        src = Path("src/models/interpretability.py").read_text()
        assert "ORDER BY confidence DESC LIMIT $2" not in src
        assert "stratified_sample(candidates, n_samples, seed)" in src
