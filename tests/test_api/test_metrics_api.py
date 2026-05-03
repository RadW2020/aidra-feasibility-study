"""
Integration tests for GET /api/metrics (Prometheus endpoint).

Tier 1: Validates that the metrics endpoint returns the correct content
type and contains aidra-prefixed metric names.
"""

from __future__ import annotations

# ------------------------------------------------------------------
# test_metrics_returns_200
# ------------------------------------------------------------------


async def test_metrics_returns_200(client):
    """GET /api/metrics returns HTTP 200."""
    resp = await client.get("/api/metrics")
    assert resp.status_code == 200


# ------------------------------------------------------------------
# test_metrics_content_type
# ------------------------------------------------------------------


async def test_metrics_content_type(client):
    """Response content type is text/plain or openmetrics."""
    resp = await client.get("/api/metrics")
    content_type = resp.headers.get("content-type", "")

    # prometheus_client returns either text/plain or openmetrics format
    assert any(
        ct in content_type
        for ct in ("text/plain", "text/openmetrics", "application/openmetrics")
    ), f"Unexpected content-type: {content_type}"


# ------------------------------------------------------------------
# test_metrics_contains_aidra
# ------------------------------------------------------------------


async def test_metrics_contains_aidra(client):
    """Response body contains at least one 'aidra_' prefixed metric."""
    resp = await client.get("/api/metrics")
    body = resp.text

    assert "aidra_" in body, (
        "No aidra_ prefixed metrics found in response body"
    )


# ------------------------------------------------------------------
# test_metrics_contains_known_metrics
# ------------------------------------------------------------------


async def test_metrics_contains_known_metrics(client):
    """Response contains the core AIDRA metric names."""
    resp = await client.get("/api/metrics")
    body = resp.text

    expected_metrics = [
        "aidra_pipeline_runs_total",
        "aidra_detections_total",
        "aidra_inference_duration_seconds",
        "aidra_system",
    ]

    for metric_name in expected_metrics:
        assert metric_name in body, (
            f"Expected metric '{metric_name}' not found in response"
        )


# ------------------------------------------------------------------
# test_metrics_body_not_empty
# ------------------------------------------------------------------


async def test_metrics_body_not_empty(client):
    """Response body is non-empty (contains at least HELP/TYPE lines)."""
    resp = await client.get("/api/metrics")
    body = resp.text

    assert len(body) > 100, (
        f"Metrics body suspiciously short: {len(body)} bytes"
    )
    assert "# HELP" in body, "No HELP comments found in metrics output"
    assert "# TYPE" in body, "No TYPE declarations found in metrics output"


# ------------------------------------------------------------------
# test_metrics_openmetrics_negotiation
# ------------------------------------------------------------------


async def test_metrics_openmetrics_negotiation(client):
    """Asking for OpenMetrics changes the content type and adds # EOF."""
    resp = await client.get(
        "/api/metrics",
        headers={"Accept": "application/openmetrics-text"},
    )
    assert resp.status_code == 200
    content_type = resp.headers.get("content-type", "")
    assert "openmetrics-text" in content_type, (
        f"Did not honour Accept header: {content_type}"
    )
    body = resp.text
    # OpenMetrics 1.0.0 always ends with # EOF — Prometheus text never
    # has it. This is the marker that proves negotiation worked.
    assert body.rstrip().endswith("# EOF"), (
        "OpenMetrics body must end with the '# EOF' sentinel"
    )


async def test_metrics_run_id_exemplar_emitted(client):
    """Pipeline run-counter exemplar carries the execution_id (I-TRACE-3)."""
    from src.observability.prometheus_metrics import (
        PIPELINE_RUNS_TOTAL,
    )

    PIPELINE_RUNS_TOTAL.labels(
        profile="ground", model_variant="test-exemplar", status="success"
    ).inc(1, exemplar={"trace_id": "1234567890abcdef1234567890abcdef"})

    resp = await client.get(
        "/api/metrics",
        headers={"Accept": "application/openmetrics-text"},
    )
    body = resp.text
    assert "1234567890abcdef1234567890abcdef" in body, (
        "Exemplar trace_id missing from OpenMetrics output — I-TRACE-3 broken"
    )
    assert "test-exemplar" in body
