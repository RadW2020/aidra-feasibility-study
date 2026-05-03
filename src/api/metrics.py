"""
Prometheus metrics endpoint.

Exposes all AIDRA metrics in Prometheus text exposition format for
scraping by the Prometheus server configured in the observability stack.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

from src.observability.prometheus_metrics import generate_metrics_response

router = APIRouter(tags=["monitoring"])


@router.get("/metrics")
async def prometheus_metrics(request: Request) -> Response:
    """Return all Prometheus metrics in text exposition format.

    Honours the ``Accept`` header for OpenMetrics negotiation: when a
    scraper requests ``application/openmetrics-text`` (Prometheus 2.x
    with exemplar support, Grafana Tempo correlations, etc.) the
    response uses OpenMetrics 1.0.0 and includes the ``run_id``
    exemplar attached by the pipeline.

    This endpoint is scraped by the Prometheus instance configured in
    ``docker-compose.yml``.  It returns counters, histograms, gauges,
    and info metrics registered by the AIDRA pipeline, including:

    - ``aidra_pipeline_runs_total`` — pipeline execution counts
      (with ``trace_id`` exemplar = run_id, OpenMetrics only)
    - ``aidra_detections_total`` — detection counts by source
    - ``aidra_inference_duration_seconds`` — inference latency histogram
      (with ``trace_id`` exemplar = run_id, OpenMetrics only)
    - ``aidra_peak_ram_mb`` — peak RAM gauge
    - ``aidra_model_size_mb`` — model file sizes
    - ``aidra_active_cues`` — pending Tip & Cue entries
    - ``aidra_system_info`` — system metadata
    """
    accept = request.headers.get("accept")
    body, content_type = generate_metrics_response(accept)
    return Response(content=body, media_type=content_type)
