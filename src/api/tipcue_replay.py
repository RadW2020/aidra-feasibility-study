"""
Tip & Cue replay endpoints (Wow effect #3).

For a given tasking_queue entry, returns the full chain:

    T0  Original execution that triggered the cue (low-confidence detection)
    Cue Tasking entry (target bbox, reason, priority, scheduled_at)
    T1  Re-execution that confirmed (or refuted) the detection

Two endpoints:

    GET /api/tipcue/replay/{tasking_id}.json
        Structured payload (machine readable; consumed by Grafana / scripts).

    GET /api/tipcue/replay/{tasking_id}.html
        Self-contained HTML viewer (visual demo). Renders three columns
        with the SAR thumbnails of the triggering detection (T0) and the
        confirmation detection (T1) side by side, plus latency metrics.

This is the SatCen-tender-extra Tip & Cue value prop, made visible in
under five seconds.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from html import escape
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from src.db.connection import db

logger = logging.getLogger("aidra.api.tipcue_replay")

router = APIRouter(tags=["tipcue"])


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------


_SELECT_TASKING_BY_ID = """
    SELECT
        t.id,
        t.created_at AS cue_created_at,
        t.trigger_type,
        t.triggered_by,
        t.triggering_detections,
        ST_AsGeoJSON(t.target_bbox) AS target_bbox_geojson,
        t.target_zone,
        t.priority,
        t.reason,
        t.status,
        t.scheduled_at,
        t.executed_at,
        t.execution_id,
        t.result_status,
        t.confirmed_detections,
        t.attempts
    FROM tasking_queue t
    WHERE t.id = $1
"""


_SELECT_EXECUTION_BRIEF = """
    SELECT
        e.id,
        e.image_id,
        e.image_title,
        e.image_sensing_date,
        e.created_at,
        e.search_zone,
        e.model_name,
        e.constraint_profile,
        e.num_detections,
        e.avg_confidence,
        e.polarisation,
        e.orbit_direction,
        e.relative_orbit
    FROM execution_log e
    WHERE e.id = $1
"""


_SELECT_DETECTIONS_BY_IDS = """
    SELECT
        d.id,
        d.confidence,
        d.source,
        d.on_land,
        d.cluster_anomaly,
        d.thumbnail_path,
        ST_X(d.center_geo) AS longitude,
        ST_Y(d.center_geo) AS latitude
    FROM detections d
    WHERE d.id = ANY($1::uuid[])
    ORDER BY d.confidence DESC
"""


_SELECT_TOP_DETECTIONS_FOR_EXECUTION = """
    SELECT
        d.id,
        d.confidence,
        d.source,
        d.on_land,
        d.cluster_anomaly,
        d.thumbnail_path,
        ST_X(d.center_geo) AS longitude,
        ST_Y(d.center_geo) AS latitude
    FROM detections d
    WHERE d.execution_id = $1
    ORDER BY d.confidence DESC
    LIMIT 12
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detection_to_dict(row: Any, base_url: str) -> dict[str, Any]:
    det_id = row["id"]
    has_thumb = bool(row.get("thumbnail_path"))
    return {
        "id": str(det_id),
        "confidence": float(row["confidence"]),
        "source": row["source"],
        "lat": float(row["latitude"]),
        "lon": float(row["longitude"]),
        "on_land": bool(row.get("on_land", False)),
        "cluster_anomaly": bool(row.get("cluster_anomaly", False)),
        "thumbnail_url": (
            f"{base_url}/api/detections/{det_id}/thumbnail.png"
            if has_thumb
            else None
        ),
    }


async def _execution_brief(exec_id: UUID | None) -> dict[str, Any] | None:
    if exec_id is None:
        return None
    row = await db.fetchrow(_SELECT_EXECUTION_BRIEF, exec_id)
    if row is None:
        return None
    return {
        "execution_id": str(row["id"]),
        "image_id": row.get("image_id"),
        "image_title": row.get("image_title"),
        "sensing_date": (
            row["image_sensing_date"].isoformat()
            if row.get("image_sensing_date")
            else None
        ),
        "processed_at": (
            row["created_at"].isoformat() if row.get("created_at") else None
        ),
        "search_zone": row.get("search_zone"),
        "model_name": row.get("model_name"),
        "constraint_profile": row.get("constraint_profile"),
        "num_detections": row.get("num_detections", 0),
        "avg_confidence": (
            float(row["avg_confidence"])
            if row.get("avg_confidence") is not None
            else None
        ),
        "sar": {
            "polarisation": row.get("polarisation"),
            "orbit_direction": row.get("orbit_direction"),
            "relative_orbit": row.get("relative_orbit"),
        },
    }


def _seconds_between(later: datetime | None, earlier: datetime | None) -> float | None:
    if later is None or earlier is None:
        return None
    return float((later - earlier).total_seconds())


# ---------------------------------------------------------------------------
# Replay assembly
# ---------------------------------------------------------------------------


async def _build_replay(tasking_id: UUID, base_url: str) -> dict[str, Any]:
    cue = await db.fetchrow(_SELECT_TASKING_BY_ID, tasking_id)
    if cue is None:
        raise HTTPException(status_code=404, detail="Tasking entry not found")

    # T0: original execution that triggered the cue.
    t0 = await _execution_brief(cue.get("triggered_by"))

    # T0 triggering detections.
    triggering: list[dict[str, Any]] = []
    raw_triggering = cue.get("triggering_detections") or []
    if raw_triggering:
        rows = await db.fetch(
            _SELECT_DETECTIONS_BY_IDS, list(raw_triggering)
        )
        triggering = [_detection_to_dict(r, base_url) for r in rows]

    # T1: re-execution that confirmed (or not) the detection.
    t1 = await _execution_brief(cue.get("execution_id"))

    confirming: list[dict[str, Any]] = []
    if cue.get("execution_id") is not None:
        rows = await db.fetch(
            _SELECT_TOP_DETECTIONS_FOR_EXECUTION, cue["execution_id"]
        )
        confirming = [_detection_to_dict(r, base_url) for r in rows]

    # Cue stage.
    bbox_geojson = cue.get("target_bbox_geojson")
    bbox_obj: dict[str, Any] | None = None
    if bbox_geojson:
        try:
            bbox_obj = json.loads(bbox_geojson) if isinstance(
                bbox_geojson, str
            ) else bbox_geojson
        except (TypeError, json.JSONDecodeError):
            bbox_obj = None

    cue_stage = {
        "tasking_id": str(cue["id"]),
        "created_at": (
            cue["cue_created_at"].isoformat()
            if cue.get("cue_created_at")
            else None
        ),
        "scheduled_at": (
            cue["scheduled_at"].isoformat()
            if cue.get("scheduled_at")
            else None
        ),
        "executed_at": (
            cue["executed_at"].isoformat() if cue.get("executed_at") else None
        ),
        "trigger_type": cue.get("trigger_type"),
        "target_zone": cue.get("target_zone"),
        "target_bbox": bbox_obj,
        "priority": cue.get("priority", 0),
        "reason": cue.get("reason"),
        "status": cue.get("status"),
        "result_status": cue.get("result_status"),
        "confirmed_detections": cue.get("confirmed_detections"),
        "attempts": cue.get("attempts", 0),
    }

    # Latency metrics.
    metrics: dict[str, Any] = {}
    t0_sensed = (
        datetime.fromisoformat(t0["sensing_date"])
        if t0 and t0.get("sensing_date")
        else None
    )
    t1_sensed = (
        datetime.fromisoformat(t1["sensing_date"])
        if t1 and t1.get("sensing_date")
        else None
    )
    metrics["t0_to_cue_seconds"] = _seconds_between(
        cue.get("cue_created_at"),
        datetime.fromisoformat(t0["processed_at"])
        if t0 and t0.get("processed_at")
        else None,
    )
    metrics["t0_to_t1_sensing_seconds"] = _seconds_between(t1_sensed, t0_sensed)
    metrics["t0_to_t1_processed_seconds"] = _seconds_between(
        datetime.fromisoformat(t1["processed_at"])
        if t1 and t1.get("processed_at")
        else None,
        datetime.fromisoformat(t0["processed_at"])
        if t0 and t0.get("processed_at")
        else None,
    )
    if triggering and confirming:
        avg_t0 = sum(d["confidence"] for d in triggering) / len(triggering)
        avg_t1 = sum(d["confidence"] for d in confirming) / len(confirming)
        metrics["avg_confidence_t0"] = round(avg_t0, 4)
        metrics["avg_confidence_t1"] = round(avg_t1, 4)
        metrics["confidence_uplift"] = round(avg_t1 - avg_t0, 4)

    return {
        "stage_t0": {
            "execution": t0,
            "triggering_detections": triggering,
        },
        "stage_cue": cue_stage,
        "stage_t1": {
            "execution": t1,
            "confirming_detections": confirming,
        },
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# JSON endpoint
# ---------------------------------------------------------------------------


@router.get("/tipcue/replay/{tasking_id}.json")
async def replay_json(tasking_id: UUID) -> dict[str, Any]:
    """Returns the full Tip & Cue chain (T0 → Cue → T1) as JSON."""
    return await _build_replay(tasking_id, base_url="http://localhost:8000")


# ---------------------------------------------------------------------------
# HTML viewer (the wow demo)
# ---------------------------------------------------------------------------


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Tip &amp; Cue Replay — {tasking_id}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0e1117; color: #e6e6e6; margin: 0; padding: 24px; }}
    h1 {{ font-size: 22px; margin: 0 0 4px 0; color: #58a6ff; }}
    .subtitle {{ color: #8b949e; font-size: 13px; margin-bottom: 24px; }}
    .lane {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px;
             align-items: stretch; }}
    .stage {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
              padding: 16px; }}
    .stage h2 {{ margin: 0 0 8px 0; font-size: 16px; color: #58a6ff;
                 border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
    .stage h2 small {{ color: #8b949e; font-weight: normal; font-size: 11px;
                       margin-left: 8px; }}
    .meta {{ font-size: 12px; color: #c9d1d9; margin: 8px 0; }}
    .meta b {{ color: #8b949e; font-weight: 500; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, 96px);
             gap: 6px; margin-top: 10px; }}
    .det {{ width: 96px; text-align: center; }}
    .det img {{ width: 96px; height: 96px; object-fit: cover;
                border: 1px solid #30363d; border-radius: 4px;
                background: #000; }}
    .det .conf {{ font-size: 11px; margin-top: 4px;
                  padding: 1px 4px; border-radius: 3px;
                  display: inline-block; color: #fff; }}
    .conf.low  {{ background: #da3633; }}
    .conf.med  {{ background: #d29922; color: #000; }}
    .conf.high {{ background: #238636; }}
    .arrow {{ font-size: 28px; color: #58a6ff; text-align: center;
              line-height: 1; align-self: center; }}
    .metrics {{ background: #161b22; border: 1px solid #30363d;
                border-radius: 8px; padding: 16px; margin-top: 16px;
                display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }}
    .metric {{ text-align: center; }}
    .metric .v {{ font-size: 22px; font-weight: 600; color: #58a6ff; }}
    .metric .v.up {{ color: #238636; }}
    .metric .v.down {{ color: #da3633; }}
    .metric .l {{ font-size: 11px; color: #8b949e; margin-top: 4px; }}
    .empty {{ color: #8b949e; font-style: italic; padding: 12px 0; }}
    .reason {{ background: #1c2128; padding: 8px; border-radius: 4px;
               font-family: ui-monospace, SFMono-Regular, monospace;
               font-size: 11px; color: #c9d1d9; margin-top: 8px; }}
  </style>
</head>
<body>
  <h1>Tip &amp; Cue Replay</h1>
  <div class="subtitle">tasking_id: <code>{tasking_id}</code></div>

  <div class="lane">
    <!-- T0 -->
    <div class="stage">
      <h2>T0 · Initial detection <small>{t0_sensed}</small></h2>
      {t0_meta}
      <div class="grid">{t0_grid}</div>
    </div>

    <!-- Cue -->
    <div class="stage">
      <h2>Cue · Re-tasking <small>{cue_created}</small></h2>
      {cue_meta}
      <div class="reason">{cue_reason}</div>
    </div>

    <!-- T1 -->
    <div class="stage">
      <h2>T1 · Confirmation <small>{t1_sensed}</small></h2>
      {t1_meta}
      <div class="grid">{t1_grid}</div>
    </div>
  </div>

  <div class="metrics">
    <div class="metric">
      <div class="v">{m_t0_t1_sensing}</div>
      <div class="l">T0 → T1 sensing gap</div>
    </div>
    <div class="metric">
      <div class="v">{m_t0_cue}</div>
      <div class="l">T0 → Cue created</div>
    </div>
    <div class="metric">
      <div class="v {uplift_class}">{m_uplift}</div>
      <div class="l">Confidence uplift (T1 − T0)</div>
    </div>
    <div class="metric">
      <div class="v">{m_attempts}</div>
      <div class="l">Re-tasking attempts</div>
    </div>
  </div>
</body>
</html>
"""


def _render_detections(dets: list[dict[str, Any]]) -> str:
    if not dets:
        return '<div class="empty">— no detections recorded —</div>'
    cells: list[str] = []
    for d in dets[:24]:
        conf = d["confidence"]
        cls = "high" if conf >= 0.8 else ("med" if conf >= 0.5 else "low")
        thumb = d.get("thumbnail_url") or ""
        img = (
            f'<img src="{escape(thumb)}" alt="">'
            if thumb
            else '<div style="width:96px;height:96px;background:#21262d;'
            'border:1px dashed #30363d;border-radius:4px;'
            'display:flex;align-items:center;justify-content:center;'
            'color:#484f58;font-size:11px;">no thumb</div>'
        )
        cells.append(
            f'<div class="det">{img}'
            f'<div class="conf {cls}">{conf:.2f}</div></div>'
        )
    return "".join(cells)


def _render_meta(execution: dict[str, Any] | None) -> str:
    if execution is None:
        return '<div class="empty">— stage not reached yet —</div>'
    rows = [
        ("Image", execution.get("image_id") or "—"),
        ("Model", execution.get("model_name") or "—"),
        ("Profile", execution.get("constraint_profile") or "—"),
        ("Detections", str(execution.get("num_detections") or 0)),
        ("Avg conf", f"{execution.get('avg_confidence') or 0:.3f}"),
        ("Pol.", execution.get("sar", {}).get("polarisation") or "—"),
        ("Orbit", execution.get("sar", {}).get("orbit_direction") or "—"),
    ]
    items = "".join(
        f'<div class="meta"><b>{escape(k)}:</b> {escape(str(v))}</div>'
        for k, v in rows
    )
    return items


def _format_seconds(s: float | None) -> str:
    if s is None:
        return "—"
    if abs(s) < 90:
        return f"{s:.0f} s"
    if abs(s) < 5400:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.2f} h"


def _format_uplift(v: float | None) -> tuple[str, str]:
    if v is None:
        return "—", ""
    sign = "+" if v >= 0 else ""
    cls = "up" if v > 0 else ("down" if v < 0 else "")
    return f"{sign}{v:.2f}", cls


@router.get(
    "/tipcue/replay/{tasking_id}.html",
    response_class=HTMLResponse,
)
async def replay_html(tasking_id: UUID) -> HTMLResponse:
    """Self-contained HTML viewer for a Tip & Cue chain.

    Renders three columns (T0 / Cue / T1) with side-by-side SAR
    thumbnails and a footer with latency + confidence-uplift metrics.
    """
    payload = await _build_replay(tasking_id, base_url="http://localhost:8000")

    t0 = payload["stage_t0"]["execution"]
    t1 = payload["stage_t1"]["execution"]
    cue = payload["stage_cue"]
    metrics = payload["metrics"]

    uplift_str, uplift_cls = _format_uplift(metrics.get("confidence_uplift"))

    cue_meta = (
        f'<div class="meta"><b>Status:</b> {escape(cue.get("status") or "—")}</div>'
        f'<div class="meta"><b>Priority:</b> {cue.get("priority", 0)}</div>'
        f'<div class="meta"><b>Zone:</b> {escape(cue.get("target_zone") or "—")}</div>'
    )

    html = _HTML_TEMPLATE.format(
        tasking_id=escape(str(tasking_id)),
        t0_sensed=escape(
            (t0 or {}).get("sensing_date") or "—"
        ),
        t0_meta=_render_meta(t0),
        t0_grid=_render_detections(payload["stage_t0"]["triggering_detections"]),
        cue_created=escape(cue.get("created_at") or "—"),
        cue_meta=cue_meta,
        cue_reason=escape(cue.get("reason") or "no reason recorded"),
        t1_sensed=escape((t1 or {}).get("sensing_date") or "—"),
        t1_meta=_render_meta(t1),
        t1_grid=_render_detections(payload["stage_t1"]["confirming_detections"]),
        m_t0_t1_sensing=_format_seconds(metrics.get("t0_to_t1_sensing_seconds")),
        m_t0_cue=_format_seconds(metrics.get("t0_to_cue_seconds")),
        m_uplift=uplift_str,
        uplift_class=uplift_cls,
        m_attempts=str(cue.get("attempts", 0)),
    )
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# List of replayable chains
# ---------------------------------------------------------------------------


_LIST_REPLAYABLE = """
    SELECT
        t.id AS tasking_id,
        t.created_at AS cue_created_at,
        t.status,
        t.target_zone,
        t.priority,
        t.reason,
        t.triggered_by AS t0_execution_id,
        t.execution_id AS t1_execution_id,
        cardinality(t.triggering_detections) AS triggering_count,
        t.confirmed_detections,
        t.result_status
    FROM tasking_queue t
    ORDER BY t.created_at DESC
    LIMIT $1
"""


@router.get("/tipcue/replays")
async def list_replays(limit: int = 50) -> list[dict[str, Any]]:
    """Lists tip-cue chains available for replay (most recent first)."""
    rows = await db.fetch(_LIST_REPLAYABLE, limit)
    return [
        {
            "tasking_id": str(r["tasking_id"]),
            "created_at": r["cue_created_at"].isoformat(),
            "status": r["status"],
            "target_zone": r.get("target_zone"),
            "priority": r.get("priority"),
            "reason": r.get("reason"),
            "t0_execution_id": (
                str(r["t0_execution_id"]) if r.get("t0_execution_id") else None
            ),
            "t1_execution_id": (
                str(r["t1_execution_id"]) if r.get("t1_execution_id") else None
            ),
            "triggering_count": r.get("triggering_count") or 0,
            "confirmed_detections": r.get("confirmed_detections"),
            "result_status": r.get("result_status"),
            "view_url": (
                f"/api/tipcue/replay/{r['tasking_id']}.html"
            ),
        }
        for r in rows
    ]
