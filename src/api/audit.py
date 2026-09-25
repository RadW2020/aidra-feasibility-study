"""Audit trail of mutating API calls: who did what, on which resource, and how it went.

The middleware writes one ``api_audit_log`` row (migration 021), one JSON log
line to Loki and one ``aidra_api_actions_total`` increment for every non-GET
``/api`` request — including the ones refused by auth, validation or the
rate limit, because "why was my request rejected?" is half of the question.

Handlers name the resource they touched with :func:`note_resource`; the
error handlers record the error code. Nothing here can fail a request: a DB
outage costs audit rows, not API availability.

``GET /api/audit/actions`` reads the trail back (scope ``read``).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query, Request

from src.api.errors import ApiError
from src.db.connection import db
from src.observability.loki_logger import StructuredLogger
from src.observability.prometheus_metrics import API_ACTIONS_TOTAL

logger = logging.getLogger("aidra.api.audit")
_log = StructuredLogger("aidra.api.audit")

router = APIRouter(prefix="/audit", tags=["audit"])

_INSERT_AUDIT = """
    INSERT INTO api_audit_log (
        request_id, actor, actor_scope, authenticated, client,
        method, path, operation, status_code, outcome, error_code,
        duration_ms, resource_type, resource_id, idempotency_key, idempotent_replay
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
"""

_SELECT_AUDIT = """
    SELECT id::text AS id, created_at, request_id, actor, actor_scope, client,
           method, path, operation, status_code, outcome, error_code, duration_ms,
           resource_type, resource_id, idempotency_key, idempotent_replay
    FROM api_audit_log
    WHERE ($1::text IS NULL OR actor = $1)
      AND ($2::text IS NULL OR outcome = $2)
      AND ($3::timestamptz IS NULL OR created_at >= $3)
      AND ($4::text IS NULL OR resource_id = $4)
    ORDER BY created_at DESC
    LIMIT $5
"""

OUTCOMES = ("success", "rejected", "error")


def note_resource(request: Request, resource_type: str, resource_id: Any) -> None:
    """Tell the audit middleware which resource this call created or touched."""
    request.state.audit_resource = (resource_type, str(resource_id))


def _outcome(status_code: int) -> str:
    if status_code >= 500:
        return "error"
    if status_code >= 400:
        return "rejected"
    return "success"


def _operation(request: Request) -> str:
    """``METHOD /api/path/{param}``: the request path with its path parameters templated.

    Built from the path, not from ``scope["route"].path``: from FastAPI 0.14x
    a route included through a router keeps a path relative to that router
    (``/tasking/cue``), so the route object would drop the ``/api`` prefix.
    """
    path = request.url.path
    for name, value in (request.scope.get("path_params") or {}).items():
        path = path.replace(f"/{value}", f"/{{{name}}}", 1)
    return f"{request.method} {path}"


async def audit_middleware(request: Request, call_next):
    if request.method in {"GET", "HEAD", "OPTIONS"} or not request.url.path.startswith("/api/"):
        return await call_next(request)

    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        await _record(request, status_code, (time.perf_counter() - start) * 1000.0)


async def _record(request: Request, status_code: int, duration_ms: float) -> None:
    state = request.state
    principal = getattr(state, "principal", None)
    actor = principal.name if principal else "anonymous"
    resource_type, resource_id = getattr(state, "audit_resource", (None, None))
    outcome = _outcome(status_code)
    operation = _operation(request)
    fields = {
        "request_id": getattr(state, "request_id", None) or "-",
        "actor": actor,
        "actor_scope": principal.scope if principal else None,
        "authenticated": bool(principal and principal.authenticated),
        "client": (request.headers.get("x-aidra-client") or "")[:200] or None,
        "method": request.method,
        "path": request.url.path,
        "operation": operation,
        "status_code": status_code,
        "outcome": outcome,
        "error_code": getattr(state, "error_code", None),
        "duration_ms": round(duration_ms, 2),
        "resource_type": resource_type,
        "resource_id": resource_id,
        "idempotency_key": (request.headers.get("idempotency-key") or "")[:200] or None,
        "idempotent_replay": bool(getattr(state, "idempotent_replay", False)),
    }
    try:
        API_ACTIONS_TOTAL.labels(operation=operation, actor=actor, outcome=outcome).inc()
    except Exception:  # noqa: BLE001 - metrics must never break a request
        logger.debug("audit metric failed", exc_info=True)
    _log.info("api_action", extra={"event": "api_action", **fields})
    try:
        await db.execute(_INSERT_AUDIT, *fields.values())
    except Exception:  # noqa: BLE001 - audit must never break a request
        logger.warning("Could not persist audit row for %s", operation, exc_info=True)


@router.get("/actions")
async def list_actions(
    actor: str | None = Query(None, description="Token name recorded as the actor (e.g. claude-agent, operator)."),
    outcome: str | None = Query(None, description="success | rejected | error"),
    since: datetime | None = Query(None, description="ISO 8601; only actions at or after this instant."),
    resource_id: str | None = Query(None, description="Only actions on this resource (execution id, cue id, ...)."),
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    """Mutating API calls, newest first: who, which operation, which resource, outcome.

    Needs a token with scope ``read`` or higher when tokens are configured.
    """
    if outcome is not None and outcome not in OUTCOMES:
        raise ApiError(422, "invalid_parameters", f"Unknown outcome '{outcome}'", valid_values=list(OUTCOMES))
    rows = await db.fetch(_SELECT_AUDIT, actor, outcome, since, resource_id, limit)
    items = []
    for r in rows:
        rec = dict(r)
        if rec.get("created_at") is not None:
            rec["created_at"] = rec["created_at"].isoformat()
        items.append(rec)
    return {"items": items, "count": len(items), "limit": limit}
