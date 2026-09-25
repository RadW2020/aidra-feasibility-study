"""Principals, scopes and the per-route permission table.

Three scopes, ordered — each includes the ones before it:

``read``
    Authenticated reads of sensitive views (the audit log). Everything else
    under ``GET /api`` stays public, as before: the deployment is a public
    demonstrator (Grafana public dashboards, STAC/OGC for QGIS).
``run``
    Produce new evidence through the normal workflow: launch a pipeline run,
    queue a Tip & Cue observation, run synthetic validation or an orbital
    simulation. This is the scope to hand an agent.
``admin``
    Change what future evidence rests on or operate the service: fetch or
    overwrite weights, build D3 bundles, import validation reports, reset the
    pipeline flag, run interpretability. Any mutating route not listed in
    ``ROUTE_SCOPES`` defaults to ``admin`` (secure default).

Tokens come from ``Settings.aidra_api_tokens`` (``name:scope:token,...``) plus
the legacy ``Settings.aidra_api_token``, which keeps its historical meaning:
principal ``operator`` with ``admin``. With no token configured at all (local
development, the unit tests) every call is allowed as ``anonymous``, exactly
the previous behaviour.
"""

from __future__ import annotations

import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from fastapi import Request
from fastapi.responses import JSONResponse

from src.api.errors import error_payload
from src.config import Settings

SCOPES: tuple[str, ...] = ("read", "run", "admin")
_RANK = {s: i for i, s in enumerate(SCOPES)}

# (method, path) -> required scope. Paths are exact; ``*`` matches a suffix.
ROUTE_SCOPES: dict[tuple[str, str], str | None] = {
    # Read-only POSTs: public, like their GET counterparts.
    ("POST", "/api/stac/search"): None,
    ("POST", "/api/pipeline/preview"): None,
    # Produce evidence through the normal workflow.
    ("POST", "/api/pipeline/trigger"): "run",
    ("POST", "/api/pipeline/trigger-all-profiles"): "run",
    ("POST", "/api/tasking/cue"): "run",
    ("POST", "/api/validation/synthetic"): "run",
    ("POST", "/api/orbital/*"): "run",
    # Sensitive reads.
    ("GET", "/api/audit/*"): "read",
}

LEGACY_TOKEN_ACTOR = "operator"
ANONYMOUS = "anonymous"


@dataclass(frozen=True)
class Principal:
    name: str
    scope: str
    authenticated: bool

    def allows(self, required: str | None) -> bool:
        return required is None or _RANK[self.scope] >= _RANK[required]


class TokenConfigError(ValueError):
    pass


def parse_tokens(settings: Settings) -> dict[str, Principal]:
    """Map bearer secret -> principal. Raises on a malformed entry."""
    table: dict[str, Principal] = {}
    for raw in (settings.aidra_api_tokens or "").split(","):
        entry = raw.strip()
        if not entry:
            continue
        parts = entry.split(":", 2)
        if len(parts) != 3 or not all(parts):
            raise TokenConfigError("AIDRA_API_TOKENS entries must be name:scope:token")
        name, scope, token = parts
        if scope not in _RANK:
            raise TokenConfigError(f"unknown scope {scope!r} for {name!r}; use one of {SCOPES}")
        table[token] = Principal(name=name, scope=scope, authenticated=True)
    if settings.aidra_api_token:
        table.setdefault(
            settings.aidra_api_token,
            Principal(name=LEGACY_TOKEN_ACTOR, scope="admin", authenticated=True),
        )
    return table


def required_scope(method: str, path: str) -> str | None:
    """Scope a request needs; ``None`` means public."""
    method = method.upper()
    if (method, path) in ROUTE_SCOPES:
        return ROUTE_SCOPES[(method, path)]
    for (m, pattern), scope in ROUTE_SCOPES.items():
        if m == method and pattern.endswith("*") and path.startswith(pattern[:-1]):
            return scope
    if method in {"GET", "HEAD", "OPTIONS"}:
        return None
    return "admin"


def _match(table: dict[str, Principal], provided: str) -> Principal | None:
    if not provided.startswith("Bearer "):
        return None
    candidate = provided[len("Bearer "):].strip()
    found = None
    # Compare against every secret so timing does not reveal which one matched.
    for token, principal in table.items():
        if secrets.compare_digest(candidate, token):
            found = principal
    return found


def confined_dir(requested: str | None, root: str) -> Path:
    """Resolve a caller-supplied output directory inside ``root`` or refuse it.

    Relative paths are taken relative to ``root``; absolute paths must already
    be inside it. Stops ``out_dir`` on bundle / interpretability requests from
    writing anywhere in the container.
    """
    from src.api.errors import ApiError

    base = Path(root).resolve()
    candidate = Path(requested) if requested else base
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve()
    if resolved != base and base not in resolved.parents:
        raise ApiError(
            422, "path_not_allowed",
            f"out_dir must be inside {base}",
            hint="Omit out_dir to use the default, or pass a sub-directory of it.",
            allowed_root=str(base),
        )
    return resolved


# ---------------------------------------------------------------------------
# Rate limit on mutating calls (per actor, sliding one-minute window)
# ---------------------------------------------------------------------------

_write_log: dict[str, deque[float]] = defaultdict(deque)


def _rate_limited(actor: str, limit_per_minute: int) -> int | None:
    """Return seconds to wait if ``actor`` is over the limit, else record the call."""
    if limit_per_minute <= 0:
        return None
    now = time.monotonic()
    window = _write_log[actor]
    while window and now - window[0] >= 60.0:
        window.popleft()
    if len(window) >= limit_per_minute:
        return max(1, int(60.0 - (now - window[0])) + 1)
    window.append(now)
    return None


def reset_rate_limits() -> None:
    _write_log.clear()


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


async def auth_middleware(request: Request, call_next):
    """Resolve the caller, enforce the route's scope, rate-limit writes.

    Sets ``request.state.principal`` even when the call is rejected, so the
    audit middleware (outside this one) can record who was refused.
    """
    path = request.url.path
    if not path.startswith("/api/"):
        return await call_next(request)

    settings = Settings()
    try:
        table = parse_tokens(settings)
    except TokenConfigError as exc:
        request.state.principal = Principal(ANONYMOUS, "read", False)
        body = error_payload(
            request, 500, detail="API token configuration is invalid",
            code="auth_misconfigured", hint=str(exc),
        )
        return JSONResponse(status_code=500, content=body)

    need = required_scope(request.method, path)
    provided = request.headers.get("authorization", "")

    if not table:
        # No tokens configured: local development, previous behaviour.
        request.state.principal = Principal(ANONYMOUS, "admin", False)
    else:
        principal = _match(table, provided)
        request.state.principal = principal or Principal(ANONYMOUS, "read", False)
        if need is not None and principal is None:
            body = error_payload(
                request, 401,
                detail="Missing or invalid API bearer token",
                code="unauthenticated",
                hint=f"This operation needs a bearer token with scope '{need}' or higher.",
                extra={"required_scope": need},
            )
            return JSONResponse(status_code=401, content=body, headers={"WWW-Authenticate": "Bearer"})
        if principal is not None and not principal.allows(need):
            body = error_payload(
                request, 403,
                detail=f"Token '{principal.name}' has scope '{principal.scope}'; this operation needs '{need}'",
                code="insufficient_scope",
                hint="Do not retry with the same token. Ask an operator to run it or to grant the scope.",
                retryable=False,
                extra={"required_scope": need, "granted_scope": principal.scope},
            )
            return JSONResponse(status_code=403, content=body)

    if request.method not in {"GET", "HEAD", "OPTIONS"} and need is not None:
        wait = _rate_limited(request.state.principal.name, settings.api_write_rate_limit_per_minute)
        if wait is not None:
            body = error_payload(
                request, 429,
                detail=f"More than {settings.api_write_rate_limit_per_minute} mutating calls in a minute",
                code="rate_limited",
                hint=f"Wait {wait} s. Repeated identical calls should reuse an Idempotency-Key instead.",
                extra={"retry_after_seconds": wait},
            )
            return JSONResponse(status_code=429, content=body, headers={"Retry-After": str(wait)})

    return await call_next(request)
