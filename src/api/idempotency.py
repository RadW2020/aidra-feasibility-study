"""``Idempotency-Key`` support for the mutating calls an agent is likely to retry.

A client that times out on ``POST /api/pipeline/trigger`` cannot know whether
the run started. Without a key, retrying either starts a second 50-minute run
or queues a duplicate cue. With a key:

* same actor + key + same body within 24 h -> the stored response is replayed
  (header ``Idempotent-Replay: true``); nothing is executed again;
* same key, different body -> 422 ``idempotency_key_reused``;
* same key while the first call is still being handled -> 409
  ``idempotency_in_progress``.

Only successful (2xx) responses are stored: a rejected call can be fixed and
retried with the same key.

Usage in a handler::

    claim = await idempotency.begin(request, body_dict)
    if claim.replay is not None:
        return claim.replay
    ... do the work ...
    await idempotency.complete(claim, 200, response_dict)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from src.api.errors import ApiError
from src.db.connection import db

logger = logging.getLogger("aidra.api.idempotency")

_KEY_RE = re.compile(r"^[A-Za-z0-9._:-]{1,200}$")
TTL_HOURS = 24

_CLAIM = """
    INSERT INTO api_idempotency_keys (actor, key, method, path, request_hash)
    VALUES ($1, $2, $3, $4, $5)
    ON CONFLICT (actor, key) DO UPDATE
        SET method = EXCLUDED.method, path = EXCLUDED.path,
            request_hash = EXCLUDED.request_hash, status_code = NULL,
            response_json = NULL, created_at = NOW(), completed_at = NULL
        WHERE api_idempotency_keys.created_at < NOW() - make_interval(hours => $6::int)
    RETURNING actor
"""

_EXISTING = """
    SELECT method, path, request_hash, status_code, response_json
    FROM api_idempotency_keys WHERE actor = $1 AND key = $2
"""

_COMPLETE = """
    UPDATE api_idempotency_keys
    SET status_code = $3, response_json = $4::jsonb, completed_at = NOW()
    WHERE actor = $1 AND key = $2
"""

_RELEASE = "DELETE FROM api_idempotency_keys WHERE actor = $1 AND key = $2 AND completed_at IS NULL"


@dataclass
class Claim:
    actor: str | None = None
    key: str | None = None
    replay: JSONResponse | None = None

    @property
    def active(self) -> bool:
        return self.key is not None and self.replay is None


def _request_hash(method: str, path: str, body: dict[str, Any]) -> str:
    canonical = json.dumps(jsonable_encoder(body), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{method} {path} {canonical}".encode()).hexdigest()


async def begin(request: Request, body: dict[str, Any]) -> Claim:
    """Claim the request's ``Idempotency-Key`` (if any) or build its replay."""
    key = request.headers.get("idempotency-key")
    if not key:
        return Claim()
    if not _KEY_RE.match(key):
        raise ApiError(
            422, "invalid_idempotency_key",
            "Idempotency-Key must be 1-200 characters of [A-Za-z0-9._:-]",
        )
    principal = getattr(request.state, "principal", None)
    actor = principal.name if principal else "anonymous"
    method, path = request.method, request.url.path
    digest = _request_hash(method, path, body)

    claimed = await db.fetchval(_CLAIM, actor, key, method, path, digest, TTL_HOURS)
    if claimed:
        return Claim(actor=actor, key=key)

    row = await db.fetchrow(_EXISTING, actor, key)
    if row is None:  # vanished between the two statements: treat as fresh
        return Claim(actor=actor, key=key)
    if row["request_hash"] != digest or row["path"] != path:
        raise ApiError(
            422, "idempotency_key_reused",
            f"Idempotency-Key '{key}' was already used for a different request",
            hint="Use a new key for a different request; reuse a key only to retry the identical call.",
        )
    if row["status_code"] is None:
        raise ApiError(
            409, "idempotency_in_progress",
            f"A request with Idempotency-Key '{key}' is still being processed",
            hint="Wait a few seconds and retry with the same key to get its result.",
            retryable=True,
        )
    stored = row["response_json"]
    content = json.loads(stored) if isinstance(stored, str) else stored
    request.state.idempotent_replay = True
    return Claim(
        actor=actor,
        key=key,
        replay=JSONResponse(
            status_code=row["status_code"],
            content=content,
            headers={"Idempotent-Replay": "true"},
        ),
    )


async def complete(claim: Claim, status_code: int, response: dict[str, Any]) -> None:
    if not claim.active:
        return
    try:
        await db.execute(
            _COMPLETE, claim.actor, claim.key, status_code,
            json.dumps(jsonable_encoder(response)),
        )
    except Exception:  # noqa: BLE001 - the call itself succeeded
        logger.warning("Could not store idempotent response for key %s", claim.key, exc_info=True)


async def release(claim: Claim) -> None:
    """Forget an unfinished claim so the caller can fix the request and retry."""
    if not claim.active:
        return
    try:
        await db.execute(_RELEASE, claim.actor, claim.key)
    except Exception:  # noqa: BLE001
        logger.warning("Could not release idempotency key %s", claim.key, exc_info=True)
