"""Structured error envelope and request ids for every ``/api`` response.

Every error keeps FastAPI's ``detail`` field — existing clients and tests read
it — and gains a machine-readable ``error`` object::

    {
      "detail": "Unknown search zone 'gibraltar_strait'",
      "error": {
        "code": "unknown_zone",            # stable contract; branch on this
        "message": "Unknown search zone 'gibraltar_strait'",
        "hint": "Tip & Cue zone 'gibraltar_strait' maps to search zone 'gibraltar'",
        "valid_values": ["gibraltar", ...],
        "retryable": false,
        "request_id": "5f0c..."
      }
    }

``code`` tells an autonomous client what to do next — fix the input
(``invalid_parameters``, ``unknown_*``), wait (``pipeline_busy``,
``rate_limited``), escalate (``insufficient_scope``) — without parsing prose.
Status codes are unchanged for backward compatibility.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from fastapi import HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("aidra.api.errors")

ERROR_CODE_BY_STATUS: dict[int, str] = {
    400: "bad_request",
    401: "unauthenticated",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    410: "gone",
    413: "payload_too_large",
    422: "invalid_parameters",
    429: "rate_limited",
    500: "internal_error",
    502: "upstream_error",
    503: "service_unavailable",
    504: "upstream_timeout",
}

RETRYABLE_STATUS: frozenset[int] = frozenset({408, 429, 502, 503, 504})

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,128}$")


class ApiError(HTTPException):
    """An anticipated failure with a stable ``code`` and guidance for the caller.

    ``detail`` stays the plain message so legacy readers of ``detail`` keep
    working; the structured fields travel in ``error``.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        hint: str | None = None,
        valid_values: list[Any] | None = None,
        retryable: bool | None = None,
        headers: dict[str, str] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(status_code=status_code, detail=message, headers=headers)
        self.code = code
        self.message = message
        self.hint = hint
        self.valid_values = valid_values
        self.retryable = status_code in RETRYABLE_STATUS if retryable is None else retryable
        self.extra = extra


def request_id_of(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def error_payload(
    request: Request,
    status_code: int,
    *,
    detail: Any,
    code: str | None = None,
    message: str | None = None,
    hint: str | None = None,
    valid_values: list[Any] | None = None,
    retryable: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``{"detail", "error"}`` body and note the code for the audit log."""
    code = code or ERROR_CODE_BY_STATUS.get(status_code, "error")
    if message is None:
        message = detail if isinstance(detail, str) else str(detail)
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "retryable": status_code in RETRYABLE_STATUS if retryable is None else retryable,
        "request_id": request_id_of(request),
    }
    if hint:
        error["hint"] = hint
    if valid_values is not None:
        error["valid_values"] = valid_values
    if extra:
        error.update(jsonable_encoder(extra))
    request.state.error_code = code
    return {"detail": jsonable_encoder(detail), "error": error}


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    if isinstance(exc, ApiError):
        body = error_payload(
            request,
            exc.status_code,
            detail=exc.detail,
            code=exc.code,
            message=exc.message,
            hint=exc.hint,
            valid_values=exc.valid_values,
            retryable=exc.retryable,
            extra=exc.extra,
        )
    else:
        detail = exc.detail
        code = None
        message = None
        extra: dict[str, Any] | None = None
        if isinstance(detail, dict):
            # Legacy structured details (e.g. /pipeline/reset's 409) keep their
            # shape in ``detail``; a ``code`` / ``message`` inside is honoured.
            code = detail.get("code")
            message = detail.get("message")
            extra = {k: v for k, v in detail.items() if k not in {"code", "message"}}
        body = error_payload(
            request, exc.status_code, detail=detail, code=code, message=message, extra=extra
        )
    return JSONResponse(status_code=exc.status_code, content=body, headers=getattr(exc, "headers", None))


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = jsonable_encoder(exc.errors())
    fields = [
        {
            "field": ".".join(str(p) for p in e.get("loc", ()) if p not in ("body", "query", "path")),
            "location": (e.get("loc") or ["?"])[0],
            "problem": e.get("msg"),
        }
        for e in errors
    ]
    names = ", ".join(f["field"] or f["location"] for f in fields)
    body = error_payload(
        request,
        422,
        detail=errors,
        code="invalid_parameters",
        message=f"Invalid parameters: {names}" if names else "Invalid parameters",
        hint="Fix the listed fields; GET /api/vocabulary and GET /api/catalog list valid values.",
        extra={"fields": fields},
    )
    return JSONResponse(status_code=422, content=body)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last resort: JSON instead of Starlette's text/plain 500, without internals."""
    logger.error(
        "Unhandled error on %s %s (request_id=%s)",
        request.method,
        request.url.path,
        request_id_of(request),
        exc_info=exc,
    )
    body = error_payload(
        request,
        500,
        detail="Internal server error",
        code="internal_error",
        hint="Not caused by the request. Report the request_id; retrying is unlikely to help.",
    )
    return JSONResponse(status_code=500, content=body)


async def request_id_middleware(request: Request, call_next):
    """Accept a caller's ``X-Request-ID`` (if well-formed) or mint one; echo it back."""
    incoming = request.headers.get("x-request-id", "")
    request.state.request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


def install(app: Any) -> None:
    """Register the handlers on a FastAPI app (``src.main`` and test apps)."""
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
