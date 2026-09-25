"""HTTP client for the AIDRA API, used by the MCP tools.

Every non-2xx answer becomes an :class:`AidraApiError` carrying the API's
``error`` envelope (``code``, ``message``, ``hint``, ``valid_values``,
``retryable``, ``request_id``) plus the HTTP status, so a tool can hand the
model something it can act on instead of a stack trace.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.mcp_server import __version__


class AidraApiError(Exception):
    """An API call that did not succeed, with the structured reason."""

    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(payload.get("message", "AIDRA API error"))
        self.payload = payload

    @property
    def code(self) -> str:
        return str(self.payload.get("code", "error"))

    def to_json(self) -> str:
        return json.dumps({"error": self.payload}, default=str)


@dataclass
class AidraConfig:
    base_url: str = "http://localhost:8000"
    token: str | None = None
    client_name: str = f"aidra-mcp/{__version__}"
    mode: str = "read-only"  # read-only | operator
    timeout_s: float = 30.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> AidraConfig:
        mode = os.environ.get("AIDRA_MCP_MODE", "read-only").strip().lower()
        if mode not in ("read-only", "operator"):
            raise ValueError("AIDRA_MCP_MODE must be 'read-only' or 'operator'")
        return cls(
            base_url=os.environ.get("AIDRA_API_URL", "http://localhost:8000").rstrip("/"),
            token=os.environ.get("AIDRA_API_TOKEN") or None,
            client_name=os.environ.get("AIDRA_MCP_CLIENT_NAME", f"aidra-mcp/{__version__}"),
            mode=mode,
            timeout_s=float(os.environ.get("AIDRA_MCP_TIMEOUT_S", "30")),
        )


class AidraClient:
    def __init__(self, config: AidraConfig, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.config = config
        self._transport = transport
        self.peer: str | None = None  # the MCP client (e.g. claude-code), once known

    def _headers(self, idempotency_key: str | None = None) -> dict[str, str]:
        client = self.config.client_name + (f" via {self.peer}" if self.peer else "")
        headers = {"X-AIDRA-Client": client, "Accept": "application/json", **self.config.extra_headers}
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    def url(self, path: str) -> str:
        return f"{self.config.base_url}{path}"

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return await self._request("GET", path, params=params)

    async def post(self, path: str, body: dict[str, Any], idempotency_key: str | None = None) -> Any:
        return await self._request("POST", path, body=body, idempotency_key=idempotency_key)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> Any:
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        try:
            async with httpx.AsyncClient(
                base_url=self.config.base_url, timeout=self.config.timeout_s, transport=self._transport,
            ) as http:
                resp = await http.request(
                    method, path, params=clean, json=body, headers=self._headers(idempotency_key),
                )
        except httpx.TimeoutException as exc:
            raise AidraApiError({
                "code": "api_timeout",
                "message": f"AIDRA API did not answer within {self.config.timeout_s:g} s ({method} {path})",
                "retryable": True,
                "hint": "For writes, retry with the same idempotency_key so a run is never started twice.",
            }) from exc
        except httpx.HTTPError as exc:
            raise AidraApiError({
                "code": "api_unreachable",
                "message": f"Cannot reach the AIDRA API at {self.config.base_url}: {type(exc).__name__}",
                "retryable": True,
                "hint": "Start it with `docker compose up` (or set AIDRA_API_URL); GET /api/health should answer.",
            }) from exc

        if resp.status_code < 400:
            if "json" in resp.headers.get("content-type", ""):
                data = resp.json()
                if isinstance(data, dict) and resp.headers.get("idempotent-replay") == "true":
                    # Same key, same request: the API returned the original result; nothing ran again.
                    data["idempotent_replay"] = True
                return data
            return {"content_type": resp.headers.get("content-type"), "text": resp.text[:2000]}

        try:
            data = resp.json()
        except ValueError:
            data = {"detail": resp.text[:500]}
        error = dict(data.get("error") or {})
        error.setdefault("code", f"http_{resp.status_code}")
        detail = data.get("detail")
        error.setdefault("message", detail if isinstance(detail, str) else json.dumps(detail, default=str)[:500])
        error["http_status"] = resp.status_code
        error.setdefault("request_id", resp.headers.get("x-request-id"))
        raise AidraApiError(error)
