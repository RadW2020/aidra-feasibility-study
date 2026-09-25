"""Helpers for the agent-interface API tests: route mocked DB calls by SQL text."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock

from tests.test_api.conftest import FakeRecord


def route_db(
    mock_db: Any,
    *,
    fetch: dict[str, Any] | None = None,
    fetchrow: dict[str, Any] | None = None,
    fetchval: dict[str, Any] | None = None,
) -> Any:
    """Make each DB method answer by the first key found in the SQL.

    Values may be callables taking the query's positional args. Unmatched
    queries return the mock_db defaults ([], None, 0).
    """

    def make(table: dict[str, Any] | None, default: Any) -> Callable[..., Any]:
        async def answer(query: str, *args: Any) -> Any:
            for needle, value in (table or {}).items():
                if needle in query:
                    return value(*args) if callable(value) else value
            return default

        return answer

    mock_db.fetch = AsyncMock(side_effect=make(fetch, []))
    mock_db.fetchrow = AsyncMock(side_effect=make(fetchrow, None))
    mock_db.fetchval = AsyncMock(side_effect=make(fetchval, 0))
    return mock_db


def model_row(name: str, version: str, *, status: str = "active", classes=("ship",), reason=None) -> FakeRecord:
    return FakeRecord(
        name=name, version=version, format="pytorch", file_path=f"/app/models/{name}.pt",
        compression_technique="none" if version == "v1.0" else "static_int8",
        classes=list(classes), status=status, rejection_reason=reason,
    )
