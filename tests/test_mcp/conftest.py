"""Reuse the API test fixtures: the MCP tools are exercised against the real app with a mocked DB."""

from tests.test_api.conftest import mock_db, test_app  # noqa: F401
