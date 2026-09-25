"""MCP adapter over the AIDRA HTTP API.

A thin client: tools map typed arguments to REST calls, trim payloads for a
model's context window and turn the API's error envelope into tool errors.
It never imports the backend (``tests/test_mcp`` enforces that) and never
touches the database, so agents go through the same auth, audit and
validation as every other client. Needs only ``mcp`` and ``httpx``.
"""

__version__ = "1.0.0"
