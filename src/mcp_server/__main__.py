"""Run the AIDRA MCP server.

    python -m src.mcp_server                       # stdio (Claude Code, Cursor, Codex)
    python -m src.mcp_server --transport http      # streamable HTTP on 127.0.0.1:8765/mcp

Configuration comes from the environment:

    AIDRA_API_URL          API base URL (default http://localhost:8000)
    AIDRA_API_TOKEN        bearer token; its scope decides what writes succeed
    AIDRA_MCP_MODE         read-only (default) | operator (registers the write tools)
    AIDRA_MCP_CLIENT_NAME  label sent as X-AIDRA-Client (recorded in the audit log)

Logs go to stderr: on stdio, stdout is the protocol channel.
"""

import argparse
import logging
import sys

from src.mcp_server.client import AidraConfig
from src.mcp_server.server import build_server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="aidra-mcp", description=__doc__.split("\n\n")[0])
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = AidraConfig.from_env()
    logging.getLogger("aidra.mcp").info(
        "AIDRA MCP server: api=%s mode=%s auth=%s transport=%s",
        config.base_url, config.mode, "token" if config.token else "none", args.transport,
    )
    server = build_server(config)
    if args.transport == "stdio":
        server.run("stdio")
    else:
        from mcp.server.transport_security import TransportSecuritySettings

        # Behind docker port mapping the Host header is localhost:<published port>.
        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["localhost:*", "127.0.0.1:*"],
            allowed_origins=["http://localhost:*", "http://127.0.0.1:*"],
        )
        server.run("streamable-http", host=args.host, port=args.port, transport_security=security)


if __name__ == "__main__":
    main()
