"""
Logger estructurado que envia logs a Loki.

Cada log incluye:
- timestamp
- level (INFO, WARNING, ERROR)
- message
- module (aidra.pipeline, aidra.detection, etc.)
- extra fields (execution_id, profile, model, etc.)

Formato: JSON para Loki, texto para stdout.

Architecture note:
    Promtail (running as a sidecar container) collects Docker stdout logs
    automatically and pushes them to Loki.  Therefore this module does NOT
    open an HTTP connection to Loki directly.  Instead it formats log records
    as JSON so that Loki / Promtail can parse structured fields from the log
    stream.

Usage:
    from src.observability.loki_logger import setup_logging, StructuredLogger

    setup_logging(settings)

    log = StructuredLogger("aidra.pipeline", execution_id="abc-123", profile="ground")
    log.info("Pipeline started", extra={"zone": "gibraltar"})
    log.warning("Retrying download", extra={"attempt": 2})
    log.error("Download failed", extra={"error": str(e)}, exc_info=True)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from src.config import Settings


class _JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for Loki / Promtail ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=UTC
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": self.formatMessage(record),
        }

        # Merge any extra fields attached by StructuredLogger or direct calls.
        # We skip internal LogRecord attributes to avoid noise.
        _INTERNAL_ATTRS = logging.LogRecord(
            "", 0, "", 0, "", (), None
        ).__dict__.keys()
        for key, value in record.__dict__.items():
            if key not in _INTERNAL_ATTRS and key not in (
                "message",
                "msg",
                "args",
                "exc_info",
                "exc_text",
                "stack_info",
                "taskName",
            ):
                log_entry[key] = value

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)

        if record.exc_text:
            log_entry["exception"] = record.exc_text

        return json.dumps(log_entry, default=str)


class _TextFormatter(logging.Formatter):
    """Human-readable text formatter for local development / stdout."""

    FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self.FORMAT)


def setup_logging(settings: Settings) -> None:
    """Configure logging for the entire AIDRA application.

    Steps:
        1. Root logger ``aidra``: level set from ``settings.log_level``.
        2. Stream handler: human-readable text to stdout.  Docker captures
           stdout, and Promtail ships it to Loki.
        3. A JSON formatter is attached so that structured fields can be
           parsed by Loki even when consumed via Promtail.

    Note:
        Promtail collects Docker container logs automatically, so there is
        no need for an HTTP handler pushing directly to Loki.  The JSON
        format in the stream handler allows Loki to parse structured fields.

    Args:
        settings: Application settings (used for ``log_level``).
    """
    root = logging.getLogger("aidra")
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root.setLevel(level)

    # Avoid adding duplicate handlers when called multiple times (e.g. tests).
    if root.handlers:
        return

    # Stream handler (stdout -> Docker logs -> Promtail -> Loki)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(_TextFormatter())
    root.addHandler(stream_handler)

    # JSON handler on stderr for Promtail structured ingestion
    json_handler = logging.StreamHandler(stream=sys.stderr)
    json_handler.setFormatter(_JSONFormatter())
    root.addHandler(json_handler)


class StructuredLogger:
    """Logger with persistent context for pipeline operations.

    The *context* kwargs provided at construction time are merged into every
    log call, so callers don't need to repeat ``execution_id``, ``profile``,
    etc. on each message.

    Example::

        log = StructuredLogger("aidra.pipeline", execution_id=uuid, profile="ground")
        log.info("Pipeline started", extra={"zone": "gibraltar"})
        log.error("Download failed", extra={"error": str(e)})
    """

    def __init__(self, name: str, **context: Any) -> None:
        self.logger: logging.Logger = logging.getLogger(name)
        self.context: dict[str, Any] = context

    def _merged_extra(self, extra: dict[str, Any] | None) -> dict[str, Any]:
        """Return context merged with call-site extra fields."""
        merged = {**self.context}
        if extra:
            merged.update(extra)
        return merged

    def info(self, msg: str, extra: dict[str, Any] | None = None) -> None:
        """Log an INFO-level message with merged context."""
        self.logger.info(msg, extra=self._merged_extra(extra))

    def warning(self, msg: str, extra: dict[str, Any] | None = None) -> None:
        """Log a WARNING-level message with merged context."""
        self.logger.warning(msg, extra=self._merged_extra(extra))

    def error(
        self,
        msg: str,
        extra: dict[str, Any] | None = None,
        exc_info: bool = False,
    ) -> None:
        """Log an ERROR-level message with merged context."""
        self.logger.error(
            msg, extra=self._merged_extra(extra), exc_info=exc_info
        )

    def debug(self, msg: str, extra: dict[str, Any] | None = None) -> None:
        """Log a DEBUG-level message with merged context."""
        self.logger.debug(msg, extra=self._merged_extra(extra))
