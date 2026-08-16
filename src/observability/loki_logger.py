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
    Production (``docker-compose.coolify.yml``) deliberately omits Promtail:
    shipping Docker logs would require mounting ``docker.sock`` on a shared
    host, exposing sibling projects' containers.  Without Promtail the
    stdout -> Loki chain has no transport, so this module pushes to Loki
    itself over the internal network (``LOKI_URL``), closing **I-TRACE-3**.

    Label cardinality: only ``service``, ``level`` and ``logger`` become Loki
    stream labels.  ``execution_id`` / ``run_id`` stay inside the JSON line
    body — a unique label per run would create one Loki stream per pipeline
    execution and degrade the index.  The run is still reachable with::

        {service="aidra"} | json | execution_id="<uuid>"

Usage:
    from src.observability.loki_logger import setup_logging, StructuredLogger

    setup_logging(settings)

    log = StructuredLogger("aidra.pipeline", execution_id="abc-123", profile="ground")
    log.info("Pipeline started", extra={"zone": "gibraltar"})
    log.warning("Retrying download", extra={"attempt": 2})
    log.error("Download failed", extra={"error": str(e)}, exc_info=True)
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import queue
import sys
import threading
from datetime import UTC, datetime
from typing import Any

import requests

from src.config import Settings


class _JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for Loki / Promtail ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        # ``formatMessage`` reads ``record.message``, which the stdlib sets in
        # ``Formatter.format``. Since we override ``format`` we must set it
        # ourselves: relying on another handler having formatted the record
        # first makes this formatter silently order-dependent.
        record.message = record.getMessage()
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


class LokiHandler(logging.Handler):
    """Ships log records to Loki's push API from a background thread.

    Design constraints, in priority order:

    1. **Never block the caller.**  ``emit`` only enqueues; all I/O happens on
       the worker thread.  The pipeline's latency measurements must not
       include log shipping.
    2. **Never raise into the caller.**  A dead or slow Loki degrades
       observability, never the detection run itself.  Failures increment
       :attr:`dropped` and are otherwise silent.
    3. **Bounded memory.**  The queue is capped; once full, new records are
       dropped rather than growing without limit while Loki is unreachable.

    Labels are intentionally low-cardinality (see module docstring).
    """

    #: Loki stream labels; everything else travels in the JSON line body.
    _LABEL_FIELDS = ("level", "logger")

    def __init__(
        self,
        url: str,
        service: str = "aidra",
        batch_size: int = 100,
        flush_interval: float = 2.0,
        queue_size: int = 10_000,
        timeout: float = 5.0,
    ) -> None:
        super().__init__()
        self.url = url.rstrip("/") + "/loki/api/v1/push"
        self.service = service
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.timeout = timeout
        self.dropped = 0
        self._queue: queue.Queue[tuple[str, dict[str, str], str]] = queue.Queue(
            maxsize=queue_size
        )
        self._stopping = threading.Event()
        self._session = requests.Session()
        self._thread = threading.Thread(
            target=self._run, name="loki-shipper", daemon=True
        )
        self._thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        """Enqueue a record. Drops it if the queue is saturated."""
        try:
            line = self.format(record)
            labels = {
                "service": self.service,
                "level": record.levelname,
                "logger": record.name,
            }
            # Loki timestamps are nanoseconds since epoch, as a string.
            ts = str(int(record.created * 1_000_000_000))
            self._queue.put_nowait((ts, labels, line))
        except queue.Full:
            self.dropped += 1
        except Exception:  # noqa: BLE001 - logging must never break callers
            self.dropped += 1

    def _run(self) -> None:
        """Worker loop: drain the queue and push batches to Loki."""
        batch: list[tuple[str, dict[str, str], str]] = []
        while not self._stopping.is_set() or not self._queue.empty():
            with contextlib.suppress(queue.Empty):
                batch.append(self._queue.get(timeout=self.flush_interval))
            if batch and (
                len(batch) >= self.batch_size or self._queue.empty()
            ):
                self._push(batch)
                batch = []
        if batch:
            self._push(batch)

    def _push(self, batch: list[tuple[str, dict[str, str], str]]) -> None:
        """POST one batch, grouping entries into streams by label set."""
        streams: dict[tuple[tuple[str, str], ...], list[list[str]]] = {}
        for ts, labels, line in batch:
            key = tuple(sorted(labels.items()))
            streams.setdefault(key, []).append([ts, line])

        payload = {
            "streams": [
                {"stream": dict(key), "values": values}
                for key, values in streams.items()
            ]
        }
        try:
            resp = self._session.post(
                self.url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code >= 300:
                self.dropped += len(batch)
        except Exception:  # noqa: BLE001 - Loki being down must not propagate
            self.dropped += len(batch)

    def close(self) -> None:
        """Flush pending records and stop the worker thread."""
        self._stopping.set()
        self._thread.join(timeout=self.timeout + self.flush_interval)
        with contextlib.suppress(Exception):
            self._session.close()
        super().close()


def setup_logging(settings: Settings) -> None:
    """Configure logging for the entire AIDRA application.

    Steps:
        1. Root logger ``aidra``: level set from ``settings.log_level``.
        2. Stream handler: human-readable text to stdout (Docker log inspection).
        3. JSON handler on stderr: structured fields for local consumption.
        4. :class:`LokiHandler`: pushes the same JSON lines to Loki over HTTP
           when ``settings.loki_enabled`` is on and ``loki_url`` is set.

    Step 4 is what satisfies **I-TRACE-3**: without it Loki has no ingestion
    path in production (Promtail is intentionally absent) and ``run_id``
    never reaches the log store.

    Args:
        settings: Application settings (``log_level``, ``loki_url``,
            ``loki_enabled``).
    """
    root = logging.getLogger("aidra")
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root.setLevel(level)

    # Avoid adding duplicate handlers when called multiple times (e.g. tests).
    if root.handlers:
        return

    # Stream handler (stdout -> Docker logs, for `docker logs` inspection)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(_TextFormatter())
    root.addHandler(stream_handler)

    # JSON handler on stderr for structured local consumption
    json_handler = logging.StreamHandler(stream=sys.stderr)
    json_handler.setFormatter(_JSONFormatter())
    root.addHandler(json_handler)

    # Direct push to Loki (no Promtail in production — see module docstring)
    if settings.loki_enabled and settings.loki_url:
        loki_handler = LokiHandler(settings.loki_url)
        loki_handler.setFormatter(_JSONFormatter())
        root.addHandler(loki_handler)
        atexit.register(loki_handler.close)


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
