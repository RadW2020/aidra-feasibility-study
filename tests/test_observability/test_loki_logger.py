"""Tests del transporte directo a Loki (invariante I-TRACE-3).

Se levanta un servidor HTTP real que imita el endpoint ``/loki/api/v1/push``
en lugar de mockear ``requests``: el contrato que importa es el payload que
sale por el socket, no la llamada Python.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.observability.loki_logger import LokiHandler, StructuredLogger, _JSONFormatter


class _FakeLoki(BaseHTTPRequestHandler):
    """Recoge los pushes recibidos en ``server.received``."""

    def do_POST(self) -> None:  # noqa: N802 - firma impuesta por BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        self.server.received.append(body)  # type: ignore[attr-defined]
        self.send_response(self.server.status_code)  # type: ignore[attr-defined]
        self.end_headers()

    def log_message(self, *args: object) -> None:
        """Silencia el logging por defecto de http.server."""


@pytest.fixture
def loki_server():
    """Servidor Loki de mentira. ``status_code`` es ajustable por el test."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeLoki)
    server.received = []  # type: ignore[attr-defined]
    server.status_code = 204  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _make_handler(server, **kwargs) -> LokiHandler:
    host, port = server.server_address
    handler = LokiHandler(f"http://{host}:{port}", flush_interval=0.1, **kwargs)
    handler.setFormatter(_JSONFormatter())
    return handler


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _all_entries(server) -> list[tuple[dict, dict]]:
    """Aplana los pushes recibidos a pares (labels, línea JSON parseada)."""
    out = []
    for body in server.received:
        for stream in body["streams"]:
            for _ts, line in stream["values"]:
                out.append((stream["stream"], json.loads(line)))
    return out


def test_records_reach_loki_with_push_api_shape(loki_server):
    """Un log emitido llega a Loki con la forma que espera su push API."""
    handler = _make_handler(loki_server)
    logger = logging.getLogger("aidra.test.shape")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        logger.info("pipeline started")
        assert _wait_for(lambda: _all_entries(loki_server))
    finally:
        logger.removeHandler(handler)
        handler.close()

    body = loki_server.received[0]
    assert "streams" in body
    stream = body["streams"][0]
    assert stream["stream"]["service"] == "aidra"
    ts, line = stream["values"][0]
    # Loki exige timestamp en nanosegundos, como string.
    assert ts.isdigit() and len(ts) == 19
    assert json.loads(line)["message"] == "pipeline started"


def test_execution_id_travels_in_body_not_in_labels(loki_server):
    """I-TRACE-3: el run_id llega a Loki, pero nunca como label.

    Un label por ejecución crearía un stream de Loki por run y degradaría
    el índice. Debe ser consultable con ``| json | execution_id="..."``.
    """
    handler = _make_handler(loki_server)
    logger = logging.getLogger("aidra.test.trace")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    run_id = "3f2b91c4-0000-4a1e-9b77-000000000001"
    try:
        StructuredLogger(
            "aidra.test.trace", execution_id=run_id, profile="ground"
        ).info("detection finished")
        assert _wait_for(lambda: _all_entries(loki_server))
    finally:
        logger.removeHandler(handler)
        handler.close()

    labels, line = _all_entries(loki_server)[0]
    assert line["execution_id"] == run_id, "el run_id debe viajar en el cuerpo"
    assert line["profile"] == "ground"
    assert "execution_id" not in labels, "el run_id NO puede ser label de Loki"
    assert set(labels) == {"service", "level", "logger"}


def test_loki_unreachable_does_not_raise(loki_server):
    """Loki caído degrada la observabilidad, nunca el pipeline."""
    handler = _make_handler(loki_server)
    # Apunta a un puerto cerrado: el push fallará en el hilo worker.
    handler.url = "http://127.0.0.1:1/loki/api/v1/push"
    logger = logging.getLogger("aidra.test.down")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        logger.info("esto no debe explotar")
        assert _wait_for(lambda: handler.dropped > 0)
    finally:
        logger.removeHandler(handler)
        handler.close()

    assert handler.dropped >= 1
    assert not loki_server.received


def test_error_status_counts_as_dropped(loki_server):
    """Un 4xx/5xx de Loki se contabiliza, no se reintenta en bucle."""
    loki_server.status_code = 500
    handler = _make_handler(loki_server)
    logger = logging.getLogger("aidra.test.status")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        logger.info("rechazado por loki")
        assert _wait_for(lambda: handler.dropped > 0)
    finally:
        logger.removeHandler(handler)
        handler.close()

    assert handler.dropped >= 1


def test_saturated_queue_drops_instead_of_growing(loki_server):
    """La cola es acotada: con Loki lento se tiran logs, no se agota la RAM."""
    handler = _make_handler(loki_server, queue_size=1)
    handler.url = "http://127.0.0.1:1/loki/api/v1/push"
    logger = logging.getLogger("aidra.test.full")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        for i in range(500):
            logger.info("flood %d", i)
        assert handler.dropped > 0
    finally:
        logger.removeHandler(handler)
        handler.close()


def test_json_formatter_is_order_independent():
    """Regresión: el formatter no puede depender de que otro corriera antes.

    ``formatMessage`` lee ``record.message``, que la stdlib fija dentro de
    ``Formatter.format``. Al sobrescribir ``format`` había que fijarlo a mano;
    sin ello el formatter sólo funcionaba si el handler de texto había
    formateado el mismo record primero, y fallaba en cualquier orden distinto.
    """
    record = logging.LogRecord(
        "aidra.x", logging.INFO, __file__, 1, "hola %s", ("mundo",), None
    )
    entry = json.loads(_JSONFormatter().format(record))
    assert entry["message"] == "hola mundo"


def test_emit_never_propagates_formatter_errors(loki_server):
    """Un formatter roto incrementa `dropped`, no rompe al llamante."""
    handler = _make_handler(loki_server)

    class _Boom(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            raise ValueError("formatter roto")

    handler.setFormatter(_Boom())
    record = logging.LogRecord("aidra.test", logging.INFO, __file__, 1, "x", (), None)
    try:
        handler.emit(record)  # no debe lanzar
        assert handler.dropped == 1
    finally:
        handler.close()
