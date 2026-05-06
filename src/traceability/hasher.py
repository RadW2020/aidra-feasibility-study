"""
Calculo de hashes SHA256 para garantizar integridad.

Se hashean:
- Imagenes satelitales descargadas (input)
- Archivos de pesos del modelo (.pt, .onnx)
- Resultados de detecciones (GeoJSON)

El hash se calcula en streaming (chunks de 64 KB) para no cargar
archivos grandes en memoria.  Para resultados, se serializa a JSON
ordenado antes de hashear para garantizar determinismo.

Usage:
    from src.traceability.hasher import (
        compute_sha256,
        compute_array_hash,
        compute_result_hash,
    )

    file_hash = compute_sha256(Path("/data/images/scene.tif"))
    array_hash = compute_array_hash(np_array)
    result_hash = compute_result_hash(detections_list)
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Calcula el SHA256 de un archivo en modo streaming.

    Lee el archivo en chunks de ``chunk_size`` bytes para evitar
    cargar archivos grandes (multi-GB) completamente en memoria.

    Parameters
    ----------
    file_path:
        Ruta al archivo a hashear.
    chunk_size:
        Tamano del chunk de lectura en bytes (default: 64 KB).

    Returns
    -------
    str
        Hex digest SHA256 del archivo (64 caracteres hexadecimales).

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    PermissionError
        Si no hay permisos de lectura.
    """
    sha256 = hashlib.sha256()
    path = Path(file_path)

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)

    return sha256.hexdigest()


def compute_array_hash(array: np.ndarray) -> str:
    """Calcula el SHA256 de un numpy array.

    Usa la representacion en bytes del array (``tobytes()``) para
    producir un hash determinista.  El array se hace contiguo en
    memoria antes de hashear para garantizar consistencia
    independientemente del layout.

    Parameters
    ----------
    array:
        Numpy array a hashear (cualquier dtype y shape).

    Returns
    -------
    str
        Hex digest SHA256 del contenido del array.
    """
    sha256 = hashlib.sha256()

    # Ensure contiguous memory layout for deterministic byte repr
    contiguous = np.ascontiguousarray(array)
    sha256.update(contiguous.tobytes())

    # Include shape and dtype in the hash for collision safety
    metadata = f"{contiguous.shape}:{contiguous.dtype}".encode()
    sha256.update(metadata)

    return sha256.hexdigest()


def compute_result_hash(detections: list[dict[str, Any]]) -> str:
    """Calcula un hash determinista de una lista de detecciones.

    Serializa las detecciones a JSON con claves ordenadas y
    separadores compactos, luego calcula el SHA256.  Esto garantiza
    que el mismo conjunto de detecciones — independientemente del
    orden de insercion de las claves en cada dict — produce
    siempre el mismo hash.

    Las detecciones se ordenan por una clave compuesta de
    ``(longitude, latitude, confidence)`` antes de serializar,
    de modo que el orden de la lista tambien es determinista.

    Parameters
    ----------
    detections:
        Lista de diccionarios de detecciones.  Cada deteccion debe
        tener al menos ``longitude``, ``latitude`` y ``confidence``
        para un ordenamiento determinista, pero se acepta cualquier
        estructura.

    Returns
    -------
    str
        Hex digest SHA256 de las detecciones serializadas.
    """
    # Sort detections deterministically
    sorted_detections = sorted(
        detections,
        key=_detection_sort_key,
    )

    # Serialize to JSON with sorted keys and compact separators
    json_bytes = json.dumps(
        sorted_detections,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")

    return hashlib.sha256(json_bytes).hexdigest()


def _detection_sort_key(detection: dict[str, Any]) -> tuple[float, float, float]:
    """Genera una clave de ordenamiento para una deteccion.

    Parameters
    ----------
    detection:
        Diccionario de deteccion.

    Returns
    -------
    tuple[float, float, float]
        Tupla ``(longitude, latitude, confidence)`` para ordenamiento.
    """
    return (
        float(detection.get("longitude", 0.0)),
        float(detection.get("latitude", 0.0)),
        float(detection.get("confidence", 0.0)),
    )


def _json_default(obj: Any) -> Any:
    """Serializador JSON personalizado para tipos no nativos.

    Soporta UUID, datetime, numpy types y Path.

    Parameters
    ----------
    obj:
        Objeto a serializar.

    Returns
    -------
    Any
        Representacion serializable del objeto.

    Raises
    ------
    TypeError
        Si el tipo no es soportado.
    """
    import uuid
    from datetime import datetime

    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def compute_input_params_hash(params: dict[str, Any]) -> str:
    """Hash determinista de los parametros de entrada de un run.

    Serializa ``params`` con ``sort_keys=True`` y separadores compactos
    para que el mismo conjunto de parametros (Settings + request)
    produzca siempre el mismo hash, independientemente del orden.

    Parameters
    ----------
    params:
        Diccionario con todos los parametros que afectan al resultado
        (thresholds, perfil, tile size, modelo, zona, fechas, etc.).
        Cierra el invariante I-TRACE-4.

    Returns
    -------
    str
        Hex digest SHA256 (64 chars).
    """
    json_bytes = json.dumps(
        params,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(json_bytes).hexdigest()


@functools.lru_cache(maxsize=1)
def get_commit_sha() -> str:
    """Devuelve el commit SHA del codigo en ejecucion (cacheado por proceso).

    Orden de resolucion (de mas a menos fiable):

    1. ``SOURCE_COMMIT`` — env var auto-establecida por Coolify (y otros
       PaaS tipo Heroku/Dokku) en cada deploy. Es la fuente de verdad
       cuando hay CI/CD: refleja el SHA del build actual.
    2. ``AIDRA_COMMIT_SHA`` — env var de override manual. Solo deberia
       usarse si el operador necesita anclar el SHA a algo distinto del
       build (raro). Si esta hardcoded en la config del PaaS, queda
       desactualizada — por eso ``SOURCE_COMMIT`` tiene precedencia.
    3. ``git rev-parse HEAD`` — fallback para entornos de desarrollo
       local con .git accesible.
    4. ``"unknown"`` — ultimo recurso (contenedor minimal sin .git, sin
       envs, sin git binary).

    Se cachea porque el SHA no cambia durante la vida del proceso.
    Llamadas a ``get_commit_sha.cache_clear()`` son utiles en tests.

    Returns
    -------
    str
        SHA hex (al menos 7 chars; tipicamente 40) o ``"unknown"``.
    """
    import os

    for env_name in ("SOURCE_COMMIT", "AIDRA_COMMIT_SHA"):
        value = os.getenv(env_name)
        if value:
            stripped = value.strip()
            if stripped:
                return stripped

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Could not resolve git commit SHA: %s", exc)
        return "unknown"
    if result.returncode != 0:
        logger.warning("git rev-parse HEAD failed: %s", result.stderr.strip())
        return "unknown"
    return result.stdout.strip() or "unknown"
