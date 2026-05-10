"""
Gestion centralizada de variantes de modelo.

Responsabilidades:
1. Descubrir modelos disponibles en el directorio models/
2. Registrar modelos en la tabla models_registry
3. Cargar/descargar modelos bajo demanda
4. Calcular y verificar hashes SHA256
5. Proporcionar interfaz unica para obtener cualquier variante

Dependencias:
- src.models.yolo.YOLODetector (wrapper de modelo)
- src.db.connection.Database (acceso a PostgreSQL)
- src.db.queries (SQL parametrizado)
- src.db.models.ModelInfo (schema Pydantic)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from src.db.connection import Database
from src.db.models import ModelInfo
from src.db.queries import SELECT_ALL_MODELS, UPSERT_MODEL
from src.models.base import BaseDetector
from src.models.yolo import YOLODetector

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# SQL queries not defined in src/db/queries.py
# ------------------------------------------------------------------

SELECT_MODEL_BY_HASH = """
    SELECT * FROM models_registry WHERE file_hash = $1
"""

SELECT_MODEL_BY_NAME_VERSION = """
    SELECT * FROM models_registry
    WHERE name = $1 AND version = $2
    LIMIT 1
"""

SELECT_MODELS_BY_NAME = """
    SELECT * FROM models_registry
    WHERE name = $1
    ORDER BY registered_at DESC
"""

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

# Version tags that identify compression variants in the filename.
# Order matters: longer suffixes must come first so they are matched
# before shorter prefixes.
_COMPRESSION_SUFFIXES: list[tuple[str, str, str]] = [
    # (suffix_pattern, version_tag, compression_technique)
    ("-int8-dynamic", "int8-dynamic", "dynamic_int8"),
    ("-int8-static", "int8-static", "static_int8"),
    ("-int8", "int8", "int8"),
    ("-fp16", "fp16", "fp16"),
    ("-pruned30-int8", "pruned30-int8", "pruned30_int8"),
    ("-pruned50-int8", "pruned50-int8", "pruned50_int8"),
    ("-pruned30-ft", "pruned30-ft", "l1_unstructured_finetune"),
    ("-pruned50-ft", "pruned50-ft", "l1_unstructured_finetune"),
    ("-pruned30", "pruned30", "l1_unstructured"),
    ("-pruned50", "pruned50", "l1_unstructured"),
    ("-structured20", "structured20", "l2_structured_channel"),
    ("-structured30", "structured30", "l2_structured_channel"),
    ("-distilled", "distilled", "knowledge_distillation"),
]

# Regex that captures a generic pruned/structured tag with percentage
_PRUNED_RE = re.compile(r"-pruned(\d+)(-ft)?$")
_STRUCTURED_RE = re.compile(r"-structured(\d+)$")


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 hex digest of a file.

    Parameters
    ----------
    path:
        Path to the file.
    chunk_size:
        Read buffer size in bytes.

    Returns
    -------
    Hexadecimal SHA-256 digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _file_size_mb(path: Path) -> float:
    """Return file size in megabytes."""
    return path.stat().st_size / (1024 * 1024)


def _row_to_model_info(row: Any) -> ModelInfo:
    """Convert an asyncpg Record to a ModelInfo Pydantic model.

    Parameters
    ----------
    row:
        asyncpg.Record from the ``models_registry`` table.

    Returns
    -------
    ModelInfo instance.
    """
    return ModelInfo(
        id=row["id"],
        name=row["name"],
        version=row["version"],
        format=row["format"],
        file_hash=row["file_hash"],
        size_mb=row["size_mb"],
        base_model=row.get("base_model"),
        compression_technique=row.get("compression_technique", "none"),
        num_params=row.get("num_params"),
        input_size=list(row["input_size"]) if row.get("input_size") else [640, 640],
        classes=list(row["classes"]) if row.get("classes") else ["vessel"],
    )


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------


class ModelManager:
    """Centralised manager for discovering, registering, and loading
    YOLO model variants.

    Scans the models directory for ``.pt`` and ``.onnx`` files,
    registers them in the ``models_registry`` database table, and
    provides a cached loader that returns ready-to-use
    ``YOLODetector`` instances.

    Parameters
    ----------
    models_dir:
        Directory containing model weight files.
    db:
        Database singleton for registry operations.
    """

    def __init__(self, models_dir: Path | str, db: Database, max_cached_models: int = 2) -> None:
        self.models_dir = Path(models_dir)
        self.db = db
        self.max_cached_models = max_cached_models
        self._cache: dict[str, BaseDetector] = {}
        self._load_order: list[str] = []  # To track LRU
        logger.info(
            "ModelManager initialised: models_dir=%s, max_cache=%d",
            self.models_dir,
            self.max_cached_models,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scan_and_register(self) -> list[ModelInfo]:
        """Scan the models directory and register every model found.

        Computes SHA-256 hashes, extracts metadata (parameter count,
        input size, classes), and upserts each entry into the
        ``models_registry`` table.

        Should be called once during application startup.

        Returns
        -------
        List of ``ModelInfo`` for all registered models.
        """
        if not self.models_dir.exists():
            logger.warning("Models directory does not exist: %s", self.models_dir)
            return []

        model_files = sorted(
            p
            for p in self.models_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".pt", ".onnx"}
        )

        if not model_files:
            logger.warning("No model files found in %s", self.models_dir)
            return []

        logger.info("Scanning %d model files in %s", len(model_files), self.models_dir)

        registered: list[ModelInfo] = []

        for model_file in model_files:
            try:
                info = await self._register_model(model_file)
                registered.append(info)
                logger.info(
                    "Registered model: %s v%s (%s, %.2f MB, hash=%s...)",
                    info.name,
                    info.version,
                    info.format,
                    info.size_mb,
                    info.file_hash[:12],
                )
            except Exception as exc:
                logger.error(
                    "Failed to register model %s: %s",
                    model_file.name,
                    exc,
                    exc_info=True,
                )

        logger.info(
            "Model scan complete: %d/%d models registered",
            len(registered),
            len(model_files),
        )
        return registered

    async def get_model(
        self,
        name: str,
        version: str | None = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ) -> BaseDetector:
        """Return a detector for the requested model variant.

        Uses an LRU cache. If the cache is full, the least recently used
        model is evicted to free memory.

        Parameters
        ----------
        name:
            Model name (e.g. "yolov8n-sar", "cfar-default").
        version:
            Model version/variant (e.g. "int8").
        """
        cache_key = f"{name}:{version or 'latest'}"

        if cache_key in self._cache:
            # Move to end of load order (most recently used)
            if cache_key in self._load_order:
                self._load_order.remove(cache_key)
            self._load_order.append(cache_key)
            logger.debug("Returning cached model: %s", cache_key)
            return self._cache[cache_key]

        # 1. Enforce cache limit (LRU Eviction)
        while len(self._cache) >= self.max_cached_models:
            if not self._load_order:
                break
            oldest_key = self._load_order.pop(0)
            logger.info("Evicting model from cache (LRU): %s", oldest_key)
            del self._cache[oldest_key]
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 2. Resolve model path.  If several variants share the same base
        # name, callers must request the exact version instead of relying on
        # "latest registered" order.
        row = None
        if version is None and not name.startswith("cfar"):
            rows = await self.db.fetch(SELECT_MODELS_BY_NAME, name)
            if len(rows) > 1:
                variants = ", ".join(sorted(str(r["version"]) for r in rows))
                raise ValueError(
                    f"Ambiguous model '{name}': registered versions are "
                    f"{variants}. Pass model_version explicitly."
                )
            if len(rows) == 1:
                row = rows[0]
        elif version is not None:
            row = await self.db.fetchrow(SELECT_MODEL_BY_NAME_VERSION, name, version)
        if row is None:
            model_path = self._find_model_file(name, version)
            if model_path is None:
                # If it's a built-in detector like CFAR, we might not need a file
                if name.startswith("cfar"):
                    return await self._load_cfar_detector(name, version)
                # I-AIA-1 / I-MOD-4: ningun modelo puede entrar al
                # pipeline sin haber sido registrado explicitamente. El
                # antiguo fallback silencioso fue retirado porque rompia
                # la trazabilidad (otro modelo se ejecutaba sin que el
                # operador lo supiera).
                raise FileNotFoundError(
                    f"Model not found: {name}:{version}. "
                    "Asegurate de que existe el binario en models/ "
                    "con su MODEL_CARD.md (gate ai-act-card)."
                )
        else:
            model_path = Path(row["file_path"])
            if not model_path.exists():
                model_path = self.models_dir / model_path.name

        # 3. Load appropriate detector type
        detector: BaseDetector
        if name.startswith("cfar"):
            detector = await self._load_cfar_detector(name, version)
        else:
            logger.info("Loading YOLO model: %s (path=%s)", cache_key, model_path)
            detector = YOLODetector(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                device=device,
            )
            if row is not None:
                detector.model_name = row["name"]
                detector.model_version = row["version"]
            elif version is not None:
                detector.model_version = version

        # Propagate compression_technique to the detector so the engine
        # can persist it on every execution_log row (Q3 compresion
        # benchmarks need the technique label, not just the model name).
        # Source priority: models_registry row > filename suffix.
        technique = "none"
        if row is not None:
            technique = (row.get("compression_technique") or "none").strip()
        elif version:
            for _sfx, vtag, t in _COMPRESSION_SUFFIXES:
                if version == vtag:
                    technique = t
                    break
        if hasattr(detector, "compression_technique"):
            detector.compression_technique = technique

        self._cache[cache_key] = detector
        self._load_order.append(cache_key)
        return detector

    async def _load_cfar_detector(self, name: str, version: str | None) -> BaseDetector:
        """Specialized loader for CFAR algorithm."""
        from src.models.cfar import CFARDetector

        logger.info("Loading dynamic CFAR detector: %s", name)
        # Note: CFARDetector should implement BaseDetector
        return CFARDetector()  # type: ignore

    def unload_model(self, cache_key: str) -> None:
        """Manually remove a model from memory."""
        if cache_key in self._cache:
            logger.info("Manually unloading model: %s", cache_key)
            del self._cache[cache_key]
            if cache_key in self._load_order:
                self._load_order.remove(cache_key)

    async def list_models(self) -> list[ModelInfo]:
        """List all models registered in the database.

        Returns
        -------
        List of ``ModelInfo`` ordered by name and version.
        """
        rows = await self.db.fetch(SELECT_ALL_MODELS)
        return [_row_to_model_info(r) for r in rows]

    async def get_model_by_hash(self, file_hash: str) -> ModelInfo | None:
        """Find a model by its SHA-256 file hash.

        Parameters
        ----------
        file_hash:
            Full hexadecimal SHA-256 hash of the model file.

        Returns
        -------
        ``ModelInfo`` if found, otherwise ``None``.
        """
        row = await self.db.fetchrow(SELECT_MODEL_BY_HASH, file_hash)
        if row is None:
            return None
        return _row_to_model_info(row)

    # ------------------------------------------------------------------
    # Filename parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_model_name(filename: str) -> tuple[str, str, str]:
        """Extract ``(name, version, format)`` from a model filename.

        Naming convention examples::

            yolov8n-sar.pt           -> ("yolov8n-sar",  "v1.0",        "pytorch")
            yolov8n-sar.onnx         -> ("yolov8n-sar",  "v1.0",        "onnx")
            yolov8n-sar-int8.onnx    -> ("yolov8n-sar",  "int8",        "onnx")
            yolov8n-sar-fp16.onnx    -> ("yolov8n-sar",  "fp16",        "onnx")
            yolov8n-sar-pruned30.pt  -> ("yolov8n-sar",  "pruned30",    "pytorch")
            yolov8n-sar-distilled.pt -> ("yolov8n-sar",  "distilled",   "pytorch")

        Parameters
        ----------
        filename:
            Model filename (with or without directory components).

        Returns
        -------
        Tuple of ``(base_name, version_tag, format_name)``.
        """
        path = Path(filename)
        stem = path.stem  # e.g. "yolov8n-sar-int8"
        suffix = path.suffix.lower()  # e.g. ".onnx"

        # Determine format
        fmt = "onnx" if suffix == ".onnx" else "pytorch"

        # Try known compression suffixes
        for sfx, version_tag, _technique in _COMPRESSION_SUFFIXES:
            if stem.endswith(sfx):
                base_name = stem[: -len(sfx)]
                return base_name, version_tag, fmt

        # Try generic pruned pattern (e.g. -pruned42, -pruned42-ft)
        match = _PRUNED_RE.search(stem)
        if match:
            pct = match.group(1)
            ft = match.group(2) or ""
            base_name = stem[: match.start()]
            version_tag = f"pruned{pct}{ft}"
            return base_name, version_tag, fmt

        # Try generic structured pattern (e.g. -structured25)
        match = _STRUCTURED_RE.search(stem)
        if match:
            pct = match.group(1)
            base_name = stem[: match.start()]
            version_tag = f"structured{pct}"
            return base_name, version_tag, fmt

        # No compression suffix found -> baseline version
        return stem, "v1.0", fmt

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _register_model(self, model_path: Path) -> ModelInfo:
        """Register a single model file in the database.

        Computes hash and metadata, then upserts into ``models_registry``.

        Parameters
        ----------
        model_path:
            Path to the model weight file.

        Returns
        -------
        ModelInfo for the registered model.
        """
        name, version, fmt = self._parse_model_name(model_path.name)
        # I-AIA-1: gate ai-act-card. Sin MODEL_CARD.md no se registra.
        self._require_model_card(name, model_path)
        file_hash = _sha256_file(model_path)
        size_mb = round(_file_size_mb(model_path), 2)

        # Determine compression technique from version tag
        compression_technique = "none"
        for _sfx, vtag, technique in _COMPRESSION_SUFFIXES:
            if version == vtag:
                compression_technique = technique
                break
        if compression_technique == "none" and version.startswith("pruned"):
            compression_technique = "l1_unstructured"
        if compression_technique == "none" and version.startswith("structured"):
            compression_technique = "l2_structured_channel"

        # Determine base model (strip compression suffix)
        base_model = name if version != "v1.0" else None

        # Extract model metadata (params, layers, classes)
        num_params: int | None = None
        num_layers: int | None = None
        input_size: list[int] = [640, 640]
        classes: list[str] = ["vessel"]

        try:
            from ultralytics import YOLO

            yolo = YOLO(str(model_path))
            model_obj = yolo.model
            if hasattr(model_obj, "parameters"):
                num_params = sum(p.numel() for p in model_obj.parameters())
            if hasattr(model_obj, "model"):
                num_layers = len(list(model_obj.model.modules()))

            # Extract class names
            names = yolo.names
            if isinstance(names, dict):
                classes = list(names.values())
            elif isinstance(names, (list, tuple)):
                classes = list(names)
        except Exception:
            logger.debug("Could not extract metadata from %s", model_path.name, exc_info=True)

        # Build compression params JSON
        compression_params: dict[str, Any] | None = None
        if compression_technique != "none":
            compression_params = {"technique": compression_technique, "version": version}

        # Upsert into database
        # UPSERT_MODEL expects: $1...$14
        await self.db.execute(
            UPSERT_MODEL,
            name,  # $1: name
            version,  # $2: version
            fmt,  # $3: format
            str(model_path.resolve()),  # $4: file_path
            file_hash,  # $5: file_hash
            size_mb,  # $6: size_mb
            base_model,  # $7: base_model
            compression_technique,  # $8: compression_technique
            json.dumps(compression_params)
            if compression_params
            else None,  # $9: compression_params (JSONB)
            num_params,  # $10: num_params
            num_layers,  # $11: num_layers
            input_size,  # $12: input_size
            classes,  # $13: classes
            json.dumps({"source": "scan_and_register"}),  # $14: metadata (JSONB)
        )

        # Fetch the inserted/updated row to get the generated UUID
        row = await self.db.fetchrow(SELECT_MODEL_BY_HASH, file_hash)
        if row is None:
            raise RuntimeError(f"Failed to retrieve model after upsert: {name} v{version}")
        return _row_to_model_info(row)

    # I-AIA-1: ai-act-card gate. Sin ficha MODEL_CARD.md el modelo no
    # se registra en models_registry y, por tanto, no se ejecuta.
    _MODEL_CARDS_DIR: Path = Path("models/cards")

    def _require_model_card(self, name: str, model_path: Path) -> None:
        """Asegura que existe ``models/cards/<name>.MODEL_CARD.md``.

        Busca en orden: ficha específica del stem del archivo (cubre
        variantes comprimidas como vesseltracker-sar-yolov8-int8-dynamic),
        luego ficha genérica por nombre de modelo. Lanza
        ``FileNotFoundError`` si ninguna existe.
        """
        candidates = [
            # Ficha específica de variante (stem del archivo de pesos)
            self._MODEL_CARDS_DIR / f"{model_path.stem}.MODEL_CARD.md",
            # Ficha genérica por nombre de modelo (FP32 baseline)
            self._MODEL_CARDS_DIR / f"{name}.MODEL_CARD.md",
        ]
        for c in candidates:
            if c.exists():
                return
        raise FileNotFoundError(
            f"AI Act gate (I-AIA-1): no MODEL_CARD.md for '{name}' "
            f"(stem={model_path.stem}). "
            f"Crea uno de: {', '.join(str(c) for c in candidates)} "
            "antes de registrar el modelo."
        )

    def _find_model_file(self, name: str, version: str | None) -> Path | None:
        """Search the models directory for a file matching the given
        name and version.

        Parameters
        ----------
        name:
            Model base name (e.g. ``"yolov8n-sar"``).
        version:
            Version/variant tag, or ``None`` for the base model.

        Returns
        -------
        Path to the model file, or ``None`` if not found.
        """
        if not self.models_dir.exists():
            return None

        candidates: list[Path] = []
        for p in self.models_dir.iterdir():
            if not p.is_file() or p.suffix.lower() not in {".pt", ".onnx"}:
                continue
            parsed_name, parsed_version, _fmt = self._parse_model_name(p.name)
            if parsed_name == name and (version is None or parsed_version == version):
                candidates.append(p)

        if not candidates:
            return None

        # Prefer .pt over .onnx if multiple matches, then pick the most
        # recently modified file.
        candidates.sort(key=lambda p: (p.suffix != ".pt", -p.stat().st_mtime))
        return candidates[0]
