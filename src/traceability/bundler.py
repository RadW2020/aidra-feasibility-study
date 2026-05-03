"""
Bundler de evidencia D3 para AIDRA.

Empaqueta el bundle de entregable D3 (Evidence Package) a partir de la
trazabilidad persistida en ``execution_log`` + ``detections``. El bundle
filtra runs por rango temporal / zona / modelo y produce un directorio
auto-contenido + tar.gz con:

  - ``executions.csv``       — runs filtrados (un row por execution)
  - ``detections.csv``       — detecciones planas con flags I-DET-2/3
  - ``detections.geojson``   — FeatureCollection 4326 (apt para QGIS)
  - ``settings.json``        — Settings vigente del proceso
  - ``models/``              — copia de MODEL_CARD.md por modelo usado
  - ``prometheus_snapshot.txt`` (best-effort, si endpoint accesible)
  - ``MANIFEST.json``        — SHA256 de cada artefacto + metadatos del run

El bundle es lo que se entrega al evaluador: con ``settings.json`` +
``commit_sha`` (presente en ``executions.csv``) + ``MANIFEST.json``,
cualquier auditor puede reproducir y verificar la cadena entera.

Cierra: criterio Q3 — Trazabilidad / paquete D3.

Usage::

    from src.traceability.bundler import EvidenceBundler

    bundler = EvidenceBundler(db=db, settings=settings)
    out_path = await bundler.build(
        out_dir=Path("/tmp/d3"),
        model_name="yolov8n-vessel",
        date_from=datetime(2026, 1, 1, tzinfo=UTC),
        zone="gibraltar",
    )
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import Settings
from src.db.connection import Database
from src.traceability.hasher import (
    compute_sha256,
    get_commit_sha,
)

logger = logging.getLogger(__name__)


# ====================================================================
# Manifest dataclass
# ====================================================================


@dataclass
class BundleManifest:
    """Metadatos del bundle D3 (volcado a MANIFEST.json)."""

    bundle_id: str
    created_at: str
    commit_sha: str
    filters: dict[str, Any]
    counts: dict[str, int]
    files: dict[str, str] = field(default_factory=dict)  # filename -> sha256
    settings_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "commit_sha": self.commit_sha,
            "filters": self.filters,
            "counts": self.counts,
            "settings_hash": self.settings_hash,
            "files": self.files,
        }


# ====================================================================
# SQL
# ====================================================================


_SELECT_EXECUTIONS_BUNDLE = """
    SELECT
        e.id,
        e.created_at,
        e.image_id,
        e.image_title,
        e.image_hash,
        ST_AsGeoJSON(e.image_bbox) AS image_bbox_geojson,
        e.image_sensing_date,
        e.image_size_mb,
        e.search_zone,
        e.model_name,
        e.model_version,
        e.model_hash,
        e.model_size_mb,
        e.model_format,
        e.compression_technique,
        e.confidence_threshold,
        e.iou_threshold,
        e.constraint_profile,
        e.tile_size,
        e.tile_overlap,
        e.num_detections,
        e.avg_confidence,
        e.total_duration_ms,
        e.peak_ram_mb,
        e.output_hash,
        e.input_params_hash,
        e.commit_sha,
        e.status,
        e.trigger_type,
        e.pipeline_version,
        e.hostname
    FROM execution_log e
    WHERE ($1::timestamptz IS NULL OR e.created_at >= $1)
      AND ($2::timestamptz IS NULL OR e.created_at <= $2)
      AND ($3::text IS NULL OR e.search_zone = $3)
      AND ($4::text IS NULL OR e.model_name = $4)
      AND ($5::text IS NULL OR e.constraint_profile = $5)
    ORDER BY e.created_at ASC
"""


_SELECT_DETECTIONS_BUNDLE = """
    SELECT
        d.id AS detection_id,
        d.execution_id,
        d.created_at,
        ST_X(d.center_geo) AS longitude,
        ST_Y(d.center_geo) AS latitude,
        ST_AsGeoJSON(d.center_geo) AS center_geojson,
        ST_AsGeoJSON(d.bbox_geo) AS bbox_geojson,
        d.bbox_pixel,
        d.confidence,
        d.source,
        d.cfar_snr,
        d.yolo_score,
        d.class_name,
        d.tile_index,
        d.on_land,
        d.cluster_anomaly,
        d.thumbnail_path,
        e.image_id,
        e.model_name,
        e.constraint_profile
    FROM detections d
    JOIN execution_log e ON e.id = d.execution_id
    WHERE d.execution_id = ANY($1::uuid[])
    ORDER BY d.created_at ASC, d.confidence DESC
"""


# ====================================================================
# Bundler
# ====================================================================


class EvidenceBundler:
    """Constructor del bundle D3.

    Parameters
    ----------
    db:
        Instancia de ``Database`` (asyncpg pool).
    settings:
        Snapshot de ``Settings`` a serializar dentro del bundle.
    models_cards_dir:
        Ruta a ``models/cards/`` (donde viven los MODEL_CARD.md).
    """

    def __init__(
        self,
        db: Database,
        settings: Settings,
        models_cards_dir: Path | None = None,
    ) -> None:
        self._db = db
        self._settings = settings
        self._models_cards_dir = models_cards_dir or Path("models/cards")

    async def build(
        self,
        out_dir: Path,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        zone: str | None = None,
        model_name: str | None = None,
        constraint_profile: str | None = None,
        archive: bool = True,
    ) -> Path:
        """Construye el bundle D3 y devuelve su ruta.

        Parameters
        ----------
        out_dir:
            Directorio donde crear el bundle (se crea si no existe).
        date_from / date_to:
            Rango de ``execution_log.created_at`` (UTC).
        zone:
            Filtro por ``search_zone``.
        model_name:
            Filtro por ``model_name``.
        constraint_profile:
            Filtro por ``constraint_profile``.
        archive:
            Si ``True`` (default) genera tambien ``<bundle>.tar.gz``.

        Returns
        -------
        Path
            Ruta al directorio del bundle (o al .tar.gz si ``archive``).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        bundle_id = datetime.utcnow().strftime("d3-%Y%m%dT%H%M%SZ")
        bundle_dir = out_dir / bundle_id
        bundle_dir.mkdir(parents=True, exist_ok=False)
        (bundle_dir / "models").mkdir(parents=True, exist_ok=True)
        (bundle_dir / "thumbnails").mkdir(parents=True, exist_ok=True)

        filters = {
            "date_from": date_from.isoformat() if date_from else None,
            "date_to": date_to.isoformat() if date_to else None,
            "zone": zone,
            "model_name": model_name,
            "constraint_profile": constraint_profile,
        }
        manifest = BundleManifest(
            bundle_id=bundle_id,
            created_at=datetime.utcnow().isoformat() + "Z",
            commit_sha=get_commit_sha(),
            filters=filters,
            counts={},
        )

        executions = await self._fetch_executions(
            date_from, date_to, zone, model_name, constraint_profile
        )
        manifest.counts["executions"] = len(executions)
        logger.info(
            "Bundle %s: %d executions match filters", bundle_id, len(executions)
        )

        execution_ids = [row["id"] for row in executions]
        detections = (
            await self._fetch_detections(execution_ids) if execution_ids else []
        )
        manifest.counts["detections"] = len(detections)

        # 1. executions.csv
        self._write_executions_csv(bundle_dir / "executions.csv", executions)

        # 2. detections.csv
        self._write_detections_csv(bundle_dir / "detections.csv", detections)

        # 3. detections.geojson
        self._write_detections_geojson(
            bundle_dir / "detections.geojson", detections
        )

        # 4. settings.json
        self._write_settings(bundle_dir / "settings.json", manifest)

        # 5. model cards used (by name AND by model_hash match)
        used_models = {row["model_name"] for row in executions}
        self._copy_model_cards(
            bundle_dir / "models", used_models, manifest, executions
        )

        # 5b. SAR thumbnails (Wow effect #1)
        self._copy_thumbnails(
            bundle_dir / "thumbnails", detections, manifest
        )

        # 6. prometheus snapshot (best-effort)
        self._maybe_write_prometheus_snapshot(
            bundle_dir / "prometheus_snapshot.txt", manifest
        )

        # 7. MANIFEST.json (file hashes, last)
        self._populate_file_hashes(bundle_dir, manifest)
        manifest_path = bundle_dir / "MANIFEST.json"
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        # 7b. Sign the bundle root: SHA256 of MANIFEST.json itself.
        # Tamper-evidence at the archive level — auditor only needs to
        # cite this single line to vouch for the entire bundle.
        manifest_sha = compute_sha256(manifest_path)
        (bundle_dir / "MANIFEST.sha256").write_text(
            f"{manifest_sha}  MANIFEST.json\n",
            encoding="utf-8",
        )
        logger.info("Bundle root SHA256: %s", manifest_sha)

        if archive:
            archive_path = self._archive(bundle_dir)
            logger.info("Bundle archive ready: %s", archive_path)
            return archive_path

        logger.info("Bundle dir ready: %s", bundle_dir)
        return bundle_dir

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    async def _fetch_executions(
        self,
        date_from: datetime | None,
        date_to: datetime | None,
        zone: str | None,
        model_name: str | None,
        constraint_profile: str | None,
    ) -> list[dict[str, Any]]:
        rows = await self._db.fetch(
            _SELECT_EXECUTIONS_BUNDLE,
            date_from,
            date_to,
            zone,
            model_name,
            constraint_profile,
        )
        return [dict(r) for r in rows]

    async def _fetch_detections(
        self, execution_ids: list[Any]
    ) -> list[dict[str, Any]]:
        rows = await self._db.fetch(_SELECT_DETECTIONS_BUNDLE, execution_ids)
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_executions_csv(
        path: Path, executions: list[dict[str, Any]]
    ) -> None:
        if not executions:
            path.write_text("", encoding="utf-8")
            return
        keys = list(executions[0].keys())
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in executions:
                writer.writerow(
                    {k: _to_csv_value(row.get(k)) for k in keys}
                )

    @staticmethod
    def _write_detections_csv(
        path: Path, detections: list[dict[str, Any]]
    ) -> None:
        if not detections:
            path.write_text("", encoding="utf-8")
            return
        keys = list(detections[0].keys())
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in detections:
                writer.writerow(
                    {k: _to_csv_value(row.get(k)) for k in keys}
                )

    @staticmethod
    def _write_detections_geojson(
        path: Path, detections: list[dict[str, Any]]
    ) -> None:
        features: list[dict[str, Any]] = []
        for d in detections:
            geom = (
                json.loads(d["center_geojson"])
                if d.get("center_geojson")
                else None
            )
            if geom is None:
                continue
            features.append(
                {
                    "type": "Feature",
                    "id": str(d["detection_id"]),
                    "geometry": geom,
                    "properties": {
                        "execution_id": str(d["execution_id"]),
                        "image_id": d.get("image_id"),
                        "model_name": d.get("model_name"),
                        "constraint_profile": d.get("constraint_profile"),
                        "confidence": float(d["confidence"]),
                        "source": d.get("source"),
                        "cfar_snr": d.get("cfar_snr"),
                        "yolo_score": d.get("yolo_score"),
                        "class_name": d.get("class_name", "vessel"),
                        "on_land": bool(d.get("on_land", False)),
                        "cluster_anomaly": bool(
                            d.get("cluster_anomaly", False)
                        ),
                        "detected_at": (
                            d["created_at"].isoformat()
                            if d.get("created_at")
                            else None
                        ),
                    },
                }
            )
        fc = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
            },
            "features": features,
        }
        path.write_text(
            json.dumps(fc, default=str, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_settings(self, path: Path, manifest: BundleManifest) -> None:
        # pydantic v2: model_dump
        try:
            data = self._settings.model_dump()
        except AttributeError:
            data = self._settings.dict()
        # Censor secrets: name-based + URL-embedded credentials.
        for k in list(data.keys()):
            data[k] = _censor_secret(k, data[k])
        payload = json.dumps(data, indent=2, sort_keys=True, default=str)
        path.write_text(payload, encoding="utf-8")
        # settings_hash for traceability
        import hashlib

        manifest.settings_hash = hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()

    def _copy_model_cards(
        self,
        models_dir: Path,
        used_models: set[str],
        manifest: BundleManifest,
        executions: list[dict[str, Any]] | None = None,
    ) -> None:
        """Copy MODEL_CARDs into the bundle keyed by name AND model_hash.

        AI Act requires every model that produced any record in the
        bundle to ship its card. We resolve in two passes:
          1. By exact name match (legacy behaviour).
          2. By scanning all .MODEL_CARD.md files for `weights_sha256`
             matching any `model_hash` from execution_log → catches
             cards renamed/aliased.
        Closes the AI Act half of Q3 demo+trace+IA.
        """
        if not self._models_cards_dir.exists():
            logger.warning(
                "Model cards dir not found: %s — skipping",
                self._models_cards_dir,
            )
            return

        copied: dict[str, Path] = {}  # filename → src path

        # Pass 1: by name.
        for model in sorted(used_models):
            if not model:
                continue
            candidate = self._models_cards_dir / f"{model}.MODEL_CARD.md"
            if candidate.exists():
                copied[candidate.name] = candidate

        # Pass 2: by model_hash inside cards (matches even if renamed).
        if executions:
            wanted_hashes = {
                row.get("model_hash") for row in executions if row.get("model_hash")
            }
            for card in self._models_cards_dir.glob("*.MODEL_CARD.md"):
                try:
                    text = card.read_text(encoding="utf-8")
                except OSError:
                    continue
                for h in wanted_hashes:
                    if h and h in text:
                        copied[card.name] = card
                        break

        for name, src in copied.items():
            try:
                shutil.copy2(src, models_dir / name)
            except OSError as exc:
                logger.warning("Could not copy MODEL_CARD %s: %s", src, exc)

        manifest.counts["model_cards"] = len(copied)
        manifest.counts["model_cards_by_hash_match"] = sum(
            1 for src in copied.values()
            if executions and any(
                src.read_text(encoding="utf-8").find(h) >= 0
                for h in {r.get("model_hash") for r in executions if r.get("model_hash")}
            )
        )

    @staticmethod
    def _copy_thumbnails(
        thumbs_dir: Path,
        detections: list[dict[str, Any]],
        manifest: BundleManifest,
    ) -> None:
        """Copy referenced detection thumbnails into the bundle.

        Thumbnails live outside the project tree (typically
        /data/thumbnails). The bundle copies just those referenced by
        the filtered detections so an auditor can open the .tar.gz and
        see the actual SAR crop next to each row of detections.csv.
        """
        copied = 0
        for d in detections:
            src = d.get("thumbnail_path")
            if not src:
                continue
            src_path = Path(src)
            if not src_path.exists():
                continue
            try:
                shutil.copy2(src_path, thumbs_dir / src_path.name)
                copied += 1
            except OSError as exc:
                logger.warning(
                    "Could not copy thumbnail %s: %s", src_path, exc
                )
        manifest.counts["thumbnails"] = copied

    @staticmethod
    def _maybe_write_prometheus_snapshot(
        path: Path, manifest: BundleManifest
    ) -> None:
        import os
        from urllib.request import urlopen

        url = os.getenv(
            "AIDRA_PROMETHEUS_URL",
            "http://aidra-prom:9090/metrics",
        )
        try:
            with urlopen(url, timeout=3) as resp:  # noqa: S310 (intra-host, env-driven)
                content = resp.read().decode("utf-8", errors="replace")
            path.write_text(content, encoding="utf-8")
            manifest.counts["prometheus_snapshot_bytes"] = len(content)
        except Exception as exc:
            logger.info(
                "Prometheus snapshot skipped (url=%s): %s", url, exc
            )
            path.write_text(
                f"# Prometheus snapshot unavailable from {url}: {exc}\n",
                encoding="utf-8",
            )

    @staticmethod
    def _populate_file_hashes(
        bundle_dir: Path, manifest: BundleManifest
    ) -> None:
        for f in sorted(bundle_dir.rglob("*")):
            if f.is_file() and f.name != "MANIFEST.json":
                rel = str(f.relative_to(bundle_dir))
                manifest.files[rel] = compute_sha256(f)

    @staticmethod
    def _archive(bundle_dir: Path) -> Path:
        archive_path = bundle_dir.with_suffix(".tar.gz")
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(bundle_dir, arcname=bundle_dir.name)
        return archive_path


# ====================================================================
# Helpers
# ====================================================================


_URL_CRED_RE = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+\-.]*://)([^:/@]+):([^@/]+)@")


def _censor_secret(key: str, value: Any) -> Any:
    """Censor secrets in Settings before writing to bundle.

    Strategy:
      * Field name contains 'password' or 'secret' or 'token' or
        'api_key' → mask completely.
      * String value embeds ``user:pass@host`` (URL credentials) → mask
        the password component, keep scheme + user + host.
    """
    name = key.lower()
    if any(s in name for s in ("password", "secret", "token", "api_key")):
        return "***"
    if isinstance(value, str) and "://" in value and "@" in value:
        return _URL_CRED_RE.sub(r"\g<scheme>\2:***@", value)
    return value


def _to_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return json.dumps(list(value), default=str)
    return value
