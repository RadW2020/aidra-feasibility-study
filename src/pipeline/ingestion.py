"""
Modulo de ingesta de imagenes satelitales Sentinel-1 desde Copernicus.

Responsabilidades:
1. Autenticacion OAuth2 con Copernicus Data Space
2. Busqueda de productos Sentinel-1 GRD por area y fecha
3. Descarga del producto (archivo .zip, ~500 MB - 1 GB)
4. Extraccion y validacion del producto descargado
5. Calculo de hash SHA256 del archivo descargado

Dependencias externas:
- httpx (async HTTP)
- pydantic (modelos de datos)

Notas:
- Los tokens OAuth2 de Copernicus expiran en 600 segundos (10 minutos)
- La cuota gratuita permite 12 TB/mes de descarga
- Los productos Sentinel-1 GRD tienen ~500 MB - 1 GB por escena
- El area de busqueda se define como bounding box [lon_min, lat_min, lon_max, lat_max]
"""

from __future__ import annotations

import hashlib
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from src.observability.loki_logger import StructuredLogger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
ODATA_BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1"
DOWNLOAD_BASE = "https://zipper.dataspace.copernicus.eu/odata/v1"
TOKEN_LIFETIME_SECONDS = 600  # Copernicus tokens expire after 10 min
TOKEN_REFRESH_MARGIN_SECONDS = 60  # refresh 1 min before expiry
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
HASH_CHUNK_SIZE = 64 * 1024  # 64 KB

# Copernicus quotas (free tier):
# - 4 concurrent connections max (S3/OData/STAC)
# - 20 MB/s per connection
# - 12 TB/month rolling window
# - Token expires in 10 min, refresh within 60 min
PARALLEL_DOWNLOAD_WORKERS = 4  # max concurrent connections
CHUNK_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB per chunk for parallel download

SEARCH_ZONES: dict[str, dict[str, Any]] = {
    "gibraltar": {
        "name": "Estrecho de Gibraltar",
        "bbox": [-5.8, 35.7, -5.2, 36.2],
        "description": "Alto trafico maritimo, estrecho natural",
    },
    "mediterranean_west": {
        "name": "Mediterraneo Occidental",
        "bbox": [-1.0, 36.5, 4.0, 39.5],
        "description": "Ruta comercial principal, costas Espana-Argelia",
    },
    "suez_approach": {
        "name": "Aproximacion Canal de Suez",
        "bbox": [32.0, 29.5, 34.0, 31.5],
        "description": "Zona de espera, alta densidad de barcos",
    },
    "english_channel": {
        "name": "Canal de la Mancha",
        "bbox": [-2.0, 49.5, 2.0, 51.5],
        "description": "Ruta comercial Europa del Norte",
    },
    "north_adriatic": {
        "name": "Norte del Adriatico",
        "bbox": [12.0, 44.5, 14.0, 45.8],
        "description": "Zona portuaria, Venecia-Trieste",
    },
}

# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------


class CopernicusSearchResult(BaseModel):
    """Represents a single product returned by the Copernicus OData search."""

    product_id: str = Field(..., description="Product UUID in Copernicus catalogue")
    title: str = Field(..., description="Product name / filename stem")
    sensing_date: datetime = Field(..., description="Acquisition datetime (UTC)")
    footprint: dict[str, Any] = Field(
        default_factory=dict,
        description="GeoJSON geometry of the product footprint",
    )
    size_mb: float = Field(0.0, description="Estimated size in megabytes")
    download_url: str = Field(..., description="Direct download URL")
    online: bool = Field(True, description="Whether the product is online")


# ---------------------------------------------------------------------------
# OAuth2 authentication
# ---------------------------------------------------------------------------


class CopernicusAuth:
    """Manages OAuth2 token lifecycle for Copernicus Data Space.

    Tokens are obtained via the *resource-owner password* grant and cached
    in memory.  ``get_token`` transparently refreshes the token when it is
    about to expire.
    """

    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password
        self._token: str | None = None
        self._token_expiry: datetime | None = None
        self._log = StructuredLogger("aidra.ingestion.auth")

    # -- public API --

    async def get_token(self) -> str:
        """Return a valid access token, refreshing if necessary.

        Returns:
            A bearer access-token string.

        Raises:
            httpx.HTTPStatusError: If the token endpoint returns an error.
        """
        now = datetime.now(tz=UTC)
        if (
            self._token is not None
            and self._token_expiry is not None
            and now < self._token_expiry
        ):
            return self._token
        return await self._refresh_token()

    # -- internal --

    async def _refresh_token(self) -> str:
        """Request a new access token from Copernicus OAuth2 endpoint.

        Returns:
            The fresh access-token string.

        Raises:
            httpx.HTTPStatusError: On non-2xx response from token endpoint.
        """
        self._log.info("Requesting new OAuth2 token from Copernicus")
        payload = {
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
            "client_id": "cdse-public",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(TOKEN_URL, data=payload)
            resp.raise_for_status()

        data = resp.json()
        self._token = data["access_token"]
        expires_in = int(data.get("expires_in", TOKEN_LIFETIME_SECONDS))
        self._token_expiry = datetime.now(tz=UTC) + timedelta(
            seconds=expires_in - TOKEN_REFRESH_MARGIN_SECONDS
        )
        self._log.info(
            "OAuth2 token acquired",
            extra={"expires_in": expires_in},
        )
        return self._token


# ---------------------------------------------------------------------------
# Image ingester
# ---------------------------------------------------------------------------


class ImageIngester:
    """Orchestrates search and download of Sentinel-1 GRD products.

    Args:
        auth: A :class:`CopernicusAuth` instance for bearer-token management.
        images_dir: Local directory to store downloaded products.
    """

    def __init__(self, auth: CopernicusAuth, images_dir: Path) -> None:
        self.auth = auth
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._log = StructuredLogger("aidra.ingestion")

    # -- Search --

    async def search(
        self,
        bbox: list[float],
        start_date: str,
        end_date: str,
        max_results: int = 5,
        sensor: str = "s1",
    ) -> list[CopernicusSearchResult]:
        """Search Sentinel products intersecting a bounding box.

        Args:
            bbox: ``[lon_min, lat_min, lon_max, lat_max]``.
            start_date: ISO date string ``YYYY-MM-DD`` (inclusive).
            end_date: ISO date string ``YYYY-MM-DD`` (inclusive).
            max_results: Maximum number of products to return.
            sensor: ``"s1"`` for Sentinel-1 GRD (SAR) or ``"s2"`` for
                Sentinel-2 L2A (optical). Default ``"s1"``.

        Returns:
            A list of :class:`CopernicusSearchResult` ordered by sensing date
            descending.

        Raises:
            httpx.HTTPStatusError: On non-2xx catalogue response.
        """
        lon_min, lat_min, lon_max, lat_max = bbox

        # Build WKT polygon (counter-clockwise ring)
        polygon_wkt = (
            f"POLYGON(("
            f"{lon_min} {lat_min},"
            f"{lon_max} {lat_min},"
            f"{lon_max} {lat_max},"
            f"{lon_min} {lat_max},"
            f"{lon_min} {lat_min}"
            f"))"
        )

        # Sensor-specific OData filter
        if sensor == "s2":
            collection_filter = (
                "Collection/Name eq 'SENTINEL-2'"
                " and Attributes/OData.CSC.StringAttribute/any("
                "att:att/Name eq 'productType'"
                " and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')"
                " and Attributes/OData.CSC.DoubleAttribute/any("
                "att:att/Name eq 'cloudCover'"
                " and att/OData.CSC.DoubleAttribute/Value lt 20)"
            )
        else:
            collection_filter = (
                "Collection/Name eq 'SENTINEL-1'"
                " and Attributes/OData.CSC.StringAttribute/any("
                "att:att/Name eq 'productType'"
                " and att/OData.CSC.StringAttribute/Value eq 'GRD')"
            )

        odata_filter = (
            f"{collection_filter}"
            f" and OData.CSC.Intersects(area=geography'SRID=4326;{polygon_wkt}')"
            f" and ContentDate/Start gt {start_date}T00:00:00.000Z"
            f" and ContentDate/Start lt {end_date}T23:59:59.999Z"
        )

        params: dict[str, str] = {
            "$filter": odata_filter,
            "$orderby": "ContentDate/Start desc",
            "$top": str(max_results),
        }

        self._log.info(
            "Searching Copernicus OData catalogue",
            extra={
                "bbox": bbox,
                "start_date": start_date,
                "end_date": end_date,
                "max_results": max_results,
            },
        )

        token = await self.auth.get_token()
        headers = {"Authorization": f"Bearer {token}"}

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(
                f"{ODATA_BASE}/Products",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()

        data = resp.json()
        results: list[CopernicusSearchResult] = []
        for item in data.get("value", []):
            product_id = item["Id"]
            content_length = item.get("ContentLength", 0)
            size_mb = content_length / (1024 * 1024) if content_length else 0.0

            # Parse footprint GeoJSON if present
            footprint: dict[str, Any] = {}
            geo = item.get("GeoFootprint") or item.get("Footprint")
            if isinstance(geo, dict):
                footprint = geo
            elif isinstance(geo, str):
                footprint = {"type": "Polygon", "raw": geo}

            sensing_str = item.get("ContentDate", {}).get(
                "Start", item.get("OriginDate", "")
            )
            sensing_date = datetime.fromisoformat(
                sensing_str.replace("Z", "+00:00")
            )

            results.append(
                CopernicusSearchResult(
                    product_id=product_id,
                    title=item.get("Name", ""),
                    sensing_date=sensing_date,
                    footprint=footprint,
                    size_mb=round(size_mb, 2),
                    download_url=f"{DOWNLOAD_BASE}/Products({product_id})/$value",
                    online=item.get("Online", True),
                )
            )

        self._log.info(
            f"Search returned {len(results)} products",
            extra={"count": len(results)},
        )
        return results

    # -- Search by ID --

    async def search_by_id(self, product_id: str) -> CopernicusSearchResult:
        """Retrieve a single product by its Copernicus catalogue ID.

        Args:
            product_id: The product UUID in the Copernicus catalogue.

        Returns:
            A :class:`CopernicusSearchResult` for the requested product.

        Raises:
            httpx.HTTPStatusError: On non-2xx catalogue response.
            ValueError: If the product is not found.
        """
        self._log.info(
            "Fetching product by ID from Copernicus OData catalogue",
            extra={"product_id": product_id},
        )

        token = await self.auth.get_token()
        headers = {"Authorization": f"Bearer {token}"}

        url = f"{ODATA_BASE}/Products({product_id})"

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()

        item = resp.json()

        content_length = item.get("ContentLength", 0)
        size_mb = content_length / (1024 * 1024) if content_length else 0.0

        footprint: dict[str, Any] = {}
        geo = item.get("GeoFootprint") or item.get("Footprint")
        if isinstance(geo, dict):
            footprint = geo
        elif isinstance(geo, str):
            footprint = {"type": "Polygon", "raw": geo}

        sensing_str = item.get("ContentDate", {}).get(
            "Start", item.get("OriginDate", "")
        )
        sensing_date = datetime.fromisoformat(
            sensing_str.replace("Z", "+00:00")
        )

        result = CopernicusSearchResult(
            product_id=item.get("Id", product_id),
            title=item.get("Name", ""),
            sensing_date=sensing_date,
            footprint=footprint,
            size_mb=round(size_mb, 2),
            download_url=f"{DOWNLOAD_BASE}/Products({product_id})/$value",
            online=item.get("Online", True),
        )

        self._log.info(
            "Product fetched by ID",
            extra={
                "product_id": product_id,
                "title": result.title,
            },
        )
        return result

    # -- Download --

    async def download(self, product: CopernicusSearchResult) -> Path:
        """Download a product zip archive via streaming.

        Downloads to a temporary ``.part`` file first, validates it is a
        valid ZIP archive, then atomically renames to the final path.
        Partial files are cleaned up on failure so retries start fresh.

        Args:
            product: The search result to download.

        Returns:
            The local path to the downloaded ``.zip`` file.

        Raises:
            httpx.HTTPStatusError: On non-2xx download response.
            zipfile.BadZipFile: If the downloaded file is not a valid zip.
        """
        zip_path = self.images_dir / f"{product.title}.zip"
        if zip_path.exists():
            self._log.info(
                "Product already downloaded, skipping",
                extra={"path": str(zip_path)},
            )
            return zip_path

        part_path = zip_path.with_suffix(".zip.part")

        self._log.info(
            "Starting product download",
            extra={
                "product_id": product.product_id,
                "title": product.title,
                "size_mb": product.size_mb,
            },
        )

        try:
            token = await self.auth.get_token()

            # Step 1: HEAD request to get Content-Length
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout=60.0, connect=30.0),
                follow_redirects=True,
            ) as client:
                head_resp = await client.head(
                    product.download_url,
                    headers={"Authorization": f"Bearer {token}"},
                )
                total_size = int(head_resp.headers.get("content-length", 0))

            if total_size == 0:
                # Fallback: single-stream download
                self._log.warning("Cannot determine file size; falling back to single-stream")
                await self._download_single_stream(product, part_path, token)
            elif total_size < CHUNK_SIZE_BYTES * 2:
                # Small file: single stream is fine
                await self._download_single_stream(product, part_path, token)
            else:
                # Large file: parallel chunked download
                self._log.info(
                    "Starting parallel download",
                    extra={
                        "total_mb": round(total_size / (1024 * 1024), 1),
                        "workers": PARALLEL_DOWNLOAD_WORKERS,
                        "chunk_mb": CHUNK_SIZE_BYTES // (1024 * 1024),
                    },
                )
                await self._download_parallel(
                    product, part_path, token, total_size
                )

            # Validate ZIP
            if not zipfile.is_zipfile(part_path):
                raise zipfile.BadZipFile(
                    f"Downloaded file is not a valid ZIP: {part_path}"
                )

            # Atomic rename
            part_path.rename(zip_path)

        except Exception:
            raise

        file_size = part_path.stat().st_size if not zip_path.exists() else zip_path.stat().st_size
        downloaded_mb = round(file_size / (1024 * 1024), 2) if file_size else 0
        self._log.info(
            "Download complete",
            extra={
                "product_id": product.product_id,
                "downloaded_mb": downloaded_mb,
                "path": str(zip_path),
            },
        )
        return zip_path

    # -- Download helpers --

    async def _download_single_stream(
        self,
        product: CopernicusSearchResult,
        dest: Path,
        token: str,
    ) -> None:
        """Download using a single HTTP stream with resume support."""
        headers: dict[str, str] = {"Authorization": f"Bearer {token}"}
        downloaded = 0
        mode = "wb"

        # Resume if partial exists
        if dest.exists() and dest.stat().st_size > 0:
            downloaded = dest.stat().st_size
            headers["Range"] = f"bytes={downloaded}-"
            mode = "ab"
            self._log.info("Resuming single-stream", extra={"from_bytes": downloaded})

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout=1800.0, connect=30.0),
            follow_redirects=True,
        ) as client, client.stream(
            "GET", product.download_url, headers=headers
        ) as resp:
            if resp.status_code == 200 and mode == "ab":
                mode = "wb"  # server doesn't support Range
                downloaded = 0
            elif resp.status_code not in (200, 206):
                resp.raise_for_status()

            with open(dest, mode) as fh:
                async for chunk in resp.aiter_bytes(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    fh.write(chunk)
                    downloaded += len(chunk)

    async def _download_parallel(
        self,
        product: CopernicusSearchResult,
        dest: Path,
        token: str,
        total_size: int,
    ) -> None:
        """Download using parallel HTTP Range requests (max 4 connections).

        Splits the file into chunks, downloads each in parallel using
        asyncio semaphore, then concatenates into the final file.
        Copernicus allows 4 concurrent connections per user.
        """
        import asyncio as _aio

        chunks_dir = dest.parent / f".chunks_{dest.stem}"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        # Build chunk list: [(start, end, chunk_path), ...]
        chunks: list[tuple[int, int, Path]] = []
        offset = 0
        chunk_idx = 0
        while offset < total_size:
            end = min(offset + CHUNK_SIZE_BYTES - 1, total_size - 1)
            chunk_path = chunks_dir / f"chunk_{chunk_idx:04d}"
            chunks.append((offset, end, chunk_path))
            offset = end + 1
            chunk_idx += 1

        self._log.info(
            "Parallel download plan",
            extra={"total_chunks": len(chunks), "workers": PARALLEL_DOWNLOAD_WORKERS},
        )

        sem = _aio.Semaphore(PARALLEL_DOWNLOAD_WORKERS)
        errors: list[str] = []

        async def _download_chunk(start: int, end: int, path: Path) -> None:
            """Download a single byte range to a file."""
            async with sem:
                # Skip if chunk already downloaded with correct size
                expected = end - start + 1
                if path.exists() and path.stat().st_size == expected:
                    return

                try:
                    # Get fresh token for each chunk (tokens expire in 10 min)
                    tok = await self.auth.get_token()
                    hdrs = {
                        "Authorization": f"Bearer {tok}",
                        "Range": f"bytes={start}-{end}",
                    }
                    async with httpx.AsyncClient(
                        timeout=httpx.Timeout(timeout=600.0, connect=30.0),
                        follow_redirects=True,
                    ) as client, client.stream(
                        "GET", product.download_url, headers=hdrs
                    ) as resp:
                        if resp.status_code not in (200, 206):
                            resp.raise_for_status()
                        with open(path, "wb") as fh:
                            async for data in resp.aiter_bytes(
                                chunk_size=DOWNLOAD_CHUNK_SIZE
                            ):
                                fh.write(data)
                except Exception as exc:
                    errors.append(f"Chunk {start}-{end}: {exc}")
                    self._log.warning(
                        "Chunk download failed",
                        extra={"start": start, "end": end, "error": str(exc)},
                    )

        # Launch all chunks (semaphore limits to 4 concurrent)
        tasks = [_download_chunk(s, e, p) for s, e, p in chunks]
        await _aio.gather(*tasks)

        if errors:
            # Retry failed chunks once with single stream
            self._log.warning(
                "Some chunks failed, retrying",
                extra={"failed": len(errors)},
            )
            errors.clear()
            failed_tasks = [
                _download_chunk(s, e, p)
                for s, e, p in chunks
                if not p.exists() or p.stat().st_size != (e - s + 1)
            ]
            await _aio.gather(*failed_tasks)

        if errors:
            import shutil
            shutil.rmtree(chunks_dir, ignore_errors=True)
            raise RuntimeError(
                f"Parallel download failed: {len(errors)} chunks failed"
            )

        # Concatenate chunks into final file
        self._log.info("Concatenating chunks")
        with open(dest, "wb") as out:
            for _, _, chunk_path in chunks:
                with open(chunk_path, "rb") as inp:
                    while True:
                        data = inp.read(DOWNLOAD_CHUNK_SIZE)
                        if not data:
                            break
                        out.write(data)

        # Verify size
        actual = dest.stat().st_size
        if actual != total_size:
            raise RuntimeError(
                f"Size mismatch after concatenation: expected {total_size}, got {actual}"
            )

        # Cleanup chunks
        import shutil
        shutil.rmtree(chunks_dir, ignore_errors=True)

        self._log.info(
            "Parallel download complete",
            extra={"total_mb": round(total_size / (1024 * 1024), 1)},
        )

    # -- Extract --

    async def extract(self, zip_path: Path) -> Path:
        """Extract a product zip to a directory alongside the archive.

        Validates each entry to prevent zip-slip attacks (path traversal
        via ``..`` components).

        Args:
            zip_path: Path to the downloaded ``.zip`` file.

        Returns:
            Path to the extracted product directory (contains ``.tiff`` etc.).

        Raises:
            zipfile.BadZipFile: If the file is not a valid zip archive.
            ValueError: If a zip entry attempts path traversal.
        """
        extract_dir = zip_path.with_suffix("")  # strip .zip
        if extract_dir.exists():
            self._log.info(
                "Product already extracted, skipping",
                extra={"path": str(extract_dir)},
            )
            return extract_dir

        self._log.info(
            "Extracting product",
            extra={"zip_path": str(zip_path)},
        )

        with zipfile.ZipFile(zip_path, "r") as zf:
            # Validate all entries before extracting any
            resolved_extract = extract_dir.resolve()
            for member in zf.namelist():
                member_path = (extract_dir / member).resolve()
                if not member_path.is_relative_to(resolved_extract):
                    raise ValueError(
                        f"Zip entry attempts path traversal: {member!r}"
                    )
            zf.extractall(extract_dir)

        self._log.info(
            "Extraction complete",
            extra={"path": str(extract_dir)},
        )
        return extract_dir

    # -- Hash --

    async def compute_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file, reading in 64 KB chunks.

        Args:
            file_path: Path to the file to hash.

        Returns:
            Hex-encoded SHA-256 digest string.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as fh:
            while True:
                chunk = fh.read(HASH_CHUNK_SIZE)
                if not chunk:
                    break
                sha256.update(chunk)

        digest = sha256.hexdigest()
        self._log.info(
            "Computed file hash",
            extra={"file": str(file_path), "sha256": digest},
        )
        return digest
