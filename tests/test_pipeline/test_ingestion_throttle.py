"""Tests for the aggregate download throttle (src/pipeline/ingestion).

The throttle exists because of the 2026-09-17 incident: four parallel
Copernicus connections pulled ~50 Mbps for six minutes, OCI answered by
dropping ~81 000 inbound packets on the instance VNIC, and every service
sharing that host started timing out. What matters here is therefore the
*aggregate* rate across workers, not the per-connection one.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from src.config import Settings
from src.pipeline.ingestion import (
    DOWNLOAD_CHUNK_SIZE,
    MIN_THROTTLED_READ_SIZE,
    CopernicusAuth,
    DownloadThrottle,
    ImageIngester,
)


class TestDownloadThrottle:
    """Token bucket behaviour."""

    def test_zero_rate_disables_the_throttle(self) -> None:
        throttle = DownloadThrottle(0)

        assert throttle.enabled is False
        assert throttle.rate_mbps == 0
        # Unthrottled downloads keep reading in full 8 MB chunks.
        assert throttle.read_size == DOWNLOAD_CHUNK_SIZE

    async def test_disabled_throttle_never_sleeps(self) -> None:
        throttle = DownloadThrottle(0)

        started = time.monotonic()
        for _ in range(50):
            await throttle.consume(DOWNLOAD_CHUNK_SIZE)

        assert time.monotonic() - started < 0.1

    def test_read_size_is_a_fraction_of_the_budget(self) -> None:
        # 8 MB/s -> reads of 1 MB, so pacing happens ~8 times per second
        # instead of in one burst at line rate.
        throttle = DownloadThrottle(8 * 1024 * 1024)

        assert throttle.read_size == 1024 * 1024

    def test_read_size_has_a_floor_and_a_ceiling(self) -> None:
        assert DownloadThrottle(1024).read_size == MIN_THROTTLED_READ_SIZE
        assert DownloadThrottle(10**9).read_size == DOWNLOAD_CHUNK_SIZE

    async def test_sustained_rate_does_not_exceed_the_ceiling(self) -> None:
        rate = 256 * 1024  # 256 KB/s
        throttle = DownloadThrottle(rate)
        payload = 192 * 1024  # 0.75 s of traffic at that rate

        started = time.monotonic()
        # The first read is paid from the initial burst; the next three
        # have to wait for the bucket to refill.
        for _ in range(4):
            await throttle.consume(payload)
        elapsed = time.monotonic() - started

        transferred = 4 * payload
        # The bucket allows one second of burst, so the floor is what is
        # left after spending it.
        assert elapsed >= (transferred - rate) / rate
        assert transferred / (elapsed + 1.0) <= rate

    async def test_budget_is_shared_across_concurrent_workers(self) -> None:
        rate = 256 * 1024
        throttle = DownloadThrottle(rate)
        # Four workers, each asking for half a second of traffic: a
        # per-connection limit would let them all through at once.
        payload = rate // 2

        started = time.monotonic()
        await asyncio.gather(*(throttle.consume(payload) for _ in range(4)))
        elapsed = time.monotonic() - started

        assert elapsed >= (4 * payload - rate) / rate


class TestIngesterWiring:
    """The ceiling reaches the ingester from Settings."""

    def test_default_ingester_is_unthrottled(self, tmp_path: Path) -> None:
        ingester = ImageIngester(
            auth=CopernicusAuth("u", "p"),
            images_dir=tmp_path,
        )

        assert ingester.throttle.enabled is False

    def test_settings_value_becomes_the_ceiling(self, tmp_path: Path) -> None:
        ingester = ImageIngester(
            auth=CopernicusAuth("u", "p"),
            images_dir=tmp_path,
            rate_limit_mbps=20.0,
        )

        assert ingester.throttle.enabled is True
        assert ingester.throttle.rate_mbps == pytest.approx(20.0)

    def test_settings_default_is_conservative(self) -> None:
        # I-DET-4 style: the ceiling lives in Settings, not in the logic.
        assert Settings().download_rate_limit_mbps == 20.0


class TestThrottledDownloadPath:
    """The real download loop honours the throttle."""

    async def test_single_stream_paces_the_socket_reads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.pipeline.ingestion as ingestion

        rate = 256 * 1024
        total = 256 * 1024  # one second of traffic
        requested_chunk_sizes: list[int] = []

        class _FakeResponse:
            status_code = 200

            async def aiter_bytes(self, chunk_size: int):
                requested_chunk_sizes.append(chunk_size)
                sent = 0
                while sent < total:
                    block = min(chunk_size, total - sent)
                    sent += block
                    yield b"\0" * block

            def raise_for_status(self) -> None:  # pragma: no cover - not hit
                raise AssertionError("unexpected error path")

        class _FakeStream:
            async def __aenter__(self):
                return _FakeResponse()

            async def __aexit__(self, *exc_info) -> None:
                return None

        class _FakeClient:
            def __init__(self, *args, **kwargs) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc_info) -> None:
                return None

            def stream(self, *args, **kwargs):
                return _FakeStream()

        monkeypatch.setattr(ingestion.httpx, "AsyncClient", _FakeClient)

        ingester = ImageIngester(
            auth=CopernicusAuth("u", "p"),
            images_dir=tmp_path,
            rate_limit_mbps=rate * 8 / 1_000_000,
        )
        # Spend the initial burst so the transfer has to be paid for.
        await ingester.throttle.consume(rate)

        product = ingestion.CopernicusSearchResult(
            product_id="p-1",
            title="S1A_TEST",
            sensing_date=ingestion.datetime.now(ingestion.UTC),
            download_url="https://example.invalid/product.zip",
        )
        dest = tmp_path / "product.zip.part"

        started = time.monotonic()
        await ingester._download_single_stream(product, dest, "token")
        elapsed = time.monotonic() - started

        assert dest.stat().st_size == total
        # Reads are paced, not requested in one 8 MB gulp.
        assert requested_chunk_sizes == [ingester.throttle.read_size]
        assert ingester.throttle.read_size < DOWNLOAD_CHUNK_SIZE
        # One second of traffic with an empty bucket cannot arrive instantly.
        assert elapsed >= 0.5
