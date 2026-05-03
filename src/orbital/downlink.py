"""
Analisis de downlink: cuantifica el ahorro de ancho de banda
que OBDP (On-Board Data Processing) proporciona respecto al
downlink de imagenes crudas.

El argumento central: OBDP multiplica por ~50 000x la capacidad
efectiva de un satelite al enviar solo detecciones (JSON, ~10 KB)
en lugar de imagenes SAR crudas (~800 MB).

Usage:
    from src.orbital.downlink import DownlinkAnalyzer

    analyzer = DownlinkAnalyzer()
    result = analyzer.analyze_single_image(
        image_size_mb=800.0,
        result_size_kb=10.0,
    )
    print(result.compression_ratio)
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.db.models import ExecutionRecord
from src.orbital.orbit_params import DOWNLINK_PROFILES

logger = logging.getLogger(__name__)

__all__ = [
    "DOWNLINK_PROFILES",
    "DownlinkAnalyzer",
    "DownlinkAnalysis",
    "DailyCapacityAnalysis",
    "OBDPValueReport",
]


# ====================================================================
# Pydantic models
# ====================================================================


class DownlinkAnalysis(BaseModel):
    """Comparison of downlink with vs. without OBDP for a single image."""

    downlink_profile: str
    image_size_mb: float
    result_size_kb: float

    # Without OBDP
    raw_downlink_seconds: float = Field(
        description="Time to downlink the full raw image",
    )
    raw_images_per_window: float = Field(
        description="Raw images that fit in one contact window",
    )
    raw_images_per_day: float = Field(
        description="Raw images per day across all passes",
    )

    # With OBDP
    obdp_downlink_seconds: float = Field(
        description="Time to downlink the processed result",
    )
    obdp_results_per_window: float = Field(
        description="Processed results that fit in one window",
    )
    obdp_results_per_day: float = Field(
        description="Processed results per day across all passes",
    )

    # Ratios
    compression_ratio: float = Field(
        description="image_size / result_size",
    )
    bandwidth_saving_percent: float = Field(
        description="(1 - result_size/image_size) * 100",
    )
    capacity_multiplier: float = Field(
        description="obdp_per_day / raw_per_day",
    )
    time_saving_percent: float = Field(
        description="(1 - obdp_time/raw_time) * 100",
    )


class DailyCapacityAnalysis(BaseModel):
    """Daily downlink capacity with vs. without OBDP."""

    downlink_profile: str
    images_per_day_requested: int
    image_size_mb: float
    result_size_kb: float

    # Total daily data volumes
    raw_daily_data_gb: float
    obdp_daily_data_mb: float

    # Capacity
    raw_daily_capacity_images: float = Field(
        description="Max raw images per day given the downlink",
    )
    obdp_daily_capacity_results: float = Field(
        description="Max results per day given the downlink",
    )

    # Can we keep up?
    raw_can_keep_up: bool = Field(
        description="True if raw downlink can handle the requested volume",
    )
    obdp_can_keep_up: bool = Field(
        description="True if OBDP downlink can handle the requested volume",
    )

    # Backlog
    raw_backlog_images_per_day: float = Field(
        description="Images that cannot be downlinked per day (raw)",
    )
    obdp_backlog_results_per_day: float = Field(
        description="Results that cannot be downlinked per day (OBDP)",
    )


class OBDPValueReport(BaseModel):
    """Executive summary of OBDP value based on real execution data."""

    avg_compression_ratio: float
    avg_image_size_mb: float
    avg_result_size_kb: float
    total_images_analyzed: int
    total_bandwidth_saved_gb: float = Field(
        description="Total bandwidth saved if all images had been downlinked raw",
    )
    equivalent_extra_capacity: str = Field(
        description='e.g. "With OBDP, a CubeSat S-Band equals a SmallSat X-Band"',
    )
    recommendations: list[str]


# ====================================================================
# DownlinkAnalyzer
# ====================================================================


class DownlinkAnalyzer:
    """Stateless calculator for downlink analyses.

    All methods are pure functions operating on the given parameters
    and the ``DOWNLINK_PROFILES`` reference table.
    """

    # ------------------------------------------------------------------
    # Single image
    # ------------------------------------------------------------------

    def analyze_single_image(
        self,
        image_size_mb: float,
        result_size_kb: float,
        thumbnail_size_kb: float = 50.0,
        metadata_size_kb: float = 5.0,
        downlink_profile: str = "smallsat_xband",
    ) -> DownlinkAnalysis:
        """Compare downlink requirements with vs. without OBDP.

        Parameters
        ----------
        image_size_mb:
            Size of the raw SAR image in megabytes.
        result_size_kb:
            Size of the processed detection result (JSON) in kilobytes.
        thumbnail_size_kb:
            Optional thumbnail of the detection area (kilobytes).
        metadata_size_kb:
            Execution metadata (kilobytes).
        downlink_profile:
            Key in ``DOWNLINK_PROFILES``.

        Returns
        -------
        DownlinkAnalysis
        """
        profile = self._get_profile(downlink_profile)
        bw_mbps: float = float(profile["bandwidth_mbps"])
        window_min: float = float(profile["window_minutes"])
        passes: int = int(profile["passes_per_day"])

        window_seconds = window_min * 60.0
        bw_mb_per_sec = bw_mbps / 8.0  # megabits -> megabytes

        # Without OBDP: send the full raw image
        raw_dl_sec = image_size_mb / bw_mb_per_sec if bw_mb_per_sec > 0 else float("inf")
        raw_per_window = window_seconds / raw_dl_sec if raw_dl_sec > 0 else 0.0
        raw_per_day = raw_per_window * passes

        # With OBDP: send result + thumbnail + metadata
        obdp_total_kb = result_size_kb + thumbnail_size_kb + metadata_size_kb
        obdp_total_mb = obdp_total_kb / 1024.0
        obdp_dl_sec = obdp_total_mb / bw_mb_per_sec if bw_mb_per_sec > 0 else float("inf")
        obdp_per_window = window_seconds / obdp_dl_sec if obdp_dl_sec > 0 else 0.0
        obdp_per_day = obdp_per_window * passes

        # Ratios
        image_size_kb = image_size_mb * 1024.0
        compression_ratio = image_size_kb / obdp_total_kb if obdp_total_kb > 0 else 0.0
        bw_saving_pct = (1.0 - obdp_total_kb / image_size_kb) * 100.0 if image_size_kb > 0 else 0.0
        capacity_mult = obdp_per_day / raw_per_day if raw_per_day > 0 else float("inf")
        time_saving_pct = (1.0 - obdp_dl_sec / raw_dl_sec) * 100.0 if raw_dl_sec > 0 else 0.0

        return DownlinkAnalysis(
            downlink_profile=downlink_profile,
            image_size_mb=image_size_mb,
            result_size_kb=result_size_kb,
            raw_downlink_seconds=round(raw_dl_sec, 3),
            raw_images_per_window=round(raw_per_window, 2),
            raw_images_per_day=round(raw_per_day, 2),
            obdp_downlink_seconds=round(obdp_dl_sec, 6),
            obdp_results_per_window=round(obdp_per_window, 1),
            obdp_results_per_day=round(obdp_per_day, 1),
            compression_ratio=round(compression_ratio, 1),
            bandwidth_saving_percent=round(bw_saving_pct, 4),
            capacity_multiplier=round(capacity_mult, 1),
            time_saving_percent=round(time_saving_pct, 4),
        )

    # ------------------------------------------------------------------
    # Daily capacity
    # ------------------------------------------------------------------

    def analyze_daily_capacity(
        self,
        images_per_day: int,
        image_size_mb: float,
        result_size_kb: float,
        downlink_profile: str = "smallsat_xband",
    ) -> DailyCapacityAnalysis:
        """Calculate whether the downlink can keep up with the capture rate.

        Parameters
        ----------
        images_per_day:
            Number of images the sensor captures per day.
        image_size_mb:
            Size of each raw image.
        result_size_kb:
            Size of each processed result.
        downlink_profile:
            Key in ``DOWNLINK_PROFILES``.

        Returns
        -------
        DailyCapacityAnalysis
        """
        profile = self._get_profile(downlink_profile)
        bw_mbps: float = float(profile["bandwidth_mbps"])
        window_min: float = float(profile["window_minutes"])
        passes: int = int(profile["passes_per_day"])

        bw_mb_per_sec = bw_mbps / 8.0
        total_dl_seconds_per_day = window_min * 60.0 * passes
        total_dl_mb_per_day = bw_mb_per_sec * total_dl_seconds_per_day

        # Raw
        raw_daily_data_gb = images_per_day * image_size_mb / 1024.0
        raw_capacity = total_dl_mb_per_day / image_size_mb if image_size_mb > 0 else 0.0
        raw_can_keep_up = raw_capacity >= images_per_day
        raw_backlog = max(0.0, images_per_day - raw_capacity)

        # OBDP
        result_size_mb = result_size_kb / 1024.0
        obdp_daily_data_mb = images_per_day * result_size_mb
        obdp_capacity = total_dl_mb_per_day / result_size_mb if result_size_mb > 0 else 0.0
        obdp_can_keep_up = obdp_capacity >= images_per_day
        obdp_backlog = max(0.0, images_per_day - obdp_capacity)

        return DailyCapacityAnalysis(
            downlink_profile=downlink_profile,
            images_per_day_requested=images_per_day,
            image_size_mb=image_size_mb,
            result_size_kb=result_size_kb,
            raw_daily_data_gb=round(raw_daily_data_gb, 3),
            obdp_daily_data_mb=round(obdp_daily_data_mb, 4),
            raw_daily_capacity_images=round(raw_capacity, 2),
            obdp_daily_capacity_results=round(obdp_capacity, 1),
            raw_can_keep_up=raw_can_keep_up,
            obdp_can_keep_up=obdp_can_keep_up,
            raw_backlog_images_per_day=round(raw_backlog, 2),
            obdp_backlog_results_per_day=round(obdp_backlog, 2),
        )

    # ------------------------------------------------------------------
    # Compare all profiles
    # ------------------------------------------------------------------

    def analyze_all_profiles(
        self,
        image_size_mb: float,
        result_size_kb: float,
    ) -> list[DownlinkAnalysis]:
        """Generate a comparison table across all downlink profiles.

        Parameters
        ----------
        image_size_mb:
            Size of the raw image.
        result_size_kb:
            Size of the processed result.

        Returns
        -------
        list[DownlinkAnalysis]
            One entry per downlink profile, sorted by bandwidth (ascending).
        """
        analyses: list[DownlinkAnalysis] = []
        for profile_key in DOWNLINK_PROFILES:
            analysis = self.analyze_single_image(
                image_size_mb=image_size_mb,
                result_size_kb=result_size_kb,
                downlink_profile=profile_key,
            )
            analyses.append(analysis)

        analyses.sort(key=lambda a: a.raw_downlink_seconds, reverse=True)
        return analyses

    # ------------------------------------------------------------------
    # OBDP value report from real executions
    # ------------------------------------------------------------------

    def generate_obdp_value_report(
        self,
        execution_records: list[ExecutionRecord],
    ) -> OBDPValueReport:
        """Generate an executive OBDP-value report from execution history.

        Uses actual ``image_size_mb`` from executions and estimates the
        result size from detected objects to calculate the real
        compression ratio and bandwidth savings.

        Parameters
        ----------
        execution_records:
            List of completed execution records.

        Returns
        -------
        OBDPValueReport
        """
        if not execution_records:
            return OBDPValueReport(
                avg_compression_ratio=0.0,
                avg_image_size_mb=0.0,
                avg_result_size_kb=0.0,
                total_images_analyzed=0,
                total_bandwidth_saved_gb=0.0,
                equivalent_extra_capacity="No execution data available.",
                recommendations=["Run pipeline executions to generate data."],
            )

        total_image_mb = 0.0
        total_result_kb = 0.0
        valid_count = 0

        for rec in execution_records:
            img_mb = rec.image_size_mb if rec.image_size_mb is not None else 0.0
            if img_mb <= 0:
                continue
            # Estimate result size: ~0.5 KB per detection + 2 KB base metadata
            result_kb = 2.0 + rec.num_detections * 0.5
            total_image_mb += img_mb
            total_result_kb += result_kb
            valid_count += 1

        if valid_count == 0:
            return OBDPValueReport(
                avg_compression_ratio=0.0,
                avg_image_size_mb=0.0,
                avg_result_size_kb=0.0,
                total_images_analyzed=0,
                total_bandwidth_saved_gb=0.0,
                equivalent_extra_capacity="No valid images with size data.",
                recommendations=["Ensure image_size_mb is populated."],
            )

        avg_image_mb = total_image_mb / valid_count
        avg_result_kb = total_result_kb / valid_count
        avg_compression = (avg_image_mb * 1024.0) / avg_result_kb if avg_result_kb > 0 else 0.0

        # Total bandwidth saved: sum(image_sizes) - sum(result_sizes)
        total_saved_mb = total_image_mb - (total_result_kb / 1024.0)
        total_saved_gb = total_saved_mb / 1024.0

        # Equivalent capacity comparison
        equivalent = self._compute_equivalent_capacity(avg_image_mb, avg_result_kb)

        # Recommendations
        recommendations = self._generate_recommendations(
            avg_compression, avg_image_mb, avg_result_kb, valid_count
        )

        return OBDPValueReport(
            avg_compression_ratio=round(avg_compression, 1),
            avg_image_size_mb=round(avg_image_mb, 2),
            avg_result_size_kb=round(avg_result_kb, 2),
            total_images_analyzed=valid_count,
            total_bandwidth_saved_gb=round(total_saved_gb, 3),
            equivalent_extra_capacity=equivalent,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_profile(name: str) -> dict[str, Any]:
        """Look up a downlink profile by key."""
        if name not in DOWNLINK_PROFILES:
            available = ", ".join(DOWNLINK_PROFILES.keys())
            raise KeyError(
                f"Downlink profile '{name}' not found. Available: {available}"
            )
        return DOWNLINK_PROFILES[name]

    def _compute_equivalent_capacity(
        self,
        avg_image_mb: float,
        avg_result_kb: float,
    ) -> str:
        """Determine which higher-tier downlink an OBDP-equipped low-tier equals."""
        # Calculate raw images/day for cubesat_sband
        sband = DOWNLINK_PROFILES["cubesat_sband"]
        sband_bw = float(sband["bandwidth_mbps"]) / 8.0
        sband_window = float(sband["window_minutes"]) * 60.0
        sband_passes = int(sband["passes_per_day"])
        sband_raw_per_day = (
            sband_bw * sband_window * sband_passes / avg_image_mb
            if avg_image_mb > 0
            else 0.0
        )

        # Calculate OBDP results/day for cubesat_sband
        result_mb = avg_result_kb / 1024.0
        sband_obdp_per_day = (
            sband_bw * sband_window * sband_passes / result_mb
            if result_mb > 0
            else 0.0
        )

        # Find which raw-downlink profile matches the OBDP capacity
        best_match = "unknown"
        for _pkey, pval in DOWNLINK_PROFILES.items():
            bw = float(pval["bandwidth_mbps"]) / 8.0
            win = float(pval["window_minutes"]) * 60.0
            passes = int(pval["passes_per_day"])
            raw_cap = bw * win * passes / avg_image_mb if avg_image_mb > 0 else 0.0
            if raw_cap <= sband_obdp_per_day:
                best_match = str(pval["name"])

        return (
            f"With OBDP, a CubeSat S-Band ({sband_raw_per_day:.0f} raw imgs/day) "
            f"achieves {sband_obdp_per_day:.0f} results/day, "
            f"equivalent to raw capacity of a {best_match}."
        )

    @staticmethod
    def _generate_recommendations(
        avg_compression: float,
        avg_image_mb: float,
        avg_result_kb: float,
        count: int,
    ) -> list[str]:
        """Generate actionable recommendations based on results."""
        recs: list[str] = []

        if avg_compression > 10000:
            recs.append(
                f"Compression ratio of {avg_compression:.0f}:1 is excellent. "
                "OBDP is highly justified for bandwidth-constrained missions."
            )
        elif avg_compression > 1000:
            recs.append(
                f"Compression ratio of {avg_compression:.0f}:1 is very good. "
                "OBDP provides significant bandwidth savings."
            )
        else:
            recs.append(
                f"Compression ratio of {avg_compression:.0f}:1 is moderate. "
                "Consider reducing result payload size."
            )

        if avg_result_kb > 100:
            recs.append(
                "Average result size exceeds 100 KB. "
                "Consider reducing thumbnail resolution or excluding optional fields."
            )

        if count < 10:
            recs.append(
                f"Only {count} executions analysed. "
                "Run more pipeline executions for statistically robust results."
            )

        recs.append(
            "For missions with UHF-only downlink, OBDP is not optional "
            "but mandatory to achieve any useful data throughput."
        )

        return recs
