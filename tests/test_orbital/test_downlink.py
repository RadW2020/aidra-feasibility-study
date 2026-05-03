"""
Tests for the orbital downlink analysis module.

Covers:
- Compression ratio for typical SAR images
- Bandwidth saving percentage
- Analysis across all downlink profiles
- Daily capacity improvement with OBDP
"""

from __future__ import annotations

from src.orbital.downlink import DownlinkAnalyzer
from src.orbital.orbit_params import DOWNLINK_PROFILES

# ====================================================================
# Single-image analysis
# ====================================================================


class TestSingleImageAnalysis:
    """Tests for analyze_single_image."""

    def test_compression_ratio(self):
        """Compression ratio must be > 1000 for a typical SAR image
        (~800 MB raw) vs. a detection result (~10 KB + thumbnail + metadata).
        """
        analyzer = DownlinkAnalyzer()
        result = analyzer.analyze_single_image(
            image_size_mb=800.0,
            result_size_kb=10.0,
            thumbnail_size_kb=50.0,
            metadata_size_kb=5.0,
        )

        # 800 MB = 819200 KB, result = 10+50+5 = 65 KB
        # ratio = 819200 / 65 ~ 12603
        assert result.compression_ratio > 1000, (
            f"Expected compression ratio > 1000, got {result.compression_ratio}"
        )

    def test_bandwidth_saving(self):
        """Bandwidth saving must exceed 99% for typical SAR vs. detection."""
        analyzer = DownlinkAnalyzer()
        result = analyzer.analyze_single_image(
            image_size_mb=800.0,
            result_size_kb=10.0,
        )

        assert result.bandwidth_saving_percent > 99.0, (
            f"Expected > 99% bandwidth saving, "
            f"got {result.bandwidth_saving_percent:.2f}%"
        )

    def test_obdp_downlink_faster(self):
        """OBDP downlink time must be drastically less than raw downlink."""
        analyzer = DownlinkAnalyzer()
        result = analyzer.analyze_single_image(
            image_size_mb=800.0,
            result_size_kb=10.0,
        )

        assert result.obdp_downlink_seconds < result.raw_downlink_seconds
        assert result.time_saving_percent > 99.0

    def test_capacity_multiplier(self):
        """OBDP must multiply the effective downlink capacity significantly."""
        analyzer = DownlinkAnalyzer()
        result = analyzer.analyze_single_image(
            image_size_mb=800.0,
            result_size_kb=10.0,
        )

        assert result.capacity_multiplier > 100


# ====================================================================
# All profiles
# ====================================================================


class TestAllProfiles:
    """Tests for analyze_all_profiles."""

    def test_all_profiles(self):
        """Must return one analysis per downlink profile."""
        analyzer = DownlinkAnalyzer()
        analyses = analyzer.analyze_all_profiles(
            image_size_mb=800.0,
            result_size_kb=10.0,
        )

        assert len(analyses) == len(DOWNLINK_PROFILES)

        # Each analysis should reference a valid profile
        profile_names = {a.downlink_profile for a in analyses}
        assert profile_names == set(DOWNLINK_PROFILES.keys())

    def test_all_profiles_returns_4_analyses(self):
        """There are exactly 4 downlink profiles defined."""
        analyzer = DownlinkAnalyzer()
        analyses = analyzer.analyze_all_profiles(
            image_size_mb=800.0,
            result_size_kb=10.0,
        )
        assert len(analyses) == 4


# ====================================================================
# Daily capacity
# ====================================================================


class TestDailyCapacity:
    """Tests for analyze_daily_capacity."""

    def test_daily_capacity_with_obdp_higher(self):
        """OBDP daily capacity must far exceed raw daily capacity."""
        analyzer = DownlinkAnalyzer()
        capacity = analyzer.analyze_daily_capacity(
            images_per_day=50,
            image_size_mb=800.0,
            result_size_kb=10.0,
            downlink_profile="smallsat_xband",
        )

        assert capacity.obdp_daily_capacity_results > capacity.raw_daily_capacity_images
        # OBDP should be able to keep up even if raw cannot
        assert capacity.obdp_can_keep_up is True

    def test_daily_capacity_raw_backlog(self):
        """When requesting more images than raw downlink can handle,
        raw_backlog should be positive.
        """
        analyzer = DownlinkAnalyzer()

        # Request a very high volume on a low-bandwidth link
        capacity = analyzer.analyze_daily_capacity(
            images_per_day=100,
            image_size_mb=800.0,
            result_size_kb=10.0,
            downlink_profile="cubesat_sband",
        )

        # CubeSat S-Band cannot handle 100 * 800 MB raw images per day
        assert capacity.raw_can_keep_up is False
        assert capacity.raw_backlog_images_per_day > 0
