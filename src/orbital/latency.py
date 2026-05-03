"""
Simulacion de latencia orbital end-to-end.

Modela el tiempo total desde que el sensor SAR captura la imagen
hasta que el resultado esta disponible en tierra, bajo diferentes
escenarios (con/sin OBDP, diferentes orbitas, diferentes estaciones).

En vigilancia maritima un barco se mueve ~20 nudos; en 1 hora
recorre ~37 km.  La latencia determina si la deteccion es
*accionable* (actionable).

Usage:
    from src.orbital.latency import OrbitalLatencySimulator

    sim = OrbitalLatencySimulator()
    without = sim.simulate_without_obdp()
    with_obdp = sim.simulate_with_obdp(inference_ms=150.0)
    print(f"Speedup: {without.total_minutes / with_obdp.total_minutes:.1f}x")
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from src.orbital.orbit_params import (
    DOWNLINK_PROFILES,
    GROUND_PROCESSING,
    ORBIT_PARAMS,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ORBIT_PARAMS",
    "GROUND_PROCESSING",
    "OrbitalLatencySimulator",
    "LatencyBreakdown",
    "LatencyComparison",
    "ActionabilityResult",
]


# ====================================================================
# Pydantic models
# ====================================================================


class LatencyBreakdown(BaseModel):
    """Detailed breakdown of end-to-end latency for one scenario."""

    scenario: str = Field(
        description='"with_obdp" or "without_obdp"',
    )
    orbit: str
    capture_s: float = 0.0
    onboard_processing_s: float = Field(
        description="0 without OBDP; inference_ms with OBDP",
    )
    wait_for_contact_s: float = Field(
        description="Dominant latency component (orbit-dependent)",
    )
    downlink_s: float = Field(
        description="Drastically different with/without OBDP",
    )
    ground_ingest_s: float
    ground_processing_s: float = Field(
        description="0 with OBDP (already processed on-board)",
    )
    dissemination_s: float
    total_seconds: float
    total_minutes: float


class LatencyComparison(BaseModel):
    """Side-by-side comparison of a single orbit/downlink combination."""

    orbit: str
    downlink_profile: str
    ground_chain: str
    without_obdp_minutes: float
    with_obdp_minutes: float
    speedup_factor: float = Field(
        description="without / with",
    )
    time_saved_minutes: float
    actionability_without: str = Field(
        description='"low", "medium", or "high"',
    )
    actionability_with: str = Field(
        description='"low", "medium", or "high"',
    )


class ActionabilityResult(BaseModel):
    """How useful a detection is, given the latency."""

    latency_minutes: float
    vessel_speed_knots: float
    distance_moved_km: float
    actionability: str = Field(
        description='"high" (<10 km), "medium" (10-50 km), "low" (>50 km)',
    )
    search_radius_km: float = Field(
        description="Radius needed to re-locate the vessel",
    )
    notes: str


# ====================================================================
# OrbitalLatencySimulator
# ====================================================================


class OrbitalLatencySimulator:
    """Simulates end-to-end latency with and without OBDP.

    The simulator is stateless; each method returns an independent result.
    """

    # ------------------------------------------------------------------
    # Without OBDP
    # ------------------------------------------------------------------

    def simulate_without_obdp(
        self,
        orbit: str = "sso_700",
        ground_chain: str = "standard_nrt",
        image_size_mb: float = 800.0,
        downlink_profile: str = "smallsat_xband",
    ) -> LatencyBreakdown:
        """Simulate latency when the raw image is downlinked and processed on the ground.

        Steps:
        1. Capture (reference t=0).
        2. On-board storage (negligible).
        3. Wait for ground-station contact window.
        4. Downlink the raw image.
        5. Ground ingest.
        6. Ground processing.
        7. Dissemination.

        Parameters
        ----------
        orbit:
            Key in ``ORBIT_PARAMS``.
        ground_chain:
            Key in ``GROUND_PROCESSING``.
        image_size_mb:
            Size of the raw SAR image.
        downlink_profile:
            Key in ``DOWNLINK_PROFILES``.

        Returns
        -------
        LatencyBreakdown
        """
        orb = self._get_orbit(orbit)
        gnd = self._get_ground_chain(ground_chain)
        dl = self._get_downlink(downlink_profile)

        # Wait: average is period/4 for LEO with global ground stations
        wait_s = float(orb["period_min"]) * 60.0 / 4.0

        # Downlink raw image
        bw_mb_s = float(dl["bandwidth_mbps"]) / 8.0
        downlink_s = image_size_mb / bw_mb_s if bw_mb_s > 0 else float("inf")

        # Ground processing chain
        ingest_s = float(gnd["ingest_minutes"]) * 60.0
        processing_s = float(gnd["processing_minutes"]) * 60.0
        dissemination_s = float(gnd["dissemination_minutes"]) * 60.0

        total_s = wait_s + downlink_s + ingest_s + processing_s + dissemination_s

        return LatencyBreakdown(
            scenario="without_obdp",
            orbit=orbit,
            capture_s=0.0,
            onboard_processing_s=0.0,
            wait_for_contact_s=round(wait_s, 2),
            downlink_s=round(downlink_s, 3),
            ground_ingest_s=round(ingest_s, 2),
            ground_processing_s=round(processing_s, 2),
            dissemination_s=round(dissemination_s, 2),
            total_seconds=round(total_s, 2),
            total_minutes=round(total_s / 60.0, 2),
        )

    # ------------------------------------------------------------------
    # With OBDP
    # ------------------------------------------------------------------

    def simulate_with_obdp(
        self,
        orbit: str = "sso_700",
        inference_ms: float = 150.0,
        result_size_kb: float = 10.0,
        downlink_profile: str = "smallsat_xband",
    ) -> LatencyBreakdown:
        """Simulate latency when OBDP processes the image on-board.

        Steps:
        1. Capture (reference t=0).
        2. On-board inference (< 1 second typically).
        3. Wait for ground-station contact window.
        4. Downlink the small result payload (~instant).
        5. Ground ingest (~1 min for small payload).
        6. Dissemination (~1 min).

        Ground processing is eliminated because inference happened on-board.

        Parameters
        ----------
        orbit:
            Key in ``ORBIT_PARAMS``.
        inference_ms:
            On-board inference time in milliseconds.
        result_size_kb:
            Size of the result to downlink (JSON + thumbnails).
        downlink_profile:
            Key in ``DOWNLINK_PROFILES``.

        Returns
        -------
        LatencyBreakdown
        """
        orb = self._get_orbit(orbit)
        dl = self._get_downlink(downlink_profile)

        onboard_s = inference_ms / 1000.0

        # Wait: same orbital mechanics apply
        wait_s = float(orb["period_min"]) * 60.0 / 4.0

        # Downlink tiny result
        bw_mb_s = float(dl["bandwidth_mbps"]) / 8.0
        result_mb = result_size_kb / 1024.0
        downlink_s = result_mb / bw_mb_s if bw_mb_s > 0 else float("inf")

        # Minimal ground processing: just ingest + disseminate
        ingest_s = 60.0   # ~1 min for a small payload
        dissemination_s = 60.0  # ~1 min

        total_s = onboard_s + wait_s + downlink_s + ingest_s + dissemination_s

        return LatencyBreakdown(
            scenario="with_obdp",
            orbit=orbit,
            capture_s=0.0,
            onboard_processing_s=round(onboard_s, 4),
            wait_for_contact_s=round(wait_s, 2),
            downlink_s=round(downlink_s, 6),
            ground_ingest_s=round(ingest_s, 2),
            ground_processing_s=0.0,
            dissemination_s=round(dissemination_s, 2),
            total_seconds=round(total_s, 2),
            total_minutes=round(total_s / 60.0, 2),
        )

    # ------------------------------------------------------------------
    # Compare all scenarios
    # ------------------------------------------------------------------

    def compare_scenarios(
        self,
        inference_ms: float,
        image_size_mb: float,
        result_size_kb: float,
    ) -> list[LatencyComparison]:
        """Generate a comparison table across all orbit x ground x downlink combos.

        Parameters
        ----------
        inference_ms:
            On-board inference time in milliseconds.
        image_size_mb:
            Raw image size.
        result_size_kb:
            Processed result size.

        Returns
        -------
        list[LatencyComparison]
            One entry per combination, sorted by speedup factor (descending).
        """
        comparisons: list[LatencyComparison] = []

        for orbit_key in ORBIT_PARAMS:
            for dl_key in DOWNLINK_PROFILES:
                for gnd_key in GROUND_PROCESSING:
                    without = self.simulate_without_obdp(
                        orbit=orbit_key,
                        ground_chain=gnd_key,
                        image_size_mb=image_size_mb,
                        downlink_profile=dl_key,
                    )
                    with_obdp = self.simulate_with_obdp(
                        orbit=orbit_key,
                        inference_ms=inference_ms,
                        result_size_kb=result_size_kb,
                        downlink_profile=dl_key,
                    )

                    speedup = (
                        without.total_minutes / with_obdp.total_minutes
                        if with_obdp.total_minutes > 0
                        else float("inf")
                    )
                    time_saved = without.total_minutes - with_obdp.total_minutes

                    act_without = self.calculate_actionability(
                        latency_minutes=without.total_minutes
                    )
                    act_with = self.calculate_actionability(
                        latency_minutes=with_obdp.total_minutes
                    )

                    comparisons.append(
                        LatencyComparison(
                            orbit=orbit_key,
                            downlink_profile=dl_key,
                            ground_chain=gnd_key,
                            without_obdp_minutes=without.total_minutes,
                            with_obdp_minutes=with_obdp.total_minutes,
                            speedup_factor=round(speedup, 2),
                            time_saved_minutes=round(time_saved, 2),
                            actionability_without=act_without.actionability,
                            actionability_with=act_with.actionability,
                        )
                    )

        comparisons.sort(key=lambda c: c.speedup_factor, reverse=True)
        return comparisons

    # ------------------------------------------------------------------
    # Actionability
    # ------------------------------------------------------------------

    def calculate_actionability(
        self,
        latency_minutes: float,
        vessel_speed_knots: float = 20.0,
    ) -> ActionabilityResult:
        """Determine how actionable a detection is given the latency.

        A vessel moving at *vessel_speed_knots* will have moved a certain
        distance by the time the detection reaches an operator.

        - < 10 km: high actionability (vessel still in the vicinity)
        - 10-50 km: medium (broad-area search needed)
        - > 50 km: low (vessel likely unreachable)

        Parameters
        ----------
        latency_minutes:
            Total end-to-end latency in minutes.
        vessel_speed_knots:
            Target vessel speed in knots.

        Returns
        -------
        ActionabilityResult
        """
        latency_hours = latency_minutes / 60.0
        # 1 knot = 1.852 km/h
        distance_km = vessel_speed_knots * latency_hours * 1.852

        if distance_km < 10.0:
            actionability = "high"
            notes = (
                f"Vessel has moved only {distance_km:.1f} km. "
                "Detection is highly actionable; vessel remains in the vicinity."
            )
        elif distance_km < 50.0:
            actionability = "medium"
            notes = (
                f"Vessel has moved {distance_km:.1f} km. "
                "Broad-area search is required to re-locate the vessel."
            )
        else:
            actionability = "low"
            notes = (
                f"Vessel has moved {distance_km:.1f} km. "
                "Detection has limited operational value; vessel is likely unreachable."
            )

        # Search radius: vessel could have gone in any direction
        search_radius = distance_km

        return ActionabilityResult(
            latency_minutes=round(latency_minutes, 2),
            vessel_speed_knots=vessel_speed_knots,
            distance_moved_km=round(distance_km, 2),
            actionability=actionability,
            search_radius_km=round(search_radius, 2),
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_orbit(name: str) -> dict[str, object]:
        """Look up orbit parameters by key."""
        if name not in ORBIT_PARAMS:
            available = ", ".join(ORBIT_PARAMS.keys())
            raise KeyError(f"Orbit '{name}' not found. Available: {available}")
        return ORBIT_PARAMS[name]

    @staticmethod
    def _get_ground_chain(name: str) -> dict[str, object]:
        """Look up ground-processing chain by key."""
        if name not in GROUND_PROCESSING:
            available = ", ".join(GROUND_PROCESSING.keys())
            raise KeyError(
                f"Ground chain '{name}' not found. Available: {available}"
            )
        return GROUND_PROCESSING[name]

    @staticmethod
    def _get_downlink(name: str) -> dict[str, object]:
        """Look up downlink profile by key."""
        if name not in DOWNLINK_PROFILES:
            available = ", ".join(DOWNLINK_PROFILES.keys())
            raise KeyError(
                f"Downlink profile '{name}' not found. Available: {available}"
            )
        return DOWNLINK_PROFILES[name]
