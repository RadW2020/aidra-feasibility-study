"""
Motor de decision autonomo.

En orbita, el sistema debe decidir sin intervencion humana:
1. Que modelo usar (segun recursos disponibles).
2. Si procesar o no (segun energia/bateria).
3. Que hacer si falla (fallback a CFAR o skip).
4. Si la salida es sospechosa (drift detection via Z-score).

Este modulo simula ese comportamiento de decision para demostrar
capacidad autonoma en la propuesta.

Usage:
    from src.orbital.decision_engine import DecisionEngine, DecisionConfig

    engine = DecisionEngine(
        models=model_list,
        energy_profiler=profiler,
    )
    decision = engine.decide_model(
        available_cpu=1.0,
        available_ram_mb=1024,
        available_energy_wh=5.0,
    )
    print(decision.action, decision.selected_model)
"""

from __future__ import annotations

import logging

import numpy as np
from pydantic import BaseModel, Field

from src.db.models import ExecutionRecord, ModelInfo
from src.orbital.energy import EnergyProfiler

logger = logging.getLogger(__name__)

__all__ = [
    "DecisionEngine",
    "DecisionConfig",
    "DecisionResult",
    "DriftResult",
    "OrbitSimulationResult",
]


# ====================================================================
# Pydantic models
# ====================================================================


class DecisionConfig(BaseModel):
    """Configuration for the autonomous decision engine."""

    min_battery_reserve_pct: float = Field(
        default=20.0,
        description="Do not process if battery below this percentage",
    )
    prefer_precision_over_speed: bool = Field(
        default=True,
        description="When True, prefer larger/more accurate models when resources allow",
    )
    enable_cfar_fallback: bool = Field(
        default=True,
        description="Allow fallback to CFAR (signal processing, no AI) when models do not fit",
    )
    drift_detection_enabled: bool = Field(
        default=True,
        description="Enable Z-score anomaly detection on execution results",
    )
    drift_z_threshold: float = Field(
        default=3.0,
        description="Z-score threshold above which drift is flagged",
    )
    max_consecutive_skips: int = Field(
        default=5,
        description="Alert ground if more than this many consecutive images are skipped",
    )


class DecisionResult(BaseModel):
    """Result of an autonomous model-selection decision."""

    action: str = Field(
        description='"process", "fallback_cfar", or "skip"',
    )
    selected_model: str | None = Field(
        default=None,
        description="Name of the selected model (None if action is skip)",
    )
    selected_profile: str | None = Field(
        default=None,
        description="Constraint profile applied",
    )
    reason: str = Field(
        description="Human-readable explanation of the decision",
    )
    estimated_energy_wh: float = Field(
        description="Estimated energy for this decision",
    )
    estimated_latency_ms: float = Field(
        description="Estimated inference latency",
    )
    confidence_estimate: str = Field(
        description='"high", "medium", or "low"',
    )


class DriftResult(BaseModel):
    """Result of drift detection analysis."""

    is_drifting: bool
    metric: str = Field(
        description='"num_detections", "avg_confidence", or "spatial"',
    )
    z_score: float
    recent_mean: float
    historical_mean: float
    recommendation: str = Field(
        description='"continue", "recalibrate", "switch_model", or "alert_ground"',
    )


class OrbitSimulationResult(BaseModel):
    """Result of a full orbit simulation with battery tracking."""

    total_images: int
    processed_images: int
    skipped_images: int
    cfar_fallback_count: int
    models_used: dict[str, int] = Field(
        description="Count of images processed per model",
    )
    battery_timeline: list[float] = Field(
        description="Battery level (Wh) at each decision step",
    )
    decisions: list[DecisionResult]
    final_battery_wh: float
    energy_efficiency: float = Field(
        description="processed_images / total_energy_consumed_wh",
    )


# ====================================================================
# Default model-energy heuristics (used when no profiler data exists)
# ====================================================================

# Approximate inference time (ms) and RAM (MB) per common model variant
_MODEL_HEURISTICS: dict[str, dict[str, float]] = {
    "yolov8n-sar": {"inference_ms": 50.0, "ram_mb": 200.0, "energy_wh": 0.00004},
    "yolov8s-sar": {"inference_ms": 120.0, "ram_mb": 400.0, "energy_wh": 0.0001},
    "yolov8m-sar": {"inference_ms": 300.0, "ram_mb": 800.0, "energy_wh": 0.00025},
    "yolov8n-sar-int8": {"inference_ms": 30.0, "ram_mb": 100.0, "energy_wh": 0.00002},
    "yolov8n-sar-pruned": {"inference_ms": 40.0, "ram_mb": 150.0, "energy_wh": 0.00003},
}


# ====================================================================
# DecisionEngine
# ====================================================================


class DecisionEngine:
    """Autonomous model-selection and anomaly-detection engine.

    Parameters
    ----------
    models:
        List of available model descriptors, ordered by size (ascending).
    energy_profiler:
        An ``EnergyProfiler`` instance for energy estimates.
    config:
        Decision-engine configuration.  Uses defaults if not given.
    """

    def __init__(
        self,
        models: list[ModelInfo],
        energy_profiler: EnergyProfiler | None = None,
        config: DecisionConfig | None = None,
    ) -> None:
        self.models = sorted(models, key=lambda m: m.size_mb)
        self.energy_profiler = energy_profiler or EnergyProfiler()
        self.config = config or DecisionConfig()
        logger.info(
            "DecisionEngine initialised with %d models, config=%s",
            len(self.models),
            self.config.model_dump_json(indent=None),
        )

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def decide_model(
        self,
        available_cpu: float,
        available_ram_mb: int,
        available_energy_wh: float,
        priority: int = 0,
        battery_capacity_wh: float = 60.0,
    ) -> DecisionResult:
        """Choose the best model given current resource constraints.

        Decision logic:
        1. Filter models that fit in available RAM.
        2. Filter by energy (discard if one inference exceeds available energy).
        3. Priority 2 (urgent): pick the fastest (lowest latency).
        4. Priority 1 (high): pick the most accurate that fits.
        5. Priority 0 (normal): maximise precision-per-watt.
        6. If no model fits: fallback to CFAR (no AI, signal processing only).
        7. If not even CFAR: skip (save image for later downlink).

        Parameters
        ----------
        available_cpu:
            Available CPU cores.
        available_ram_mb:
            Available RAM in megabytes.
        available_energy_wh:
            Remaining battery energy in watt-hours.
        priority:
            0 = normal, 1 = high, 2 = urgent.
        battery_capacity_wh:
            Total battery capacity for reserve-percentage check.

        Returns
        -------
        DecisionResult
        """
        # Check battery reserve
        battery_pct = (available_energy_wh / battery_capacity_wh * 100.0) if battery_capacity_wh > 0 else 0.0
        if battery_pct < self.config.min_battery_reserve_pct:
            return DecisionResult(
                action="skip",
                selected_model=None,
                selected_profile=None,
                reason=(
                    f"Battery at {battery_pct:.1f}% (< {self.config.min_battery_reserve_pct}% reserve). "
                    "Skipping to preserve power for critical subsystems."
                ),
                estimated_energy_wh=0.0,
                estimated_latency_ms=0.0,
                confidence_estimate="low",
            )

        # Filter models by RAM
        ram_candidates = [
            m for m in self.models
            if m.size_mb * 3.0 <= available_ram_mb  # model + activations ~ 3x model size
        ]

        # Filter by energy
        energy_candidates: list[tuple[ModelInfo, float, float]] = []
        for model in ram_candidates:
            heuristic = _MODEL_HEURISTICS.get(model.name, {})
            est_energy_wh = heuristic.get("energy_wh", model.size_mb * 0.00005)
            est_latency_ms = heuristic.get("inference_ms", model.size_mb * 5.0)
            if est_energy_wh <= available_energy_wh:
                energy_candidates.append((model, est_energy_wh, est_latency_ms))

        if not energy_candidates:
            # No AI model fits — try CFAR fallback
            if self.config.enable_cfar_fallback:
                cfar_energy = 0.00001  # CFAR is very lightweight
                if cfar_energy <= available_energy_wh:
                    return DecisionResult(
                        action="fallback_cfar",
                        selected_model="cfar",
                        selected_profile=None,
                        reason=(
                            "No AI model fits within RAM/energy constraints. "
                            "Falling back to CFAR signal-processing detection."
                        ),
                        estimated_energy_wh=cfar_energy,
                        estimated_latency_ms=10.0,
                        confidence_estimate="low",
                    )
            return DecisionResult(
                action="skip",
                selected_model=None,
                selected_profile=None,
                reason=(
                    "No model (including CFAR) fits within available resources. "
                    f"RAM={available_ram_mb} MB, energy={available_energy_wh:.6f} Wh."
                ),
                estimated_energy_wh=0.0,
                estimated_latency_ms=0.0,
                confidence_estimate="low",
            )

        # Select based on priority
        if priority >= 2:
            # Urgent: pick fastest
            selected, energy, latency = min(energy_candidates, key=lambda t: t[2])
            reason = f"URGENT priority: selected fastest model '{selected.name}' (latency ~{latency:.0f} ms)."
            confidence = "medium"
        elif priority == 1:
            # High: pick most accurate (largest that fits)
            selected, energy, latency = max(energy_candidates, key=lambda t: t[0].size_mb)
            reason = f"HIGH priority: selected most accurate model '{selected.name}' ({selected.size_mb:.1f} MB)."
            confidence = "high"
        else:
            # Normal: maximise precision per watt (size_mb as proxy for precision)
            best = max(
                energy_candidates,
                key=lambda t: t[0].size_mb / t[1] if t[1] > 0 else 0.0,
            )
            selected, energy, latency = best
            reason = (
                f"NORMAL priority: selected '{selected.name}' "
                f"(best precision/energy ratio: {selected.size_mb:.1f} MB / {energy:.6f} Wh)."
            )
            confidence = "high" if selected.size_mb > 10.0 else "medium"

        return DecisionResult(
            action="process",
            selected_model=selected.name,
            selected_profile=None,
            reason=reason,
            estimated_energy_wh=round(energy, 8),
            estimated_latency_ms=round(latency, 2),
            confidence_estimate=confidence,
        )

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def detect_drift(
        self,
        recent_executions: list[ExecutionRecord],
        window_size: int = 10,
    ) -> DriftResult:
        """Detect anomalies in recent executions using Z-score analysis.

        Compares the last *window_size* executions against the full
        historical distribution.  If Z > threshold on any metric, drift
        is flagged.

        Parameters
        ----------
        recent_executions:
            List of execution records, ordered by time (newest first).
        window_size:
            Number of recent executions to treat as the "recent" window.

        Returns
        -------
        DriftResult
        """
        if not self.config.drift_detection_enabled:
            return DriftResult(
                is_drifting=False,
                metric="disabled",
                z_score=0.0,
                recent_mean=0.0,
                historical_mean=0.0,
                recommendation="continue",
            )

        if len(recent_executions) < window_size + 5:
            return DriftResult(
                is_drifting=False,
                metric="insufficient_data",
                z_score=0.0,
                recent_mean=0.0,
                historical_mean=0.0,
                recommendation="continue",
            )

        # Split into recent vs. historical
        recent = recent_executions[:window_size]
        historical = recent_executions[window_size:]

        # Check num_detections
        recent_dets = np.array([r.num_detections for r in recent], dtype=np.float64)
        hist_dets = np.array([r.num_detections for r in historical], dtype=np.float64)

        drift_result = self._z_score_test(
            recent_vals=recent_dets,
            historical_vals=hist_dets,
            metric="num_detections",
        )
        if drift_result.is_drifting:
            return drift_result

        # Check avg_confidence
        recent_conf = np.array(
            [r.avg_confidence for r in recent if r.avg_confidence is not None],
            dtype=np.float64,
        )
        hist_conf = np.array(
            [r.avg_confidence for r in historical if r.avg_confidence is not None],
            dtype=np.float64,
        )
        if len(recent_conf) >= 3 and len(hist_conf) >= 3:
            drift_result = self._z_score_test(
                recent_vals=recent_conf,
                historical_vals=hist_conf,
                metric="avg_confidence",
            )
            if drift_result.is_drifting:
                return drift_result

        # No drift detected
        return DriftResult(
            is_drifting=False,
            metric="all",
            z_score=0.0,
            recent_mean=float(np.mean(recent_dets)),
            historical_mean=float(np.mean(hist_dets)),
            recommendation="continue",
        )

    # ------------------------------------------------------------------
    # Orbit simulation
    # ------------------------------------------------------------------

    def simulate_orbit_sequence(
        self,
        num_images: int = 20,
        initial_battery_wh: float = 60.0,
        solar_recharge_w: float = 5.0,
        orbit_period_min: float = 95.0,
        image_interval_min: float = 10.0,
        battery_capacity_wh: float = 60.0,
    ) -> OrbitSimulationResult:
        """Simulate a sequence of autonomous decisions during one orbit.

        Every *image_interval_min* minutes the sensor captures an image.
        The decision engine decides whether to process (and with which
        model) or skip.  The battery drains with each processing and
        recharges from solar panels (during sunlit portions).

        Parameters
        ----------
        num_images:
            Number of images captured during the simulation.
        initial_battery_wh:
            Starting battery level in watt-hours.
        solar_recharge_w:
            Solar-panel recharge power in watts (applied during sunlit time).
        orbit_period_min:
            Orbital period in minutes.
        image_interval_min:
            Time between image captures in minutes.
        battery_capacity_wh:
            Maximum battery capacity in watt-hours.

        Returns
        -------
        OrbitSimulationResult
        """
        battery_wh = initial_battery_wh
        timeline: list[float] = [battery_wh]
        decisions: list[DecisionResult] = []
        models_used: dict[str, int] = {}
        processed = 0
        skipped = 0
        cfar_count = 0
        total_energy_consumed = 0.0
        consecutive_skips = 0

        sunlit_fraction = 0.6  # Typical LEO

        for i in range(num_images):
            # Determine if currently in sunlight (simple model)
            time_in_orbit_min = (i * image_interval_min) % orbit_period_min
            is_sunlit = time_in_orbit_min < (orbit_period_min * sunlit_fraction)

            # Solar recharge since last image
            if i > 0:
                recharge_hours = image_interval_min / 60.0
                recharge_wh = solar_recharge_w * recharge_hours if is_sunlit else 0.0
                battery_wh = min(battery_capacity_wh, battery_wh + recharge_wh)

            # Make decision
            decision = self.decide_model(
                available_cpu=1.0,
                available_ram_mb=1024,
                available_energy_wh=battery_wh,
                priority=0,
                battery_capacity_wh=battery_capacity_wh,
            )

            # Apply decision
            if decision.action == "process":
                battery_wh -= decision.estimated_energy_wh
                total_energy_consumed += decision.estimated_energy_wh
                processed += 1
                model_name = decision.selected_model or "unknown"
                models_used[model_name] = models_used.get(model_name, 0) + 1
                consecutive_skips = 0
            elif decision.action == "fallback_cfar":
                battery_wh -= decision.estimated_energy_wh
                total_energy_consumed += decision.estimated_energy_wh
                cfar_count += 1
                models_used["cfar"] = models_used.get("cfar", 0) + 1
                consecutive_skips = 0
            else:
                skipped += 1
                consecutive_skips += 1
                if consecutive_skips > self.config.max_consecutive_skips:
                    logger.warning(
                        "Alert: %d consecutive images skipped (> max %d). "
                        "Consider alerting ground station.",
                        consecutive_skips,
                        self.config.max_consecutive_skips,
                    )

            battery_wh = max(0.0, battery_wh)
            timeline.append(round(battery_wh, 6))
            decisions.append(decision)

        energy_efficiency = (
            processed / total_energy_consumed
            if total_energy_consumed > 0
            else 0.0
        )

        return OrbitSimulationResult(
            total_images=num_images,
            processed_images=processed,
            skipped_images=skipped,
            cfar_fallback_count=cfar_count,
            models_used=models_used,
            battery_timeline=[round(b, 4) for b in timeline],
            decisions=decisions,
            final_battery_wh=round(battery_wh, 6),
            energy_efficiency=round(energy_efficiency, 2),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _z_score_test(
        self,
        recent_vals: np.ndarray,
        historical_vals: np.ndarray,
        metric: str,
    ) -> DriftResult:
        """Run a Z-score test comparing recent values against history."""
        hist_mean = float(np.mean(historical_vals))
        hist_std = float(np.std(historical_vals))
        recent_mean = float(np.mean(recent_vals))

        if hist_std == 0:
            z_score = 0.0 if recent_mean == hist_mean else float("inf")
        else:
            z_score = abs(recent_mean - hist_mean) / hist_std

        is_drifting = z_score > self.config.drift_z_threshold

        if is_drifting:
            if z_score > self.config.drift_z_threshold * 2:
                recommendation = "alert_ground"
            elif z_score > self.config.drift_z_threshold * 1.5:
                recommendation = "switch_model"
            else:
                recommendation = "recalibrate"
            logger.warning(
                "Drift detected on '%s': z=%.2f (threshold=%.2f), "
                "recent_mean=%.2f, hist_mean=%.2f",
                metric,
                z_score,
                self.config.drift_z_threshold,
                recent_mean,
                hist_mean,
            )
        else:
            recommendation = "continue"

        return DriftResult(
            is_drifting=is_drifting,
            metric=metric,
            z_score=round(z_score, 4),
            recent_mean=round(recent_mean, 4),
            historical_mean=round(hist_mean, 4),
            recommendation=recommendation,
        )
