"""
Orbital value endpoints: energy profiling, downlink analysis,
latency simulation, and resilience testing.

These endpoints expose the orbital differentiator modules (M9-M12)
that quantify the operational value of AI-OBDP for satellite deployment.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.db.connection import db
from src.db.queries import SELECT_BENCHMARKS_BY_MODEL

logger = logging.getLogger("aidra.api.orbital")

router = APIRouter(prefix="/orbital", tags=["orbital"])


# ---------------------------------------------------------------------------
# Energy profiling (M9)
# ---------------------------------------------------------------------------


@router.get("/energy")
async def energy_profile(
    model: str | None = Query(None, description="Model name filter"),
    profile: str | None = Query(None, description="Constraint profile filter"),
    processor: str = Query("oci_arm_a1", description="Reference processor for estimation"),
) -> list[dict[str, Any]]:
    """Estimate energy consumption per model variant and profile.

    Uses execution_log data (inference_ms, cpu_usage_pct) combined with
    reference processor TDP to estimate joules per inference.
    """
    try:
        from src.orbital.energy import EnergyProfiler

        profiler = EnergyProfiler(reference_processor=processor)

        rows = await db.fetch(SELECT_BENCHMARKS_BY_MODEL, model, profile)
        if not rows:
            return []

        results = []
        for row in rows:
            if row["avg_inference_ms"] is None:
                continue
            cpu_time_s = row["avg_inference_ms"] / 1000.0
            cpu_pct = row["avg_cpu_pct"] or 50.0
            estimate = profiler.estimate_inference_energy(
                cpu_time_seconds=cpu_time_s,
                cpu_cores_used=cpu_pct / 100.0 * 4,  # Scale by utilization
                processor=processor,
            )
            results.append({
                "model_name": row["model_name"],
                "model_version": row["model_version"],
                "constraint_profile": row["constraint_profile"],
                "processor": processor,
                "inference_ms": row["avg_inference_ms"],
                **estimate.model_dump(),
            })

            # Emit Prometheus metric
            from src.observability.prometheus_metrics import ENERGY_JOULES
            ENERGY_JOULES.labels(
                profile=row["constraint_profile"],
                model_variant=row["model_name"],
                processor=processor,
            ).set(estimate.energy_joules)

        return results

    except ImportError:
        raise HTTPException(503, "Orbital energy module not available") from None


@router.get("/energy/budget")
async def orbital_budget(
    model: str = Query(..., description="Model name"),
    profile: str = Query("ground", description="Constraint profile"),
    satellite: str = Query("cubesat_6u", description="Satellite type"),
    processor: str = Query("xilinx_zynq_ultrascale", description="Target processor"),
) -> dict[str, Any]:
    """Calculate orbital energy budget: how many images can be processed per orbit."""
    try:
        from src.orbital.energy import EnergyProfiler

        profiler = EnergyProfiler(reference_processor=processor)

        rows = await db.fetch(SELECT_BENCHMARKS_BY_MODEL, model, profile)
        if not rows:
            raise HTTPException(404, f"No benchmark data for model={model}, profile={profile}")

        row = rows[0]
        cpu_time_s = row["avg_inference_ms"] / 1000.0
        energy = profiler.estimate_inference_energy(
            cpu_time_seconds=cpu_time_s,
            cpu_cores_used=1.0,
            processor=processor,
        )
        budget = profiler.calculate_orbital_budget(
            energy_per_image_joules=energy.energy_joules,
            satellite_type=satellite,
        )

        # Emit metrics
        from src.observability.prometheus_metrics import IMAGES_PER_ORBIT, TOPS_PER_WATT
        tops = profiler.calculate_tops_per_watt(
            model_flops=8_700_000_000,  # ~8.7 GFLOPs for YOLOv8n
            inference_seconds=cpu_time_s,
            processor=processor,
        )
        TOPS_PER_WATT.labels(model_variant=model, processor=processor).set(tops)
        IMAGES_PER_ORBIT.labels(
            model_variant=model, profile=profile, satellite_type=satellite,
        ).set(budget.max_images_per_orbit)

        return budget.model_dump()

    except ImportError:
        raise HTTPException(503, "Orbital energy module not available") from None
    except KeyError as e:
        raise HTTPException(400, f"Unknown satellite type or processor: {e}") from e


# ---------------------------------------------------------------------------
# Downlink analysis (M10)
# ---------------------------------------------------------------------------


@router.get("/downlink")
async def downlink_analysis(
    image_id: str | None = Query(None, description="Execution image_id to look up sizes from DB"),
    image_size_mb: float = Query(800.0, description="SAR image size in MB (used if no image_id)"),
    result_size_kb: float = Query(10.0, description="Processed result size in KB (used if no image_id)"),
    downlink_profile: str | None = Query(None, description="Specific downlink profile"),
) -> list[dict[str, Any]]:
    """Compare downlink requirements with vs without OBDP.

    If image_id is provided, looks up actual image_size_mb and result size
    from execution_log. Otherwise uses the explicit parameters.
    """
    try:
        from src.orbital.downlink import DownlinkAnalyzer

        analyzer = DownlinkAnalyzer()

        # Look up from DB if image_id provided
        if image_id:
            row = await db.fetchrow(
                "SELECT image_size_mb, num_detections FROM execution_log "
                "WHERE image_id = $1 AND status = 'success' LIMIT 1",
                image_id,
            )
            if row and row["image_size_mb"]:
                image_size_mb = row["image_size_mb"]
                # Estimate result size: ~1KB base + 0.5KB per detection
                result_size_kb = 1.0 + (row["num_detections"] or 0) * 0.5

        if downlink_profile:
            result = analyzer.analyze_single_image(
                image_size_mb=image_size_mb,
                result_size_kb=result_size_kb,
                downlink_profile=downlink_profile,
            )
            return [result.model_dump()]
        else:
            results = analyzer.analyze_all_profiles(
                image_size_mb=image_size_mb,
                result_size_kb=result_size_kb,
            )
            return [r.model_dump() for r in results]

    except ImportError:
        raise HTTPException(503, "Orbital downlink module not available") from None
    except KeyError as e:
        raise HTTPException(400, f"Unknown downlink profile: {e}") from e


@router.get("/downlink/value")
async def obdp_value_report() -> dict[str, Any]:
    """Generate OBDP value report from real execution data."""
    try:
        from src.orbital.downlink import DownlinkAnalyzer

        analyzer = DownlinkAnalyzer()

        rows = await db.fetch(
            "SELECT image_size_mb, num_detections, avg_confidence, id "
            "FROM execution_log WHERE status = 'success' AND image_size_mb IS NOT NULL "
            "ORDER BY created_at DESC LIMIT 100"
        )

        if not rows:
            return {"message": "No execution data available yet. Run the pipeline first."}

        # Build mock execution records for the report
        records = []
        for row in rows:
            records.append(type("R", (), {
                "image_size_mb": row["image_size_mb"],
                "num_detections": row["num_detections"],
                "avg_confidence": row["avg_confidence"],
                "id": row["id"],
            })())

        report = analyzer.generate_obdp_value_report(records)
        return report.model_dump()

    except ImportError:
        raise HTTPException(503, "Orbital downlink module not available") from None


# ---------------------------------------------------------------------------
# Latency simulation (M11)
# ---------------------------------------------------------------------------


@router.get("/latency")
async def latency_comparison(
    orbit: str | None = Query(None, description="Orbit type (leo_500, sso_700, leo_350_isstyle)"),
    downlink: str | None = Query(None, description="Downlink profile (cubesat_uhf, cubesat_sband, smallsat_xband, highcap_ka)"),
    inference_ms: float = Query(150.0, description="On-board inference time in ms"),
    image_size_mb: float = Query(800.0, description="Image size in MB"),
    result_size_kb: float = Query(10.0, description="Result size in KB"),
) -> list[dict[str, Any]]:
    """Compare latency scenarios with vs without OBDP."""
    try:
        from src.orbital.latency import OrbitalLatencySimulator

        simulator = OrbitalLatencySimulator()
        comparisons = simulator.compare_scenarios(
            inference_ms=inference_ms,
            image_size_mb=image_size_mb,
            result_size_kb=result_size_kb,
        )

        results = [c.model_dump() for c in comparisons]

        if orbit:
            results = [r for r in results if r.get("orbit") == orbit]
        if downlink:
            results = [r for r in results if r.get("downlink_profile") == downlink]

        return results

    except ImportError:
        raise HTTPException(503, "Orbital latency module not available") from None


# ---------------------------------------------------------------------------
# Resilience (M12)
# ---------------------------------------------------------------------------


class BitFlipRequest(BaseModel):
    """Request body for bit-flip resilience sweep."""
    model: str = "yolov8n-sar"
    flip_counts: list[int] = [0, 1, 5, 10, 50, 100]
    runs_per_count: int = 3


@router.post("/resilience/bitflip")
async def bitflip_sweep(request: BitFlipRequest) -> dict[str, Any]:
    """Run bit-flip resilience sweep on a model.

    Injects increasing numbers of bit-flips into model weights
    and measures detection degradation.
    """
    try:
        from src.config import Settings
        from src.models.yolo import YOLODetector
        from src.orbital.resilience import BitFlipSimulator
        from src.pipeline.preprocessing import generate_synthetic_sar_tile

        settings = Settings()
        from pathlib import Path
        model_path = Path(settings.models_dir) / f"{request.model}.pt"
        if not model_path.exists():
            raise HTTPException(404, f"Model file not found: {model_path}")

        detector = YOLODetector(model_path=model_path)

        # Get weights via public API
        weights = detector.get_weights_dict()

        simulator = BitFlipSimulator(model_weights=weights)

        # Generate test image
        test_image, _ = generate_synthetic_sar_tile(size=640, num_vessels=5, seed=42)

        counts = request.flip_counts

        # Run baseline inference to get baseline metrics
        baseline_dets = detector.predict(test_image)
        baseline_detections = len(baseline_dets)
        baseline_confidence = (
            float(sum(d["confidence"] for d in baseline_dets) / len(baseline_dets))
            if baseline_dets else 0.0
        )

        result = simulator.sweep_bitflips(
            image=test_image,
            model=detector,
            flip_counts=counts,
            runs_per_count=request.runs_per_count,
            baseline_detections=baseline_detections,
            baseline_confidence=baseline_confidence,
            model_name=request.model,
        )

        # Emit BITFLIP_DEGRADATION metrics
        from src.observability.prometheus_metrics import BITFLIP_DEGRADATION
        for entry in result.results:
            BITFLIP_DEGRADATION.labels(
                num_flips=str(entry["num_flips"]),
                model_variant=request.model,
            ).set(entry.get("degradation_pct", 0.0))

        return result.model_dump()

    except ImportError as e:
        raise HTTPException(503, f"Resilience module not available: {e}") from e


class OrbitSimulationRequest(BaseModel):
    """Request body for orbit simulation."""
    num_images: int = 20
    satellite: str = "cubesat_6u"


@router.post("/resilience/simulate-orbit")
async def simulate_orbit(request: OrbitSimulationRequest) -> dict[str, Any]:
    """Simulate a full orbit with autonomous decision-making."""
    try:
        from src.orbital.decision_engine import DecisionConfig, DecisionEngine
        from src.orbital.orbit_params import SATELLITE_POWER_BUDGETS

        if request.satellite not in SATELLITE_POWER_BUDGETS:
            raise HTTPException(400, f"Unknown satellite: {request.satellite}. Options: {list(SATELLITE_POWER_BUDGETS.keys())}")

        sat = SATELLITE_POWER_BUDGETS[request.satellite]

        # Build model infos from registry
        from src.db.queries import SELECT_ALL_MODELS
        rows = await db.fetch(SELECT_ALL_MODELS)

        from src.db.models import ModelInfo
        models = []
        for row in rows:
            models.append(ModelInfo(
                id=row["id"],
                name=row["name"],
                version=row["version"],
                format=row["format"],
                file_hash=row["file_hash"],
                size_mb=row["size_mb"],
                num_params=row.get("num_params"),
            ))

        if not models:
            return {"message": "No models registered. Run the pipeline first to populate model registry."}

        config = DecisionConfig()
        engine = DecisionEngine(models=models, energy_profiler=None, config=config)

        result = engine.simulate_orbit_sequence(
            num_images=request.num_images,
            initial_battery_wh=sat["battery_wh"],
            solar_recharge_w=sat["total_w"] * sat["sunlit_fraction"] * 0.3,
            orbit_period_min=sat["orbit_period_min"],
        )

        # Emit battery metric
        from src.observability.prometheus_metrics import BATTERY_LEVEL_WH
        BATTERY_LEVEL_WH.set(result.final_battery_wh)

        return result.model_dump()

    except ImportError as e:
        raise HTTPException(503, f"Decision engine not available: {e}") from e


@router.get("/resilience/drift")
async def drift_status(
    window_size: int = Query(10, ge=3, le=100, description="Number of recent executions to analyze"),
) -> dict[str, Any]:
    """Check for drift in recent pipeline executions."""
    try:
        from src.orbital.decision_engine import DecisionConfig, DecisionEngine

        rows = await db.fetch(
            "SELECT * FROM execution_log WHERE status = 'success' "
            "ORDER BY created_at DESC LIMIT $1",
            window_size + 50,  # Get extra for historical comparison
        )

        if len(rows) < window_size:
            return {
                "status": "insufficient_data",
                "message": f"Need at least {window_size} successful executions for drift detection. Have {len(rows)}.",
            }

        from src.db.models import ExecutionRecord

        records = []
        for row in rows:
            records.append(ExecutionRecord(
                id=row["id"],
                created_at=row["created_at"],
                image_id=row["image_id"],
                image_hash=row["image_hash"],
                model_name=row["model_name"],
                model_version=row["model_version"],
                model_hash=row["model_hash"],
                model_size_mb=row["model_size_mb"],
                num_detections=row["num_detections"],
                avg_confidence=row["avg_confidence"],
                output_hash=row["output_hash"],
                status=row["status"],
                inference_ms=row.get("inference_ms"),
                peak_ram_mb=row.get("peak_ram_mb"),
                cpu_usage_pct=row.get("cpu_usage_pct"),
            ))

        config = DecisionConfig()
        engine = DecisionEngine(models=[], energy_profiler=None, config=config)
        result = engine.detect_drift(recent_executions=records, window_size=window_size)

        # Emit Prometheus metric if drifting
        if result.is_drifting:
            from src.observability.prometheus_metrics import DRIFT_ALERTS
            DRIFT_ALERTS.labels(metric=result.metric).inc()

        return result.model_dump()

    except ImportError as e:
        raise HTTPException(503, f"Decision engine not available: {e}") from e


class DecisionRequest(BaseModel):
    """Request body for decision engine query."""
    available_cpu: float = 1.0
    available_ram_mb: int = 1024
    available_energy_wh: float = 5.0
    priority: int = 0


@router.post("/decision")
async def query_decision_engine(request: DecisionRequest) -> dict[str, Any]:
    """Query the autonomous decision engine for model selection."""
    try:
        from src.db.queries import SELECT_ALL_MODELS
        from src.orbital.decision_engine import DecisionConfig, DecisionEngine

        rows = await db.fetch(SELECT_ALL_MODELS)

        from src.db.models import ModelInfo
        models = []
        for row in rows:
            models.append(ModelInfo(
                id=row["id"],
                name=row["name"],
                version=row["version"],
                format=row["format"],
                file_hash=row["file_hash"],
                size_mb=row["size_mb"],
                num_params=row.get("num_params"),
            ))

        if not models:
            return {"action": "skip", "reason": "No models registered", "selected_model": None}

        config = DecisionConfig()
        engine = DecisionEngine(models=models, energy_profiler=None, config=config)

        result = engine.decide_model(
            available_cpu=request.available_cpu,
            available_ram_mb=request.available_ram_mb,
            available_energy_wh=request.available_energy_wh,
            priority=request.priority,
        )

        # Emit Prometheus metric
        from src.observability.prometheus_metrics import DECISION_ACTION
        DECISION_ACTION.labels(action=result.action).inc()

        return result.model_dump()

    except ImportError as e:
        raise HTTPException(503, f"Decision engine not available: {e}") from e
