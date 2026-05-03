"""
Simulacion de resiliencia: que pasa cuando las cosas van mal en orbita.

Modela tres dimensiones de resiliencia:
1. **Bit-flips (SEU)**: corrupcion de pesos del modelo por radiacion
   cosmica (Single Event Upsets).
2. **Degradacion medible**: cuantos bit-flips puede tolerar un modelo
   antes de que las detecciones se degraden significativamente.
3. **MTBF (Mean Time Between Failures)**: estimacion de cuanto tiempo
   puede operar un modelo en orbita antes de necesitar recarga de pesos.

Usage:
    import numpy as np
    from src.orbital.resilience import BitFlipSimulator

    weights = {"layer1": np.random.randn(64, 64).astype(np.float32)}
    sim = BitFlipSimulator(model_weights=weights)
    corrupted, records = sim.inject_bitflips(num_flips=5)
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from src.orbital.orbit_params import ORBIT_PARAMS

logger = logging.getLogger(__name__)

__all__ = [
    "BitFlipSimulator",
    "BitFlipRecord",
    "BitFlipSweepResult",
    "MTBFEstimate",
]

# SEU rates by orbit (bit-flips per bit per day).
# Based on published data for commercial SRAM behind typical Al shielding.
_SEU_RATES: dict[str, float] = {
    "leo_500": 1e-6,
    "sso_700": 5e-6,
    "leo_350_isstyle": 5e-7,
}

# Shielding attenuation factor per mm of aluminium (simplified model).
# Each mm of Al reduces the SEU rate by roughly a factor of 2-3.
_SHIELDING_ATTENUATION_PER_MM: float = 0.4


# ====================================================================
# Pydantic models
# ====================================================================


class BitFlipRecord(BaseModel):
    """Record of a single injected bit-flip."""

    layer_name: str
    tensor_index: list[int] = Field(
        description="Multi-dimensional index into the weight tensor",
    )
    original_value: float
    corrupted_value: float
    bit_position: int = Field(
        description="0-31 for float32",
    )
    bit_significance: str = Field(
        description='"sign", "exponent", or "mantissa"',
    )


class BitFlipSweepResult(BaseModel):
    """Result of sweeping multiple bit-flip counts."""

    baseline_detections: int
    baseline_confidence: float
    results: list[dict[str, Any]] = Field(
        description=(
            "List of dicts: {num_flips, avg_detections, avg_confidence, "
            "std_detections, degradation_pct}"
        ),
    )
    critical_threshold: int = Field(
        description="Number of flips where degradation > 20%",
    )
    model_name: str
    model_size_bytes: int


class MTBFEstimate(BaseModel):
    """Estimated Mean Time Between Failures for a model in orbit."""

    orbit: str
    model_size_bits: int
    seu_rate_per_bit_per_day: float
    expected_flips_per_day: float
    expected_flips_per_orbit: float
    critical_threshold: int = Field(
        description="From BitFlipSweepResult.critical_threshold",
    )
    estimated_mtbf_days: float = Field(
        description="Days until accumulated flips reach critical_threshold",
    )
    mitigation_recommendations: list[str]


# ====================================================================
# BitFlipSimulator
# ====================================================================


class BitFlipSimulator:
    """Simulates Single Event Upsets (SEU) in model weight tensors.

    In LEO the SEU rate is approximately 1e-7 to 1e-5 bit-flips per bit
    per day, depending on orbit altitude and shielding thickness.

    Parameters
    ----------
    model_weights:
        Dictionary mapping layer names to numpy arrays (float32).
        Typically the ``state_dict`` of a PyTorch model converted to numpy.
    """

    def __init__(self, model_weights: dict[str, np.ndarray]) -> None:
        if not model_weights:
            raise ValueError("model_weights must be a non-empty dict")
        self._original_weights = model_weights
        self._layer_names = list(model_weights.keys())
        self._total_params = sum(w.size for w in model_weights.values())
        self._total_bytes = sum(w.nbytes for w in model_weights.values())
        logger.info(
            "BitFlipSimulator initialised: %d layers, %d params, %d bytes",
            len(self._layer_names),
            self._total_params,
            self._total_bytes,
        )

    # ------------------------------------------------------------------
    # Inject bit-flips
    # ------------------------------------------------------------------

    def inject_bitflips(
        self,
        num_flips: int = 1,
        target_layers: list[str] | None = None,
        bit_position: str = "random",
        rng: np.random.Generator | None = None,
    ) -> tuple[dict[str, np.ndarray], list[BitFlipRecord]]:
        """Inject *num_flips* bit-flips into a copy of the model weights.

        Parameters
        ----------
        num_flips:
            Number of independent bit-flips to inject.
        target_layers:
            If given, only flip bits in these layers.  Otherwise select
            layers uniformly at random.
        bit_position:
            ``"random"`` (uniform 0-31), ``"msb"`` (bit 31, sign bit --
            most destructive), or ``"lsb"`` (bit 0, least destructive).
        rng:
            Optional numpy random generator for reproducibility.

        Returns
        -------
        tuple[dict[str, np.ndarray], list[BitFlipRecord]]
            ``(corrupted_weights, flip_records)``
        """
        if rng is None:
            rng = np.random.default_rng()

        corrupted = deepcopy(self._original_weights)
        records: list[BitFlipRecord] = []
        available_layers = target_layers if target_layers else self._layer_names

        for _ in range(num_flips):
            # 1. Select random layer
            layer_name = available_layers[rng.integers(0, len(available_layers))]
            tensor = corrupted[layer_name]

            # Ensure float32
            if tensor.dtype != np.float32:
                tensor = tensor.astype(np.float32)
                corrupted[layer_name] = tensor

            # 2. Select random element
            flat_idx = rng.integers(0, tensor.size)
            multi_idx = np.unravel_index(flat_idx, tensor.shape)

            # 3. Get binary representation (view float32 as uint32)
            original_val = float(tensor[multi_idx])
            uint_view = tensor.view(np.uint32)
            original_bits = uint_view[multi_idx]

            # 4. Choose bit to flip
            if bit_position == "msb":
                bit_pos = 31
            elif bit_position == "lsb":
                bit_pos = 0
            else:
                bit_pos = int(rng.integers(0, 32))

            # 5. XOR to flip the bit
            mask = np.uint32(1 << bit_pos)
            flipped_bits = original_bits ^ mask
            uint_view[multi_idx] = flipped_bits

            # 6. Read back the corrupted value
            corrupted_val = float(tensor[multi_idx])

            # Determine bit significance
            significance = self._bit_significance(bit_pos)

            records.append(
                BitFlipRecord(
                    layer_name=layer_name,
                    tensor_index=list(int(i) for i in multi_idx),
                    original_value=original_val,
                    corrupted_value=corrupted_val,
                    bit_position=bit_pos,
                    bit_significance=significance,
                )
            )

        logger.debug("Injected %d bit-flip(s) into %d layer(s)", num_flips, len(available_layers))
        return corrupted, records

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------

    def sweep_bitflips(
        self,
        image: np.ndarray,
        model: Any,
        flip_counts: list[int] | None = None,
        runs_per_count: int = 5,
        baseline_detections: int | None = None,
        baseline_confidence: float | None = None,
        model_name: str = "unknown",
    ) -> BitFlipSweepResult:
        """Run inference at multiple bit-flip counts and measure degradation.

        The caller must provide a *model* object with the following
        interface:

        - ``model.load_weights(weights: dict[str, np.ndarray]) -> None``
        - ``model.predict(image: np.ndarray) -> dict`` returning at
          least ``{"num_detections": int, "avg_confidence": float}``.

        If the model does not implement this interface, pass
        ``baseline_detections`` and ``baseline_confidence`` and the sweep
        will run the bit-flip injection and track weight-delta statistics
        instead of running actual inference.

        Parameters
        ----------
        image:
            Input image as a numpy array.
        model:
            Model object (see interface above).
        flip_counts:
            List of flip counts to test.
        runs_per_count:
            Number of independent runs per flip count for averaging.
        baseline_detections:
            Pre-computed baseline detection count (skip baseline inference).
        baseline_confidence:
            Pre-computed baseline confidence.
        model_name:
            Name of the model for reporting.

        Returns
        -------
        BitFlipSweepResult
        """
        if flip_counts is None:
            flip_counts = [0, 1, 5, 10, 50, 100, 500, 1000]

        rng = np.random.default_rng(42)

        # Run baseline if not provided
        # Check for model inference capability — support both load_weights and load_weights_dict
        _load_fn = None
        if hasattr(model, "load_weights_dict"):
            _load_fn = model.load_weights_dict
        elif hasattr(model, "load_weights"):
            _load_fn = model.load_weights
        has_predict = hasattr(model, "predict") and _load_fn is not None

        if baseline_detections is None or baseline_confidence is None:
            if has_predict:
                _load_fn(self._original_weights)
                baseline_preds = model.predict(image)
                if isinstance(baseline_preds, list):
                    baseline_detections = len(baseline_preds)
                    baseline_confidence = float(
                        sum(d.get("confidence", 0) for d in baseline_preds) / max(len(baseline_preds), 1)
                    )
                else:
                    baseline_detections = int(baseline_preds.get("num_detections", 0))
                    baseline_confidence = float(baseline_preds.get("avg_confidence", 0.0))
            else:
                baseline_detections = baseline_detections or 0
                baseline_confidence = baseline_confidence or 0.0

        results: list[dict[str, Any]] = []
        critical_threshold = flip_counts[-1] if flip_counts else 0

        for n_flips in flip_counts:
            det_counts: list[int] = []
            conf_values: list[float] = []

            for _ in range(runs_per_count):
                if n_flips == 0:
                    det_counts.append(baseline_detections)
                    conf_values.append(baseline_confidence)
                    continue

                corrupted_weights, _ = self.inject_bitflips(
                    num_flips=n_flips,
                    rng=rng,
                )

                if has_predict:
                    _load_fn(corrupted_weights)
                    pred = model.predict(image)
                    if isinstance(pred, list):
                        det_counts.append(len(pred))
                        conf_values.append(float(
                            sum(d.get("confidence", 0) for d in pred) / max(len(pred), 1)
                        ) if pred else 0.0)
                    else:
                        det_counts.append(int(pred.get("num_detections", 0)))
                        conf_values.append(float(pred.get("avg_confidence", 0.0)))
                else:
                    # Statistical estimation: weight perturbation magnitude
                    # correlates with detection degradation
                    total_delta = 0.0
                    total_elements = 0
                    for layer_name, orig in self._original_weights.items():
                        corr = corrupted_weights[layer_name]
                        diff = np.abs(corr - orig)
                        # Bit-flips can produce inf/nan; treat those as large perturbations
                        diff = np.where(np.isfinite(diff), diff, 1e6)
                        total_delta += float(np.sum(diff))
                        total_elements += orig.size
                    avg_perturbation = total_delta / total_elements if total_elements > 0 else 0.0

                    # Heuristic degradation model: detections degrade
                    # proportionally to perturbation magnitude
                    degradation_factor = min(1.0, avg_perturbation * 10.0)
                    est_detections = max(0, int(baseline_detections * (1.0 - degradation_factor)))
                    est_confidence = max(0.0, baseline_confidence * (1.0 - degradation_factor * 0.5))
                    det_counts.append(est_detections)
                    conf_values.append(est_confidence)

            avg_det = float(np.mean(det_counts))
            avg_conf = float(np.mean(conf_values))
            std_det = float(np.std(det_counts))
            degradation_pct = (
                (1.0 - avg_det / baseline_detections) * 100.0
                if baseline_detections > 0
                else 0.0
            )

            results.append(
                {
                    "num_flips": n_flips,
                    "avg_detections": round(avg_det, 2),
                    "avg_confidence": round(avg_conf, 4),
                    "std_detections": round(std_det, 2),
                    "degradation_pct": round(degradation_pct, 2),
                }
            )

            # Track critical threshold (first time degradation > 20%)
            if degradation_pct > 20.0 and n_flips < critical_threshold:
                critical_threshold = n_flips

        return BitFlipSweepResult(
            baseline_detections=baseline_detections,
            baseline_confidence=baseline_confidence,
            results=results,
            critical_threshold=critical_threshold,
            model_name=model_name,
            model_size_bytes=self._total_bytes,
        )

    # ------------------------------------------------------------------
    # MTBF estimation
    # ------------------------------------------------------------------

    def estimate_mtbf(
        self,
        orbit: str = "leo_500",
        model_size_bytes: int | None = None,
        shielding_mm_al: float = 1.0,
        critical_threshold: int = 100,
    ) -> MTBFEstimate:
        """Estimate Mean Time Between Failures for the model in orbit.

        A "failure" is defined as accumulating enough bit-flips to
        exceed the *critical_threshold* (i.e., degradation > 20%).

        Parameters
        ----------
        orbit:
            Key in ``ORBIT_PARAMS``.
        model_size_bytes:
            Model size in bytes.  Defaults to the size of the loaded weights.
        shielding_mm_al:
            Aluminium shielding thickness in millimetres.
        critical_threshold:
            Number of bit-flips that cause > 20% degradation (from sweep).

        Returns
        -------
        MTBFEstimate
        """
        if orbit not in ORBIT_PARAMS:
            available = ", ".join(ORBIT_PARAMS.keys())
            raise KeyError(f"Orbit '{orbit}' not found. Available: {available}")

        if model_size_bytes is None:
            model_size_bytes = self._total_bytes

        model_size_bits = model_size_bytes * 8

        # Base SEU rate for the orbit
        base_rate = _SEU_RATES.get(orbit, 1e-6)

        # Apply shielding attenuation
        effective_rate = base_rate * (_SHIELDING_ATTENUATION_PER_MM ** shielding_mm_al)

        flips_per_day = effective_rate * model_size_bits
        orbit_period_min = float(ORBIT_PARAMS[orbit]["period_min"])
        orbits_per_day = 24.0 * 60.0 / orbit_period_min
        flips_per_orbit = flips_per_day / orbits_per_day

        # MTBF: days until accumulated flips reach critical_threshold
        mtbf_days = critical_threshold / flips_per_day if flips_per_day > 0 else float("inf")

        # Mitigation recommendations
        mitigations: list[str] = []
        if mtbf_days < 1.0:
            mitigations.append(
                "CRITICAL: Model corruption expected within hours. "
                "Use radiation-hardened memory or TMR (Triple Modular Redundancy)."
            )
        if mtbf_days < 30.0:
            mitigations.append(
                "Implement periodic weight re-loading from protected storage (e.g. NAND flash with ECC)."
            )
        if mtbf_days < 365.0:
            mitigations.append(
                "Use model quantisation (INT8) to reduce weight memory footprint and exposure."
            )
        mitigations.append(
            "Deploy checksum verification on weight tensors before each inference."
        )
        mitigations.append(
            "Consider using smaller models (fewer parameters = fewer exposed bits)."
        )
        if shielding_mm_al < 3.0:
            mitigations.append(
                f"Increase shielding from {shielding_mm_al} mm to 3+ mm Al "
                f"to reduce SEU rate by ~{(1.0 / _SHIELDING_ATTENUATION_PER_MM ** (3.0 - shielding_mm_al)):.1f}x."
            )

        return MTBFEstimate(
            orbit=orbit,
            model_size_bits=model_size_bits,
            seu_rate_per_bit_per_day=effective_rate,
            expected_flips_per_day=round(flips_per_day, 4),
            expected_flips_per_orbit=round(flips_per_orbit, 6),
            critical_threshold=critical_threshold,
            estimated_mtbf_days=round(mtbf_days, 2),
            mitigation_recommendations=mitigations,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bit_significance(bit_pos: int) -> str:
        """Classify a bit position in IEEE 754 float32.

        Bit 31 = sign, bits 30-23 = exponent, bits 22-0 = mantissa.
        """
        if bit_pos == 31:
            return "sign"
        if 23 <= bit_pos <= 30:
            return "exponent"
        return "mantissa"
