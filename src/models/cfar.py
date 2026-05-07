"""
Detector CFAR (Constant False Alarm Rate) para imagenes SAR.

El CFAR detecta pixeles cuyo valor de backscatter es significativamente
superior al fondo local.  Es el algoritmo estandar para detectar barcos
en SAR porque los barcos producen reflexiones metalicas muy fuertes
(corner reflector effect).

Algoritmo:
1. Para cada pixel "bajo test" (CUT):
   a. Definir ventana de guarda (guard cells): excluye pixeles adyacentes
   b. Definir ventana de entrenamiento (training cells): estima el fondo
   c. Calcular media/varianza del fondo en training cells
   d. Calcular umbral adaptativo: threshold = mean + k * std
   e. Si CUT > threshold -> deteccion

Variantes implementadas:
- CA-CFAR (Cell-Averaging): media simple del fondo
- OS-CFAR (Ordered Statistics): usa percentil del fondo (mas robusto a clutter)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import generic_filter, uniform_filter
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


class CFARDetector:
    """Constant False Alarm Rate detector for SAR imagery.

    Parameters
    ----------
    guard_size:
        Radius of the guard window around the cell under test (CUT).
    training_size:
        Radius of the training window (must be > guard_size).
    pfa:
        Desired probability of false alarm.  Lower values yield fewer
        false positives but may miss weaker targets.
    method:
        ``"ca"`` for Cell-Averaging CFAR or ``"os"`` for Ordered-Statistics
        CFAR.
    os_percentile:
        Percentile (0-1) used for OS-CFAR background estimation.  Ignored
        when *method* is ``"ca"``.
    """

    VALID_METHODS = ("ca", "os")
    VALID_DISTRIBUTIONS = ("exponential", "gaussian")

    def __init__(
        self,
        guard_size: int = 8,
        training_size: int = 20,
        pfa: float = 1e-5,
        method: str = "ca",
        os_percentile: float = 0.75,
        distribution: str = "exponential",
    ) -> None:
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Invalid CFAR method '{method}'. Must be one of {self.VALID_METHODS}"
            )
        if distribution not in self.VALID_DISTRIBUTIONS:
            raise ValueError(
                f"Invalid distribution '{distribution}'. Must be one of "
                f"{self.VALID_DISTRIBUTIONS}"
            )
        if training_size <= guard_size:
            raise ValueError(
                f"training_size ({training_size}) must be > guard_size ({guard_size})"
            )
        if not 0.0 < pfa < 1.0:
            raise ValueError(f"pfa must be in (0, 1), got {pfa}")
        if not 0.0 < os_percentile < 1.0:
            raise ValueError(
                f"os_percentile must be in (0, 1), got {os_percentile}"
            )

        self.guard_size = guard_size
        self.training_size = training_size
        self.pfa = pfa
        self.method = method
        self.os_percentile = os_percentile
        self.distribution = distribution

        # Pre-compute the number of training cells and threshold factors.
        self._training_window_side = 2 * training_size + 1
        self._guard_window_side = 2 * guard_size + 1
        self._num_training_cells = (
            self._training_window_side ** 2 - self._guard_window_side ** 2
        )

        # Gaussian k (legacy, used when distribution="gaussian")
        self._k = float(np.sqrt(2.0 * np.log(self._num_training_cells / pfa)))

        # Exponential / Rayleigh-power α: T = α * mean.
        # Closed form for sample-mean CA-CFAR with N training cells under
        # exponential clutter: α = N * (PFA^(-1/N) - 1).
        # Approaches -ln(pfa) asymptotically for large N.
        n = self._num_training_cells
        self._alpha = float(n * (pfa ** (-1.0 / n) - 1.0))

        logger.info(
            "CFARDetector initialised: method=%s, dist=%s, guard=%d, "
            "training=%d, pfa=%.2e, k=%.3f, alpha=%.3f, num_training_cells=%d",
            method,
            distribution,
            guard_size,
            training_size,
            pfa,
            self._k,
            self._alpha,
            self._num_training_cells,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        image: NDArray[np.floating],
        valid_mask: NDArray[np.bool_] | None = None,
    ) -> list[dict[str, Any]]:
        """Run CFAR detection on a calibrated SAR image.

        Parameters
        ----------
        image:
            2-D array of calibrated backscatter values (sigma-0 or intensity).
            Should already be in linear power scale (not dB).
        valid_mask:
            Optional 2-D boolean array (same shape as ``image``).  Pixels where
            the mask is False are excluded from CFAR detection — typically used
            to mask out land before CFAR runs (CFAR assumes a Rayleigh
            sea-clutter background and fires aggressively on bright land
            features).  ``None`` means no masking.

        Returns
        -------
        List of pixel-level detections, each containing:
        ``x`` (column), ``y`` (row), ``intensity``, ``snr``, ``method``.
        """
        image = np.asarray(image, dtype=np.float64)
        if image.ndim != 2:
            raise ValueError(f"Expected 2-D image, got shape {image.shape}")

        rows, cols = image.shape
        logger.debug(
            "CFAR detect on image %dx%d, method=%s", rows, cols, self.method
        )

        if self.method == "ca":
            threshold_map, background_map = self._ca_cfar(image)
        else:
            threshold_map, background_map = self._os_cfar(image)

        # Identify detections: CUT > threshold
        detection_mask = image > threshold_map

        # Exclude border pixels where the training window is incomplete
        border = self.training_size
        detection_mask[:border, :] = False
        detection_mask[-border:, :] = False
        detection_mask[:, :border] = False
        detection_mask[:, -border:] = False

        # Exclude masked pixels (e.g. land) — CFAR's Rayleigh sea-clutter
        # assumption breaks on land and would otherwise produce ~90% false
        # positives on Strait-of-Gibraltar-style mixed-coverage scenes.
        if valid_mask is not None:
            mask = np.asarray(valid_mask, dtype=bool)
            if mask.shape != image.shape:
                raise ValueError(
                    f"valid_mask shape {mask.shape} does not match image {image.shape}"
                )
            detection_mask &= mask

        det_ys, det_xs = np.nonzero(detection_mask)
        method_str = "ca-cfar" if self.method == "ca" else "os-cfar"

        detections: list[dict[str, Any]] = []
        for y, x in zip(det_ys, det_xs, strict=False):
            intensity = float(image[y, x])
            bg = float(background_map[y, x])
            snr = intensity / bg if bg > 0 else float("inf")
            detections.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "intensity": intensity,
                    "snr": snr,
                    "method": method_str,
                }
            )

        logger.info(
            "CFAR %s detected %d pixels on %dx%d image",
            method_str,
            len(detections),
            rows,
            cols,
        )
        return detections

    def detect_with_clustering(
        self,
        image: NDArray[np.floating],
        min_cluster_size: int = 3,
        eps: float = 2.0,
        min_mean_snr: float = 0.0,
        valid_mask: NDArray[np.bool_] | None = None,
    ) -> list[dict[str, Any]]:
        """Run CFAR followed by DBSCAN clustering of adjacent detected pixels.

        Adjacent bright pixels are grouped into single vessel detections and
        a bounding box is computed for each cluster.

        Parameters
        ----------
        image:
            2-D calibrated SAR image (linear scale).
        min_cluster_size:
            Minimum number of CFAR-detected pixels to form a valid cluster.
        eps:
            Maximum distance (in pixels) between two detected pixels to be
            considered neighbours by DBSCAN.
        valid_mask:
            Optional 2-D boolean array forwarded to :meth:`detect` — pixels
            where the mask is False are excluded from CFAR (used to mask
            land before detection).

        Returns
        -------
        List of clustered detections, each with ``bbox``, ``center``,
        ``num_pixels``, ``mean_intensity``, ``max_intensity``,
        ``mean_snr``, and ``method``.
        """
        pixel_detections = self.detect(image, valid_mask=valid_mask)

        if not pixel_detections:
            logger.info("No pixel detections to cluster")
            return []

        # Build coordinate matrix for DBSCAN
        coords = np.array(
            [[d["x"], d["y"]] for d in pixel_detections], dtype=np.float64
        )

        clustering = DBSCAN(
            eps=eps, min_samples=min_cluster_size, metric="euclidean"
        ).fit(coords)

        labels = clustering.labels_
        unique_labels = set(labels)
        unique_labels.discard(-1)  # noise label

        method_str = "ca-cfar" if self.method == "ca" else "os-cfar"
        clustered: list[dict[str, Any]] = []

        for label in sorted(unique_labels):
            mask = labels == label
            cluster_dets = [
                d for d, m in zip(pixel_detections, mask, strict=False) if m
            ]

            xs = [d["x"] for d in cluster_dets]
            ys = [d["y"] for d in cluster_dets]
            intensities = [d["intensity"] for d in cluster_dets]
            snrs = [d["snr"] for d in cluster_dets]

            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            mean_snr = float(np.mean(snrs))

            if mean_snr < min_mean_snr:
                continue

            clustered.append(
                {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "center": [
                        int(round(np.mean(xs))),
                        int(round(np.mean(ys))),
                    ],
                    "num_pixels": len(cluster_dets),
                    "mean_intensity": float(np.mean(intensities)),
                    "max_intensity": float(np.max(intensities)),
                    "mean_snr": mean_snr,
                    "method": method_str,
                }
            )

        noise_count = int(np.sum(labels == -1))
        logger.info(
            "DBSCAN clustering: %d clusters from %d pixels (%d noise)",
            len(clustered),
            len(pixel_detections),
            noise_count,
        )
        return clustered

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ca_cfar(
        self, image: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Cell-Averaging CFAR via fast uniform-filter convolution.

        For ``distribution="exponential"`` (default) — appropriate for
        linear sigma0 power on SAR — the threshold is multiplicative:
        ``T = α · mean`` with ``α = N · (PFA^(-1/N) − 1)``.

        For ``distribution="gaussian"`` (legacy / dB scale) the original
        ``T = mean + k·std`` formula is retained.

        Returns (threshold_map, background_mean_map).
        """
        ts = self._training_window_side
        gs = self._guard_window_side
        n_train = self._num_training_cells

        # Sum over training window = sum_over_outer - sum_over_guard
        sum_outer = uniform_filter(image, size=ts, mode="reflect") * (ts * ts)
        sum_guard = uniform_filter(image, size=gs, mode="reflect") * (gs * gs)
        background_sum = sum_outer - sum_guard
        background_mean = background_sum / n_train

        if self.distribution == "exponential":
            threshold_map = self._alpha * background_mean
            return threshold_map, background_mean

        # Gaussian path (legacy)
        sq_sum_outer = uniform_filter(image ** 2, size=ts, mode="reflect") * (
            ts * ts
        )
        sq_sum_guard = uniform_filter(image ** 2, size=gs, mode="reflect") * (
            gs * gs
        )
        background_sq_mean = (sq_sum_outer - sq_sum_guard) / n_train
        background_var = np.maximum(
            background_sq_mean - background_mean ** 2, 0.0
        )
        background_std = np.sqrt(background_var)

        threshold_map = background_mean + self._k * background_std
        return threshold_map, background_mean

    def _os_cfar(
        self, image: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Ordered-Statistics CFAR using a generic filter.

        More robust to clutter edges than CA-CFAR but slower because it
        must sort the training cells for every pixel.

        Returns (threshold_map, background_mean_map).
        """
        ts = self.training_size
        gs = self.guard_size
        percentile = self.os_percentile
        k = self._k

        def _os_kernel(values: NDArray[np.float64]) -> float:
            """Kernel applied to each pixel neighbourhood."""
            side = 2 * ts + 1
            len(values) // 2
            vals_2d = values.reshape(side, side)

            # Build a mask that keeps only training cells (outer - guard)
            mask = np.ones((side, side), dtype=bool)
            g_start = ts - gs
            g_end = ts + gs + 1
            mask[g_start:g_end, g_start:g_end] = False

            training_vals = vals_2d[mask]

            # Ordered-statistics background estimate
            sorted_vals = np.sort(training_vals)
            idx = int(percentile * (len(sorted_vals) - 1))
            bg_estimate = float(sorted_vals[idx])

            return bg_estimate

        size = 2 * ts + 1
        background_map: NDArray[np.float64] = generic_filter(
            image, _os_kernel, size=size, mode="reflect"
        )

        # For OS-CFAR the threshold uses the same k factor applied to the
        # ordered-statistic background estimate.  A common simplification
        # is threshold = k * background_os (multiplicative).
        threshold_map = k * background_map

        return threshold_map, background_map
