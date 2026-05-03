"""
Verificacion de reproducibilidad de ejecuciones del pipeline.

Dado un ``execution_id``, re-ejecuta el pipeline con los mismos
inputs y parametros, y compara el ``output_hash``.  Si coincide,
el resultado es reproducible (determinista).

Los modelos de deep learning no son 100% deterministas en CPU
(diferencias de precision aritmetica, orden de operaciones).
Diferencias menores en confianza (<0.01) se consideran aceptables
y no invalidan la reproducibilidad.

Usage:
    from src.traceability.verifier import ReproducibilityVerifier

    verifier = ReproducibilityVerifier(recorder=recorder, engine=engine)
    result = await verifier.verify(execution_id)
    print(result.reproducible, result.confidence_diff_mean)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from src.pipeline.engine import PipelineRequest, PipelineResult
from src.traceability.recorder import ExecutionRecorder

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Protocol for PipelineEngine (avoid circular imports)
# ------------------------------------------------------------------


class PipelineEngineProtocol(Protocol):
    """Protocolo minimo del engine del pipeline para verificacion.

    Se define como Protocol para evitar importaciones circulares.
    """

    async def run(
        self,
        request: PipelineRequest,
    ) -> PipelineResult:
        """Ejecuta el pipeline de deteccion."""
        ...


# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------


@dataclass
class ReproducibilityResult:
    """Resultado de la verificacion de reproducibilidad.

    Attributes
    ----------
    execution_id:
        UUID de la ejecucion original verificada.
    verification_id:
        UUID de la re-ejecucion de verificacion.
    reproducible:
        ``True`` si los output_hash coinciden, o las diferencias
        en detecciones estan dentro de la tolerancia aceptable.
    original_hash:
        output_hash de la ejecucion original.
    verification_hash:
        output_hash de la re-ejecucion.
    original_detections:
        Numero de detecciones en la ejecucion original.
    verification_detections:
        Numero de detecciones en la re-ejecucion.
    matching_detections:
        Numero de detecciones con IoU > 0.9 y confianza similar
        entre original y verificacion.
    confidence_diff_mean:
        Diferencia media absoluta en confianza entre detecciones
        emparejadas.
    notes:
        Notas textuales sobre el resultado de la verificacion.
    """

    execution_id: UUID | None = None
    verification_id: UUID | None = None
    reproducible: bool = False
    original_hash: str = ""
    verification_hash: str = ""
    original_detections: int = 0
    verification_detections: int = 0
    matching_detections: int = 0
    confidence_diff_mean: float = 0.0
    notes: str = ""


# ------------------------------------------------------------------
# Verifier
# ------------------------------------------------------------------

# Tolerance for confidence differences (deep learning float imprecision)
_CONFIDENCE_TOLERANCE: float = 0.01

# Minimum IoU to consider two detections as "matching"
_IOU_THRESHOLD: float = 0.9


class ReproducibilityVerifier:
    """Verificador de reproducibilidad de ejecuciones del pipeline.

    Re-ejecuta el pipeline con los mismos parametros que una ejecucion
    anterior y compara los resultados para evaluar el determinismo.

    Parameters
    ----------
    recorder:
        Instancia de ``ExecutionRecorder`` para acceder a los registros
        de ejecucion.
    engine:
        Motor del pipeline que implementa ``PipelineEngineProtocol``.
    """

    def __init__(
        self,
        recorder: ExecutionRecorder,
        engine: PipelineEngineProtocol,
    ) -> None:
        self._recorder: ExecutionRecorder = recorder
        self._engine: PipelineEngineProtocol = engine

    async def verify(self, execution_id: UUID) -> ReproducibilityResult:
        """Verifica la reproducibilidad de una ejecucion.

        1. Obtiene el registro original de ``execution_log``
        2. Re-ejecuta el pipeline con los mismos parametros
        3. Compara ``output_hash`` del original vs re-ejecucion
        4. Si los hashes no coinciden, compara detecciones
           individualmente con tolerancia

        Parameters
        ----------
        execution_id:
            UUID de la ejecucion a verificar.

        Returns
        -------
        ReproducibilityResult
            Resultado completo de la verificacion.

        Raises
        ------
        ValueError
            Si la ejecucion original no existe o no tiene status
            ``"success"``.
        """
        # 1. Fetch original execution
        original = await self._recorder.get(execution_id)
        if original is None:
            raise ValueError(f"Execution {execution_id} not found")

        if original.status != "success":
            raise ValueError(
                f"Execution {execution_id} has status '{original.status}' "
                f"— only 'success' executions can be verified"
            )

        logger.info(
            "Verifying reproducibility of execution %s "
            "(model=%s, profile=%s, image=%s)",
            execution_id,
            original.model_name,
            original.constraint_profile,
            original.image_id,
        )

        result = ReproducibilityResult(
            execution_id=execution_id,
            original_hash=original.output_hash,
            original_detections=original.num_detections,
        )

        # 2. Re-run pipeline with same parameters
        try:
            verification_request = PipelineRequest(
                image_id=original.image_id,
                model=original.model_name,
                profile=original.constraint_profile,
                confidence_threshold=original.confidence_threshold,
                iou_threshold=original.iou_threshold,
                trigger_type="verification",
                triggered_by=execution_id,
                zone=original.search_zone or "gibraltar",
            )
            verification_output = await self._engine.run(verification_request)
        except Exception as exc:
            result.notes = f"Re-execution failed: {exc}"
            logger.exception(
                "Verification re-execution failed for %s", execution_id
            )
            return result

        # 3. Extract verification results
        result.verification_id = verification_output.execution_id

        verification_hash = verification_output.output_hash
        result.verification_hash = verification_hash
        result.verification_detections = verification_output.num_detections

        # 4. Compare hashes
        if original.output_hash and original.output_hash == verification_hash:
            result.reproducible = True
            result.matching_detections = result.verification_detections
            result.confidence_diff_mean = 0.0
            result.notes = (
                "Exact match: output hashes are identical — "
                "fully reproducible execution."
            )
            logger.info(
                "Execution %s is reproducible (exact hash match)", execution_id
            )
            return result

        # 5. Hashes differ — compare detections with tolerance
        logger.info(
            "Hash mismatch for %s (original=%s, verification=%s) — "
            "comparing detections with tolerance",
            execution_id,
            original.output_hash[:16],
            verification_hash[:16],
        )

        # Load original detections from DB for comparison
        original_det_rows = await self._recorder._db.fetch(
            "SELECT *, ST_X(center_geo) AS longitude, ST_Y(center_geo) AS latitude "
            "FROM detections WHERE execution_id = $1 ORDER BY confidence DESC",
            execution_id,
        )
        original_dets: list[dict[str, Any]] = [dict(row) for row in original_det_rows]
        verification_dets = [d.model_dump() for d in verification_output.detections]

        matching, avg_diff = self._compare_detections(
            original_dets, verification_dets
        )

        result.matching_detections = matching
        result.confidence_diff_mean = avg_diff

        # Determine if "effectively reproducible" within tolerance
        if (
            result.original_detections == result.verification_detections
            and matching == result.original_detections
            and avg_diff <= _CONFIDENCE_TOLERANCE
        ):
            result.reproducible = True
            result.notes = (
                f"Hash differs but detections match within tolerance "
                f"(avg_diff={avg_diff:.4f} <= {_CONFIDENCE_TOLERANCE}). "
                f"Effectively reproducible."
            )
        elif matching > 0:
            detection_match_pct = (
                matching / max(result.original_detections, 1) * 100
            )
            result.notes = (
                f"Partial match: {matching}/{result.original_detections} "
                f"detections match ({detection_match_pct:.1f}%), "
                f"avg confidence diff={avg_diff:.4f}. "
                f"Non-deterministic but consistent."
            )
        else:
            result.notes = (
                "No matching detections found — "
                "significantly different results."
            )

        logger.info(
            "Verification result for %s: reproducible=%s, "
            "matching=%d/%d, avg_diff=%.4f",
            execution_id,
            result.reproducible,
            matching,
            result.original_detections,
            avg_diff,
        )

        return result

    # ------------------------------------------------------------------
    # Detection comparison
    # ------------------------------------------------------------------

    @staticmethod
    def _compare_detections(
        original: list[dict[str, Any]],
        verification: list[dict[str, Any]],
    ) -> tuple[int, float]:
        """Compara dos listas de detecciones con tolerancia.

        Empareja detecciones por IoU de bounding box (>0.9) y calcula
        la diferencia media de confianza.

        Parameters
        ----------
        original:
            Detecciones de la ejecucion original.
        verification:
            Detecciones de la re-ejecucion.

        Returns
        -------
        tuple[int, float]
            ``(matching_count, avg_confidence_diff)`` — numero de
            detecciones emparejadas y diferencia media de confianza.
        """
        if not original or not verification:
            return 0, 0.0

        matched_count = 0
        confidence_diffs: list[float] = []
        used_verification: set[int] = set()

        for orig_det in original:
            orig_bbox = orig_det.get("bbox_pixel", [])
            orig_conf = orig_det.get("confidence", 0.0)

            best_iou = 0.0
            best_idx = -1
            best_conf_diff = 0.0

            for v_idx, ver_det in enumerate(verification):
                if v_idx in used_verification:
                    continue

                ver_bbox = ver_det.get("bbox_pixel", [])
                iou = _compute_iou(orig_bbox, ver_bbox)

                if iou > best_iou:
                    best_iou = iou
                    best_idx = v_idx
                    best_conf_diff = abs(
                        orig_conf - ver_det.get("confidence", 0.0)
                    )

            if best_iou >= _IOU_THRESHOLD and best_idx >= 0:
                matched_count += 1
                confidence_diffs.append(best_conf_diff)
                used_verification.add(best_idx)

        avg_diff = (
            sum(confidence_diffs) / len(confidence_diffs)
            if confidence_diffs
            else 0.0
        )
        return matched_count, avg_diff


def _compute_iou(
    bbox_a: list[float],
    bbox_b: list[float],
) -> float:
    """Calcula Intersection over Union (IoU) entre dos bounding boxes.

    Bounding boxes en formato ``[x1, y1, x2, y2]`` (esquinas
    superior-izquierda e inferior-derecha).

    Parameters
    ----------
    bbox_a:
        Primer bounding box.
    bbox_b:
        Segundo bounding box.

    Returns
    -------
    float
        IoU en rango [0.0, 1.0].  Retorna 0.0 si alguna bbox
        es invalida o no hay interseccion.
    """
    if len(bbox_a) < 4 or len(bbox_b) < 4:
        return 0.0

    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0

    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union
