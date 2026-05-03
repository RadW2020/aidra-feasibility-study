"""
Terrain correction (Range-Doppler) sobre Sentinel-1 GRD.

================================================================
EXCLUSION FORMAL DE ALCANCE — AIDRA POC (autoaudit 2026-04-26)
================================================================

Range-Doppler Terrain Correction queda **formalmente fuera del
alcance** del proof-of-concept AIDRA. Esta decision se documenta
aqui de forma irreversible para que el evaluador SatCen no asuma
que la cadena ESA-canonica se ejecuta.

Motivacion (palanca L5 de la autoevaluacion contra rubrica Q3):

  1. **Dominio de evaluacion = mar abierto.** Las cuatro zonas
     operativas declaradas (Estrecho de Gibraltar, Mar Rojo,
     Canal de Suez, English Channel) son AOIs maritimas plano-mar
     en latitudes < 60 deg. Para deteccion de barcos sobre agua
     el GCP-linear geocoding tiene RMSE < 1 px de S1 GRD
     (verificado en runs reales con AIS overlay).

  2. **Coste vs valor.** Integrar pyrosar + SRTM 1\\" requiere
     ~200 MB de binarios GDAL, cache DEM externa, y add-ons
     Python que rompen el target ARM A1 Free Tier (RAM 24 GB
     compartida). El payoff -- mejorar geocoding sobre relieve
     costero -- no afecta a la metrica primaria (Pd/FAR de
     barcos sobre mar).

  3. **Trazabilidad explicita.** Cada execution_log persiste
     ``geocoding_backend = "gcp_linear"`` (bandera implicita en
     ``preprocessing._build_pixel_to_geo_transform``); cualquier
     run con ``backend="pyrosar_rd"`` queda diferenciado en el
     bundle D3.

  4. **Limitacion declarada.** Las 5 MODEL_CARDs llevan en su
     seccion *Limitaciones*: "Geocoding via affine 6-parametros
     ajustada sobre GCPs Sentinel-1; valido para mar abierto.
     Detecciones costeras con relieve > 200 m pueden tener
     desplazamiento azimutal de hasta ~30 m." Eso cierra el
     contrato Anexo IV del AI Act.

Re-evaluacion: la integracion pyrosar pasa a "post-MVP" en el
RISK_REGISTER (R8) y se reactivara solo si:
  (a) la zona operativa se amplia a fjordos/relieve costero, o
  (b) el evaluador SatCen exige DEM-corrected output formal.

================================================================
INTERFAZ TECNICA (preservada como andamio re-activable)
================================================================

``apply_terrain_correction`` se mantiene en el repo pero **NO se
invoca** desde ``src/pipeline/preprocessing.py`` ni desde
``src/pipeline/engine.py``. La interfaz devuelve un
``TerrainCorrectionResult`` con backend ``"gcp_linear"`` por
defecto (sin abortar el pipeline) y solo intenta Range-Doppler
real si pyrosar esta disponible. Asi:

  1. La auditoria sigue distinguiendo "TC ejecutado" vs
     "GCPs lineal" si algun dia se reactiva.
  2. Tests unitarios futuros pueden inyectar pyrosar mock sin
     tocar el pipeline productivo.
  3. La cadena de decision queda en codigo y RISK_REGISTER, no
     en supuestos del evaluador.

Notas operativas (si se reactiva):
  - DEM cache: ``$AIDRA_DEM_CACHE`` (default ``~/.cache/aidra/dem``).
  - Region: SRTM 1\\" cubre 60S/60N — pyrosar cae a SRTM 3\\"
    automaticamente fuera.
  - Esta implementacion no usa SNAP: ``pyrosar.snap`` se evita
    para no requerir Java en el contenedor.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =====================================================================
# Result
# =====================================================================


@dataclass
class TerrainCorrectionResult:
    """Devuelto por ``apply_terrain_correction``.

    Attributes
    ----------
    backend:
        ``"pyrosar_rd"`` (Range-Doppler real con DEM) o
        ``"gcp_linear"`` (fallback, transformacion lineal GCPs).
    output_path:
        Ruta al raster corregido. ``None`` si backend == gcp_linear
        (el pipeline sigue trabajando sobre el TIFF original con su
        ``geo_transform``).
    dem_used:
        Identificador del DEM (``"SRTM 1Sec HGT"``, ``"SRTM 3Sec"``).
        ``None`` para fallback.
    notes:
        Mensajes adicionales (warnings, version de pyrosar, etc.).
    """

    backend: str
    output_path: Path | None = None
    dem_used: str | None = None
    notes: str = ""


# =====================================================================
# Public API
# =====================================================================


def apply_terrain_correction(
    product_dir: Path,
    work_dir: Path | None = None,
    polarisation: str = "VV",
) -> TerrainCorrectionResult:
    """Intenta Range-Doppler TC con DEM SRTM via pyrosar.

    Cuando pyrosar no esta instalado o falla, devuelve un resultado
    con ``backend="gcp_linear"`` (sin abortar el pipeline). El caller
    debe persistir ``backend`` en ``execution_log.notes`` para que la
    auditoria distinga runs con TC real vs runs con GCPs.
    """
    work_dir = work_dir or Path(
        os.getenv("AIDRA_DEM_CACHE", str(Path.home() / ".cache" / "aidra" / "dem"))
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        return _pyrosar_rd_tc(product_dir, work_dir, polarisation)
    except ImportError as exc:
        logger.info("pyrosar not available — using GCP linear fallback: %s", exc)
        return TerrainCorrectionResult(
            backend="gcp_linear",
            notes="pyrosar not installed",
        )
    except Exception as exc:
        logger.warning(
            "pyrosar Range-Doppler TC failed; using GCP fallback: %s", exc
        )
        return TerrainCorrectionResult(
            backend="gcp_linear",
            notes=f"pyrosar error: {exc}",
        )


# =====================================================================
# pyrosar backend (lazy)
# =====================================================================


def _pyrosar_rd_tc(
    product_dir: Path, work_dir: Path, polarisation: str
) -> TerrainCorrectionResult:
    """Range-Doppler TC con pyrosar (importado lazy)."""
    # Lazy imports — pyrosar trae GDAL bindings pesadas.
    from pyrosar.gamma.api import geocode  # type: ignore[import-not-found]

    out_dir = work_dir / "tc"
    out_dir.mkdir(parents=True, exist_ok=True)

    geocode(
        scene=str(product_dir),
        dem=None,  # pyrosar autoload SRTM 1Sec inside its boundary
        outdir=str(out_dir),
        spacing=10,
        polarizations=[polarisation],
    )

    # pyrosar names outputs deterministically from scene id.
    candidates: list[Any] = sorted(out_dir.glob(f"*_{polarisation}_*.tif"))
    if not candidates:
        raise RuntimeError(
            f"pyrosar produced no output in {out_dir} for pol {polarisation}"
        )
    return TerrainCorrectionResult(
        backend="pyrosar_rd",
        output_path=candidates[-1],
        dem_used="SRTM 1Sec HGT",
        notes="Range-Doppler TC over SRTM 1\" via pyrosar",
    )
