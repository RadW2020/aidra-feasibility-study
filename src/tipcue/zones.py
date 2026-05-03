"""
Definicion de zonas de interes para Tip & Cue.

Las zonas de interes son areas geograficas donde las detecciones
tienen mayor relevancia y pueden generar cues automaticos.  Cada zona
se define con un bounding box y una prioridad que influye en el orden
de procesamiento de los cues generados.

Usage:
    from src.tipcue.zones import DEFAULT_ZONES, Zone

    for zone in DEFAULT_ZONES:
        print(zone.name, zone.bbox)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class Zone(BaseModel):
    """Zona de interes geoespacial para Tip & Cue.

    Attributes
    ----------
    id:
        Identificador unico de la zona (slug, e.g. ``"gibraltar_strait"``).
    name:
        Nombre descriptivo legible.
    bbox:
        Bounding box ``[lon_min, lat_min, lon_max, lat_max]`` en WGS84.
    geometry:
        Representacion GeoJSON Polygon del bounding box.
        Se genera automaticamente a partir de ``bbox`` si no se proporciona.
    priority:
        Prioridad de la zona: 0=normal, 1=alta, 2=urgente.
    description:
        Descripcion textual del interes de la zona.
    active:
        Si la zona esta activa para evaluacion de tips.
    """

    id: str
    name: str
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box [lon_min, lat_min, lon_max, lat_max]",
    )
    geometry: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=0, ge=0, le=2)
    description: str = ""
    active: bool = True

    @model_validator(mode="after")
    def _build_geometry(self) -> Zone:
        """Genera el GeoJSON Polygon a partir del bbox si no se proporciono."""
        if not self.geometry:
            lon_min, lat_min, lon_max, lat_max = self.bbox
            self.geometry = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [lon_min, lat_min],
                        [lon_max, lat_min],
                        [lon_max, lat_max],
                        [lon_min, lat_max],
                        [lon_min, lat_min],
                    ]
                ],
            }
        return self

    def contains_point(self, lon: float, lat: float) -> bool:
        """Comprueba si un punto (lon, lat) esta dentro del bbox.

        Parameters
        ----------
        lon:
            Longitud del punto.
        lat:
            Latitud del punto.

        Returns
        -------
        bool
            ``True`` si el punto cae dentro del bounding box.
        """
        lon_min, lat_min, lon_max, lat_max = self.bbox
        return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max

    def to_geojson_bbox(self) -> str:
        """Serializa el bbox como un GeoJSON Polygon string.

        Returns
        -------
        str
            Representacion JSON del poligono para uso en queries SQL
            con ``ST_GeomFromGeoJSON``.
        """
        import json

        return json.dumps(self.geometry)


# ------------------------------------------------------------------
# Default zones
# ------------------------------------------------------------------

DEFAULT_ZONES: list[Zone] = [
    Zone(
        id="gibraltar_strait",
        name="Estrecho de Gibraltar - Zona de transito",
        bbox=[-5.6, 35.8, -5.3, 36.1],
        priority=1,
        description="Punto de paso obligatorio entre Atlantico y Mediterraneo",
    ),
    Zone(
        id="algeciras_port",
        name="Puerto de Algeciras - Zona de fondeo",
        bbox=[-5.5, 36.05, -5.35, 36.15],
        priority=1,
        description="Zona de fondeo del puerto de Algeciras, alta densidad",
    ),
    Zone(
        id="med_patrol",
        name="Patrulla Mediterraneo Central",
        bbox=[10.0, 33.0, 16.0, 38.0],
        priority=0,
        description="Ruta migratoria y de trafico, Tunez-Sicilia",
    ),
]


def get_zone(zone_id: str) -> Zone | None:
    """Busca una zona por su identificador.

    Parameters
    ----------
    zone_id:
        Identificador de la zona (e.g. ``"gibraltar_strait"``).

    Returns
    -------
    Zone | None
        La zona encontrada o ``None`` si no existe.
    """
    for zone in DEFAULT_ZONES:
        if zone.id == zone_id:
            return zone
    return None


def get_active_zones() -> list[Zone]:
    """Retorna todas las zonas activas.

    Returns
    -------
    list[Zone]
        Lista de zonas con ``active=True``.
    """
    return [z for z in DEFAULT_ZONES if z.active]
