"""
Parametros orbitales centralizados.

Exporta constantes de referencia usadas por todos los modulos orbitales:
procesadores, presupuestos energeticos de satelites, perfiles de downlink
y parametros de orbita.  Sirve como fuente unica de verdad para los
valores compartidos.

Usage:
    from src.orbital.orbit_params import (
        PROCESSOR_TDP_WATTS,
        SATELLITE_POWER_BUDGETS,
        DOWNLINK_PROFILES,
        ORBIT_PARAMS,
    )
"""

from __future__ import annotations

# ====================================================================
# Procesadores de referencia (TDP en Watts)
# ====================================================================

PROCESSOR_TDP_WATTS: dict[str, float] = {
    "oci_arm_a1": 3.0,
    "xilinx_zynq_ultrascale": 5.0,
    "intel_myriad_x": 1.5,
    "google_coral_tpu": 2.0,
    "nvidia_jetson_nano": 5.0,
    "raspberry_pi4_arm": 3.5,
    "leon3_gr740": 1.5,
    "unibap_ix10": 15.0,
}

# ====================================================================
# Presupuestos energeticos por tipo de satelite
# ====================================================================

SATELLITE_POWER_BUDGETS: dict[str, dict[str, object]] = {
    "cubesat_3u": {
        "total_w": 6.0,
        "payload_w": 2.0,
        "battery_wh": 30.0,
        "orbit_period_min": 95,
        "sunlit_fraction": 0.6,
        "description": "CubeSat 3U (ej: PhiSat-1)",
    },
    "cubesat_6u": {
        "total_w": 15.0,
        "payload_w": 5.0,
        "battery_wh": 60.0,
        "orbit_period_min": 95,
        "sunlit_fraction": 0.6,
        "description": "CubeSat 6U",
    },
    "small_sat": {
        "total_w": 80.0,
        "payload_w": 30.0,
        "battery_wh": 300.0,
        "orbit_period_min": 100,
        "sunlit_fraction": 0.6,
        "description": "Small satellite (100-500 kg)",
    },
    "medium_sat": {
        "total_w": 300.0,
        "payload_w": 100.0,
        "battery_wh": 2000.0,
        "orbit_period_min": 100,
        "sunlit_fraction": 0.6,
        "description": "Medium satellite (ej: Sentinel-1 class)",
    },
}

# ====================================================================
# Perfiles de downlink
# ====================================================================

DOWNLINK_PROFILES: dict[str, dict[str, object]] = {
    "cubesat_uhf": {
        "name": "CubeSat UHF",
        "bandwidth_mbps": 0.009,
        "window_minutes": 8,
        "passes_per_day": 4,
        "description": "CubeSat con radio UHF basica",
    },
    "cubesat_sband": {
        "name": "CubeSat S-Band",
        "bandwidth_mbps": 2.0,
        "window_minutes": 8,
        "passes_per_day": 4,
        "description": "CubeSat con S-Band",
    },
    "smallsat_xband": {
        "name": "SmallSat X-Band",
        "bandwidth_mbps": 100.0,
        "window_minutes": 10,
        "passes_per_day": 6,
        "description": "SmallSat con X-Band (ej: Sentinel-1 class)",
    },
    "highcap_ka": {
        "name": "High-Capacity Ka-Band",
        "bandwidth_mbps": 800.0,
        "window_minutes": 10,
        "passes_per_day": 8,
        "description": "Satelite grande con Ka-Band + red EDRS",
    },
}

# ====================================================================
# Parametros orbitales
# ====================================================================

ORBIT_PARAMS: dict[str, dict[str, object]] = {
    "leo_500": {
        "altitude_km": 500,
        "period_min": 94.6,
        "velocity_km_s": 7.6,
        "ground_track_km_s": 6.9,
        "max_contact_min": 10,
        "avg_revisit_hours": 12,
        "description": "LEO 500 km (tipica EO)",
    },
    "sso_700": {
        "altitude_km": 700,
        "period_min": 98.8,
        "velocity_km_s": 7.5,
        "ground_track_km_s": 6.8,
        "max_contact_min": 12,
        "avg_revisit_hours": 6,
        "description": "SSO 700 km (Sentinel-1)",
    },
    "leo_350_isstyle": {
        "altitude_km": 350,
        "period_min": 91.4,
        "velocity_km_s": 7.7,
        "ground_track_km_s": 7.1,
        "max_contact_min": 7,
        "avg_revisit_hours": 24,
        "description": "LEO baja 350 km (ISS-like)",
    },
}

# ====================================================================
# Cadenas de procesamiento en tierra
# ====================================================================

GROUND_PROCESSING: dict[str, dict[str, object]] = {
    "fast_automated": {
        "ingest_minutes": 5,
        "processing_minutes": 10,
        "dissemination_minutes": 2,
        "description": "Cadena automatizada rapida",
    },
    "standard_nrt": {
        "ingest_minutes": 15,
        "processing_minutes": 30,
        "dissemination_minutes": 10,
        "description": "Near Real-Time estandar (ESA NRT)",
    },
    "manual_analysis": {
        "ingest_minutes": 30,
        "processing_minutes": 120,
        "dissemination_minutes": 30,
        "description": "Analisis con intervencion humana",
    },
}


# ====================================================================
# Helper functions
# ====================================================================


def list_processors() -> list[str]:
    """Return the names of all reference processors."""
    return list(PROCESSOR_TDP_WATTS.keys())


def list_satellite_types() -> list[str]:
    """Return the names of all satellite power-budget types."""
    return list(SATELLITE_POWER_BUDGETS.keys())


def list_downlink_profiles() -> list[str]:
    """Return the names of all downlink profiles."""
    return list(DOWNLINK_PROFILES.keys())


def list_orbits() -> list[str]:
    """Return the names of all orbit parameter sets."""
    return list(ORBIT_PARAMS.keys())


def list_ground_chains() -> list[str]:
    """Return the names of all ground-processing chains."""
    return list(GROUND_PROCESSING.keys())


def get_processor_tdp(name: str) -> float:
    """Return TDP in watts for the given processor.

    Raises
    ------
    KeyError
        If the processor name is not found.
    """
    if name not in PROCESSOR_TDP_WATTS:
        available = ", ".join(PROCESSOR_TDP_WATTS.keys())
        raise KeyError(
            f"Processor '{name}' not found. Available: {available}"
        )
    return PROCESSOR_TDP_WATTS[name]


def get_satellite_budget(name: str) -> dict[str, object]:
    """Return the power-budget dict for the given satellite type.

    Raises
    ------
    KeyError
        If the satellite type is not found.
    """
    if name not in SATELLITE_POWER_BUDGETS:
        available = ", ".join(SATELLITE_POWER_BUDGETS.keys())
        raise KeyError(
            f"Satellite type '{name}' not found. Available: {available}"
        )
    return SATELLITE_POWER_BUDGETS[name]


def get_downlink_profile(name: str) -> dict[str, object]:
    """Return the downlink-profile dict for the given profile name.

    Raises
    ------
    KeyError
        If the profile name is not found.
    """
    if name not in DOWNLINK_PROFILES:
        available = ", ".join(DOWNLINK_PROFILES.keys())
        raise KeyError(
            f"Downlink profile '{name}' not found. Available: {available}"
        )
    return DOWNLINK_PROFILES[name]


def get_orbit_params(name: str) -> dict[str, object]:
    """Return the orbit-parameter dict for the given orbit name.

    Raises
    ------
    KeyError
        If the orbit name is not found.
    """
    if name not in ORBIT_PARAMS:
        available = ", ".join(ORBIT_PARAMS.keys())
        raise KeyError(
            f"Orbit '{name}' not found. Available: {available}"
        )
    return ORBIT_PARAMS[name]
