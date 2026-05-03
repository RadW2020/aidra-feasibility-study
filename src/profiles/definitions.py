"""
Definicion de perfiles de restriccion para simulacion espacial.

Cada perfil representa un nivel de recursos computacionales que simula
un tipo de procesador — desde una estacion terrena sin restricciones
hasta un CubeSat con recursos extremadamente limitados.

Los perfiles se usan para ejecutar el pipeline de deteccion bajo
condiciones controladas y medir la degradacion en precision y
rendimiento a medida que se reducen los recursos.

Usage:
    from src.profiles.definitions import PROFILES, ConstraintProfile

    profile = PROFILES["sat-mid"]
    print(profile.cpu_limit)       # 1.0
    print(profile.memory_limit_mb) # 2048
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConstraintProfile:
    """Perfil de restriccion de recursos computacionales.

    Attributes
    ----------
    name:
        Identificador unico del perfil (e.g. ``"ground"``, ``"sat-low"``).
    display_name:
        Nombre legible para UIs y reportes.
    description:
        Descripcion breve en castellano del perfil.
    cpu_limit:
        Numero maximo de CPUs (puede ser fraccion, e.g. ``0.5``).
    memory_limit_mb:
        Limite de RAM en megabytes.
    docker_cpus:
        Valor para el flag ``--cpus`` de Docker.
    docker_memory:
        Valor para el flag ``--memory`` de Docker.
    simulates:
        Descripcion del hardware real que este perfil simula.
    """

    name: str
    display_name: str
    description: str
    cpu_limit: float
    memory_limit_mb: int
    docker_cpus: str
    docker_memory: str
    simulates: str
    # ---- Energy budget (Q3 simulation) ------------------------------
    # ``tdp_watts`` is the assumed thermal design power of the simulated
    # processor at 100% CPU.  Used by ``ResourceCollector`` to derive
    # an estimated energy cost (Joules) per pipeline run, weighted by
    # the average CPU usage during sampling.  Values are public-data
    # nominal references — not a substitute for in-orbit telemetry.
    tdp_watts: float | None = None


# Orden jerarquico: de mas recursos a menos recursos.
# El pipeline debe ejecutarse exitosamente al menos hasta ``sat-mid``.
PROFILES: dict[str, ConstraintProfile] = {
    "ground": ConstraintProfile(
        name="ground",
        display_name="Ground Station",
        description="Sin restricciones, estacion terrena",
        cpu_limit=4.0,
        memory_limit_mb=24576,
        docker_cpus="4",
        docker_memory="24g",
        simulates="Ground processing station (baseline)",
        # Generic x86/ARM workstation — energy is irrelevant for the
        # ground baseline; we still emit a value so the metric is
        # comparable across profiles.
        tdp_watts=65.0,
    ),
    "sat-high": ConstraintProfile(
        name="sat-high",
        display_name="Satellite High-End",
        description="Satelite gama alta (Xilinx Zynq / Unibap iX10)",
        cpu_limit=2.0,
        memory_limit_mb=4096,
        docker_cpus="2",
        docker_memory="4g",
        simulates="High-end satellite processor (e.g. Xilinx Zynq UltraScale+)",
        # Xilinx Zynq UltraScale+ MPSoC ZU3EG — typical board power 8–12 W
        # (see ZCU104 board user guide, AMD/Xilinx UG1267).
        tdp_watts=10.0,
    ),
    "sat-mid": ConstraintProfile(
        name="sat-mid",
        display_name="Satellite Mid-Range",
        description="Satelite gama media",
        cpu_limit=1.0,
        memory_limit_mb=2048,
        docker_cpus="1",
        docker_memory="2g",
        simulates="Mid-range satellite processor",
        # NXP LX2160A / Unibap iX5-100 class. Reference: ~5 W typical.
        tdp_watts=5.0,
    ),
    "sat-low": ConstraintProfile(
        name="sat-low",
        display_name="Satellite Low-End / CubeSat",
        description="Satelite gama baja o CubeSat",
        cpu_limit=0.5,
        memory_limit_mb=1024,
        docker_cpus="0.5",
        docker_memory="1g",
        simulates="Low-end processor / CubeSat (e.g. Raspberry Pi class)",
        # Raspberry Pi 4 class — ~2.7 W active load
        # (RPi Foundation power benchmarks, 2020).
        tdp_watts=2.5,
    ),
    "sat-extreme": ConstraintProfile(
        name="sat-extreme",
        display_name="Extreme Constraint",
        description="Limite inferior: donde se rompe el pipeline",
        cpu_limit=0.25,
        memory_limit_mb=512,
        docker_cpus="0.25",
        docker_memory="512m",
        simulates="Extreme constraint — find breaking point",
        # Cortex-M / RP2040 class breakeven assumption.
        tdp_watts=1.5,
    ),
}

# Orden de perfiles de mas a menos recursos (util para iteraciones).
PROFILE_ORDER: list[str] = [
    "ground",
    "sat-high",
    "sat-mid",
    "sat-low",
    "sat-extreme",
]


def get_profile(name: str) -> ConstraintProfile:
    """Obtiene un perfil por nombre.

    Parameters
    ----------
    name:
        Nombre del perfil (e.g. ``"ground"``).

    Returns
    -------
    ConstraintProfile

    Raises
    ------
    KeyError
        Si el perfil no existe.
    """
    if name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise KeyError(
            f"Profile '{name}' not found. Available profiles: {available}"
        )
    return PROFILES[name]
