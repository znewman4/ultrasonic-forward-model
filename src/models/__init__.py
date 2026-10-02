"""Canonical reflector models.

M0 is the original point-reflector model. M1 is the equal-weight geometric
circular-boundary model. The M0 physics ladder (M0-TD+, M0-FD, ...) adds
spreading, directivity and attenuation to M0.
"""

from .boundary_circle import (
    CircularReflector,
    circular_boundary_points,
    simulate_circular_reflector_fmc,
)
from .elastic_sdh import ElasticSDHFMCResult, ElasticSideDrilledHole, simulate_elastic_sdh_fmc
from .point_reflector import (
    PointReflector,
    reflector_coordinate,
    simulate_point_reflector_ascan,
    simulate_point_reflector_fmc,
)
from .point_reflector_physics import (
    FrequencyDomainFMCResult,
    LadderStep,
    PropagationPhysics,
    physics_ladder,
    simulate_ladder_step,
    simulate_point_reflector_fmc_fd,
    simulate_point_reflector_fmc_td,
)

__all__ = [
    "CircularReflector",
    "ElasticSDHFMCResult",
    "ElasticSideDrilledHole",
    "FrequencyDomainFMCResult",
    "LadderStep",
    "PointReflector",
    "PropagationPhysics",
    "circular_boundary_points",
    "physics_ladder",
    "reflector_coordinate",
    "simulate_circular_reflector_fmc",
    "simulate_elastic_sdh_fmc",
    "simulate_ladder_step",
    "simulate_point_reflector_ascan",
    "simulate_point_reflector_fmc",
    "simulate_point_reflector_fmc_fd",
    "simulate_point_reflector_fmc_td",
]
