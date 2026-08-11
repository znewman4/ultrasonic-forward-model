"""Canonical reflector models.

M0 is the original point-reflector model. M1 is the equal-weight geometric
circular-boundary model.
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

__all__ = [
    "CircularReflector",
    "ElasticSDHFMCResult",
    "ElasticSideDrilledHole",
    "PointReflector",
    "circular_boundary_points",
    "reflector_coordinate",
    "simulate_circular_reflector_fmc",
    "simulate_elastic_sdh_fmc",
    "simulate_point_reflector_ascan",
    "simulate_point_reflector_fmc",
]
