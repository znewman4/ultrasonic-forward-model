"""Backward-compatible imports for reflector forward models.

New code may import Model M0 and M1 from :mod:`src.models` directly. Existing
imports from this module retain the same public call signatures and behaviour.
"""

try:
    from .models.boundary_circle import simulate_circular_reflector_fmc
    from .models.point_reflector import (
        simulate_point_reflector_ascan,
        simulate_point_reflector_fmc,
    )
except ImportError:  # Supports importing when ``src`` is directly on sys.path.
    from models.boundary_circle import simulate_circular_reflector_fmc
    from models.point_reflector import (
        simulate_point_reflector_ascan,
        simulate_point_reflector_fmc,
    )

__all__ = [
    "simulate_circular_reflector_fmc",
    "simulate_point_reflector_ascan",
    "simulate_point_reflector_fmc",
]
