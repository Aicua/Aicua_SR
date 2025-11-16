"""
Master module for Rose SR discovered formulas.
Generated: 2025-11-16 03:12:18

Import this module to access all discovered formulas.
"""

from .petal_geometry_formulas import *
from .bone_rigging_formulas import *
from .animation_params_formulas import *


class RoseFormulaComputer:
    """
    Utility class to compute all rose parameters.
    """

    @staticmethod
    def compute_petal_geometry(**kwargs):
        """Compute all petal geometry parameters."""
        results = {}
        results["base_width"] = compute_base_width(**kwargs)
        results["length"] = compute_length(**kwargs)
        results["curvature"] = compute_curvature(**kwargs)
        results["twist_angle"] = compute_twist_angle(**kwargs)
        results["thickness"] = compute_thickness(**kwargs)
        return results

    @staticmethod
    def compute_bone_rigging(**kwargs):
        """Compute all bone rigging parameters."""
        results = {}
        results["bone_count"] = compute_bone_count(**kwargs)
        results["bone_length"] = compute_bone_length(**kwargs)
        results["joint_spacing"] = compute_joint_spacing(**kwargs)
        results["bind_weight"] = compute_bind_weight(**kwargs)
        return results

    @staticmethod
    def compute_animation_params(**kwargs):
        """Compute all animation params parameters."""
        results = {}
        results["frequency"] = compute_frequency(**kwargs)
        results["amplitude"] = compute_amplitude(**kwargs)
        results["phase_offset"] = compute_phase_offset(**kwargs)
        results["damping_factor"] = compute_damping_factor(**kwargs)
        return results
