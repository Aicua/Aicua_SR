"""
Auto-generated formulas for Bone Rigging
Generated: 2025-11-16 03:12:18

These formulas were discovered using Symbolic Regression (PySR).
DO NOT EDIT MANUALLY - regenerate using codegen_formulas.py
"""

import math


def compute_bone_count(petal_length, petal_width, flexibility_factor, layer_index):
    """
    Compute bone count.
    
    Discovered Formula: ceil(3.0 * x0 * x2)
    R² Score: 0.987957
    Complexity: 6
    
    Args:
        petal_length: Feature value
        petal_width: Feature value
        flexibility_factor: Feature value
        layer_index: Feature value
    
    Returns:
        Computed bone count value
    """
    try:
        result = math.ceil(3.0 * petal_length * flexibility_factor)
        return float(result)
    except (ValueError, ZeroDivisionError) as e:
        # Fallback for edge cases
        print(f"Warning: Error computing bone_count: {e}")
        return 0.0


def compute_bone_length(petal_length, petal_width, flexibility_factor, layer_index):
    """
    Compute bone length.
    
    Discovered Formula: x0 / (3.0 * x0 * x2 + 1) * 0.9
    R² Score: 0.985794
    Complexity: 5
    
    Args:
        petal_length: Feature value
        petal_width: Feature value
        flexibility_factor: Feature value
        layer_index: Feature value
    
    Returns:
        Computed bone length value
    """
    try:
        result = petal_length / (3.0 * petal_length * flexibility_factor + 1) * 0.9
        return float(result)
    except (ValueError, ZeroDivisionError) as e:
        # Fallback for edge cases
        print(f"Warning: Error computing bone_length: {e}")
        return 0.0


def compute_joint_spacing(petal_length, petal_width, flexibility_factor, layer_index):
    """
    Compute joint spacing.
    
    Discovered Formula: x0 / ceil(3.0 * x0 * x2)
    R² Score: 0.978798
    Complexity: 12
    
    Args:
        petal_length: Feature value
        petal_width: Feature value
        flexibility_factor: Feature value
        layer_index: Feature value
    
    Returns:
        Computed joint spacing value
    """
    try:
        result = petal_length / math.ceil(3.0 * petal_length * flexibility_factor)
        return float(result)
    except (ValueError, ZeroDivisionError) as e:
        # Fallback for edge cases
        print(f"Warning: Error computing joint_spacing: {e}")
        return 0.0


def compute_bind_weight(petal_length, petal_width, flexibility_factor, layer_index):
    """
    Compute bind weight.
    
    Discovered Formula: x2 * (0.5 + x3 * 0.5)
    R² Score: 0.957870
    Complexity: 12
    
    Args:
        petal_length: Feature value
        petal_width: Feature value
        flexibility_factor: Feature value
        layer_index: Feature value
    
    Returns:
        Computed bind weight value
    """
    try:
        result = flexibility_factor * (0.5 + layer_index * 0.5)
        return float(result)
    except (ValueError, ZeroDivisionError) as e:
        # Fallback for edge cases
        print(f"Warning: Error computing bind_weight: {e}")
        return 0.0

