"""
Auto-generated formulas for Petal Geometry
Generated: 2025-11-16 03:12:18

These formulas were discovered using Symbolic Regression (PySR).
DO NOT EDIT MANUALLY - regenerate using codegen_formulas.py
"""

import math


def compute_base_width(base_size, layer_index, petal_index, opening_degree):
    """
    Compute base width.
    
    Discovered Formula: 0.3 * x0 * (0.7 + x1 * 0.1)
    R² Score: 0.952434
    Complexity: 9
    
    Args:
        base_size: Feature value
        layer_index: Feature value
        petal_index: Feature value
        opening_degree: Feature value
    
    Returns:
        Computed base width value
    """
    try:
        result = 0.3 * base_size * (0.7 + layer_index * 0.1)
        return float(result)
    except (ValueError, ZeroDivisionError) as e:
        # Fallback for edge cases
        print(f"Warning: Error computing base_width: {e}")
        return 0.0


def compute_length(base_size, layer_index, petal_index, opening_degree):
    """
    Compute length.
    
    Discovered Formula: 0.8 * x0 * (0.3 + x1 * 0.233)
    R² Score: 0.961281
    Complexity: 14
    
    Args:
        base_size: Feature value
        layer_index: Feature value
        petal_index: Feature value
        opening_degree: Feature value
    
    Returns:
        Computed length value
    """
    try:
        result = 0.8 * base_size * (0.3 + layer_index * 0.233)
        return float(result)
    except (ValueError, ZeroDivisionError) as e:
        # Fallback for edge cases
        print(f"Warning: Error computing length: {e}")
        return 0.0


def compute_curvature(base_size, layer_index, petal_index, opening_degree):
    """
    Compute curvature.
    
    Discovered Formula: (1.2 - x1 * 0.3) * (1 - x3 * 0.3)
    R² Score: 0.957999
    Complexity: 8
    
    Args:
        base_size: Feature value
        layer_index: Feature value
        petal_index: Feature value
        opening_degree: Feature value
    
    Returns:
        Computed curvature value
    """
    try:
        result = (1.2 - layer_index * 0.3) * (1 - opening_degree * 0.3)
        return float(result)
    except (ValueError, ZeroDivisionError) as e:
        # Fallback for edge cases
        print(f"Warning: Error computing curvature: {e}")
        return 0.0


def compute_twist_angle(base_size, layer_index, petal_index, opening_degree):
    """
    Compute twist angle.
    
    Discovered Formula: 137.5 * x2 / (x1 * 2 + 3)
    R² Score: 0.956054
    Complexity: 14
    
    Args:
        base_size: Feature value
        layer_index: Feature value
        petal_index: Feature value
        opening_degree: Feature value
    
    Returns:
        Computed twist angle value
    """
    try:
        result = 137.5 * petal_index / (layer_index * 2 + 3)
        return float(result)
    except (ValueError, ZeroDivisionError) as e:
        # Fallback for edge cases
        print(f"Warning: Error computing twist_angle: {e}")
        return 0.0


def compute_thickness(base_size, layer_index, petal_index, opening_degree):
    """
    Compute thickness.
    
    Discovered Formula: 0.02 * x0 * (1 + sqrt(x1/3) * 0.5)
    R² Score: 0.983441
    Complexity: 10
    
    Args:
        base_size: Feature value
        layer_index: Feature value
        petal_index: Feature value
        opening_degree: Feature value
    
    Returns:
        Computed thickness value
    """
    try:
        result = 0.02 * base_size * (1 + math.sqrt(layer_index/3) * 0.5)
        return float(result)
    except (ValueError, ZeroDivisionError) as e:
        # Fallback for edge cases
        print(f"Warning: Error computing thickness: {e}")
        return 0.0

