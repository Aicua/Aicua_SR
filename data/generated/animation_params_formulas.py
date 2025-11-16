"""
Auto-generated formulas for Animation Params
Generated: 2025-11-16 03:12:18

These formulas were discovered using Symbolic Regression (PySR).
DO NOT EDIT MANUALLY - regenerate using codegen_formulas.py
"""

import math


def compute_frequency(base_size, petal_mass, wind_speed, flexibility):
    """
    Compute frequency.
    
    Discovered Formula: 2.0 * sqrt(x3 / (x1 + 0.01)) * (1 + x2 * 0.1)
    R² Score: 0.984444
    Complexity: 10
    
    Args:
        base_size: Feature value
        petal_mass: Feature value
        wind_speed: Feature value
        flexibility: Feature value
    
    Returns:
        Computed frequency value
    """
    try:
        result = 2.0 * math.sqrt(flexibility / (petal_mass + 0.01)) * (1 + wind_speed * 0.1)
        return float(result)
    except (ValueError, ZeroDivisionError) as e:
        # Fallback for edge cases
        print(f"Warning: Error computing frequency: {e}")
        return 0.0


def compute_amplitude(base_size, petal_mass, wind_speed, flexibility):
    """
    Compute amplitude.
    
    Discovered Formula: x2 * x3 * 5.0 / sqrt(x1 + 0.1)
    R² Score: 0.987491
    Complexity: 8
    
    Args:
        base_size: Feature value
        petal_mass: Feature value
        wind_speed: Feature value
        flexibility: Feature value
    
    Returns:
        Computed amplitude value
    """
    try:
        result = wind_speed * flexibility * 5.0 / math.sqrt(petal_mass + 0.1)
        return float(result)
    except (ValueError, ZeroDivisionError) as e:
        # Fallback for edge cases
        print(f"Warning: Error computing amplitude: {e}")
        return 0.0


def compute_phase_offset(base_size, petal_mass, wind_speed, flexibility):
    """
    Compute phase offset.
    
    Discovered Formula: 2.1415
    R² Score: 0.982771
    Complexity: 11
    
    Args:
        base_size: Feature value
        petal_mass: Feature value
        wind_speed: Feature value
        flexibility: Feature value
    
    Returns:
        Computed phase offset value
    """
    try:
        result = 2.1415
        return float(result)
    except (ValueError, ZeroDivisionError) as e:
        # Fallback for edge cases
        print(f"Warning: Error computing phase_offset: {e}")
        return 0.0


def compute_damping_factor(base_size, petal_mass, wind_speed, flexibility):
    """
    Compute damping factor.
    
    Discovered Formula: 0.1 * x1 / (x3 + 0.01)
    R² Score: 0.968373
    Complexity: 10
    
    Args:
        base_size: Feature value
        petal_mass: Feature value
        wind_speed: Feature value
        flexibility: Feature value
    
    Returns:
        Computed damping factor value
    """
    try:
        result = 0.1 * petal_mass / (flexibility + 0.01)
        return float(result)
    except (ValueError, ZeroDivisionError) as e:
        # Fallback for edge cases
        print(f"Warning: Error computing damping_factor: {e}")
        return 0.0

