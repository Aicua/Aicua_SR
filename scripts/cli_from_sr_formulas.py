#!/usr/bin/env python3
"""
Generate CLI output using SR-discovered formulas.

This script demonstrates how to replace the original mathematical formulas
with symbolic regression (SR) discovered formulas from Kaggle training.

To use this script:
1. Run Kaggle training to discover formulas
2. Copy the discovered formulas into this file
3. Run the script to generate CLI output

Usage:
    python cli_from_sr_formulas.py --base_size 3.0 --opening_degree 0.8
"""

import argparse
import math


# =============================================================================
# SR-DISCOVERED FORMULAS
# Replace these placeholder formulas with actual SR-discovered formulas
# from your Kaggle training results
# =============================================================================

def sr_cp1_x(base_size, opening_degree):
    """SR formula for cp1_x - REPLACE WITH ACTUAL FORMULA"""
    # Example: return 0.0  (cp1 is always at origin)
    return 0.0


def sr_cp1_y(base_size, opening_degree):
    """SR formula for cp1_y - REPLACE WITH ACTUAL FORMULA"""
    # Example: return 0.0  (cp1 is always at origin)
    return 0.0


def sr_cp2_x(base_size, opening_degree):
    """SR formula for cp2_x - REPLACE WITH ACTUAL FORMULA"""
    # Placeholder - replace with actual SR formula
    # Example from training: -0.12 * base_size * (1.08 + 0.06 * opening_degree)
    return -0.12 * base_size * (1.08 + 0.06 * opening_degree)


def sr_cp2_y(base_size, opening_degree):
    """SR formula for cp2_y - REPLACE WITH ACTUAL FORMULA"""
    # Placeholder - replace with actual SR formula
    # Example: 0.066 * base_size * (1.2 - 0.3 * opening_degree)
    return 0.066 * base_size * (1.2 - 0.3 * opening_degree)


def sr_cp3_x(base_size, opening_degree):
    """SR formula for cp3_x - REPLACE WITH ACTUAL FORMULA"""
    return -0.2625 * base_size * (1.08 + 0.06 * opening_degree)


def sr_cp3_y(base_size, opening_degree):
    """SR formula for cp3_y - REPLACE WITH ACTUAL FORMULA"""
    return 0.25 * base_size * (1.2 - 0.3 * opening_degree)


def sr_cp4_x(base_size, opening_degree):
    """SR formula for cp4_x - REPLACE WITH ACTUAL FORMULA"""
    return -0.35 * base_size * (1.08 + 0.06 * opening_degree)


def sr_cp4_y(base_size, opening_degree):
    """SR formula for cp4_y - REPLACE WITH ACTUAL FORMULA"""
    return 0.45 * base_size * (1.2 - 0.3 * opening_degree)


def sr_cp5_x(base_size, opening_degree):
    """SR formula for cp5_x - REPLACE WITH ACTUAL FORMULA"""
    return -0.40 * base_size * (1.08 + 0.06 * opening_degree)


def sr_cp5_y(base_size, opening_degree):
    """SR formula for cp5_y - REPLACE WITH ACTUAL FORMULA"""
    return 0.62 * base_size * (1.2 - 0.3 * opening_degree)


def sr_cp6_x(base_size, opening_degree):
    """SR formula for cp6_x - REPLACE WITH ACTUAL FORMULA"""
    return -0.325 * base_size * (1.08 + 0.06 * opening_degree)


def sr_cp6_y(base_size, opening_degree):
    """SR formula for cp6_y - REPLACE WITH ACTUAL FORMULA"""
    return 0.78 * base_size * (1.2 - 0.3 * opening_degree)


def sr_cp7_x(base_size, opening_degree):
    """SR formula for cp7_x - REPLACE WITH ACTUAL FORMULA"""
    return -0.20 * base_size * (1.08 + 0.06 * opening_degree)


def sr_cp7_y(base_size, opening_degree):
    """SR formula for cp7_y - REPLACE WITH ACTUAL FORMULA"""
    return 0.92 * base_size * (1.2 - 0.3 * opening_degree)


def sr_cp8_x(base_size, opening_degree):
    """SR formula for cp8_x - REPLACE WITH ACTUAL FORMULA"""
    # Tip x-offset (usually near 0 for layer 0)
    return 0.0


def sr_cp8_y(base_size, opening_degree):
    """SR formula for cp8_y - REPLACE WITH ACTUAL FORMULA"""
    # Petal height
    return base_size * (1.2 - 0.3 * opening_degree)


def sr_cp9_x(base_size, opening_degree):
    """SR formula for cp9_x - REPLACE WITH ACTUAL FORMULA"""
    # Symmetric to cp7
    return 0.20 * base_size * (1.08 + 0.06 * opening_degree)


def sr_cp9_y(base_size, opening_degree):
    """SR formula for cp9_y - REPLACE WITH ACTUAL FORMULA"""
    return 0.92 * base_size * (1.2 - 0.3 * opening_degree)


def sr_cp10_x(base_size, opening_degree):
    """SR formula for cp10_x - REPLACE WITH ACTUAL FORMULA"""
    # Symmetric to cp6
    return 0.325 * base_size * (1.08 + 0.06 * opening_degree)


def sr_cp10_y(base_size, opening_degree):
    """SR formula for cp10_y - REPLACE WITH ACTUAL FORMULA"""
    return 0.78 * base_size * (1.2 - 0.3 * opening_degree)


def sr_cp11_x(base_size, opening_degree):
    """SR formula for cp11_x - REPLACE WITH ACTUAL FORMULA"""
    # Symmetric to cp5 - WIDEST
    return 0.40 * base_size * (1.08 + 0.06 * opening_degree)


def sr_cp11_y(base_size, opening_degree):
    """SR formula for cp11_y - REPLACE WITH ACTUAL FORMULA"""
    return 0.62 * base_size * (1.2 - 0.3 * opening_degree)


def sr_cp12_x(base_size, opening_degree):
    """SR formula for cp12_x - REPLACE WITH ACTUAL FORMULA"""
    # Symmetric to cp4
    return 0.35 * base_size * (1.08 + 0.06 * opening_degree)


def sr_cp12_y(base_size, opening_degree):
    """SR formula for cp12_y - REPLACE WITH ACTUAL FORMULA"""
    return 0.45 * base_size * (1.2 - 0.3 * opening_degree)


def sr_cp13_x(base_size, opening_degree):
    """SR formula for cp13_x - REPLACE WITH ACTUAL FORMULA"""
    # Symmetric to cp3
    return 0.2625 * base_size * (1.08 + 0.06 * opening_degree)


def sr_cp13_y(base_size, opening_degree):
    """SR formula for cp13_y - REPLACE WITH ACTUAL FORMULA"""
    return 0.25 * base_size * (1.2 - 0.3 * opening_degree)


def sr_cp14_x(base_size, opening_degree):
    """SR formula for cp14_x - REPLACE WITH ACTUAL FORMULA"""
    # Symmetric to cp2
    return 0.12 * base_size * (1.08 + 0.06 * opening_degree)


def sr_cp14_y(base_size, opening_degree):
    """SR formula for cp14_y - REPLACE WITH ACTUAL FORMULA"""
    return 0.066 * base_size * (1.2 - 0.3 * opening_degree)


def sr_cp15_x(base_size, opening_degree):
    """SR formula for cp15_x - REPLACE WITH ACTUAL FORMULA"""
    # Close spline - back to origin
    return 0.0


def sr_cp15_y(base_size, opening_degree):
    """SR formula for cp15_y - REPLACE WITH ACTUAL FORMULA"""
    # Close spline - back to origin
    return 0.0


def sr_extrude_depth(base_size, opening_degree):
    """SR formula for extrude_depth - REPLACE WITH ACTUAL FORMULA"""
    # Ultra-thin extrude
    # Example: 0.005 * base_size * (1 - 0.3 * opening_degree)
    return max(0.001, 0.005 * base_size * (1.0 - 0.3 * opening_degree))


# =============================================================================
# CLI GENERATION
# =============================================================================

def compute_petal_from_sr(base_size: float, opening_degree: float) -> dict:
    """
    Compute petal control points using SR-discovered formulas.

    Args:
        base_size: Overall rose size (2.0 - 8.0)
        opening_degree: 0.0 (closed) to 1.0 (fully open)

    Returns:
        Dictionary with control point coordinates and extrude_depth
    """

    return {
        'cp1_x': sr_cp1_x(base_size, opening_degree),
        'cp1_y': sr_cp1_y(base_size, opening_degree),
        'cp2_x': sr_cp2_x(base_size, opening_degree),
        'cp2_y': sr_cp2_y(base_size, opening_degree),
        'cp3_x': sr_cp3_x(base_size, opening_degree),
        'cp3_y': sr_cp3_y(base_size, opening_degree),
        'cp4_x': sr_cp4_x(base_size, opening_degree),
        'cp4_y': sr_cp4_y(base_size, opening_degree),
        'cp5_x': sr_cp5_x(base_size, opening_degree),
        'cp5_y': sr_cp5_y(base_size, opening_degree),
        'cp6_x': sr_cp6_x(base_size, opening_degree),
        'cp6_y': sr_cp6_y(base_size, opening_degree),
        'cp7_x': sr_cp7_x(base_size, opening_degree),
        'cp7_y': sr_cp7_y(base_size, opening_degree),
        'cp8_x': sr_cp8_x(base_size, opening_degree),
        'cp8_y': sr_cp8_y(base_size, opening_degree),
        'cp9_x': sr_cp9_x(base_size, opening_degree),
        'cp9_y': sr_cp9_y(base_size, opening_degree),
        'cp10_x': sr_cp10_x(base_size, opening_degree),
        'cp10_y': sr_cp10_y(base_size, opening_degree),
        'cp11_x': sr_cp11_x(base_size, opening_degree),
        'cp11_y': sr_cp11_y(base_size, opening_degree),
        'cp12_x': sr_cp12_x(base_size, opening_degree),
        'cp12_y': sr_cp12_y(base_size, opening_degree),
        'cp13_x': sr_cp13_x(base_size, opening_degree),
        'cp13_y': sr_cp13_y(base_size, opening_degree),
        'cp14_x': sr_cp14_x(base_size, opening_degree),
        'cp14_y': sr_cp14_y(base_size, opening_degree),
        'cp15_x': sr_cp15_x(base_size, opening_degree),
        'cp15_y': sr_cp15_y(base_size, opening_degree),
        'extrude_depth': sr_extrude_depth(base_size, opening_degree),
    }


def format_cli(control_points: dict, layer: int = 1, petal: int = 1) -> str:
    """Format control points as CLI commands."""

    # Build spline coordinates
    coords = []
    for i in range(1, 16):
        coords.append(f"{control_points[f'cp{i}_x']:.4f}")
        coords.append(f"{control_points[f'cp{i}_y']:.4f}")

    spline_str = ' '.join(coords)
    petal_name = f'petal_L{layer}_P{petal}'
    depth = control_points['extrude_depth']

    return f"""2d;
obj {petal_name};
spline {spline_str};
exit;
sketch_extrude {petal_name} {depth:.3f};"""


def main():
    parser = argparse.ArgumentParser(
        description='Generate CLI using SR-discovered formulas',
        epilog='NOTE: Replace placeholder formulas with actual SR formulas from Kaggle training'
    )

    parser.add_argument('-b', '--base_size', type=float, required=True,
                       help='Overall rose size (2.0 - 8.0)')
    parser.add_argument('-o', '--opening_degree', type=float, required=True,
                       help='Opening degree: 0.0 (closed) to 1.0 (fully open)')
    parser.add_argument('-l', '--layer', type=int, default=1,
                       help='Layer number for naming (default: 1)')
    parser.add_argument('-p', '--petal', type=int, default=1,
                       help='Petal number for naming (default: 1)')
    parser.add_argument('--show-coords', action='store_true',
                       help='Show control point coordinates')

    args = parser.parse_args()

    # Compute using SR formulas
    control_points = compute_petal_from_sr(args.base_size, args.opening_degree)

    # Show coordinates if requested
    if args.show_coords:
        print("=" * 60)
        print("SR-Computed Control Points:")
        print("=" * 60)
        for i in range(1, 16):
            x = control_points[f'cp{i}_x']
            y = control_points[f'cp{i}_y']
            print(f"  CP{i:2d}: ({x:8.4f}, {y:8.4f})")
        print(f"  Extrude: {control_points['extrude_depth']:.6f}")
        print()

    # Generate CLI
    cli = format_cli(control_points, args.layer, args.petal)

    print("=" * 60)
    print(f"CLI from SR Formulas (base_size={args.base_size}, opening={args.opening_degree})")
    print("=" * 60)
    print(cli)
    print()
    print("NOTE: Update the sr_* functions with actual formulas from Kaggle training")
    print()


if __name__ == '__main__':
    main()
