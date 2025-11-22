#!/usr/bin/env python3
"""
Generate CLI output from petal spline formulas.

This script demonstrates how to use discovered SR formulas to generate
CLI commands for petal geometry.

Usage:
    python generate_cli_from_formulas.py --base_size 3.0 --opening_degree 0.8
    python generate_cli_from_formulas.py --base_size 5.0 --opening_degree 0.5 --layer 1 --petal 2
"""

import argparse
import math


def compute_petal_spline(base_size: float, opening_degree: float,
                        layer_idx: int = 0, petal_idx: int = 0) -> dict:
    """
    Compute petal spline control points using mathematical formulas.

    This function uses the original mathematical formulas. After SR training,
    replace these with discovered formulas for better accuracy.

    Args:
        base_size: Overall rose size (2.0 - 8.0)
        opening_degree: 0.0 (closed) to 1.0 (fully open)
        layer_idx: Layer index (0-2)
        petal_idx: Petal index (0-25)

    Returns:
        Dictionary with control point coordinates and extrude_depth
    """

    # === LAYER FACTOR ===
    layer_factor = 0.8 + 0.1 * layer_idx

    # === PETAL HEIGHT ===
    petal_height = (
        base_size * layer_factor *
        (1.2 - opening_degree * 0.3)
    )

    # === WIDTH CALCULATIONS ===
    base_spread = (
        base_size * 0.30 * layer_factor *
        (1 + opening_degree * 0.2)
    )

    # Use fixed values instead of random for deterministic output
    # (SR formulas should learn these relationships)
    base_width_factor = 0.5  # Fixed at 50% (was random 0.3-0.7)
    base_width = base_spread * base_width_factor

    lower_width_factor = 1.05  # Fixed at 105% (was random 0.95-1.15)
    lower_width = base_spread * lower_width_factor

    mid_low_width = base_spread * 1.4
    upper_mid_width = base_spread * 1.6  # WIDEST
    upper_width = base_spread * 1.3
    near_tip_width = base_spread * 0.8

    # Tip x offset (for asymmetry/tilt)
    tip_x_offset = base_size * 0.02 * layer_idx * opening_degree

    # === 15 CONTROL POINTS ===

    # CP1: Base center
    cp1_x = 0.0
    cp1_y = 0.0

    # CP2: Base left - NARROW
    base_left_height = 0.055  # Fixed at 5.5% (was random 0.03-0.08)
    cp2_x = -base_width * 0.5
    cp2_y = petal_height * base_left_height

    # CP3: Lower left
    cp3_x = -lower_width * 0.5
    cp3_y = petal_height * 0.25

    # CP4: Mid-low left
    cp4_x = -mid_low_width * 0.5
    cp4_y = petal_height * 0.45

    # CP5: Upper-mid left - WIDEST
    cp5_x = -upper_mid_width * 0.5
    cp5_y = petal_height * 0.62

    # CP6: Upper left
    cp6_x = -upper_width * 0.5
    cp6_y = petal_height * 0.78

    # CP7: Near-tip left
    cp7_x = -near_tip_width * 0.5
    cp7_y = petal_height * 0.92

    # CP8: Tip center
    cp8_x = tip_x_offset
    cp8_y = petal_height

    # CP9: Near-tip right
    cp9_x = near_tip_width * 0.5
    cp9_y = petal_height * 0.92

    # CP10: Upper right
    cp10_x = upper_width * 0.5
    cp10_y = petal_height * 0.78

    # CP11: Upper-mid right - WIDEST
    cp11_x = upper_mid_width * 0.5
    cp11_y = petal_height * 0.62

    # CP12: Mid-low right
    cp12_x = mid_low_width * 0.5
    cp12_y = petal_height * 0.45

    # CP13: Lower right
    cp13_x = lower_width * 0.5
    cp13_y = petal_height * 0.25

    # CP14: Base right - NARROW
    cp14_x = base_width * 0.5
    cp14_y = petal_height * base_left_height

    # CP15: Close spline
    cp15_x = 0.0
    cp15_y = 0.0

    # === EXTRUDE DEPTH ===
    thickness_base = 0.0005  # Ultra-thin like real rose petals (0.001-0.0015)
    thickness = (
        thickness_base * base_size *
        (1 - layer_idx * 0.1) *
        (1 - opening_degree * 0.3)
    )
    extrude_depth = max(0.001, min(0.0015, thickness))  # Clamp to [0.001, 0.0015]

    return {
        'cp1_x': cp1_x, 'cp1_y': cp1_y,
        'cp2_x': cp2_x, 'cp2_y': cp2_y,
        'cp3_x': cp3_x, 'cp3_y': cp3_y,
        'cp4_x': cp4_x, 'cp4_y': cp4_y,
        'cp5_x': cp5_x, 'cp5_y': cp5_y,
        'cp6_x': cp6_x, 'cp6_y': cp6_y,
        'cp7_x': cp7_x, 'cp7_y': cp7_y,
        'cp8_x': cp8_x, 'cp8_y': cp8_y,
        'cp9_x': cp9_x, 'cp9_y': cp9_y,
        'cp10_x': cp10_x, 'cp10_y': cp10_y,
        'cp11_x': cp11_x, 'cp11_y': cp11_y,
        'cp12_x': cp12_x, 'cp12_y': cp12_y,
        'cp13_x': cp13_x, 'cp13_y': cp13_y,
        'cp14_x': cp14_x, 'cp14_y': cp14_y,
        'cp15_x': cp15_x, 'cp15_y': cp15_y,
        'extrude_depth': extrude_depth,
    }


def format_cli_output(control_points: dict, layer_idx: int = 1, petal_idx: int = 1) -> str:
    """
    Format control points as CLI commands.

    Args:
        control_points: Dictionary with cp1_x, cp1_y, ..., cp15_x, cp15_y, extrude_depth
        layer_idx: Layer number for naming (1-based)
        petal_idx: Petal number for naming (1-based)

    Returns:
        CLI commands as string
    """

    # Build spline point list
    spline_points = []
    for i in range(1, 16):
        x = control_points[f'cp{i}_x']
        y = control_points[f'cp{i}_y']
        spline_points.append(f'{x:.4f}')
        spline_points.append(f'{y:.4f}')

    spline_coords = ' '.join(spline_points)
    extrude_depth = control_points['extrude_depth']

    petal_name = f'petal_L{layer_idx}_P{petal_idx}'

    cli = f"""2d;
obj {petal_name};
spline {spline_coords};
exit;
sketch_extrude {petal_name} {extrude_depth:.4f};"""

    return cli


def main():
    parser = argparse.ArgumentParser(
        description='Generate CLI output from petal spline formulas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --base_size 3.0 --opening_degree 0.8
  %(prog)s --base_size 5.0 --opening_degree 0.5 --layer 1 --petal 2
  %(prog)s -b 4.0 -o 0.6 -l 0 -p 0
        """
    )

    parser.add_argument('-b', '--base_size', type=float, required=True,
                       help='Overall rose size (2.0 - 8.0)')
    parser.add_argument('-o', '--opening_degree', type=float, required=True,
                       help='Opening degree: 0.0 (closed) to 1.0 (fully open)')
    parser.add_argument('-l', '--layer', type=int, default=1,
                       help='Layer index for naming (default: 1)')
    parser.add_argument('-p', '--petal', type=int, default=1,
                       help='Petal index for naming (default: 1)')
    parser.add_argument('--show-coords', action='store_true',
                       help='Show individual control point coordinates')

    args = parser.parse_args()

    # Validate inputs
    if not (2.0 <= args.base_size <= 8.0):
        print(f"Warning: base_size {args.base_size} is outside typical range (2.0 - 8.0)")

    if not (0.0 <= args.opening_degree <= 1.0):
        print(f"Warning: opening_degree {args.opening_degree} is outside range (0.0 - 1.0)")

    # Compute control points
    control_points = compute_petal_spline(
        base_size=args.base_size,
        opening_degree=args.opening_degree,
        layer_idx=args.layer,
        petal_idx=args.petal
    )

    # Show coordinates if requested
    if args.show_coords:
        print("=" * 60)
        print("Control Point Coordinates:")
        print("=" * 60)
        for i in range(1, 16):
            x = control_points[f'cp{i}_x']
            y = control_points[f'cp{i}_y']
            print(f"  CP{i:2d}: ({x:8.4f}, {y:8.4f})")
        print(f"  Extrude depth: {control_points['extrude_depth']:.6f}")
        print()

    # Generate CLI output
    cli_output = format_cli_output(control_points, args.layer, args.petal)

    print("=" * 60)
    print(f"CLI Output (base_size={args.base_size}, opening_degree={args.opening_degree})")
    print("=" * 60)
    print(cli_output)
    print()


if __name__ == '__main__':
    main()
