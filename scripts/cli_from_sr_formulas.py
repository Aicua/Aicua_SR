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
    """SR formula: (-opening_degree + opening_degree)**4 (R²=1.0000)"""
    return 0.0  # Simplifies to 0


def sr_cp1_y(base_size, opening_degree):
    """SR formula: (-opening_degree + opening_degree)**2 (R²=1.0000)"""
    return 0.0  # Simplifies to 0


def sr_cp2_x(base_size, opening_degree):
    """SR formula: base_size*(-0.06660193) (R²=0.6408)"""
    return base_size * (-0.06660193)


def sr_cp2_y(base_size, opening_degree):
    """SR formula: base_size*0.046049967 (R²=0.5716)"""
    return base_size * 0.046049967


def sr_cp3_x(base_size, opening_degree):
    """SR formula: (base_size + opening_degree)*(-0.12700757) (R²=0.9532)"""
    return (base_size + opening_degree) * (-0.12700757)


def sr_cp3_y(base_size, opening_degree):
    """SR formula: base_size*(0.23950993 - 0.059578367*opening_degree) (R²=0.9917)"""
    return base_size * (0.23950993 - 0.059578367 * opening_degree)


def sr_cp4_x(base_size, opening_degree):
    """SR formula: base_size*(opening_degree*(-0.033973925) - 1*0.16772935) (R²=0.9914)"""
    return base_size * (opening_degree * (-0.033973925) - 0.16772935)


def sr_cp4_y(base_size, opening_degree):
    """SR formula: base_size*(0.43164855 - 0.10751137*opening_degree) (R²=0.9920)"""
    return base_size * (0.43164855 - 0.10751137 * opening_degree)


def sr_cp5_x(base_size, opening_degree):
    """SR formula: base_size/(opening_degree - 5.2565217) (R²=0.9910)"""
    try:
        return base_size / (opening_degree - 5.2565217)
    except ZeroDivisionError:
        return -0.19


def sr_cp5_y(base_size, opening_degree):
    """SR formula: base_size*(-0.14710286)*(opening_degree - 1*4.039837) (R²=0.9916)"""
    return base_size * (-0.14710286) * (opening_degree - 4.039837)


def sr_cp6_x(base_size, opening_degree):
    """SR formula: base_size*(opening_degree*(-0.030374026) - 0.1565289) (R²=0.9916)"""
    return base_size * (opening_degree * (-0.030374026) - 0.1565289)


def sr_cp6_y(base_size, opening_degree):
    """SR formula: 0.736865096629327*base_size/sqrt(0.542970170630548*opening_degree + 1) (R²=0.9881)"""
    return 0.736865096629327 * base_size / math.sqrt(0.542970170630548 * opening_degree + 1)


def sr_cp7_x(base_size, opening_degree):
    """SR formula: base_size*(-0.019377816*opening_degree - 0.095919825) (R²=0.9918)"""
    return base_size * (-0.019377816 * opening_degree - 0.095919825)


def sr_cp7_y(base_size, opening_degree):
    """SR formula: 0.905464601038703*base_size/sqrt(0.819866143734177*opening_degree + 1) (R²=0.9907)"""
    return 0.905464601038703 * base_size / math.sqrt(0.819866143734177 * opening_degree + 1)


def sr_cp8_x(base_size, opening_degree):
    """SR formula: 7.970367e-6/(0.20965719 - opening_degree) (R²=0.0054)"""
    # R² very low - essentially 0
    return 0.0


def sr_cp8_y(base_size, opening_degree):
    """SR formula: base_size*(0.961326 - opening_degree/4.117992) (R²=0.9915)"""
    return base_size * (0.961326 - opening_degree / 4.117992)


def sr_cp9_x(base_size, opening_degree):
    """SR formula: base_size*(opening_degree*0.01915575 + 0.09590936) (R²=0.9911)"""
    return base_size * (opening_degree * 0.01915575 + 0.09590936)


def sr_cp9_y(base_size, opening_degree):
    """SR formula: 0.892451988766659*base_size/sqrt(0.796470552253566*opening_degree + 1) (R²=0.9892)"""
    return 0.892451988766659 * base_size / math.sqrt(0.796470552253566 * opening_degree + 1)


def sr_cp10_x(base_size, opening_degree):
    """SR formula: base_size*(opening_degree*0.031619746 + 0.15594344) (R²=0.9919)"""
    return base_size * (opening_degree * 0.031619746 + 0.15594344)


def sr_cp10_y(base_size, opening_degree):
    """SR formula: base_size*(opening_degree*0.35291463 - 1*1.4014238)*(-0.53408945) (R²=0.9918)"""
    return base_size * (opening_degree * 0.35291463 - 1.4014238) * (-0.53408945)


def sr_cp11_x(base_size, opening_degree):
    """SR formula: base_size*(opening_degree*0.037792355 + 0.19234422) (R²=0.9913)"""
    return base_size * (opening_degree * 0.037792355 + 0.19234422)


def sr_cp11_y(base_size, opening_degree):
    """SR formula: base_size*(0.5965338 - 0.15095319*opening_degree) (R²=0.9916)"""
    return base_size * (0.5965338 - 0.15095319 * opening_degree)


def sr_cp12_x(base_size, opening_degree):
    """SR formula: base_size*(0.16823693 - (-0.033299472)*opening_degree) (R²=0.9917)"""
    return base_size * (0.16823693 + 0.033299472 * opening_degree)


def sr_cp12_y(base_size, opening_degree):
    """SR formula: base_size*(0.43128464 - 0.107101806*opening_degree) (R²=0.9922)"""
    return base_size * (0.43128464 - 0.107101806 * opening_degree)


def sr_cp13_x(base_size, opening_degree):
    """SR formula: base_size*(opening_degree*0.025087584 + 0.1258421) (R²=0.9620)"""
    return base_size * (opening_degree * 0.025087584 + 0.1258421)


def sr_cp13_y(base_size, opening_degree):
    """SR formula: base_size/(opening_degree*1.3537062 + 4.1148305) (R²=0.9917)"""
    return base_size / (opening_degree * 1.3537062 + 4.1148305)


def sr_cp14_x(base_size, opening_degree):
    """SR formula: base_size*0.0664584 (R²=0.6409)"""
    return base_size * 0.0664584


def sr_cp14_y(base_size, opening_degree):
    """SR formula: base_size*0.046029154 (R²=0.5704)"""
    return base_size * 0.046029154


def sr_cp15_x(base_size, opening_degree):
    """SR formula: 0 (R²=1.0000)"""
    return 0.0


def sr_cp15_y(base_size, opening_degree):
    """SR formula: -opening_degree + opening_degree (R²=1.0000)"""
    return 0.0  # Simplifies to 0


def sr_extrude_depth(base_size, opening_degree):
    """SR formula: base_size*(0.0049875244 + opening_degree*(-0.0014847366)) (R²=0.9919)
    NOTE: Scaled down 10x because SR was trained on old data with thickness_base=0.005
    """
    # Ultra-thin extrude (0.001-0.0015) - MUST stay in this range
    # Scale SR formula down 10x to match new thickness_base=0.0005
    thickness = base_size * (0.00049875244 + opening_degree * (-0.00014847366))
    return max(0.001, min(0.0015, thickness))  # Clamp to [0.001, 0.0015]


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
sketch_extrude {petal_name} {depth:.4f};"""


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


if __name__ == '__main__':
    main()
