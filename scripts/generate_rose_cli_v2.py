#!/usr/bin/env python3
"""
Generate Rose CAD CLI V2 using 5 independent bones.

Key changes from v1:
- Uses 5 independent bones (no parent-child)
- Supports deformation types: straight, s_curve, c_curve, wave
- Each bone can be rotated independently without affecting others
"""

import argparse
import sys
from pathlib import Path

# Add generated module path
sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "generated"))


class RoseCLIGeneratorV2:
    """Generate CLI commands for 3D rose with 5 independent bones."""

    def __init__(self):
        self.cli_commands = []

    def compute_bone_positions(self, petal_height: float, base_spread: float,
                                deformation_type: int, intensity: float) -> dict:
        """
        Compute positions for 5 independent bones - T-shape.

        Args:
            petal_height: Total height of petal
            base_spread: Base width parameter
            deformation_type: 0=straight, 1=s_curve, 2=c_curve, 3=wave
            intensity: Deformation intensity (0.0 - 1.0)

        Returns:
            Dictionary with bone positions
        """
        import numpy as np

        # Width at 62% (widest point)
        width_at_62 = base_spread * 1.6

        # Calculate x-offsets based on deformation type for vertical bones
        offset_45 = 0.0
        offset_78 = 0.0
        offset_100 = 0.0
        # Edge curl for horizontal bones
        curl_left = 0.0
        curl_right = 0.0

        if deformation_type == 0:  # Straight
            pass

        elif deformation_type == 1:  # S-curve
            offset_45 = base_spread * 0.2 * intensity
            offset_78 = -base_spread * 0.15 * intensity
            offset_100 = base_spread * 0.1 * intensity
            curl_left = width_at_62 * 0.1 * intensity
            curl_right = width_at_62 * 0.1 * intensity

        elif deformation_type == 2:  # C-curve
            offset_45 = -base_spread * 0.05 * intensity
            offset_78 = -base_spread * 0.15 * intensity
            offset_100 = -base_spread * 0.1 * intensity
            curl_left = width_at_62 * 0.35 * intensity
            curl_right = width_at_62 * 0.35 * intensity

        elif deformation_type == 3:  # Wave
            offset_45 = base_spread * 0.15 * np.sin(0.45 * np.pi * 2) * intensity
            offset_78 = base_spread * 0.15 * np.sin(0.78 * np.pi * 2) * intensity
            offset_100 = base_spread * 0.15 * np.sin(1.0 * np.pi * 2) * intensity
            curl_left = width_at_62 * 0.2 * np.sin(0.62 * np.pi * 3) * intensity
            curl_right = width_at_62 * 0.2 * np.sin(0.62 * np.pi * 3 + np.pi) * intensity

        return {
            # Vertical spine bones
            'bone_base': {
                'start_x': 0.0, 'start_y': 0.0,
                'end_x': offset_45, 'end_y': petal_height * 0.45
            },
            'bone_mid': {
                'start_x': offset_45, 'start_y': petal_height * 0.45,
                'end_x': offset_78, 'end_y': petal_height * 0.78
            },
            'bone_tip': {
                'start_x': offset_78, 'start_y': petal_height * 0.78,
                'end_x': offset_100, 'end_y': petal_height
            },
            # Horizontal edge bones at 62%
            'bone_left': {
                'start_x': 0.0, 'start_y': petal_height * 0.62,
                'end_x': -width_at_62 / 2 + curl_left, 'end_y': petal_height * 0.62
            },
            'bone_right': {
                'start_x': 0.0, 'start_y': petal_height * 0.62,
                'end_x': width_at_62 / 2 - curl_right, 'end_y': petal_height * 0.62
            }
        }

    def generate_petal(self, layer_idx: int, petal_idx: int, base_size: float,
                       opening_degree: float, deformation_type: int = 0,
                       intensity: float = 0.5) -> dict:
        """Generate CLI for a single petal with 5 independent bones."""

        # Calculate petal dimensions
        layer_factor = 0.8 + 0.1 * layer_idx
        petal_height = base_size * layer_factor * (1.2 - opening_degree * 0.3)
        base_spread = base_size * 0.30 * layer_factor * (1 + opening_degree * 0.2)

        # Simplified geometry parameters
        base_width = base_spread * 2
        length = petal_height
        curvature = 0.3 + opening_degree * 0.4
        twist_angle = layer_idx * 5 + petal_idx * 2
        thickness = 0.01 * base_size

        petal_name = f"petal_L{layer_idx}_P{petal_idx}"

        # Generate geometry CLI
        geometry_cli = [
            f"# {petal_name} - Layer {layer_idx}, Petal {petal_idx}",
            f"obj {petal_name};",
            f"bezier_surface {petal_name} {length:.4f} {base_width:.4f} {curvature:.4f} {twist_angle:.4f} {thickness:.6f};",
        ]

        # Compute bone positions
        bones = self.compute_bone_positions(petal_height, base_spread,
                                             deformation_type, intensity)

        armature_name = f"{petal_name}_rig"

        rigging_cli = [
            f"",
            f"# Rigging for {petal_name} (v7 - T-shape 5 bones)",
            f"create_armature {armature_name};",
        ]

        # Generate 5 independent bones - T-shape (NO parent_bone commands!)
        bone_names = ['bone_base', 'bone_mid', 'bone_tip', 'bone_left', 'bone_right']
        for bone_name in bone_names:
            pos = bones[bone_name]
            rigging_cli.append(
                f"add_bone {armature_name} {bone_name} "
                f"{pos['start_x']:.4f} {pos['start_y']:.4f} 0 "
                f"{pos['end_x']:.4f} {pos['end_y']:.4f} 0;"
            )

        rigging_cli.append(f"finalize_bones {armature_name};")
        rigging_cli.append(f"bind_armature {armature_name} {petal_name};")

        return {
            'geometry': geometry_cli,
            'rigging': rigging_cli,
            'params': {
                'base_width': base_width,
                'length': length,
                'petal_height': petal_height,
                'deformation_type': deformation_type,
                'intensity': intensity,
            }
        }

    def generate_rose(self, base_size: float = 2.0, opening_degree: float = 0.8,
                      n_layers: int = 3, deformation_type: int = 0,
                      intensity: float = 0.5) -> str:
        """
        Generate complete rose CLI with 5 independent bones per petal.

        Args:
            base_size: Overall size of rose
            opening_degree: How open (0.0 = bud, 1.0 = fully open)
            n_layers: Number of petal layers (1-3)
            deformation_type: 0=straight, 1=s_curve, 2=c_curve, 3=wave
            intensity: Deformation intensity (0.0 - 1.0)

        Returns:
            Complete CLI string
        """
        deform_names = ['straight', 's_curve', 'c_curve', 'wave']
        petals_per_layer = [5, 8, 13]

        all_cli = [
            "# Rose CAD Generation V2",
            f"# Base Size: {base_size}",
            f"# Opening Degree: {opening_degree}",
            f"# Layers: {n_layers}",
            f"# Deformation: {deform_names[deformation_type]} (intensity={intensity})",
            f"# Using 5 INDEPENDENT bones per petal (no parent-child)",
            "",
            "2d;",
            "",
        ]

        total_petals = 0

        for layer_idx in range(1, n_layers + 1):
            n_petals = petals_per_layer[layer_idx - 1]

            all_cli.append(f"# ====== LAYER {layer_idx} ({n_petals} petals) ======")
            all_cli.append("")

            for petal_idx in range(n_petals):
                petal_data = self.generate_petal(
                    layer_idx, petal_idx, base_size, opening_degree,
                    deformation_type, intensity
                )

                # Add geometry
                all_cli.extend(petal_data['geometry'])
                # Add rigging
                all_cli.extend(petal_data['rigging'])
                all_cli.append("")

                total_petals += 1

        # Exit and summary
        all_cli.append("exit;")
        all_cli.append("")
        all_cli.append(f"# Total petals: {total_petals}")
        all_cli.append(f"# Total bones: {total_petals * 5}")
        all_cli.append(f"# Each petal has 5 independent bones (no hierarchy)")

        return '\n'.join(all_cli)

    def save_cli(self, cli_text: str, output_path: str):
        """Save CLI to file."""
        with open(output_path, 'w') as f:
            f.write(cli_text)
        print(f"Saved CLI to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Rose CAD CLI V2 with 5 independent bones"
    )

    parser.add_argument(
        '--size', type=float, default=2.0,
        help='Base size of the rose (default: 2.0)'
    )
    parser.add_argument(
        '--opening', type=float, default=0.8,
        help='Opening degree 0.0-1.0 (default: 0.8)'
    )
    parser.add_argument(
        '--layers', type=int, default=3,
        help='Number of petal layers 1-3 (default: 3)'
    )
    parser.add_argument(
        '--deformation', type=int, default=0,
        help='Deformation type: 0=straight, 1=s_curve, 2=c_curve, 3=wave (default: 0)'
    )
    parser.add_argument(
        '--intensity', type=float, default=0.5,
        help='Deformation intensity 0.0-1.0 (default: 0.5)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output file path (default: stdout)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Show detailed generation info'
    )

    args = parser.parse_args()

    # Validate inputs
    if args.opening < 0 or args.opening > 1:
        print("Error: opening must be between 0.0 and 1.0")
        sys.exit(1)

    if args.layers < 1 or args.layers > 3:
        print("Error: layers must be 1, 2, or 3")
        sys.exit(1)

    if args.deformation < 0 or args.deformation > 3:
        print("Error: deformation must be 0, 1, 2, or 3")
        sys.exit(1)

    if args.intensity < 0 or args.intensity > 1:
        print("Error: intensity must be between 0.0 and 1.0")
        sys.exit(1)

    print("=" * 60)
    print("Rose CLI Generator V2 (5 Independent Bones)")
    print("=" * 60)

    deform_names = ['straight', 's_curve', 'c_curve', 'wave']

    if args.verbose:
        print(f"Parameters:")
        print(f"  Base Size: {args.size}")
        print(f"  Opening Degree: {args.opening}")
        print(f"  Layers: {args.layers}")
        print(f"  Deformation: {deform_names[args.deformation]}")
        print(f"  Intensity: {args.intensity}")
        print()

    # Generate
    generator = RoseCLIGeneratorV2()
    cli = generator.generate_rose(
        base_size=args.size,
        opening_degree=args.opening,
        n_layers=args.layers,
        deformation_type=args.deformation,
        intensity=args.intensity
    )

    # Output
    if args.output:
        generator.save_cli(cli, args.output)
    else:
        print("\nGenerated CLI:")
        print("-" * 60)
        print(cli)

    print("\nRose CLI generation complete!")
    print("\nNote: Each petal has 5 independent bones:")
    print("  - bone_base (0-25%)")
    print("  - bone_mid (25-45%)")
    print("  - bone_mid_upper (45-62%) - WIDEST")
    print("  - bone_upper (62-78%)")
    print("  - bone_tip (78-100%)")


if __name__ == "__main__":
    main()
