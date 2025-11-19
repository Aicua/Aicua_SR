#!/usr/bin/env python3
"""
Generate Rose CAD CLI using SR-discovered formulas.

This is the main application that uses discovered formulas to generate
CLI commands for creating 3D rose models with bones for animation.
"""

import argparse
import json
import sys
from pathlib import Path

# Add generated module path
sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "generated"))


def load_formulas_module():
    """Dynamically load formula functions."""
    try:
        # Import all formula modules
        import petal_geometry_formulas as petal
        import bone_rigging_formulas as bone
        import animation_params_formulas as anim
        return petal, bone, anim
    except ImportError as e:
        print(f"Error loading formulas: {e}")
        print("Make sure to run:")
        print("  1. python scripts/generate_rose_dataset.py")
        print("  2. python scripts/train_sr_models.py")
        print("  3. python scripts/codegen_formulas.py")
        sys.exit(1)


class RoseCLIGenerator:
    """Generate CLI commands for 3D rose with bones."""

    def __init__(self):
        self.petal_mod, self.bone_mod, self.anim_mod = load_formulas_module()
        self.cli_commands = []

    def generate_petal(self, layer_idx: int, petal_idx: int, base_size: float,
                       opening_degree: float) -> dict:
        """Generate CLI for a single petal with bones."""

        # Use SR-discovered formulas for geometry
        base_width = self.petal_mod.compute_base_width(
            base_size, layer_idx, petal_idx, opening_degree
        )
        length = self.petal_mod.compute_length(
            base_size, layer_idx, petal_idx, opening_degree
        )
        curvature = self.petal_mod.compute_curvature(
            base_size, layer_idx, petal_idx, opening_degree
        )
        twist_angle = self.petal_mod.compute_twist_angle(
            base_size, layer_idx, petal_idx, opening_degree
        )
        thickness = self.petal_mod.compute_thickness(
            base_size, layer_idx, petal_idx, opening_degree
        )

        petal_name = f"petal_L{layer_idx}_P{petal_idx}"

        # Generate geometry CLI
        geometry_cli = [
            f"# {petal_name} - Layer {layer_idx}, Petal {petal_idx}",
            f"obj {petal_name};",
            f"bezier_surface {petal_name} {length:.4f} {base_width:.4f} {curvature:.4f} {twist_angle:.4f} {thickness:.6f};",
        ]

        # Generate bone rigging with 5 independent bones (v6)
        flexibility = 0.5 + (3 - layer_idx) * 0.15  # Outer more flexible
        bind_weight = flexibility * [1.0, 1.5, 2.0][layer_idx - 1] if layer_idx > 0 else 1.0

        armature_name = f"{petal_name}_arm"

        # Calculate bone heights based on petal length
        # 5 bones: base (0-25%), mid (25-45%), mid_upper (45-62%), upper (62-78%), tip (78-100%)
        h_base = length * 0.25
        h_mid = length * 0.45
        h_mid_upper = length * 0.62
        h_upper = length * 0.78
        h_tip = length

        rigging_cli = [
            f"",
            f"# Rigging for {petal_name} (5 independent bones)",
            f"create_armature {armature_name};",
        ]

        # Generate 5 independent bones (NO parent_bone commands!)
        rigging_cli.append(f"add_bone {armature_name} bone_base 0 0.0000 0 0 {h_base:.4f} 0;")
        rigging_cli.append(f"add_bone {armature_name} bone_mid 0 {h_base:.4f} 0 0 {h_mid:.4f} 0;")
        rigging_cli.append(f"add_bone {armature_name} bone_mid_upper 0 {h_mid:.4f} 0 0 {h_mid_upper:.4f} 0;")
        rigging_cli.append(f"add_bone {armature_name} bone_upper 0 {h_mid_upper:.4f} 0 0 {h_upper:.4f} 0;")
        rigging_cli.append(f"add_bone {armature_name} bone_tip 0 {h_upper:.4f} 0 0 {h_tip:.4f} 0;")

        rigging_cli.append(f"finalize_bones {armature_name};")
        rigging_cli.append(f"bind_armature {armature_name} {petal_name} {bind_weight:.4f};")

        # Generate animation parameters using discovered formulas
        petal_mass = base_size * base_width * length * 0.01  # Estimated mass
        wind_speed = 3.0  # Default wind speed

        frequency = self.anim_mod.compute_frequency(
            base_size, petal_mass, wind_speed, flexibility
        )
        amplitude = self.anim_mod.compute_amplitude(
            base_size, petal_mass, wind_speed, flexibility
        )
        damping = self.anim_mod.compute_damping_factor(
            base_size, petal_mass, wind_speed, flexibility
        )

        # Clamp to reasonable ranges
        frequency = max(0.5, min(frequency, 15.0))
        amplitude = max(5.0, min(amplitude, 60.0))
        damping = max(0.01, min(damping, 0.5))

        animation_cli = [
            f"",
            f"# Animation for {petal_name} (5 independent bones)",
            f"wind_sway {armature_name} bone_base {frequency:.4f} {amplitude * 0.3:.4f} 0 1 0 {damping:.6f};",
            f"wind_sway {armature_name} bone_mid {frequency:.4f} {amplitude * 0.5:.4f} 0 1 0 {damping:.6f};",
            f"wind_sway {armature_name} bone_mid_upper {frequency:.4f} {amplitude * 0.8:.4f} 0 1 0 {damping:.6f};",
            f"wind_sway {armature_name} bone_upper {frequency:.4f} {amplitude * 0.6:.4f} 0 1 0 {damping:.6f};",
            f"wind_sway {armature_name} bone_tip {frequency:.4f} {amplitude:.4f} 0 1 0 {damping:.6f};",
        ]

        return {
            'geometry': geometry_cli,
            'rigging': rigging_cli,
            'animation': animation_cli,
            'params': {
                'base_width': base_width,
                'length': length,
                'curvature': curvature,
                'twist_angle': twist_angle,
                'thickness': thickness,
                'bone_count': 5,  # Fixed 5 bones
                'frequency': frequency,
                'amplitude': amplitude,
            }
        }

    def generate_rose(self, base_size: float = 2.0, opening_degree: float = 0.8,
                      n_layers: int = 3) -> str:
        """
        Generate complete rose CLI.

        Args:
            base_size: Overall size of rose
            opening_degree: How open (0.0 = bud, 1.0 = fully open)
            n_layers: Number of petal layers (1-3)

        Returns:
            Complete CLI string
        """
        # Fibonacci-inspired petal counts
        petals_per_layer = [5, 8, 13]

        all_cli = [
            "# Rose CAD Generation",
            f"# Base Size: {base_size}",
            f"# Opening Degree: {opening_degree}",
            f"# Layers: {n_layers}",
            f"# Generated using SR-discovered formulas",
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
                    layer_idx, petal_idx, base_size, opening_degree
                )

                # Add geometry
                all_cli.extend(petal_data['geometry'])
                # Add rigging
                all_cli.extend(petal_data['rigging'])
                # Add animation
                all_cli.extend(petal_data['animation'])
                all_cli.append("")

                total_petals += 1

        # Exit and summary
        all_cli.append("exit;")
        all_cli.append("")
        all_cli.append(f"# Total petals: {total_petals}")
        all_cli.append(f"# Total bones: {total_petals * 5}")  # 5 independent bones per petal
        all_cli.append(f"# All parameters computed using Symbolic Regression")

        return '\n'.join(all_cli)

    def save_cli(self, cli_text: str, output_path: str):
        """Save CLI to file."""
        with open(output_path, 'w') as f:
            f.write(cli_text)
        print(f"✓ Saved CLI to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Rose CAD CLI using SR-discovered formulas"
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

    print("=" * 60)
    print("Rose CLI Generator (using SR-discovered formulas)")
    print("=" * 60)

    if args.verbose:
        print(f"Parameters:")
        print(f"  Base Size: {args.size}")
        print(f"  Opening Degree: {args.opening}")
        print(f"  Layers: {args.layers}")
        print()

    # Generate
    generator = RoseCLIGenerator()
    cli = generator.generate_rose(
        base_size=args.size,
        opening_degree=args.opening,
        n_layers=args.layers
    )

    # Output
    if args.output:
        generator.save_cli(cli, args.output)
    else:
        print("\nGenerated CLI:")
        print("-" * 60)
        print(cli)

    print("\n✓ Rose CLI generation complete!")


if __name__ == "__main__":
    main()
