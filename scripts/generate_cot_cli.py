#!/usr/bin/env python3
"""
Generate Rose CAD CLI using Chain-of-Thought (CoT) Reasoning.

Dynamically decides optimal number of control points based on shape analysis.
"""

import argparse
import math
import sys
from pathlib import Path

# Add local modules
sys.path.insert(0, str(Path(__file__).parent))
from cot_reasoning import CoTReasoner


class CoTRoseCLIGenerator:
    """Generate rose CLI with CoT-based dynamic CP count."""

    def __init__(self, verbose: bool = False):
        """Initialize with CoT reasoner."""
        self.reasoner = CoTReasoner()
        self.verbose = verbose

    def generate_petal_with_cot(
        self, layer_idx, petal_idx, base_size, opening_degree, detail_level="medium",
        bloom_animation=False, bloom_duration=3000
    ):
        """Generate single petal using CoT reasoning."""

        petal_name = f"petal_L{layer_idx}_P{petal_idx}"

        # Calculate dimensions based on layer (V2: continuous layer factor)
        layer_factor = 0.8 + 0.1 * (layer_idx - 1)  # [0.8, 0.9, 1.0]
        width = base_size * 0.3 * layer_factor * (1 + opening_degree * 0.2)
        height = base_size * layer_factor * (1.2 - opening_degree * 0.3)
        tip_offset = base_size * 0.02 * (layer_idx - 1) * opening_degree
        # Ultra-thin thickness
        extrude_depth = max(0.001, base_size * 0.005 * (1 - (layer_idx - 1) * 0.1) * (1 - opening_degree * 0.3))

        # Use CoT reasoning to determine CP count and positions
        cot_result = self.reasoner.reason_and_generate(
            object_name=petal_name,
            width=width,
            height=height,
            symmetry_required=True,
            smooth_curves=True,
            detail_level=detail_level,
            tip_offset=tip_offset,
            verbose=self.verbose,
        )

        # Calculate rotation angle for spiral arrangement
        golden_angle = 137.5
        rotation_angle = (petal_idx * golden_angle) % 360

        # Generate geometry CLI
        geometry_cli = [
            f"# {petal_name} - Layer {layer_idx}, Petal {petal_idx}",
            f"# CoT Decision: {cot_result['cp_count']} CPs (confidence: {cot_result['decision'].confidence:.1%})",
            f"2d;",
            f"obj {petal_name};",
            cot_result["spline_command"],
            f"exit;",
            f"sketch_extrude {petal_name} {extrude_depth:.4f};",
        ]

        # Get petal height from control points
        cps = cot_result["control_points"]
        petal_height = max(cp[1] for cp in cps)
        petal_width = max(cp[0] for cp in cps) - min(cp[0] for cp in cps)

        # Generate bone rigging (V7: T-shape 5 independent bones)
        rig_name = f"{petal_name}_rig"
        rigging_cli = [
            f"",
            f"# Rigging for {petal_name} (v7 - T-shape 5 bones)",
            f"create_armature {rig_name};",
        ]

        # Calculate bone positions (0-based layer index for calculations)
        layer_idx_0based = layer_idx - 1
        layer_factor_bone = 0.8 + 0.1 * layer_idx_0based

        # V7 T-shape bone height positions
        h_45 = petal_height * 0.45 * layer_factor_bone   # End of bone_base
        h_62 = petal_height * 0.62 * layer_factor_bone   # Position of bone_left/bone_right
        h_78 = petal_height * 0.78 * layer_factor_bone   # End of bone_mid
        h_100 = petal_height * layer_factor_bone          # End of bone_tip

        # Width at 62% (widest point)
        half_width = petal_width / 2

        # Add 5 independent bones - T-shape (connected spine)
        # Vertical spine (tail of each bone = head of next)
        rigging_cli.append(f"add_bone {rig_name} bone_base 0 0 0 0 {h_45:.4f} 0;")
        rigging_cli.append(f"add_bone {rig_name} bone_mid 0 {h_45:.4f} 0 0 {h_78:.4f} 0;")  # connected
        rigging_cli.append(f"add_bone {rig_name} bone_tip 0 {h_78:.4f} 0 0 {h_100:.4f} 0;")
        # Horizontal edges at 62%
        rigging_cli.append(f"add_bone {rig_name} bone_left 0 {h_62:.4f} 0 {-half_width:.4f} {h_62:.4f} 0;")
        rigging_cli.append(f"add_bone {rig_name} bone_right 0 {h_62:.4f} 0 {half_width:.4f} {h_62:.4f} 0;")

        rigging_cli.append(f"finalize_bones {rig_name};")
        flexibility = 0.5 + (3 - layer_idx) * 0.15
        bind_weight = flexibility * [1.0, 1.5, 2.0][layer_idx - 1]
        rigging_cli.append(f"bind_armature {rig_name} {petal_name} {bind_weight:.4f};")

        # Rotate petal into position using base bone
        if rotation_angle > 0:
            rigging_cli.append(f"")
            rigging_cli.append(f"# Position petal in spiral arrangement (rotate base bone)")
            rigging_cli.append(f"rotate_bone {rig_name} bone_base 0 0 {rotation_angle:.2f};")

        # Generate animation (each bone animates independently)
        petal_mass = base_size * petal_width * petal_height * 0.01
        wind_speed = 3.0
        frequency = 10.0 * math.sqrt(flexibility / (petal_mass + 0.01))
        frequency = max(5.0, min(30.0, frequency))
        amplitude = wind_speed * flexibility * 3.0
        amplitude = max(10.0, min(60.0, amplitude))

        animation_cli = [
            f"",
            f"# Animation for {petal_name} (v7 - T-shape 5 bones)",
            f"wing_flap {rig_name} bone_base {frequency:.0f} {amplitude * 0.3:.1f} 0 -1 0 0;",
            f"wing_flap {rig_name} bone_mid {frequency:.0f} {amplitude * 0.6:.1f} 0 -1 0 0.05;",
            f"wing_flap {rig_name} bone_tip {frequency * 1.2:.0f} {amplitude * 0.4:.1f} 0 -1 0 0.1;",
            f"wing_flap {rig_name} bone_left {frequency * 0.8:.0f} {amplitude * 0.5:.1f} -1 0 0 0.15;",
            f"wing_flap {rig_name} bone_right {frequency * 0.8:.0f} {amplitude * 0.5:.1f} 1 0 0 0.15;",
        ]

        # Add bloom animation if requested (auto_rotate for smooth opening)
        if bloom_animation:
            # Calculate bloom angle based on layer (outer layers open more)
            bloom_angle = 15.0 + (layer_idx - 1) * 10.0  # L1: 15°, L2: 25°, L3: 35°

            animation_cli.append(f"")
            animation_cli.append(f"# Bloom animation (smooth opening)")
            animation_cli.append(
                f"auto_rotate {rig_name} bone_mid 1 0 0 {bloom_angle:.1f} {bloom_duration} smooth;"
            )

        # Generate deformation rotate_bone commands (head/tail modes)
        # Deformation type based on opening_degree
        if opening_degree < 0.3:
            deformation_type = 0  # straight
        elif opening_degree < 0.6:
            deformation_type = 1  # s_curve
        elif opening_degree < 0.8:
            deformation_type = 2  # c_curve
        else:
            deformation_type = 3  # wave

        intensity = opening_degree
        deformation_cli = [
            f"",
            f"# Deformation rotate_bone (head/tail modes)",
        ]

        if deformation_type == 1:  # S-curve
            deformation_cli.extend([
                f"rotate_bone {rig_name} bone_base {15 * intensity:.1f} 0 0 head;",
                f"rotate_bone {rig_name} bone_mid {-25 * intensity:.1f} 0 0 head;",
                f"rotate_bone {rig_name} bone_tip {15 * intensity:.1f} 0 0 head;",
                f"rotate_bone {rig_name} bone_left 0 {10 * intensity:.1f} 0 head;",
                f"rotate_bone {rig_name} bone_right 0 {-10 * intensity:.1f} 0 head;",
            ])
        elif deformation_type == 2:  # C-curve (cup shape)
            deformation_cli.extend([
                f"rotate_bone {rig_name} bone_base 0 {20 * intensity:.1f} 0 head;",
                f"rotate_bone {rig_name} bone_mid 0 {30 * intensity:.1f} 0 head;",
                f"rotate_bone {rig_name} bone_tip 0 {25 * intensity:.1f} 0 head;",
                f"rotate_bone {rig_name} bone_left 0 {40 * intensity:.1f} 0 head;",
                f"rotate_bone {rig_name} bone_right 0 {-40 * intensity:.1f} 0 head;",
            ])
        elif deformation_type == 3:  # Wave
            deformation_cli.extend([
                f"rotate_bone {rig_name} bone_base {10 * intensity:.1f} {5 * intensity:.1f} 0 head;",
                f"rotate_bone {rig_name} bone_mid {-15 * intensity:.1f} {-10 * intensity:.1f} 0 head;",
                f"rotate_bone {rig_name} bone_tip {20 * intensity:.1f} {8 * intensity:.1f} 0 head;",
                f"rotate_bone {rig_name} bone_left {5 * intensity:.1f} {15 * intensity:.1f} 0 head;",
                f"rotate_bone {rig_name} bone_right {-5 * intensity:.1f} {-15 * intensity:.1f} 0 head;",
            ])
        # straight (type 0) has no deformation

        return {
            "geometry": geometry_cli,
            "rigging": rigging_cli,
            "animation": animation_cli,
            "deformation": deformation_cli,
            "cot_result": cot_result,
        }

    def generate_rose(
        self, base_size=2.0, opening_degree=0.8, n_layers=3, detail_level="medium",
        bloom_animation=False, bloom_duration=3000
    ):
        """Generate complete rose with CoT reasoning."""

        petals_per_layer = [5, 8, 13]

        all_cli = [
            "# Rose CAD Generation - CoT (Chain-of-Thought) VERSION",
            f"# Base Size: {base_size}",
            f"# Opening Degree: {opening_degree}",
            f"# Layers: {n_layers}",
            f"# Detail Level: {detail_level}",
            f"# Bloom Animation: {bloom_animation}",
            f"# Dynamic CP count based on shape analysis",
            "",
        ]

        total_petals = 0
        cp_stats = {}

        for layer_idx in range(1, n_layers + 1):
            n_petals = petals_per_layer[layer_idx - 1]

            all_cli.append(f"# ====== LAYER {layer_idx} ({n_petals} petals) ======")
            all_cli.append("")

            for petal_idx in range(n_petals):
                petal_data = self.generate_petal_with_cot(
                    layer_idx, petal_idx, base_size, opening_degree, detail_level,
                    bloom_animation, bloom_duration
                )

                all_cli.extend(petal_data["geometry"])
                all_cli.extend(petal_data["rigging"])
                all_cli.extend(petal_data["animation"])
                all_cli.extend(petal_data["deformation"])
                all_cli.append("")

                # Track CP statistics
                cp_count = petal_data["cot_result"]["cp_count"]
                cp_stats[cp_count] = cp_stats.get(cp_count, 0) + 1

                total_petals += 1

        # Summary
        all_cli.append(f"# Total petals: {total_petals}")
        all_cli.append(f"# CP distribution: {cp_stats}")
        all_cli.append(f"# Geometry: CoT-based dynamic spline curves")
        all_cli.append(f"# Animation: wing_flap style oscillation")

        return "\n".join(all_cli)

    def save_cli(self, cli_text, output_path):
        """Save CLI to file."""
        with open(output_path, "w") as f:
            f.write(cli_text)
        print(f"✓ Saved CLI to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Rose CAD CLI with CoT Reasoning"
    )

    parser.add_argument("--size", type=float, default=2.0, help="Base size")
    parser.add_argument(
        "--opening", type=float, default=0.8, help="Opening degree 0-1"
    )
    parser.add_argument("--layers", type=int, default=3, help="Number of layers 1-3")
    parser.add_argument(
        "--detail", type=str, default="medium", help="Detail level: low/medium/high"
    )
    parser.add_argument("--bloom", action="store_true", help="Enable bloom animation (auto_rotate)")
    parser.add_argument("--bloom-duration", type=int, default=3000, help="Bloom duration in ms")
    parser.add_argument("--output", type=str, default=None, help="Output file")
    parser.add_argument("--verbose", action="store_true", help="Show CoT reasoning")

    args = parser.parse_args()

    print("=" * 60)
    print("Rose CLI Generator - CoT (Chain-of-Thought) VERSION")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  Base Size: {args.size}")
    print(f"  Opening Degree: {args.opening}")
    print(f"  Layers: {args.layers}")
    print(f"  Detail Level: {args.detail}")
    print(f"  Bloom Animation: {args.bloom}")
    if args.bloom:
        print(f"  Bloom Duration: {args.bloom_duration}ms")
    print()

    generator = CoTRoseCLIGenerator(verbose=args.verbose)
    cli = generator.generate_rose(
        base_size=args.size,
        opening_degree=args.opening,
        n_layers=args.layers,
        detail_level=args.detail,
        bloom_animation=args.bloom,
        bloom_duration=args.bloom_duration,
    )

    if args.output:
        generator.save_cli(cli, args.output)
    else:
        print("\nGenerated CLI:")
        print("-" * 60)
        print(cli)

    print("\n✓ CoT-based Rose CLI generation complete!")


if __name__ == "__main__":
    main()
