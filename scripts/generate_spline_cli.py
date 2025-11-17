#!/usr/bin/env python3
"""
Generate Rose CAD CLI using Spline geometry and SR-discovered formulas.

Output format matches:
- Spline for petal shape
- Bones for rigging
- wing_flap for animation
"""

import argparse
import math
import sys
from pathlib import Path

# Add generated module path
sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "generated"))


class SplineRoseCLIGenerator:
    """Generate spline-based rose CLI with bones and animation."""

    def __init__(self):
        """Initialize with formula modules."""
        try:
            # Try to load SR-discovered formulas
            import petal_spline_formulas as petal
            import bone_rigging_v3_formulas as bone
            import animation_wingflap_formulas as anim
            self.petal_mod = petal
            self.bone_mod = bone
            self.anim_mod = anim
            self.use_sr = True
        except ImportError:
            print("SR formulas not found. Using fallback formulas.")
            self.use_sr = False

    def compute_spline_params(self, base_size, layer_idx, petal_idx, opening_degree):
        """Compute 2D spline control points using SR formulas or fallback."""
        if self.use_sr:
            return {
                'cp1_x': self.petal_mod.compute_cp1_x(base_size, layer_idx, petal_idx, opening_degree),
                'cp1_y': self.petal_mod.compute_cp1_y(base_size, layer_idx, petal_idx, opening_degree),
                'cp2_x': self.petal_mod.compute_cp2_x(base_size, layer_idx, petal_idx, opening_degree),
                'cp2_y': self.petal_mod.compute_cp2_y(base_size, layer_idx, petal_idx, opening_degree),
                'cp3_x': self.petal_mod.compute_cp3_x(base_size, layer_idx, petal_idx, opening_degree),
                'cp3_y': self.petal_mod.compute_cp3_y(base_size, layer_idx, petal_idx, opening_degree),
                'cp4_x': self.petal_mod.compute_cp4_x(base_size, layer_idx, petal_idx, opening_degree),
                'cp4_y': self.petal_mod.compute_cp4_y(base_size, layer_idx, petal_idx, opening_degree),
                'cp5_x': self.petal_mod.compute_cp5_x(base_size, layer_idx, petal_idx, opening_degree),
                'cp5_y': self.petal_mod.compute_cp5_y(base_size, layer_idx, petal_idx, opening_degree),
                'extrude_depth': self.petal_mod.compute_extrude_depth(base_size, layer_idx, petal_idx, opening_degree),
            }
        else:
            # Fallback formulas (2D spline control points)
            layer_factor = [0.6, 0.8, 1.0][layer_idx - 1]

            base_spread = base_size * 0.3 * layer_factor * (1 + opening_degree * 0.2)
            petal_height = base_size * layer_factor * (1.2 - opening_degree * 0.3)
            tip_x_offset = base_size * 0.05 * (layer_idx - 1) * opening_degree

            return {
                'cp1_x': -base_spread / 2,
                'cp1_y': 0.0,
                'cp2_x': -base_spread / 3,
                'cp2_y': petal_height * 0.4,
                'cp3_x': tip_x_offset,
                'cp3_y': petal_height,
                'cp4_x': base_spread / 3,
                'cp4_y': petal_height * 0.4,
                'cp5_x': base_spread / 2,
                'cp5_y': 0.0,
                'extrude_depth': base_size * 0.01 * (1 + layer_idx * 0.1),
            }

    def compute_bone_params(self, tip_y, base_spread, flexibility, layer_idx):
        """Compute bone rigging parameters."""
        if self.use_sr:
            return {
                'bone_count': int(self.bone_mod.compute_bone_count(tip_y, base_spread, flexibility, layer_idx)),
                'bone_end_y': self.bone_mod.compute_bone_end_y(tip_y, base_spread, flexibility, layer_idx),
                'bind_weight': self.bone_mod.compute_bind_weight(tip_y, base_spread, flexibility, layer_idx),
            }
        else:
            bone_count = max(2, min(4, int(tip_y * flexibility * 2)))
            return {
                'bone_count': bone_count,
                'bone_end_y': tip_y * 0.4,
                'bind_weight': flexibility * [1.0, 1.5, 2.0][layer_idx - 1],
            }

    def compute_anim_params(self, base_size, petal_mass, wind_speed, flexibility, layer_idx):
        """Compute animation parameters for wing_flap."""
        if self.use_sr:
            return {
                'frequency': self.anim_mod.compute_frequency(base_size, petal_mass, wind_speed, flexibility, layer_idx),
                'amplitude': self.anim_mod.compute_amplitude(base_size, petal_mass, wind_speed, flexibility, layer_idx),
                'axis_x': int(self.anim_mod.compute_axis_x(base_size, petal_mass, wind_speed, flexibility, layer_idx)),
                'axis_y': int(self.anim_mod.compute_axis_y(base_size, petal_mass, wind_speed, flexibility, layer_idx)),
                'axis_z': int(self.anim_mod.compute_axis_z(base_size, petal_mass, wind_speed, flexibility, layer_idx)),
            }
        else:
            frequency = 10.0 * math.sqrt(flexibility / (petal_mass + 0.01))
            frequency = max(5.0, min(30.0, frequency))
            amplitude = wind_speed * flexibility * 3.0
            amplitude = max(10.0, min(60.0, amplitude))
            return {
                'frequency': frequency,
                'amplitude': amplitude,
                'axis_x': -1 if layer_idx > 1 else 0,
                'axis_y': -1,
                'axis_z': 0,
            }

    def generate_petal_spline(self, layer_idx, petal_idx, base_size, opening_degree):
        """Generate CLI for single petal using 2D spline."""

        petal_name = f"petal_L{layer_idx}_P{petal_idx}"

        # Get 2D spline control points
        sp = self.compute_spline_params(base_size, layer_idx, petal_idx, opening_degree)

        # Calculate rotation angle for spiral arrangement
        golden_angle = 137.5
        rotation_angle = (petal_idx * golden_angle) % 360

        # Generate geometry CLI with 2D spline (x y pairs)
        # Format: spline x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 x6 y6 (closed curve - last point = first point)
        geometry_cli = [
            f"# {petal_name} - Layer {layer_idx}, Petal {petal_idx}",
            f"2d;",
            f"obj {petal_name};",
            f"spline {sp['cp1_x']:.4f} {sp['cp1_y']:.4f} {sp['cp2_x']:.4f} {sp['cp2_y']:.4f} {sp['cp3_x']:.4f} {sp['cp3_y']:.4f} {sp['cp4_x']:.4f} {sp['cp4_y']:.4f} {sp['cp5_x']:.4f} {sp['cp5_y']:.4f} {sp['cp1_x']:.4f} {sp['cp1_y']:.4f};",
            f"exit;",
            f"sketch_extrude {petal_name} {sp['extrude_depth']:.4f};",
        ]

        # Generate bone rigging
        flexibility = 0.5 + (3 - layer_idx) * 0.15
        # Use cp3_y as petal height, calculate base_spread from cp1_x and cp5_x
        petal_height = sp['cp3_y']
        base_spread = sp['cp5_x'] - sp['cp1_x']
        bp = self.compute_bone_params(petal_height, base_spread, flexibility, layer_idx)

        rig_name = f"{petal_name}_rig"

        rigging_cli = [
            f"",
            f"# Rigging for {petal_name}",
            f"create_armature {rig_name};",
        ]

        # Generate bones along petal (2D: x=0, y=height, z=0)
        bone_segment = petal_height / bp['bone_count']

        for i in range(bp['bone_count']):
            bone_name = f"bone_{i}"
            start_y = i * bone_segment
            end_y = (i + 1) * bone_segment

            rigging_cli.append(
                f"add_bone {rig_name} {bone_name} 0 {start_y:.4f} 0 0 {end_y:.4f} 0;"
            )

        # Parent bones
        for i in range(1, bp['bone_count']):
            rigging_cli.append(f"parent_bone {rig_name} bone_{i} bone_{i-1};")

        rigging_cli.append(f"finalize_bones {rig_name};")
        rigging_cli.append(f"bind_armature {rig_name} {petal_name} {bp['bind_weight']:.4f};")

        # Rotate petal into position using root bone (bone_0)
        # rotate_bone on root bone rotates entire object
        if rotation_angle > 0:
            rigging_cli.append(f"")
            rigging_cli.append(f"# Position petal in spiral arrangement (rotate root bone)")
            rigging_cli.append(f"rotate_bone {rig_name} bone_0 0 0 {rotation_angle:.2f};")

        # Generate animation (wing_flap style)
        petal_mass = base_size * base_spread * petal_height * 0.01
        wind_speed = 3.0  # Default

        ap = self.compute_anim_params(base_size, petal_mass, wind_speed, flexibility, layer_idx)

        last_bone = f"bone_{bp['bone_count']-1}"

        animation_cli = [
            f"",
            f"# Animation for {petal_name}",
            f"wing_flap {rig_name} {last_bone} {ap['frequency']:.0f} {ap['amplitude']:.1f} {ap['axis_x']} {ap['axis_y']} {ap['axis_z']} 0;",
        ]

        return {
            'geometry': geometry_cli,
            'rigging': rigging_cli,
            'animation': animation_cli,
            'params': sp,
        }

    def generate_rose(self, base_size=2.0, opening_degree=0.8, n_layers=3):
        """Generate complete rose CLI with splines."""

        petals_per_layer = [5, 8, 13]

        all_cli = [
            "# Rose CAD Generation - SPLINE VERSION",
            f"# Base Size: {base_size}",
            f"# Opening Degree: {opening_degree}",
            f"# Layers: {n_layers}",
            f"# Using spline geometry with bones and wing_flap animation",
            "",
        ]

        total_petals = 0

        for layer_idx in range(1, n_layers + 1):
            n_petals = petals_per_layer[layer_idx - 1]

            all_cli.append(f"# ====== LAYER {layer_idx} ({n_petals} petals) ======")
            all_cli.append("")

            for petal_idx in range(n_petals):
                petal_data = self.generate_petal_spline(
                    layer_idx, petal_idx, base_size, opening_degree
                )

                all_cli.extend(petal_data['geometry'])
                all_cli.extend(petal_data['rigging'])
                all_cli.extend(petal_data['animation'])
                all_cli.append("")

                total_petals += 1

        # Summary
        all_cli.append(f"# Total petals: {total_petals}")
        all_cli.append(f"# Geometry: Spline-based 3D curves")
        all_cli.append(f"# Animation: wing_flap style oscillation")

        return '\n'.join(all_cli)

    def save_cli(self, cli_text, output_path):
        """Save CLI to file."""
        with open(output_path, 'w') as f:
            f.write(cli_text)
        print(f"✓ Saved CLI to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Rose CAD CLI using Spline geometry"
    )

    parser.add_argument('--size', type=float, default=2.0, help='Base size')
    parser.add_argument('--opening', type=float, default=0.8, help='Opening degree 0-1')
    parser.add_argument('--layers', type=int, default=3, help='Number of layers 1-3')
    parser.add_argument('--output', type=str, default=None, help='Output file')
    parser.add_argument('--verbose', action='store_true', help='Show details')

    args = parser.parse_args()

    print("=" * 60)
    print("Rose CLI Generator - SPLINE VERSION")
    print("=" * 60)

    if args.verbose:
        print(f"Parameters:")
        print(f"  Base Size: {args.size}")
        print(f"  Opening Degree: {args.opening}")
        print(f"  Layers: {args.layers}")
        print()

    generator = SplineRoseCLIGenerator()
    cli = generator.generate_rose(
        base_size=args.size,
        opening_degree=args.opening,
        n_layers=args.layers
    )

    if args.output:
        generator.save_cli(cli, args.output)
    else:
        print("\nGenerated CLI:")
        print("-" * 60)
        print(cli)

    print("\n✓ Spline-based Rose CLI generation complete!")


if __name__ == "__main__":
    main()
