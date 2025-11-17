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
        self.use_sr_petal = False
        self.use_sr_bone = False
        self.use_sr_anim = False

        try:
            import petal_spline_formulas as petal
            self.petal_mod = petal
            self.use_sr_petal = True
        except ImportError:
            print("Petal SR formulas not found. Using fallback.")

        try:
            import bone_rigging_v4_formulas as bone
            self.bone_mod = bone
            self.use_sr_bone = True
        except ImportError:
            print("Bone rigging v4 SR formulas not found. Using fallback.")

        try:
            import animation_wingflap_formulas as anim
            self.anim_mod = anim
            self.use_sr_anim = True
        except ImportError:
            print("Animation SR formulas not found. Using fallback.")

        # Legacy compatibility
        self.use_sr = self.use_sr_petal and self.use_sr_bone and self.use_sr_anim

    def compute_spline_params(self, base_size, layer_idx, petal_idx, opening_degree):
        """Compute 2D spline control points using SR formulas or fallback."""
        if self.use_sr_petal:
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
            # Fallback formulas (2D spline control points) - V2 MIDDLE-WIDE SHAPE
            # Continuous layer factor: layer_idx is 1-based here
            layer_factor = 0.8 + 0.1 * (layer_idx - 1)  # [0.8, 0.9, 1.0]

            base_spread = base_size * 0.3 * layer_factor * (1 + opening_degree * 0.2)
            petal_height = base_size * layer_factor * (1.2 - opening_degree * 0.3)
            tip_x_offset = base_size * 0.02 * (layer_idx - 1) * opening_degree

            # MIDDLE-WIDE SHAPE:
            # CP1/CP5: Base narrow (±1/4 spread)
            # CP2/CP4: Middle widest (±1/2 spread)
            return {
                'cp1_x': -base_spread / 4,  # Narrow base
                'cp1_y': 0.0,
                'cp2_x': -base_spread / 2,  # WIDEST middle
                'cp2_y': petal_height * 0.4,
                'cp3_x': tip_x_offset,
                'cp3_y': petal_height,
                'cp4_x': base_spread / 2,   # WIDEST middle (symmetric)
                'cp4_y': petal_height * 0.4,
                'cp5_x': base_spread / 4,   # Narrow base (symmetric)
                'cp5_y': 0.0,
                # ULTRA-THIN THICKNESS
                'extrude_depth': max(0.001, base_size * 0.005 * (1 - (layer_idx - 1) * 0.1) * (1 - opening_degree * 0.3)),
            }

    def compute_bone_params_v4(self, petal_height, petal_width, opening_degree, layer_idx, curvature_intensity=1.0):
        """Compute bone rigging parameters for v4 branching structure."""
        if self.use_sr_bone:
            return {
                'bone_root_start_x': self.bone_mod.compute_bone_root_start_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_root_start_y': self.bone_mod.compute_bone_root_start_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_root_end_x': self.bone_mod.compute_bone_root_end_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_root_end_y': self.bone_mod.compute_bone_root_end_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_middle_start_x': self.bone_mod.compute_bone_middle_start_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_middle_start_y': self.bone_mod.compute_bone_middle_start_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_middle_end_x': self.bone_mod.compute_bone_middle_end_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_middle_end_y': self.bone_mod.compute_bone_middle_end_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_left_start_x': self.bone_mod.compute_bone_left_start_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_left_start_y': self.bone_mod.compute_bone_left_start_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_left_end_x': self.bone_mod.compute_bone_left_end_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_left_end_y': self.bone_mod.compute_bone_left_end_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_right_start_x': self.bone_mod.compute_bone_right_start_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_right_start_y': self.bone_mod.compute_bone_right_start_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_right_end_x': self.bone_mod.compute_bone_right_end_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_right_end_y': self.bone_mod.compute_bone_right_end_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
            }
        else:
            # Fallback formulas for v4 branching structure
            layer_factor = [0.8, 0.9, 1.0][layer_idx]

            # Bone root: base to 30% height
            root_end_y = petal_height * 0.3 * layer_factor

            # Bone middle: 30% to 65% height
            middle_end_y = petal_height * 0.65 * layer_factor

            # Bone left/right: branches from middle, spread outward
            left_spread = petal_width * 0.4 * (0.5 + opening_degree * 0.5)
            branch_end_y = petal_height * 0.9 * layer_factor

            # Apply curvature
            curvature_factor = curvature_intensity * 0.1
            left_end_x = -left_spread * (1 + curvature_factor)
            right_end_x = left_spread * (1 + curvature_factor)

            return {
                'bone_root_start_x': 0.0,
                'bone_root_start_y': 0.0,
                'bone_root_end_x': 0.0,
                'bone_root_end_y': root_end_y,
                'bone_middle_start_x': 0.0,
                'bone_middle_start_y': root_end_y,
                'bone_middle_end_x': 0.0,
                'bone_middle_end_y': middle_end_y,
                'bone_left_start_x': 0.0,
                'bone_left_start_y': middle_end_y,
                'bone_left_end_x': left_end_x,
                'bone_left_end_y': branch_end_y,
                'bone_right_start_x': 0.0,
                'bone_right_start_y': middle_end_y,
                'bone_right_end_x': right_end_x,
                'bone_right_end_y': branch_end_y,
            }

    def compute_anim_params(self, base_size, petal_mass, wind_speed, flexibility, layer_idx):
        """Compute animation parameters for wing_flap."""
        if self.use_sr_anim:
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

        # Generate bone rigging with v4 branching structure
        # Use cp3_y as petal height, use MID-CURVE width (widest part)
        petal_height = sp['cp3_y']
        # For middle-wide shape: cp4_x - cp2_x is the widest part
        petal_width = sp['cp4_x'] - sp['cp2_x']  # Mid-curve width (widest)
        curvature_intensity = 1.0  # Default curvature

        # layer_idx is 1-based, convert to 0-based for v4
        layer_idx_0based = layer_idx - 1
        bp = self.compute_bone_params_v4(petal_height, petal_width, opening_degree, layer_idx_0based, curvature_intensity)

        rig_name = f"{petal_name}_rig"

        rigging_cli = [
            f"",
            f"# Rigging for {petal_name} (v4 branching structure)",
            f"create_armature {rig_name};",
        ]

        # Generate 4 bones with branching structure (2D coords, z=0)
        # Bone root
        rigging_cli.append(
            f"add_bone {rig_name} bone_root {bp['bone_root_start_x']:.4f} {bp['bone_root_start_y']:.4f} 0 {bp['bone_root_end_x']:.4f} {bp['bone_root_end_y']:.4f} 0;"
        )
        # Bone middle
        rigging_cli.append(
            f"add_bone {rig_name} bone_middle {bp['bone_middle_start_x']:.4f} {bp['bone_middle_start_y']:.4f} 0 {bp['bone_middle_end_x']:.4f} {bp['bone_middle_end_y']:.4f} 0;"
        )
        # Bone left
        rigging_cli.append(
            f"add_bone {rig_name} bone_left {bp['bone_left_start_x']:.4f} {bp['bone_left_start_y']:.4f} 0 {bp['bone_left_end_x']:.4f} {bp['bone_left_end_y']:.4f} 0;"
        )
        # Bone right
        rigging_cli.append(
            f"add_bone {rig_name} bone_right {bp['bone_right_start_x']:.4f} {bp['bone_right_start_y']:.4f} 0 {bp['bone_right_end_x']:.4f} {bp['bone_right_end_y']:.4f} 0;"
        )

        # Parent bones in branching structure
        rigging_cli.append(f"parent_bone {rig_name} bone_middle bone_root;")
        rigging_cli.append(f"parent_bone {rig_name} bone_left bone_middle;")
        rigging_cli.append(f"parent_bone {rig_name} bone_right bone_middle;")

        # Calculate bind weight based on flexibility
        flexibility = 0.5 + (3 - layer_idx) * 0.15
        bind_weight = flexibility * [1.0, 1.5, 2.0][layer_idx - 1]

        rigging_cli.append(f"finalize_bones {rig_name};")
        rigging_cli.append(f"bind_armature {rig_name} {petal_name} {bind_weight:.4f};")

        # Rotate petal into position using root bone
        if rotation_angle > 0:
            rigging_cli.append(f"")
            rigging_cli.append(f"# Position petal in spiral arrangement (rotate root bone)")
            rigging_cli.append(f"rotate_bone {rig_name} bone_root 0 0 {rotation_angle:.2f};")

        # Generate animation (wing_flap style)
        petal_mass = base_size * petal_width * petal_height * 0.01
        wind_speed = 3.0  # Default

        ap = self.compute_anim_params(base_size, petal_mass, wind_speed, flexibility, layer_idx)

        animation_cli = [
            f"",
            f"# Animation for {petal_name}",
            f"# bone_middle controls overall bend",
            f"wing_flap {rig_name} bone_middle {ap['frequency']:.0f} {ap['amplitude']:.1f} 0 -1 0 0;",
            f"# bone_left and bone_right create symmetric opening",
            f"wing_flap {rig_name} bone_left {ap['frequency']:.0f} {ap['amplitude'] * 0.5:.1f} -1 0 0 0.25;",
            f"wing_flap {rig_name} bone_right {ap['frequency']:.0f} {ap['amplitude'] * 0.5:.1f} 1 0 0 0.25;",
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
