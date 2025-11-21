#!/usr/bin/env python3
"""
Generate Flower CAD CLI V1 with MOVE + ROTATE positioning.

This script generates complete CLI commands for creating a flower with:
1. Petal geometry (15 CP spline or simplified bezier)
2. Bone rigging (5-bone T-shape)
3. Positioning (MOVE + ROTATE in 3D space)

Usage:
    # Generate 3 petals (test symmetry)
    python scripts/generate_flower_cli_v1.py --petals 3 --radius 5.0

    # Generate 5 petals
    python scripts/generate_flower_cli_v1.py --petals 5 --radius 6.0

    # Generate multi-layer flower
    python scripts/generate_flower_cli_v1.py --layers 3
"""

import argparse
import json
import math
import sys
from pathlib import Path


class FlowerCLIGeneratorV1:
    """Generate CLI commands for flower with positioning."""

    def __init__(self, formulas_path: str = None):
        """Initialize generator with positioning formulas."""
        if formulas_path is None:
            formulas_path = Path(__file__).parent.parent / "data" / "models" / "sr_positioning_v1" / "all_formulas.json"

        self.formulas_path = Path(formulas_path)
        self.formulas = self._load_formulas()
        self.cli_commands = []

    def _load_formulas(self) -> dict:
        """Load SR-discovered formulas for positioning."""
        if not self.formulas_path.exists():
            print(f"Warning: Formulas not found at {self.formulas_path}")
            print("Using default mathematical formulas.")
            return self._get_default_formulas()

        with open(self.formulas_path, 'r') as f:
            formulas = json.load(f)

        print(f"Loaded positioning formulas from: {self.formulas_path.name}")
        return formulas

    def _get_default_formulas(self) -> dict:
        """Get default mathematical formulas for positioning."""
        return {
            'pos_x': {'formula': 'layer_radius * cos((petal_index * 360.0) / num_petals)'},
            'pos_y': {'formula': 'layer_radius * sin((petal_index * 360.0) / num_petals)'},
            'pos_z': {'formula': 'z_variation'},
            'rotate_x': {'formula': '0.0'},
            'rotate_y': {'formula': 'base_tilt_angle * (1.0 - opening_degree)'},
            'rotate_z': {'formula': '((petal_index * 360.0) / num_petals) + 90.0'},
        }

    def calculate_position(self, layer_radius: float, num_petals: int, petal_index: int,
                          opening_degree: float, base_tilt_angle: float, z_variation: float = 0.0) -> dict:
        """
        Calculate position and rotation for a petal using SR formulas.

        Args:
            layer_radius: Distance from center
            num_petals: Total petals in this layer
            petal_index: Index of this petal (0 to num_petals-1)
            opening_degree: How open (0=closed, 1=open)
            base_tilt_angle: Maximum tilt angle
            z_variation: Height offset

        Returns:
            Dictionary with pos_x, pos_y, pos_z, rotate_x, rotate_y, rotate_z
        """
        # Calculate base angle
        angle_deg = (petal_index * 360.0) / num_petals
        angle_rad = math.radians(angle_deg)

        # Position (cylindrical to cartesian)
        pos_x = layer_radius * math.cos(angle_rad)
        pos_y = layer_radius * math.sin(angle_rad)
        pos_z = z_variation

        # Rotation
        rotate_x = 0.0  # No wobble for now
        rotate_y = base_tilt_angle * (1.0 - opening_degree)  # Cup when closed
        rotate_z = angle_deg + 90.0  # Face outward

        return {
            'pos_x': pos_x,
            'pos_y': pos_y,
            'pos_z': pos_z,
            'rotate_x': rotate_x,
            'rotate_y': rotate_y,
            'rotate_z': rotate_z,
        }

    def generate_petal_geometry_simple(self, petal_name: str, base_size: float,
                                       opening_degree: float, layer_index: int) -> list:
        """Generate simplified petal geometry using bezier_surface."""
        layer_factor = 0.8 + 0.1 * layer_index
        petal_height = base_size * layer_factor * (1.2 - opening_degree * 0.3)
        base_width = base_size * 0.30 * layer_factor * (1 + opening_degree * 0.2) * 2
        curvature = 0.3 + opening_degree * 0.4
        twist_angle = layer_index * 5
        thickness = 0.01 * base_size

        return [
            f"obj {petal_name};",
            f"bezier_surface {petal_name} {petal_height:.4f} {base_width:.4f} {curvature:.4f} {twist_angle:.4f} {thickness:.6f};",
        ]

    def generate_bone_rigging_v7(self, petal_name: str, base_size: float,
                                  opening_degree: float, layer_index: int) -> list:
        """
        Generate T-shape bone rigging (v7) with 5 independent bones.

        Bones:
        - bone_base (0% → 45%)
        - bone_mid (45% → 78%)
        - bone_tip (78% → 100%)
        - bone_left (at 62%)
        - bone_right (at 62%)
        """
        layer_factor = 0.8 + 0.1 * layer_index
        petal_height = base_size * layer_factor * (1.2 - opening_degree * 0.3)
        base_spread = base_size * 0.30 * layer_factor * (1 + opening_degree * 0.2)
        width_at_62 = base_spread * 1.6

        armature_name = f"{petal_name}_rig"

        cli = [
            f"create_armature {armature_name};",
        ]

        # Vertical spine bones (3 connected)
        bones = {
            'bone_base': {
                'start_x': 0.0, 'start_y': 0.0,
                'end_x': 0.0, 'end_y': petal_height * 0.45
            },
            'bone_mid': {
                'start_x': 0.0, 'start_y': petal_height * 0.45,
                'end_x': 0.0, 'end_y': petal_height * 0.78
            },
            'bone_tip': {
                'start_x': 0.0, 'start_y': petal_height * 0.78,
                'end_x': 0.0, 'end_y': petal_height
            },
            # Horizontal edge bones at 62% (widest point)
            'bone_left': {
                'start_x': 0.0, 'start_y': petal_height * 0.62,
                'end_x': -width_at_62 / 2, 'end_y': petal_height * 0.62
            },
            'bone_right': {
                'start_x': 0.0, 'start_y': petal_height * 0.62,
                'end_x': width_at_62 / 2, 'end_y': petal_height * 0.62
            }
        }

        for bone_name, pos in bones.items():
            cli.append(
                f"add_bone {armature_name} {bone_name} "
                f"{pos['start_x']:.4f} {pos['start_y']:.4f} 0 "
                f"{pos['end_x']:.4f} {pos['end_y']:.4f} 0;"
            )

        cli.append(f"finalize_bones {armature_name};")
        cli.append(f"bind_armature {armature_name} {petal_name};")

        return cli

    def generate_positioning(self, petal_name: str, position: dict) -> list:
        """
        Generate MOVE + ROTATE commands for positioning.

        Args:
            petal_name: Name of the petal object
            position: Dict with pos_x, pos_y, pos_z, rotate_x, rotate_y, rotate_z

        Returns:
            List of CLI commands
        """
        return [
            f"move {petal_name} {position['pos_x']:.4f} {position['pos_y']:.4f} {position['pos_z']:.4f};",
            f"rotate {petal_name} {position['rotate_x']:.2f} {position['rotate_y']:.2f} {position['rotate_z']:.2f};",
        ]

    def generate_single_petal(self, petal_index: int, layer_radius: float, num_petals: int,
                             base_size: float, opening_degree: float, base_tilt_angle: float,
                             layer_index: int = 0, z_offset: float = 0.0) -> list:
        """Generate complete CLI for a single petal (geometry + rigging + positioning)."""
        petal_name = f"petal_{petal_index}"
        if num_petals > 3:
            petal_name = f"petal_L{layer_index}_P{petal_index}"

        cli = [
            f"# === Petal {petal_index} (Layer {layer_index}) ===",
            "",
        ]

        # Calculate position first for CoT
        position = self.calculate_position(
            layer_radius=layer_radius,
            num_petals=num_petals,
            petal_index=petal_index,
            opening_degree=opening_degree,
            base_tilt_angle=base_tilt_angle,
            z_variation=z_offset
        )

        # CoT: Positioning reasoning
        angle_deg = (petal_index * 360.0) / num_petals
        cli.extend([
            f"# CoT Positioning Reasoning:",
            f"#   Target: Arrange {num_petals} petals in circular pattern",
            f"#   Angle: petal_{petal_index} = ({petal_index} * 360° / {num_petals}) = {angle_deg:.2f}°",
            f"#   Position: radius={layer_radius:.2f} → (x={position['pos_x']:.4f}, y={position['pos_y']:.4f}, z={position['pos_z']:.4f})",
            f"#   Rotation: face_outward = angle + 90° = {position['rotate_z']:.2f}°",
            f"#   Tilt: cup_shape = {base_tilt_angle:.1f}° * (1 - {opening_degree:.2f}) = {position['rotate_y']:.2f}°",
            "",
        ])

        # 1. Geometry
        cli.append("# Geometry")
        cli.extend(self.generate_petal_geometry_simple(petal_name, base_size, opening_degree, layer_index))
        cli.append("")

        # 2. Rigging
        cli.append("# Rigging (v7 - T-shape 5 bones)")
        cli.extend(self.generate_bone_rigging_v7(petal_name, base_size, opening_degree, layer_index))
        cli.append("")

        # 3. Positioning with SR formulas
        cli.append("# Positioning (MOVE + ROTATE) - Using SR-discovered formulas")
        cli.append(f"#   pos_x = layer_radius * cos(angle°) = {layer_radius:.2f} * cos({angle_deg:.2f}°)")
        cli.append(f"#   pos_y = layer_radius * sin(angle°) = {layer_radius:.2f} * sin({angle_deg:.2f}°)")
        cli.extend(self.generate_positioning(petal_name, position))
        cli.append("")

        return cli

    def generate_layer(self, layer_radius: float, num_petals: int, base_size: float,
                      opening_degree: float, base_tilt_angle: float,
                      layer_index: int = 0, z_offset: float = 0.0) -> list:
        """Generate complete CLI for a layer of petals."""
        cli = [
            f"# ====== LAYER {layer_index} ({num_petals} petals) ======",
            f"# Radius: {layer_radius:.2f}, Opening: {opening_degree:.2f}, Tilt: {base_tilt_angle:.2f}°",
            "",
        ]

        for petal_idx in range(num_petals):
            cli.extend(self.generate_single_petal(
                petal_index=petal_idx,
                layer_radius=layer_radius,
                num_petals=num_petals,
                base_size=base_size,
                opening_degree=opening_degree,
                base_tilt_angle=base_tilt_angle,
                layer_index=layer_index,
                z_offset=z_offset
            ))

        return cli

    def generate_flower(self, layers_config: list, base_size: float = 2.0) -> str:
        """
        Generate complete flower with multiple layers.

        Args:
            layers_config: List of layer configurations
                Each config: {
                    'radius': float,
                    'num_petals': int,
                    'opening_degree': float,
                    'base_tilt_angle': float,
                    'z_offset': float
                }
            base_size: Overall size scale

        Returns:
            Complete CLI string
        """
        cli = [
            "# Flower CAD Generation V1",
            "# Approach: MOVE + ROTATE (Cylindrical Positioning)",
            f"# Base Size: {base_size}",
            f"# Layers: {len(layers_config)}",
            "",
            "2d;",
            "",
        ]

        total_petals = 0

        for layer_idx, config in enumerate(layers_config):
            cli.extend(self.generate_layer(
                layer_radius=config['radius'],
                num_petals=config['num_petals'],
                base_size=base_size,
                opening_degree=config['opening_degree'],
                base_tilt_angle=config['base_tilt_angle'],
                layer_index=layer_idx,
                z_offset=config.get('z_offset', 0.0)
            ))

            total_petals += config['num_petals']

        # Summary
        cli.extend([
            "exit;",
            "",
            f"# Total petals: {total_petals}",
            f"# Total bones: {total_petals * 5} (5 per petal)",
        ])

        return '\n'.join(cli)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Flower CAD CLI V1 with MOVE + ROTATE positioning"
    )

    parser.add_argument(
        '--petals', type=int, default=3,
        help='Number of petals in single layer (default: 3)'
    )
    parser.add_argument(
        '--radius', type=float, default=5.0,
        help='Layer radius (default: 5.0)'
    )
    parser.add_argument(
        '--size', type=float, default=2.0,
        help='Base petal size (default: 2.0)'
    )
    parser.add_argument(
        '--opening', type=float, default=0.8,
        help='Opening degree 0.0-1.0 (default: 0.8)'
    )
    parser.add_argument(
        '--tilt', type=float, default=15.0,
        help='Base tilt angle in degrees (default: 15.0)'
    )
    parser.add_argument(
        '--layers', type=int, default=1,
        help='Number of layers (default: 1 for testing)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output file path (default: stdout)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Flower CLI Generator V1")
    print("Approach: MOVE + ROTATE (Cylindrical Positioning)")
    print("=" * 70)

    generator = FlowerCLIGeneratorV1()

    # Configure layers
    if args.layers == 1:
        # Single layer for testing
        layers_config = [{
            'radius': args.radius,
            'num_petals': args.petals,
            'opening_degree': args.opening,
            'base_tilt_angle': args.tilt,
            'z_offset': 0.0,
        }]
        print(f"\nGenerating single layer:")
        print(f"  {args.petals} petals at radius {args.radius}")

    else:
        # Multi-layer flower (Fibonacci-like: 5, 8, 13)
        petal_counts = [5, 8, 13]
        radii = [8.0, 5.0, 3.0]  # Outer → Inner
        openings = [0.9, 0.7, 0.5]  # Outer more open
        tilts = [10.0, 20.0, 30.0]  # Inner tilts more (cup)
        z_offsets = [0.0, -0.5, -1.0]  # Stack layers

        layers_config = []
        for i in range(min(args.layers, 3)):
            layers_config.append({
                'radius': radii[i],
                'num_petals': petal_counts[i],
                'opening_degree': openings[i],
                'base_tilt_angle': tilts[i],
                'z_offset': z_offsets[i],
            })

        print(f"\nGenerating {args.layers} layers:")
        for i, cfg in enumerate(layers_config):
            print(f"  Layer {i}: {cfg['num_petals']} petals at radius {cfg['radius']}")

    # Generate CLI
    cli_text = generator.generate_flower(layers_config, base_size=args.size)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(cli_text)
        print(f"\n✓ Saved CLI to: {output_path}")
    else:
        print("\n" + "=" * 70)
        print("Generated CLI:")
        print("=" * 70)
        print(cli_text)

    print("\n" + "=" * 70)
    print("✓ Generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
