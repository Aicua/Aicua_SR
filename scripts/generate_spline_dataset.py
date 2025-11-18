#!/usr/bin/env python3
"""
Generate training datasets for Rose SR using Spline geometry.

Spline-based petal representation instead of Bezier Surface.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import math

# Golden ratio
PHI = 1.618033988749895


def generate_spline_petal_dataset(n_samples: int = 100) -> pd.DataFrame:
    """
    Generate dataset for petal geometry using SPLINE control points.

    Spline format: spline x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 x6 y6 x7 y7 x8 y8 ...
    (2D coordinates only - pairs of x,y)

    For rose petal, we use 8 control points (realistic shape from rose_petals.jpg):
    - CP1: Base left (narrow)
    - CP2: Lower left curve
    - CP3: Upper left curve (widest)
    - CP4: Tip left (before notch)
    - CP5: Tip center
    - CP6: Tip right (after notch)
    - CP7: Upper right curve (widest)
    - CP8: Lower right curve

    Features:
        - base_size: Overall rose size
        - layer_index: 1 (inner), 2 (middle), 3 (outer)
        - petal_index: Position within layer
        - opening_degree: 0.0 (closed) to 1.0 (fully open)

    Targets (Spline 2D Control Points - 8 CPs):
        - cp1_x, cp1_y: Base left
        - cp2_x, cp2_y: Lower left curve
        - cp3_x, cp3_y: Upper left curve
        - cp4_x, cp4_y: Tip left
        - cp5_x, cp5_y: Tip center
        - cp6_x, cp6_y: Tip right
        - cp7_x, cp7_y: Upper right curve
        - cp8_x, cp8_y: Lower right curve
        - extrude_depth: Thickness when extruded
    """
    data = []

    petals_per_layer = [5, 8, 13]  # Fibonacci

    for _ in range(n_samples):
        base_size = np.random.uniform(2.0, 8.0)
        opening_degree = np.random.uniform(0.0, 1.0)

        for layer_idx in range(1, 4):
            n_petals = petals_per_layer[layer_idx - 1]

            for petal_idx in range(n_petals):
                # === LAYER FACTOR ===
                layer_factor = [0.8, 0.9, 1.0][layer_idx - 1]

                # === WIDTH CALCULATIONS ===
                base_spread = base_size * 0.35 * layer_factor * (1 + opening_degree * 0.2)
                mid_width = base_spread * 0.7
                upper_width = base_spread * 0.5

                # Petal height
                petal_height = base_size * layer_factor * (1.2 - opening_degree * 0.3)

                # Tip notch (heart-shape)
                notch_depth = petal_height * 0.05 * (1 - (layer_idx - 1) * 0.3)
                notch_width = base_spread * 0.15

                # Tip x offset (for asymmetry/tilt)
                tip_x_offset = base_size * 0.02 * (layer_idx - 1) * opening_degree

                # === 8 CONTROL POINTS ===
                # CP1: Base left (narrow)
                cp1_x = -base_spread / 4
                cp1_y = 0.0

                # CP2: Lower left curve (at 25% height)
                cp2_x = -mid_width / 2
                cp2_y = petal_height * 0.25

                # CP3: Upper left curve (widest at 60% height)
                cp3_x = -upper_width / 2
                cp3_y = petal_height * 0.6

                # CP4: Tip left (before notch)
                cp4_x = -notch_width
                cp4_y = petal_height - notch_depth

                # CP5: Tip center
                cp5_x = tip_x_offset
                cp5_y = petal_height

                # CP6: Tip right (after notch)
                cp6_x = notch_width
                cp6_y = petal_height - notch_depth

                # CP7: Upper right curve (widest)
                cp7_x = upper_width / 2
                cp7_y = petal_height * 0.6

                # CP8: Lower right curve
                cp8_x = mid_width / 2
                cp8_y = petal_height * 0.25

                # Extrude depth (ultra-thin)
                thickness_base = 0.005
                extrude_depth = (
                    thickness_base * base_size *
                    (1 - (layer_idx - 1) * 0.1) *
                    (1 - opening_degree * 0.3)
                )
                extrude_depth = max(0.001, extrude_depth)

                # Rotation angle for spiral arrangement
                golden_angle = 137.5
                rotation_angle = (petal_idx * golden_angle) % 360

                # Add noise (3%)
                noise = 0.03
                cp1_x *= (1 + np.random.normal(0, noise))
                cp2_x *= (1 + np.random.normal(0, noise))
                cp2_y *= (1 + np.random.normal(0, noise))
                cp3_x *= (1 + np.random.normal(0, noise))
                cp3_y *= (1 + np.random.normal(0, noise))
                cp4_x *= (1 + np.random.normal(0, noise))
                cp4_y *= (1 + np.random.normal(0, noise))
                cp5_x += np.random.normal(0, noise * base_size * 0.05)
                cp5_y *= (1 + np.random.normal(0, noise))
                cp6_x *= (1 + np.random.normal(0, noise))
                cp6_y *= (1 + np.random.normal(0, noise))
                cp7_x *= (1 + np.random.normal(0, noise))
                cp7_y *= (1 + np.random.normal(0, noise))
                cp8_x *= (1 + np.random.normal(0, noise))
                cp8_y *= (1 + np.random.normal(0, noise))
                extrude_depth *= (1 + np.random.normal(0, noise))
                extrude_depth = max(0.001, extrude_depth)

                data.append({
                    # Features
                    'base_size': round(base_size, 4),
                    'layer_index': layer_idx,
                    'petal_index': petal_idx,
                    'opening_degree': round(opening_degree, 4),

                    # Targets (8 Spline control points)
                    'cp1_x': round(cp1_x, 6),
                    'cp1_y': round(cp1_y, 6),
                    'cp2_x': round(cp2_x, 6),
                    'cp2_y': round(cp2_y, 6),
                    'cp3_x': round(cp3_x, 6),
                    'cp3_y': round(cp3_y, 6),
                    'cp4_x': round(cp4_x, 6),
                    'cp4_y': round(cp4_y, 6),
                    'cp5_x': round(cp5_x, 6),
                    'cp5_y': round(cp5_y, 6),
                    'cp6_x': round(cp6_x, 6),
                    'cp6_y': round(cp6_y, 6),
                    'cp7_x': round(cp7_x, 6),
                    'cp7_y': round(cp7_y, 6),
                    'cp8_x': round(cp8_x, 6),
                    'cp8_y': round(cp8_y, 6),
                    'extrude_depth': round(extrude_depth, 6),
                    'rotation_angle': round(rotation_angle, 4),
                })

    return pd.DataFrame(data)


def generate_bone_rigging_dataset_v2(n_samples: int = 100) -> pd.DataFrame:
    """
    Generate bone rigging dataset compatible with spline petals.

    Bone structure:
    - Base bone (at petal base)
    - Mid bone (middle of petal)
    - Tip bone (at petal tip)

    Similar to wing example:
    add_bone rig_left bone_left 0 0 0 -3 0 0
    """
    data = []

    for _ in range(n_samples):
        # Petal dimensions (from spline)
        tip_y = np.random.uniform(0.5, 4.0)  # Height
        base_spread = np.random.uniform(0.2, 1.5)  # Width
        flexibility = np.random.uniform(0.3, 1.0)
        layer_idx = np.random.randint(1, 4)

        # === BONE PARAMETERS ===

        # Number of bones (2-4 typically)
        # Longer petals need more bones
        bone_count = max(2, min(4, int(tip_y * flexibility * 2)))

        # Bone positions along petal length (Y axis)
        # Bones distributed along petal
        bone_length = tip_y / bone_count

        # Bind weight (how much bone affects mesh)
        layer_weight = [1.0, 1.5, 2.0][layer_idx - 1]
        bind_weight = flexibility * layer_weight

        # Start and end positions for main bone
        bone_start_y = 0.0
        bone_end_y = tip_y * 0.4  # First bone covers 40%

        # Add noise
        noise = 0.05
        bone_length *= (1 + np.random.normal(0, noise))
        bind_weight *= (1 + np.random.normal(0, noise))

        data.append({
            # Features
            'petal_tip_y': round(tip_y, 4),
            'petal_base_spread': round(base_spread, 4),
            'flexibility_factor': round(flexibility, 4),
            'layer_index': layer_idx,

            # Targets
            'bone_count': bone_count,
            'bone_start_y': round(bone_start_y, 6),
            'bone_end_y': round(bone_end_y, 6),
            'bone_segment_length': round(bone_length, 6),
            'bind_weight': round(bind_weight, 6),
        })

    return pd.DataFrame(data)


def generate_animation_dataset_v2(n_samples: int = 100) -> pd.DataFrame:
    """
    Generate animation dataset for wing_flap style movement.

    CLI format:
    wing_flap rig_name bone_name frequency amplitude axis_x axis_y axis_z phase

    Example:
    wing_flap rig_right bone_right 15 30.0 -1 -1 0 0
    """
    data = []

    for _ in range(n_samples):
        base_size = np.random.uniform(1.0, 5.0)
        petal_mass = base_size * np.random.uniform(0.05, 0.2)
        wind_speed = np.random.uniform(0.5, 10.0)
        flexibility = np.random.uniform(0.3, 1.0)
        layer_idx = np.random.randint(1, 4)

        # === ANIMATION PARAMETERS ===

        # Frequency: Oscillations per second
        # Lighter, more flexible = faster
        frequency = 10.0 * np.sqrt(flexibility / (petal_mass + 0.01))
        frequency = np.clip(frequency, 5.0, 30.0)

        # Amplitude: Maximum rotation in degrees
        # More wind = more movement
        amplitude = wind_speed * flexibility * 3.0
        amplitude = np.clip(amplitude, 10.0, 60.0)

        # Axis of rotation (for petal flapping)
        # -1 -1 0 means rotate around XY axis
        axis_x = -1 if layer_idx > 1 else 0
        axis_y = -1
        axis_z = 0

        # Phase offset for natural variation
        phase_offset = np.random.uniform(0, 1)

        # Add noise
        noise = 0.05
        frequency *= (1 + np.random.normal(0, noise))
        amplitude *= (1 + np.random.normal(0, noise))

        data.append({
            # Features
            'base_size': round(base_size, 4),
            'petal_mass': round(petal_mass, 6),
            'wind_speed': round(wind_speed, 4),
            'flexibility': round(flexibility, 4),
            'layer_index': layer_idx,

            # Targets
            'frequency': round(frequency, 4),
            'amplitude': round(amplitude, 4),
            'axis_x': axis_x,
            'axis_y': axis_y,
            'axis_z': axis_z,
            'phase_offset': round(phase_offset, 4),
        })

    return pd.DataFrame(data)


def generate_spline_cli_example():
    """Generate example CLI showing spline-based petal with rigging."""

    example = """
# Rose Petal using Spline V3 - 8 Control Points
# Layer 1, Petal 0
# Base size: 2.0, Opening: 0.8

# Geometry (Spline - 2D coordinates only)
2d;
obj petal_L1_P0;

# Spline control points (8 points + close, 2D: x y pairs)
# CP1: base_left, CP2: lower_left, CP3: upper_left, CP4: tip_left
# CP5: tip_center, CP6: tip_right, CP7: upper_right, CP8: lower_right
# Format: spline x1 y1 ... x8 y8 x1 y1 (closed curve)
spline -0.07 0.0 -0.098 0.288 -0.07 0.691 -0.021 1.095 0.0 1.152 0.021 1.095 0.07 0.691 0.098 0.288 -0.07 0.0;
exit;

# Extrude to give thickness
sketch_extrude petal_L1_P0 0.015;

# Rigging (bones along petal height)
create_armature petal_L1_P0_rig;

# Bone from base to lower (25% height)
add_bone petal_L1_P0_rig bone_0 0 0.0000 0 0 0.288 0;

# Bone from lower to upper (60% height)
add_bone petal_L1_P0_rig bone_1 0 0.288 0 0 0.691 0;

# Bone from upper to tip
add_bone petal_L1_P0_rig bone_2 0 0.691 0 0 1.152 0;

# Parent chain
parent_bone petal_L1_P0_rig bone_1 bone_0;
parent_bone petal_L1_P0_rig bone_2 bone_1;

finalize_bones petal_L1_P0_rig;
bind_armature petal_L1_P0_rig petal_L1_P0 0.8;

# Animation (wing_flap style)
# wing_flap rig bone freq amp axis_x axis_y axis_z phase
wing_flap petal_L1_P0_rig bone_2 30 10.0 0 -1 0 0;
"""
    return example.strip()


def main():
    """Generate all datasets with Spline-based geometry."""
    print("=" * 60)
    print("Rose SR Dataset Generator - SPLINE VERSION")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate spline petal dataset
    print("\n[1/3] Generating SPLINE petal geometry dataset...")
    petal_df = generate_spline_petal_dataset(n_samples=100)
    petal_path = output_dir / "petal_spline.csv"
    petal_df.to_csv(petal_path, index=False)
    print(f"  ✓ Generated {len(petal_df)} samples → {petal_path.name}")
    print(f"  Features: {list(petal_df.columns[:4])}")
    print(f"  Targets: {list(petal_df.columns[4:])}")

    # Generate bone rigging dataset
    print("\n[2/3] Generating bone rigging dataset (v2)...")
    bone_df = generate_bone_rigging_dataset_v2(n_samples=200)
    bone_path = output_dir / "bone_rigging_v2.csv"
    bone_df.to_csv(bone_path, index=False)
    print(f"  ✓ Generated {len(bone_df)} samples → {bone_path.name}")
    print(f"  Features: {list(bone_df.columns[:4])}")
    print(f"  Targets: {list(bone_df.columns[4:])}")

    # Generate animation dataset
    print("\n[3/3] Generating animation dataset (wing_flap style)...")
    anim_df = generate_animation_dataset_v2(n_samples=200)
    anim_path = output_dir / "animation_wingflap.csv"
    anim_df.to_csv(anim_path, index=False)
    print(f"  ✓ Generated {len(anim_df)} samples → {anim_path.name}")
    print(f"  Features: {list(anim_df.columns[:5])}")
    print(f"  Targets: {list(anim_df.columns[5:])}")

    # Generate CLI example
    print("\n[Bonus] Generating Spline CLI example...")
    examples_dir = Path(__file__).parent.parent / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    example = generate_spline_cli_example()
    example_path = examples_dir / "rose_spline_example.txt"
    with open(example_path, 'w') as f:
        f.write(example)
    print(f"  ✓ Example → {example_path.name}")

    print("\n" + "=" * 60)
    print("Spline-based dataset generation complete!")
    print("=" * 60)

    print("\nDataset Statistics:")
    print(f"  Petal Spline: {len(petal_df)} rows")
    print(f"    - Targets: tip_x, tip_y, tip_z, base_spread, base_z_offset, extrude_depth")
    print(f"  Bone Rigging v2: {len(bone_df)} rows")
    print(f"    - Targets: bone_count, bone_start/end_y, segment_length, bind_weight")
    print(f"  Animation (wing_flap): {len(anim_df)} rows")
    print(f"    - Targets: frequency, amplitude, axis_x/y/z, phase_offset")

    print("\nNext steps:")
    print("  1. Update configs/sr_config.yaml for new targets")
    print("  2. Run 'python scripts/train_sr_models.py' to discover spline formulas")
    print("  3. Update generate_rose_cli.py to output spline commands")


if __name__ == "__main__":
    main()
