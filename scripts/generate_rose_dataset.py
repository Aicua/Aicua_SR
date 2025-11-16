#!/usr/bin/env python3
"""
Generate training datasets for Rose Symbolic Regression.

This script creates synthetic datasets based on design rules,
which will be used by SR to discover optimal formulas.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import math

# Golden ratio - commonly used but SR might discover better
PHI = 1.618033988749895


def load_config():
    """Load rose definition config."""
    config_path = Path(__file__).parent.parent / "configs" / "rose_definition.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_petal_geometry_dataset(n_samples: int = 100) -> pd.DataFrame:
    """
    Generate dataset for petal geometry parameters.

    Features:
        - base_size: Overall rose size (1.0 to 5.0)
        - layer_index: 1 (inner), 2 (middle), 3 (outer)
        - petal_index: Position within layer (0 to n_petals-1)
        - opening_degree: 0.0 (closed/bud) to 1.0 (fully open)

    Targets:
        - base_width: Width at base of petal
        - length: Length of petal
        - curvature: How curved the petal is (0.0 to 1.0)
        - twist_angle: Rotation/twist in degrees
        - thickness: Petal thickness
    """
    data = []

    # Fibonacci-inspired petal counts
    petals_per_layer = [5, 8, 13]

    for _ in range(n_samples):
        # Random base parameters
        base_size = np.random.uniform(1.0, 5.0)
        opening_degree = np.random.uniform(0.0, 1.0)

        for layer_idx in range(1, 4):  # Layers 1, 2, 3
            n_petals = petals_per_layer[layer_idx - 1]

            for petal_idx in range(n_petals):
                # === CURRENT FORMULAS (to be discovered by SR) ===
                # These are "ground truth" - SR should discover similar or better

                # Base width: Increases with size, varies by layer
                layer_factor = [0.8, 0.9, 1.0][layer_idx - 1]
                base_width = base_size * 0.3 * layer_factor

                # Length: Outer petals are longer
                length_factor = [0.5, 0.7, 1.0][layer_idx - 1]
                length = base_size * 0.8 * length_factor * (1 + opening_degree * 0.3)

                # Curvature: Inner petals more curved
                curvature = [0.9, 0.6, 0.3][layer_idx - 1] * (1 - opening_degree * 0.3)

                # Twist: Spiral arrangement (Fibonacci angle)
                golden_angle = 137.5  # degrees
                twist_angle = petal_idx * golden_angle / n_petals * layer_idx

                # Thickness: Proportional to size
                thickness = base_size * 0.02 * (1 + curvature * 0.5)

                # Add some noise to simulate real-world variation
                noise_factor = 0.05
                base_width *= (1 + np.random.normal(0, noise_factor))
                length *= (1 + np.random.normal(0, noise_factor))
                curvature = np.clip(curvature * (1 + np.random.normal(0, noise_factor)), 0, 1)
                twist_angle *= (1 + np.random.normal(0, noise_factor / 2))
                thickness *= (1 + np.random.normal(0, noise_factor))

                data.append({
                    'base_size': round(base_size, 4),
                    'layer_index': layer_idx,
                    'petal_index': petal_idx,
                    'opening_degree': round(opening_degree, 4),
                    'base_width': round(base_width, 6),
                    'length': round(length, 6),
                    'curvature': round(curvature, 6),
                    'twist_angle': round(twist_angle, 4),
                    'thickness': round(thickness, 6),
                })

    return pd.DataFrame(data)


def generate_bone_rigging_dataset(n_samples: int = 100) -> pd.DataFrame:
    """
    Generate dataset for bone/rigging parameters.

    Each petal gets bones for animation (wind sway, blooming).

    Features:
        - petal_length: Length of the petal
        - petal_width: Width of the petal
        - flexibility_factor: How flexible (0.0 = stiff, 1.0 = very flexible)
        - layer_index: Which layer (1, 2, 3)

    Targets:
        - bone_count: Number of bones per petal
        - bone_length: Length of each bone segment
        - joint_spacing: Distance between joints
        - bind_weight: Weight painting strength
    """
    data = []

    for _ in range(n_samples):
        petal_length = np.random.uniform(0.5, 4.0)
        petal_width = np.random.uniform(0.2, 1.5)
        flexibility = np.random.uniform(0.3, 1.0)
        layer_idx = np.random.randint(1, 4)

        # === FORMULAS (to be discovered by SR) ===

        # More bones for longer petals, fewer for stiff ones
        bone_count = max(2, int(petal_length * 3 * flexibility))

        # Bone length proportional to petal length divided by count
        bone_length = petal_length / bone_count * 0.9

        # Joint spacing
        joint_spacing = petal_length / (bone_count - 1) if bone_count > 1 else petal_length

        # Bind weight: stronger for outer layers
        layer_weight = [1.0, 1.5, 2.0][layer_idx - 1]
        bind_weight = flexibility * layer_weight

        # Add noise
        noise = 0.05
        bone_length *= (1 + np.random.normal(0, noise))
        joint_spacing *= (1 + np.random.normal(0, noise))
        bind_weight *= (1 + np.random.normal(0, noise))

        data.append({
            'petal_length': round(petal_length, 4),
            'petal_width': round(petal_width, 4),
            'flexibility_factor': round(flexibility, 4),
            'layer_index': layer_idx,
            'bone_count': bone_count,
            'bone_length': round(bone_length, 6),
            'joint_spacing': round(joint_spacing, 6),
            'bind_weight': round(bind_weight, 6),
        })

    return pd.DataFrame(data)


def generate_animation_dataset(n_samples: int = 100) -> pd.DataFrame:
    """
    Generate dataset for animation parameters.

    Focus: Wind sway animation for petals.

    Features:
        - base_size: Overall rose size
        - petal_mass: Estimated mass of petal
        - wind_speed: Wind speed (0.0 to 10.0)
        - flexibility: How flexible the petal is

    Targets:
        - frequency: Oscillation frequency (Hz)
        - amplitude: Maximum displacement (degrees)
        - phase_offset: Phase shift for natural look
        - damping_factor: Energy loss factor
    """
    data = []

    for _ in range(n_samples):
        base_size = np.random.uniform(1.0, 5.0)
        petal_mass = base_size * np.random.uniform(0.05, 0.2)  # Mass proportional to size
        wind_speed = np.random.uniform(0.5, 10.0)
        flexibility = np.random.uniform(0.3, 1.0)

        # === FORMULAS (to be discovered by SR) ===

        # Frequency: Lighter and more flexible = higher frequency
        # Inspired by physics: f ∝ 1/sqrt(mass)
        frequency = 2.0 * np.sqrt(flexibility / petal_mass) * (1 + wind_speed * 0.1)
        frequency = np.clip(frequency, 0.5, 15.0)

        # Amplitude: More wind = more movement
        amplitude = wind_speed * flexibility * 5.0 / np.sqrt(petal_mass + 0.1)
        amplitude = np.clip(amplitude, 5.0, 60.0)

        # Phase offset: Random for natural variation
        phase_offset = np.random.uniform(0, 2 * np.pi)

        # Damping: Heavier petals damp faster
        damping_factor = 0.1 * petal_mass / flexibility
        damping_factor = np.clip(damping_factor, 0.01, 0.5)

        # Add noise
        noise = 0.05
        frequency *= (1 + np.random.normal(0, noise))
        amplitude *= (1 + np.random.normal(0, noise))
        damping_factor *= (1 + np.random.normal(0, noise))

        data.append({
            'base_size': round(base_size, 4),
            'petal_mass': round(petal_mass, 6),
            'wind_speed': round(wind_speed, 4),
            'flexibility': round(flexibility, 4),
            'frequency': round(frequency, 6),
            'amplitude': round(amplitude, 6),
            'phase_offset': round(phase_offset, 6),
            'damping_factor': round(damping_factor, 6),
        })

    return pd.DataFrame(data)


def generate_cli_examples() -> list:
    """
    Generate example CLI commands for reference.

    Returns list of CLI strings demonstrating the output format.
    """
    examples = []

    # Example 1: Single petal with bone
    example1 = """
# Rose Petal Layer 1, Petal 0
# Geometry
2d;
obj petal_1_0;
bezier_surface petal_1_0 0.6 0.4 0.9 15.0 0.04;
exit;

# Rigging
create_armature petal_1_0_arm;
add_bone petal_1_0_arm bone_base 0 0 0 0 0.2 0;
add_bone petal_1_0_arm bone_mid 0 0.2 0 0 0.4 0;
add_bone petal_1_0_arm bone_tip 0 0.4 0 0 0.6 0;
parent_bone petal_1_0_arm bone_mid bone_base;
parent_bone petal_1_0_arm bone_tip bone_mid;
finalize_bones petal_1_0_arm;
bind_armature petal_1_0_arm petal_1_0 1.0;

# Animation
wing_flap petal_1_0_arm bone_tip 5.2 25.0 0 1 0 0;
"""
    examples.append(example1.strip())

    # Example 2: Full layer
    example2 = """
# Rose Layer 2 (5 petals)
# Base size: 2.0, Opening: 0.8

# Petal 0
obj petal_2_0;
bezier_surface petal_2_0 0.54 1.12 0.42 0.0 0.036;
create_armature arm_2_0;
add_bone arm_2_0 b0 0 0 0 0 0.28 0;
add_bone arm_2_0 b1 0 0.28 0 0 0.56 0;
add_bone arm_2_0 b2 0 0.56 0 0 0.84 0;
add_bone arm_2_0 b3 0 0.84 0 0 1.12 0;
parent_bone arm_2_0 b1 b0;
parent_bone arm_2_0 b2 b1;
parent_bone arm_2_0 b3 b2;
bind_armature arm_2_0 petal_2_0 1.5;

# Animation params
wind_sway arm_2_0 b3 3.8 32.5 1.234 0.08;
"""
    examples.append(example2.strip())

    return examples


def main():
    """Generate all datasets."""
    print("=" * 60)
    print("Rose SR Dataset Generator")
    print("=" * 60)

    # Create output directory
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate datasets
    print("\n[1/3] Generating petal geometry dataset...")
    petal_df = generate_petal_geometry_dataset(n_samples=100)
    petal_path = output_dir / "petal_geometry.csv"
    petal_df.to_csv(petal_path, index=False)
    print(f"  ✓ Generated {len(petal_df)} samples → {petal_path.name}")
    print(f"  Features: {list(petal_df.columns[:4])}")
    print(f"  Targets: {list(petal_df.columns[4:])}")

    print("\n[2/3] Generating bone rigging dataset...")
    bone_df = generate_bone_rigging_dataset(n_samples=200)
    bone_path = output_dir / "bone_rigging.csv"
    bone_df.to_csv(bone_path, index=False)
    print(f"  ✓ Generated {len(bone_df)} samples → {bone_path.name}")
    print(f"  Features: {list(bone_df.columns[:4])}")
    print(f"  Targets: {list(bone_df.columns[4:])}")

    print("\n[3/3] Generating animation parameters dataset...")
    anim_df = generate_animation_dataset(n_samples=200)
    anim_path = output_dir / "animation_params.csv"
    anim_df.to_csv(anim_path, index=False)
    print(f"  ✓ Generated {len(anim_df)} samples → {anim_path.name}")
    print(f"  Features: {list(anim_df.columns[:4])}")
    print(f"  Targets: {list(anim_df.columns[4:])}")

    # Save CLI examples
    print("\n[Bonus] Generating CLI examples...")
    examples_dir = Path(__file__).parent.parent / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    examples = generate_cli_examples()
    for i, cli in enumerate(examples, 1):
        example_path = examples_dir / f"rose_cli_example_{i}.txt"
        with open(example_path, 'w') as f:
            f.write(cli)
        print(f"  ✓ Example {i} → {example_path.name}")

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)

    # Summary statistics
    print("\nDataset Statistics:")
    print(f"  Petal Geometry: {len(petal_df)} rows")
    print(f"  Bone Rigging: {len(bone_df)} rows")
    print(f"  Animation: {len(anim_df)} rows")

    print("\nNext steps:")
    print("  1. Review generated data in data/processed/")
    print("  2. Run 'python scripts/train_sr_models.py' to discover formulas")
    print("  3. Or upload to Kaggle for GPU-accelerated training")


if __name__ == "__main__":
    main()
