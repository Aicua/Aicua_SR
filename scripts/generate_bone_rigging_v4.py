#!/usr/bin/env python3
"""
Generate bone rigging v4 dataset with 4 bones (branching structure).

Structure:
    bone_left      bone_right
         \           /
          bone_middle
               |
          bone_root

All coordinates are 2D (x, y) since petal is created from 2D spline then extruded.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_bone_rigging_v4(n_samples: int = 500) -> pd.DataFrame:
    """
    Generate bone rigging dataset with 4 bones in branching structure.

    4 bones create natural petal deformation:
    - bone_root: Base of petal (anchored)
    - bone_middle: Main axis (controls overall bend)
    - bone_left: Left branch (controls left side opening)
    - bone_right: Right branch (controls right side opening)

    Features:
        - petal_height: Vertical extent of petal (y-axis)
        - petal_width: Horizontal width (x-axis)
        - opening_degree: How open the flower is (0=closed, 1=open)
        - layer_index: Which layer (0=inner, 1=middle, 2=outer)
        - curvature_intensity: How curved the petal is

    Targets (16 values - 4 bones x 4 coordinates):
        - bone_root_start_x, bone_root_start_y, bone_root_end_x, bone_root_end_y
        - bone_middle_start_x, bone_middle_start_y, bone_middle_end_x, bone_middle_end_y
        - bone_left_start_x, bone_left_start_y, bone_left_end_x, bone_left_end_y
        - bone_right_start_x, bone_right_start_y, bone_right_end_x, bone_right_end_y
    """
    data = []

    for _ in range(n_samples):
        # === FEATURES ===
        petal_height = np.random.uniform(0.5, 4.0)
        petal_width = np.random.uniform(0.2, 1.5)
        opening_degree = np.random.uniform(0.0, 1.0)
        layer_index = np.random.randint(0, 3)  # 0, 1, 2
        curvature_intensity = np.random.uniform(0.5, 1.5)

        # === BONE STRUCTURE (2D coordinates) ===

        # Layer factors affect bone positions
        layer_factor = [0.8, 0.9, 1.0][layer_index]

        # --- BONE ROOT ---
        # From center bottom, goes up along main axis
        root_start_x = 0.0
        root_start_y = 0.0
        root_end_x = 0.0
        root_end_y = petal_height * 0.3 * layer_factor

        # --- BONE MIDDLE ---
        # Continues from root, main axis of petal
        middle_start_x = root_end_x
        middle_start_y = root_end_y
        # Middle bone goes up to about 60-70% of petal height
        middle_end_x = 0.0
        middle_end_y = petal_height * 0.65 * layer_factor

        # --- BONE LEFT ---
        # Branches from middle, controls left side
        left_start_x = middle_end_x
        left_start_y = middle_end_y
        # Left bone extends outward and upward
        # Opening degree affects how far it spreads
        left_spread = petal_width * 0.4 * (0.5 + opening_degree * 0.5)
        left_end_x = -left_spread
        left_end_y = petal_height * 0.9 * layer_factor

        # --- BONE RIGHT ---
        # Branches from middle, controls right side (symmetric to left)
        right_start_x = middle_end_x
        right_start_y = middle_end_y
        right_end_x = left_spread  # Symmetric
        right_end_y = petal_height * 0.9 * layer_factor

        # Apply curvature intensity to bone positions
        # Higher curvature = bones spread more outward
        curvature_factor = curvature_intensity * 0.1
        left_end_x *= (1 + curvature_factor)
        right_end_x *= (1 + curvature_factor)

        # Add realistic noise
        noise = 0.03
        root_end_y *= (1 + np.random.normal(0, noise))
        middle_end_y *= (1 + np.random.normal(0, noise))
        left_end_x *= (1 + np.random.normal(0, noise))
        left_end_y *= (1 + np.random.normal(0, noise))
        right_end_x *= (1 + np.random.normal(0, noise))
        right_end_y *= (1 + np.random.normal(0, noise))

        data.append({
            # Features
            'petal_height': round(petal_height, 6),
            'petal_width': round(petal_width, 6),
            'opening_degree': round(opening_degree, 6),
            'layer_index': layer_index,
            'curvature_intensity': round(curvature_intensity, 6),

            # Targets - Bone Root
            'bone_root_start_x': round(root_start_x, 6),
            'bone_root_start_y': round(root_start_y, 6),
            'bone_root_end_x': round(root_end_x, 6),
            'bone_root_end_y': round(root_end_y, 6),

            # Targets - Bone Middle
            'bone_middle_start_x': round(middle_start_x, 6),
            'bone_middle_start_y': round(middle_start_y, 6),
            'bone_middle_end_x': round(middle_end_x, 6),
            'bone_middle_end_y': round(middle_end_y, 6),

            # Targets - Bone Left
            'bone_left_start_x': round(left_start_x, 6),
            'bone_left_start_y': round(left_start_y, 6),
            'bone_left_end_x': round(left_end_x, 6),
            'bone_left_end_y': round(left_end_y, 6),

            # Targets - Bone Right
            'bone_right_start_x': round(right_start_x, 6),
            'bone_right_start_y': round(right_start_y, 6),
            'bone_right_end_x': round(right_end_x, 6),
            'bone_right_end_y': round(right_end_y, 6),
        })

    return pd.DataFrame(data)


def main():
    """Generate bone rigging v4 dataset."""
    print("=" * 60)
    print("Bone Rigging V4 Dataset Generator")
    print("4 Bones: Root + Middle + Left + Right (Branching Structure)")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating bone rigging v4 dataset...")
    df = generate_bone_rigging_v4(n_samples=500)

    output_path = output_dir / "bone_rigging_v4.csv"
    df.to_csv(output_path, index=False)

    print(f"\nGenerated {len(df)} samples")
    print(f"Output: {output_path}")

    print("\nFeatures (5):")
    features = ['petal_height', 'petal_width', 'opening_degree', 'layer_index', 'curvature_intensity']
    for f in features:
        print(f"  - {f}")

    print("\nTargets (16 - 4 bones x 4 coordinates):")
    bones = ['bone_root', 'bone_middle', 'bone_left', 'bone_right']
    for bone in bones:
        print(f"  {bone}:")
        print(f"    - {bone}_start_x, {bone}_start_y")
        print(f"    - {bone}_end_x, {bone}_end_y")

    print("\nSample data (first 3 rows):")
    print(df.head(3).to_string())

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)

    print("\nBone Structure:")
    print("""
       bone_left      bone_right
            \\           /
             bone_middle
                  |
             bone_root
    """)

    print("Next steps:")
    print("  1. Update configs/sr_config_spline.yaml with bone_rigging_v4")
    print("  2. Train SR models to discover formulas")
    print("  3. Update CLI generator to use branching bones")


if __name__ == "__main__":
    main()
