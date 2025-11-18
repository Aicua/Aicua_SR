#!/usr/bin/env python3
"""
Generate bone rigging v5 dataset with 7 bones (fishbone structure).

Structure (like fish skeleton):

                    bone_tip
                       |
       bone_left_upper   bone_right_upper
                  \\     |     /
                   \\    |    /
                    bone_middle
                   /     |    \\
                  /      |     \\
       bone_left_lower   |   bone_right_lower
                         |
                    bone_root

Central spine (3): root → middle → tip
Left ribs (2): left_lower, left_upper
Right ribs (2): right_lower, right_upper

All coordinates are 2D (x, y) since petal is created from 2D spline then extruded.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_bone_rigging_v5(n_samples: int = 500) -> pd.DataFrame:
    """
    Generate bone rigging dataset with 7 bones in fishbone structure.

    7 bones create natural petal deformation matching 8-CP shape:
    - bone_root: Base of petal (0% → 25% height)
    - bone_middle: Middle axis (25% → 60% height)
    - bone_tip: Top of petal (60% → 100% height)
    - bone_left_lower: Lower left rib (at 25% height, toward CP2)
    - bone_left_upper: Upper left rib (at 60% height, toward CP3)
    - bone_right_lower: Lower right rib (at 25% height, toward CP8)
    - bone_right_upper: Upper right rib (at 60% height, toward CP7)

    Features:
        - petal_height: Vertical extent of petal (y-axis)
        - petal_width: Horizontal width (x-axis)
        - opening_degree: How open the flower is (0=closed, 1=open)
        - layer_index: Which layer (0=inner, 1=middle, 2=outer)
        - curvature_intensity: How curved the petal is

    Targets (28 values - 7 bones x 4 coordinates):
        - bone_root_*: start_x, start_y, end_x, end_y
        - bone_middle_*: start_x, start_y, end_x, end_y
        - bone_tip_*: start_x, start_y, end_x, end_y
        - bone_left_lower_*: start_x, start_y, end_x, end_y
        - bone_left_upper_*: start_x, start_y, end_x, end_y
        - bone_right_lower_*: start_x, start_y, end_x, end_y
        - bone_right_upper_*: start_x, start_y, end_x, end_y
    """
    data = []

    for _ in range(n_samples):
        # === FEATURES ===
        petal_height = np.random.uniform(1.5, 9.6)
        petal_width = np.random.uniform(0.4, 3.0)
        opening_degree = np.random.uniform(0.0, 1.0)
        layer_index = np.random.randint(0, 3)  # 0, 1, 2
        curvature_intensity = np.random.uniform(0.5, 1.5)

        # === BONE STRUCTURE (2D coordinates) ===

        # Layer factor: 0.8, 0.9, 1.0
        layer_factor = 0.8 + 0.1 * layer_index

        # Width calculations matching 8-CP structure
        base_spread = petal_width * 0.35 * layer_factor
        mid_width = base_spread * 0.7   # at 25% height
        upper_width = base_spread * 0.5  # at 60% height

        # === CENTRAL SPINE (3 bones) ===

        # --- BONE ROOT ---
        # From base (0%) to lower junction (25% height)
        root_start_x = 0.0
        root_start_y = 0.0
        root_end_x = 0.0
        root_end_y = petal_height * 0.25 * layer_factor

        # --- BONE MIDDLE ---
        # From lower junction (25%) to upper junction (60% height)
        middle_start_x = root_end_x
        middle_start_y = root_end_y
        middle_end_x = 0.0
        middle_end_y = petal_height * 0.6 * layer_factor

        # --- BONE TIP ---
        # From upper junction (60%) to tip (100% height)
        tip_start_x = middle_end_x
        tip_start_y = middle_end_y
        tip_end_x = 0.0
        tip_end_y = petal_height * layer_factor

        # === LEFT RIBS (2 bones) ===

        # Opening factor affects how much ribs spread
        opening_factor = 0.5 + opening_degree * 0.5

        # --- BONE LEFT LOWER ---
        # Branches from root_end toward CP2 (lower left curve)
        left_lower_start_x = root_end_x
        left_lower_start_y = root_end_y
        left_lower_end_x = -mid_width * opening_factor * curvature_intensity
        left_lower_end_y = root_end_y * 1.1  # Slightly upward angle

        # --- BONE LEFT UPPER ---
        # Branches from middle_end toward CP3 (upper left curve)
        left_upper_start_x = middle_end_x
        left_upper_start_y = middle_end_y
        left_upper_end_x = -upper_width * opening_factor * curvature_intensity
        left_upper_end_y = middle_end_y * 1.05  # Slightly upward angle

        # === RIGHT RIBS (2 bones) - Symmetric ===

        # --- BONE RIGHT LOWER ---
        right_lower_start_x = root_end_x
        right_lower_start_y = root_end_y
        right_lower_end_x = mid_width * opening_factor * curvature_intensity
        right_lower_end_y = root_end_y * 1.1

        # --- BONE RIGHT UPPER ---
        right_upper_start_x = middle_end_x
        right_upper_start_y = middle_end_y
        right_upper_end_x = upper_width * opening_factor * curvature_intensity
        right_upper_end_y = middle_end_y * 1.05

        # Add realistic noise (3%)
        noise = 0.03

        # Spine noise
        root_end_y *= (1 + np.random.normal(0, noise))
        middle_end_y *= (1 + np.random.normal(0, noise))
        tip_end_y *= (1 + np.random.normal(0, noise))

        # Left rib noise
        left_lower_end_x *= (1 + np.random.normal(0, noise))
        left_lower_end_y *= (1 + np.random.normal(0, noise))
        left_upper_end_x *= (1 + np.random.normal(0, noise))
        left_upper_end_y *= (1 + np.random.normal(0, noise))

        # Right rib noise
        right_lower_end_x *= (1 + np.random.normal(0, noise))
        right_lower_end_y *= (1 + np.random.normal(0, noise))
        right_upper_end_x *= (1 + np.random.normal(0, noise))
        right_upper_end_y *= (1 + np.random.normal(0, noise))

        data.append({
            # Features
            'petal_height': round(petal_height, 6),
            'petal_width': round(petal_width, 6),
            'opening_degree': round(opening_degree, 6),
            'layer_index': layer_index,
            'curvature_intensity': round(curvature_intensity, 6),

            # Targets - Central Spine
            'bone_root_start_x': round(root_start_x, 6),
            'bone_root_start_y': round(root_start_y, 6),
            'bone_root_end_x': round(root_end_x, 6),
            'bone_root_end_y': round(root_end_y, 6),

            'bone_middle_start_x': round(middle_start_x, 6),
            'bone_middle_start_y': round(middle_start_y, 6),
            'bone_middle_end_x': round(middle_end_x, 6),
            'bone_middle_end_y': round(middle_end_y, 6),

            'bone_tip_start_x': round(tip_start_x, 6),
            'bone_tip_start_y': round(tip_start_y, 6),
            'bone_tip_end_x': round(tip_end_x, 6),
            'bone_tip_end_y': round(tip_end_y, 6),

            # Targets - Left Ribs
            'bone_left_lower_start_x': round(left_lower_start_x, 6),
            'bone_left_lower_start_y': round(left_lower_start_y, 6),
            'bone_left_lower_end_x': round(left_lower_end_x, 6),
            'bone_left_lower_end_y': round(left_lower_end_y, 6),

            'bone_left_upper_start_x': round(left_upper_start_x, 6),
            'bone_left_upper_start_y': round(left_upper_start_y, 6),
            'bone_left_upper_end_x': round(left_upper_end_x, 6),
            'bone_left_upper_end_y': round(left_upper_end_y, 6),

            # Targets - Right Ribs
            'bone_right_lower_start_x': round(right_lower_start_x, 6),
            'bone_right_lower_start_y': round(right_lower_start_y, 6),
            'bone_right_lower_end_x': round(right_lower_end_x, 6),
            'bone_right_lower_end_y': round(right_lower_end_y, 6),

            'bone_right_upper_start_x': round(right_upper_start_x, 6),
            'bone_right_upper_start_y': round(right_upper_start_y, 6),
            'bone_right_upper_end_x': round(right_upper_end_x, 6),
            'bone_right_upper_end_y': round(right_upper_end_y, 6),
        })

    return pd.DataFrame(data)


def main():
    """Generate bone rigging v5 dataset."""
    print("=" * 60)
    print("Bone Rigging V5 Dataset Generator")
    print("7 Bones: Fishbone Structure (3 spine + 4 ribs)")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating bone rigging v5 dataset...")
    df = generate_bone_rigging_v5(n_samples=500)

    output_path = output_dir / "bone_rigging_v5.csv"
    df.to_csv(output_path, index=False)

    print(f"\nGenerated {len(df)} samples")
    print(f"Output: {output_path}")

    print("\nFeatures (5):")
    features = ['petal_height', 'petal_width', 'opening_degree', 'layer_index', 'curvature_intensity']
    for f in features:
        print(f"  - {f}")

    print("\nTargets (28 - 7 bones x 4 coordinates):")

    print("\n  Central Spine (3 bones):")
    for bone in ['bone_root', 'bone_middle', 'bone_tip']:
        print(f"    {bone}: start_x, start_y, end_x, end_y")

    print("\n  Left Ribs (2 bones):")
    for bone in ['bone_left_lower', 'bone_left_upper']:
        print(f"    {bone}: start_x, start_y, end_x, end_y")

    print("\n  Right Ribs (2 bones):")
    for bone in ['bone_right_lower', 'bone_right_upper']:
        print(f"    {bone}: start_x, start_y, end_x, end_y")

    print("\nDataset Statistics:")
    print(f"  petal_height: {df['petal_height'].min():.2f} - {df['petal_height'].max():.2f}")
    print(f"  petal_width: {df['petal_width'].min():.2f} - {df['petal_width'].max():.2f}")

    # Rib spread statistics
    left_lower_spread = abs(df['bone_left_lower_end_x'])
    left_upper_spread = abs(df['bone_left_upper_end_x'])
    print(f"  left_lower rib spread: {left_lower_spread.mean():.4f}")
    print(f"  left_upper rib spread: {left_upper_spread.mean():.4f}")

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)

    print("\nFishbone Structure:")
    print("""
                        bone_tip
                           |
           bone_left_upper   bone_right_upper
                      \\     |     /
                       \\    |    /
                        bone_middle
                       /     |    \\
                      /      |     \\
           bone_left_lower   |   bone_right_lower
                             |
                        bone_root
    """)

    print("Bone-to-CP Alignment:")
    print("  - bone_root: CP1 (base)")
    print("  - bone_left_lower/right_lower: CP2/CP8 (lower curves)")
    print("  - bone_middle: spine at 60% height")
    print("  - bone_left_upper/right_upper: CP3/CP7 (upper curves)")
    print("  - bone_tip: CP4-CP5-CP6 (tip area)")

    print("\nNext steps:")
    print("  1. Update configs/sr_config_spline_v3.yaml with bone_rigging_v5")
    print("  2. Train SR models to discover formulas")
    print("  3. Update CLI generator to use fishbone structure")


if __name__ == "__main__":
    main()
