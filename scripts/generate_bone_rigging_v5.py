#!/usr/bin/env python3
"""
Generate bone rigging v5 dataset with 12 bones (fishbone structure).

Updated to match petal_spline_v3 with 15 control points.

Structure (like fish skeleton):

                        bone_tip
                           |
           bone_left_upper   bone_right_upper (78%)
                      \     |     /
            bone_left_mid_upper  bone_right_mid_upper (62% - WIDEST)
                        \   |   /
              bone_left_mid_lower  bone_right_mid_lower (45%)
                          \ | /
                     bone_upper_mid
                           |
                     bone_lower_mid
                          / | \
               bone_left_lower  bone_right_lower (25%)
                         / | \
                    bone_root
                       (0%)

Central spine (4): root → lower_mid → upper_mid → tip
Left ribs (4): lower (25%), mid_lower (45%), mid_upper (62%), upper (78%)
Right ribs (4): symmetric

All coordinates are 2D (x, y) since petal is created from 2D spline then extruded.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_bone_rigging_v5(n_samples: int = 500) -> pd.DataFrame:
    """
    Generate bone rigging dataset with 12 bones in fishbone structure.

    Updated to match petal_spline_v3 (15 CPs):
    - bone_root: Base of petal (0% → 25% height)
    - bone_lower_mid: Lower middle axis (25% → 45% height)
    - bone_upper_mid: Upper middle axis (45% → 62% height)
    - bone_tip: Top of petal (62% → 100% height)
    - bone_left_lower: Lower left rib (at 25% height, toward CP3)
    - bone_left_mid_lower: Mid-lower left rib (at 45% height, toward CP4)
    - bone_left_mid_upper: Mid-upper left rib (at 62% height, toward CP5 - WIDEST)
    - bone_left_upper: Upper left rib (at 78% height, toward CP6)
    - bone_right_*: Symmetric right ribs

    Features (matching petal_spline_v3):
        - base_size: Overall rose size (2.0 - 8.0)
        - layer_index: Which layer (0=inner, 1=middle, 2=outer)
        - petal_index: Position within layer (0-12)
        - opening_degree: How open the flower is (0=closed, 1=open)
        - curvature_intensity: How curved the petal is

    Targets (48 values - 12 bones x 4 coordinates):
        - bone_root_*: start_x, start_y, end_x, end_y
        - bone_lower_mid_*: start_x, start_y, end_x, end_y
        - bone_upper_mid_*: start_x, start_y, end_x, end_y
        - bone_tip_*: start_x, start_y, end_x, end_y
        - bone_left_lower_*: start_x, start_y, end_x, end_y
        - bone_left_mid_lower_*: start_x, start_y, end_x, end_y
        - bone_left_mid_upper_*: start_x, start_y, end_x, end_y
        - bone_left_upper_*: start_x, start_y, end_x, end_y
        - bone_right_lower_*: start_x, start_y, end_x, end_y
        - bone_right_mid_lower_*: start_x, start_y, end_x, end_y
        - bone_right_mid_upper_*: start_x, start_y, end_x, end_y
        - bone_right_upper_*: start_x, start_y, end_x, end_y
    """
    data = []

    petals_per_layer = [5, 8, 13]  # Fibonacci - matching petal_spline_v3

    for _ in range(n_samples):
        # === FEATURES (matching petal_spline_v3) ===
        base_size = np.random.uniform(2.0, 8.0)
        opening_degree = np.random.uniform(0.0, 1.0)
        curvature_intensity = np.random.uniform(0.5, 1.5)

        for layer_idx in range(3):
            n_petals = petals_per_layer[layer_idx]

            for petal_idx in range(n_petals):
                # === LAYER FACTOR (matching petal_spline_v3) ===
                layer_factor = 0.8 + 0.1 * layer_idx

                # === PETAL HEIGHT (matching petal_spline_v3) ===
                petal_height = (
                    base_size * layer_factor *
                    (1.2 - opening_degree * 0.3)
                )

                # === WIDTH CALCULATIONS (matching petal_spline_v3) ===
                base_spread = (
                    base_size * 0.30 * layer_factor *
                    (1 + opening_degree * 0.2)
                )

                # Width at each height level (matching petal CPs)
                lower_width = base_spread * 1.05  # 25% height (avg of 0.95-1.15)
                mid_low_width = base_spread * 1.4  # 45% height
                upper_mid_width = base_spread * 1.6  # 62% height - WIDEST
                upper_width = base_spread * 1.3  # 78% height

                # === CENTRAL SPINE (4 bones) ===

                # --- BONE ROOT ---
                # From base (0%) to lower junction (25% height)
                root_start_x = 0.0
                root_start_y = 0.0
                root_end_x = 0.0
                root_end_y = petal_height * 0.25

                # --- BONE LOWER MID ---
                # From lower junction (25%) to mid junction (45% height)
                lower_mid_start_x = root_end_x
                lower_mid_start_y = root_end_y
                lower_mid_end_x = 0.0
                lower_mid_end_y = petal_height * 0.45

                # --- BONE UPPER MID ---
                # From mid junction (45%) to upper junction (62% height)
                upper_mid_start_x = lower_mid_end_x
                upper_mid_start_y = lower_mid_end_y
                upper_mid_end_x = 0.0
                upper_mid_end_y = petal_height * 0.62

                # --- BONE TIP ---
                # From upper junction (62%) to tip (100% height)
                tip_start_x = upper_mid_end_x
                tip_start_y = upper_mid_end_y
                tip_end_x = 0.0
                tip_end_y = petal_height

                # === LEFT RIBS (4 bones) ===

                # Opening factor affects how much ribs spread
                opening_factor = 0.5 + opening_degree * 0.5

                # --- BONE LEFT LOWER ---
                # Branches from root_end toward CP3 (25% height)
                left_lower_start_x = root_end_x
                left_lower_start_y = root_end_y
                left_lower_end_x = -lower_width * 0.5 * opening_factor * curvature_intensity
                left_lower_end_y = root_end_y * 1.05

                # --- BONE LEFT MID LOWER ---
                # Branches from lower_mid_end toward CP4 (45% height)
                left_mid_lower_start_x = lower_mid_end_x
                left_mid_lower_start_y = lower_mid_end_y
                left_mid_lower_end_x = -mid_low_width * 0.5 * opening_factor * curvature_intensity
                left_mid_lower_end_y = lower_mid_end_y * 1.03

                # --- BONE LEFT MID UPPER ---
                # Branches from upper_mid_end toward CP5 (62% height - WIDEST)
                left_mid_upper_start_x = upper_mid_end_x
                left_mid_upper_start_y = upper_mid_end_y
                left_mid_upper_end_x = -upper_mid_width * 0.5 * opening_factor * curvature_intensity
                left_mid_upper_end_y = upper_mid_end_y * 1.02

                # --- BONE LEFT UPPER ---
                # Branches toward CP6 (78% height)
                left_upper_start_x = 0.0
                left_upper_start_y = petal_height * 0.78
                left_upper_end_x = -upper_width * 0.5 * opening_factor * curvature_intensity
                left_upper_end_y = left_upper_start_y * 1.01

                # === RIGHT RIBS (4 bones) - Symmetric ===

                # --- BONE RIGHT LOWER ---
                right_lower_start_x = root_end_x
                right_lower_start_y = root_end_y
                right_lower_end_x = lower_width * 0.5 * opening_factor * curvature_intensity
                right_lower_end_y = root_end_y * 1.05

                # --- BONE RIGHT MID LOWER ---
                right_mid_lower_start_x = lower_mid_end_x
                right_mid_lower_start_y = lower_mid_end_y
                right_mid_lower_end_x = mid_low_width * 0.5 * opening_factor * curvature_intensity
                right_mid_lower_end_y = lower_mid_end_y * 1.03

                # --- BONE RIGHT MID UPPER ---
                right_mid_upper_start_x = upper_mid_end_x
                right_mid_upper_start_y = upper_mid_end_y
                right_mid_upper_end_x = upper_mid_width * 0.5 * opening_factor * curvature_intensity
                right_mid_upper_end_y = upper_mid_end_y * 1.02

                # --- BONE RIGHT UPPER ---
                right_upper_start_x = 0.0
                right_upper_start_y = petal_height * 0.78
                right_upper_end_x = upper_width * 0.5 * opening_factor * curvature_intensity
                right_upper_end_y = right_upper_start_y * 1.01

                # === ADD REALISTIC NOISE (3%) ===
                noise = 0.03

                # Spine noise
                root_end_y *= (1 + np.random.normal(0, noise))
                lower_mid_end_y *= (1 + np.random.normal(0, noise))
                upper_mid_end_y *= (1 + np.random.normal(0, noise))
                tip_end_y *= (1 + np.random.normal(0, noise))

                # Left rib noise
                left_lower_end_x *= (1 + np.random.normal(0, noise))
                left_lower_end_y *= (1 + np.random.normal(0, noise))
                left_mid_lower_end_x *= (1 + np.random.normal(0, noise))
                left_mid_lower_end_y *= (1 + np.random.normal(0, noise))
                left_mid_upper_end_x *= (1 + np.random.normal(0, noise))
                left_mid_upper_end_y *= (1 + np.random.normal(0, noise))
                left_upper_end_x *= (1 + np.random.normal(0, noise))
                left_upper_end_y *= (1 + np.random.normal(0, noise))

                # Right rib noise
                right_lower_end_x *= (1 + np.random.normal(0, noise))
                right_lower_end_y *= (1 + np.random.normal(0, noise))
                right_mid_lower_end_x *= (1 + np.random.normal(0, noise))
                right_mid_lower_end_y *= (1 + np.random.normal(0, noise))
                right_mid_upper_end_x *= (1 + np.random.normal(0, noise))
                right_mid_upper_end_y *= (1 + np.random.normal(0, noise))
                right_upper_end_x *= (1 + np.random.normal(0, noise))
                right_upper_end_y *= (1 + np.random.normal(0, noise))

                data.append({
                    # Features
                    'base_size': round(base_size, 6),
                    'layer_index': layer_idx,
                    'petal_index': petal_idx,
                    'opening_degree': round(opening_degree, 6),
                    'curvature_intensity': round(curvature_intensity, 6),

                    # Targets - Central Spine (4 bones)
                    'bone_root_start_x': round(root_start_x, 6),
                    'bone_root_start_y': round(root_start_y, 6),
                    'bone_root_end_x': round(root_end_x, 6),
                    'bone_root_end_y': round(root_end_y, 6),

                    'bone_lower_mid_start_x': round(lower_mid_start_x, 6),
                    'bone_lower_mid_start_y': round(lower_mid_start_y, 6),
                    'bone_lower_mid_end_x': round(lower_mid_end_x, 6),
                    'bone_lower_mid_end_y': round(lower_mid_end_y, 6),

                    'bone_upper_mid_start_x': round(upper_mid_start_x, 6),
                    'bone_upper_mid_start_y': round(upper_mid_start_y, 6),
                    'bone_upper_mid_end_x': round(upper_mid_end_x, 6),
                    'bone_upper_mid_end_y': round(upper_mid_end_y, 6),

                    'bone_tip_start_x': round(tip_start_x, 6),
                    'bone_tip_start_y': round(tip_start_y, 6),
                    'bone_tip_end_x': round(tip_end_x, 6),
                    'bone_tip_end_y': round(tip_end_y, 6),

                    # Targets - Left Ribs (4 bones)
                    'bone_left_lower_start_x': round(left_lower_start_x, 6),
                    'bone_left_lower_start_y': round(left_lower_start_y, 6),
                    'bone_left_lower_end_x': round(left_lower_end_x, 6),
                    'bone_left_lower_end_y': round(left_lower_end_y, 6),

                    'bone_left_mid_lower_start_x': round(left_mid_lower_start_x, 6),
                    'bone_left_mid_lower_start_y': round(left_mid_lower_start_y, 6),
                    'bone_left_mid_lower_end_x': round(left_mid_lower_end_x, 6),
                    'bone_left_mid_lower_end_y': round(left_mid_lower_end_y, 6),

                    'bone_left_mid_upper_start_x': round(left_mid_upper_start_x, 6),
                    'bone_left_mid_upper_start_y': round(left_mid_upper_start_y, 6),
                    'bone_left_mid_upper_end_x': round(left_mid_upper_end_x, 6),
                    'bone_left_mid_upper_end_y': round(left_mid_upper_end_y, 6),

                    'bone_left_upper_start_x': round(left_upper_start_x, 6),
                    'bone_left_upper_start_y': round(left_upper_start_y, 6),
                    'bone_left_upper_end_x': round(left_upper_end_x, 6),
                    'bone_left_upper_end_y': round(left_upper_end_y, 6),

                    # Targets - Right Ribs (4 bones)
                    'bone_right_lower_start_x': round(right_lower_start_x, 6),
                    'bone_right_lower_start_y': round(right_lower_start_y, 6),
                    'bone_right_lower_end_x': round(right_lower_end_x, 6),
                    'bone_right_lower_end_y': round(right_lower_end_y, 6),

                    'bone_right_mid_lower_start_x': round(right_mid_lower_start_x, 6),
                    'bone_right_mid_lower_start_y': round(right_mid_lower_start_y, 6),
                    'bone_right_mid_lower_end_x': round(right_mid_lower_end_x, 6),
                    'bone_right_mid_lower_end_y': round(right_mid_lower_end_y, 6),

                    'bone_right_mid_upper_start_x': round(right_mid_upper_start_x, 6),
                    'bone_right_mid_upper_start_y': round(right_mid_upper_start_y, 6),
                    'bone_right_mid_upper_end_x': round(right_mid_upper_end_x, 6),
                    'bone_right_mid_upper_end_y': round(right_mid_upper_end_y, 6),

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
    print("12 Bones: Fishbone Structure (4 spine + 8 ribs)")
    print("Updated to match petal_spline_v3 (15 CPs)")
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
    features = ['base_size', 'layer_index', 'petal_index', 'opening_degree', 'curvature_intensity']
    for f in features:
        print(f"  - {f}")

    print("\nTargets (48 - 12 bones x 4 coordinates):")

    print("\n  Central Spine (4 bones):")
    for bone in ['bone_root', 'bone_lower_mid', 'bone_upper_mid', 'bone_tip']:
        print(f"    {bone}: start_x, start_y, end_x, end_y")

    print("\n  Left Ribs (4 bones):")
    for bone in ['bone_left_lower', 'bone_left_mid_lower', 'bone_left_mid_upper', 'bone_left_upper']:
        print(f"    {bone}: start_x, start_y, end_x, end_y")

    print("\n  Right Ribs (4 bones):")
    for bone in ['bone_right_lower', 'bone_right_mid_lower', 'bone_right_mid_upper', 'bone_right_upper']:
        print(f"    {bone}: start_x, start_y, end_x, end_y")

    print("\nDataset Statistics:")
    print(f"  base_size: {df['base_size'].min():.2f} - {df['base_size'].max():.2f}")

    # Calculate petal dimensions from base_size
    petal_heights = df['base_size'] * (0.8 + 0.1 * df['layer_index']) * (1.2 - df['opening_degree'] * 0.3)
    print(f"  petal_height (calculated): {petal_heights.min():.2f} - {petal_heights.max():.2f}")

    # Rib spread statistics
    left_lower_spread = abs(df['bone_left_lower_end_x'])
    left_mid_upper_spread = abs(df['bone_left_mid_upper_end_x'])
    print(f"  left_lower rib spread (25%): {left_lower_spread.mean():.4f}")
    print(f"  left_mid_upper rib spread (62% - WIDEST): {left_mid_upper_spread.mean():.4f}")

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)

    print("\nFishbone Structure (12 bones):")
    print("""
                            bone_tip
                               |
               bone_left_upper   bone_right_upper (78%)
                          \\     |     /
                bone_left_mid_upper  bone_right_mid_upper (62% - WIDEST)
                            \\   |   /
                  bone_left_mid_lower  bone_right_mid_lower (45%)
                              \\ | /
                         bone_upper_mid
                               |
                         bone_lower_mid
                              / | \\
                   bone_left_lower  bone_right_lower (25%)
                             / | \\
                        bone_root
                           (0%)
    """)

    print("Bone-to-CP Alignment (matching petal_spline_v3):")
    print("  - bone_root: CP1 (base center)")
    print("  - bone_left_lower/right_lower (25%): CP3/CP13")
    print("  - bone_left_mid_lower/right_mid_lower (45%): CP4/CP12")
    print("  - bone_left_mid_upper/right_mid_upper (62%): CP5/CP11 (WIDEST)")
    print("  - bone_left_upper/right_upper (78%): CP6/CP10")
    print("  - bone_tip: CP8 (tip center)")

    print("\nWidth matching petal_spline_v3:")
    print("  - 25%: lower_width = base_spread * 1.05")
    print("  - 45%: mid_low_width = base_spread * 1.4")
    print("  - 62%: upper_mid_width = base_spread * 1.6 (WIDEST)")
    print("  - 78%: upper_width = base_spread * 1.3")


if __name__ == "__main__":
    main()
