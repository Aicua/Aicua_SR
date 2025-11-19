#!/usr/bin/env python3
"""
Generate bone rigging v6 dataset with 5 INDEPENDENT bones.

Key changes from v5:
- 5 bones instead of 12 (reduces complexity)
- NO parent-child relationships (independent positioning)
- Supports deformation types: straight, s_curve, c_curve, wave

Structure (5 independent bones along petal height):

    bone_tip (78%-100%)           → Controls tip & taper

    bone_upper (62%-78%)          → Controls upper region

    bone_mid_upper (45%-62%)      → Controls WIDEST region (critical)

    bone_mid (25%-45%)            → Controls middle expansion

    bone_base (0%-25%)            → Anchors base

Each bone has: start_x, start_y, end_x, end_y (4 coords)
Total: 5 bones × 4 coords = 20 targets

All coordinates are 2D (x, y) since petal is created from 2D spline then extruded.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_bone_rigging_v6(n_samples: int = 500) -> pd.DataFrame:
    """
    Generate bone rigging dataset with 5 independent bones.

    Features (6 inputs):
        - base_size: Overall rose size (2.0 - 8.0)
        - layer_index: Which layer (0=inner, 1=middle, 2=outer)
        - petal_index: Position within layer (0-12)
        - opening_degree: How open the flower is (0=closed, 1=open)
        - deformation_type: 0=straight, 1=s_curve, 2=c_curve, 3=wave
        - intensity: Deformation intensity (0.0 - 1.0)

    Targets (20 values - 5 bones × 4 coordinates):
        - bone_base_*: start_x, start_y, end_x, end_y
        - bone_mid_*: start_x, start_y, end_x, end_y
        - bone_mid_upper_*: start_x, start_y, end_x, end_y
        - bone_upper_*: start_x, start_y, end_x, end_y
        - bone_tip_*: start_x, start_y, end_x, end_y
    """
    data = []

    petals_per_layer = [5, 8, 13]  # Fibonacci - matching petal_spline_v3

    for _ in range(n_samples):
        # === FEATURES ===
        base_size = np.random.uniform(2.0, 8.0)
        opening_degree = np.random.uniform(0.0, 1.0)
        deformation_type = np.random.randint(0, 4)  # 0=straight, 1=s_curve, 2=c_curve, 3=wave
        intensity = np.random.uniform(0.0, 1.0)

        for layer_idx in range(3):
            n_petals = petals_per_layer[layer_idx]

            for petal_idx in range(n_petals):
                # === LAYER FACTOR ===
                layer_factor = 0.8 + 0.1 * layer_idx

                # === PETAL HEIGHT ===
                petal_height = (
                    base_size * layer_factor *
                    (1.2 - opening_degree * 0.3)
                )

                # === WIDTH AT EACH HEIGHT LEVEL ===
                base_spread = (
                    base_size * 0.30 * layer_factor *
                    (1 + opening_degree * 0.2)
                )

                # Width profile (matching petal_spline_v3)
                width_at_25 = base_spread * 1.05   # 25% height
                width_at_45 = base_spread * 1.4    # 45% height
                width_at_62 = base_spread * 1.6    # 62% height - WIDEST
                width_at_78 = base_spread * 1.3    # 78% height
                width_at_100 = base_spread * 0.1   # 100% height - tip

                # === DEFORMATION OFFSETS ===
                # Calculate x-offsets based on deformation type
                offset_25 = 0.0
                offset_45 = 0.0
                offset_62 = 0.0
                offset_78 = 0.0
                offset_100 = 0.0

                if deformation_type == 0:  # Straight
                    # No offset
                    pass

                elif deformation_type == 1:  # S-curve
                    # Pattern: out → in → out
                    offset_25 = width_at_25 * 0.1 * intensity
                    offset_45 = width_at_45 * 0.3 * intensity
                    offset_62 = -width_at_62 * 0.2 * intensity
                    offset_78 = -width_at_78 * 0.15 * intensity
                    offset_100 = width_at_100 * 0.25 * intensity

                elif deformation_type == 2:  # C-curve (inward curl)
                    # Pattern: consistent inward
                    offset_25 = -width_at_25 * 0.05 * intensity
                    offset_45 = -width_at_45 * 0.15 * intensity
                    offset_62 = -width_at_62 * 0.35 * intensity  # strongest
                    offset_78 = -width_at_78 * 0.3 * intensity
                    offset_100 = -width_at_100 * 0.25 * intensity

                elif deformation_type == 3:  # Wave
                    # Pattern: sinusoidal
                    offset_25 = width_at_25 * 0.2 * np.sin(0.25 * np.pi * 2) * intensity
                    offset_45 = width_at_45 * 0.2 * np.sin(0.45 * np.pi * 2) * intensity
                    offset_62 = width_at_62 * 0.2 * np.sin(0.62 * np.pi * 2) * intensity
                    offset_78 = width_at_78 * 0.2 * np.sin(0.78 * np.pi * 2) * intensity
                    offset_100 = width_at_100 * 0.2 * np.sin(1.0 * np.pi * 2) * intensity

                # === 5 INDEPENDENT BONES ===

                # --- BONE BASE (0% → 25%) ---
                # Anchors at origin, extends to 25% height
                base_start_x = 0.0
                base_start_y = 0.0
                base_end_x = offset_25
                base_end_y = petal_height * 0.25

                # --- BONE MID (25% → 45%) ---
                # Controls middle expansion
                mid_start_x = offset_25
                mid_start_y = petal_height * 0.25
                mid_end_x = offset_45
                mid_end_y = petal_height * 0.45

                # --- BONE MID UPPER (45% → 62%) ---
                # Controls WIDEST region (critical)
                mid_upper_start_x = offset_45
                mid_upper_start_y = petal_height * 0.45
                mid_upper_end_x = offset_62
                mid_upper_end_y = petal_height * 0.62

                # --- BONE UPPER (62% → 78%) ---
                # Controls upper taper
                upper_start_x = offset_62
                upper_start_y = petal_height * 0.62
                upper_end_x = offset_78
                upper_end_y = petal_height * 0.78

                # --- BONE TIP (78% → 100%) ---
                # Controls tip
                tip_start_x = offset_78
                tip_start_y = petal_height * 0.78
                tip_end_x = offset_100
                tip_end_y = petal_height

                # === ADD REALISTIC NOISE (2%) ===
                noise = 0.02

                # Apply noise to end positions only (start stays clean)
                base_end_x *= (1 + np.random.normal(0, noise))
                base_end_y *= (1 + np.random.normal(0, noise))
                mid_end_x *= (1 + np.random.normal(0, noise))
                mid_end_y *= (1 + np.random.normal(0, noise))
                mid_upper_end_x *= (1 + np.random.normal(0, noise))
                mid_upper_end_y *= (1 + np.random.normal(0, noise))
                upper_end_x *= (1 + np.random.normal(0, noise))
                upper_end_y *= (1 + np.random.normal(0, noise))
                tip_end_x *= (1 + np.random.normal(0, noise))
                tip_end_y *= (1 + np.random.normal(0, noise))

                data.append({
                    # Features
                    'base_size': round(base_size, 6),
                    'layer_index': layer_idx,
                    'petal_index': petal_idx,
                    'opening_degree': round(opening_degree, 6),
                    'deformation_type': deformation_type,
                    'intensity': round(intensity, 6),

                    # Targets - 5 Independent Bones (20 coords)
                    'bone_base_start_x': round(base_start_x, 6),
                    'bone_base_start_y': round(base_start_y, 6),
                    'bone_base_end_x': round(base_end_x, 6),
                    'bone_base_end_y': round(base_end_y, 6),

                    'bone_mid_start_x': round(mid_start_x, 6),
                    'bone_mid_start_y': round(mid_start_y, 6),
                    'bone_mid_end_x': round(mid_end_x, 6),
                    'bone_mid_end_y': round(mid_end_y, 6),

                    'bone_mid_upper_start_x': round(mid_upper_start_x, 6),
                    'bone_mid_upper_start_y': round(mid_upper_start_y, 6),
                    'bone_mid_upper_end_x': round(mid_upper_end_x, 6),
                    'bone_mid_upper_end_y': round(mid_upper_end_y, 6),

                    'bone_upper_start_x': round(upper_start_x, 6),
                    'bone_upper_start_y': round(upper_start_y, 6),
                    'bone_upper_end_x': round(upper_end_x, 6),
                    'bone_upper_end_y': round(upper_end_y, 6),

                    'bone_tip_start_x': round(tip_start_x, 6),
                    'bone_tip_start_y': round(tip_start_y, 6),
                    'bone_tip_end_x': round(tip_end_x, 6),
                    'bone_tip_end_y': round(tip_end_y, 6),
                })

    return pd.DataFrame(data)


def main():
    """Generate bone rigging v6 dataset."""
    print("=" * 60)
    print("Bone Rigging V6 Dataset Generator")
    print("5 INDEPENDENT Bones (no parent-child)")
    print("Supports: straight, s_curve, c_curve, wave deformations")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating bone rigging v6 dataset...")
    df = generate_bone_rigging_v6(n_samples=500)

    output_path = output_dir / "bone_rigging_v6.csv"
    df.to_csv(output_path, index=False)

    print(f"\nGenerated {len(df)} samples")
    print(f"Output: {output_path}")

    print("\nFeatures (6):")
    features = ['base_size', 'layer_index', 'petal_index', 'opening_degree',
                'deformation_type', 'intensity']
    for f in features:
        print(f"  - {f}")

    print("\nDeformation Types:")
    print("  0 = straight (no deformation)")
    print("  1 = s_curve (out → in → out)")
    print("  2 = c_curve (inward curl)")
    print("  3 = wave (sinusoidal)")

    print("\nTargets (20 - 5 bones × 4 coordinates):")
    for bone in ['bone_base', 'bone_mid', 'bone_mid_upper', 'bone_upper', 'bone_tip']:
        print(f"  {bone}: start_x, start_y, end_x, end_y")

    print("\nDataset Statistics:")
    print(f"  base_size: {df['base_size'].min():.2f} - {df['base_size'].max():.2f}")
    print(f"  deformation_type distribution:")
    for dtype in range(4):
        count = len(df[df['deformation_type'] == dtype])
        pct = count / len(df) * 100
        names = ['straight', 's_curve', 'c_curve', 'wave']
        print(f"    {dtype} ({names[dtype]}): {count} ({pct:.1f}%)")

    # Calculate petal dimensions
    petal_heights = df['base_size'] * (0.8 + 0.1 * df['layer_index']) * (1.2 - df['opening_degree'] * 0.3)
    print(f"\n  petal_height (calculated): {petal_heights.min():.2f} - {petal_heights.max():.2f}")

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)

    print("\n5-Bone Structure (INDEPENDENT):")
    print("""
    bone_tip (78%-100%)           → Controls tip & taper

    bone_upper (62%-78%)          → Controls upper region

    bone_mid_upper (45%-62%)      → Controls WIDEST region (critical)

    bone_mid (25%-45%)            → Controls middle expansion

    bone_base (0%-25%)            → Anchors base
    """)

    print("Height Zone Coverage:")
    print("  - bone_base:      0% → 25%")
    print("  - bone_mid:      25% → 45%")
    print("  - bone_mid_upper: 45% → 62% (WIDEST)")
    print("  - bone_upper:    62% → 78%")
    print("  - bone_tip:      78% → 100%")

    print("\nComplexity Comparison:")
    print("  - v5 (12 bones): 48 targets")
    print("  - v6 (5 bones):  20 targets (58% reduction)")


if __name__ == "__main__":
    main()
