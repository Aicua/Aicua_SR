#!/usr/bin/env python3
"""
Generate petal spline dataset v3 with:
- 8 Control Points for realistic rose petal shape
- Heart-shaped tip with optional notch
- Undulating edges for organic look
- Based on rose_petals.jpg reference
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_petal_spline_v3(n_samples: int = 500) -> pd.DataFrame:
    """
    Generate petal spline dataset with 8 control points.

    Shape: Realistic rose petal from rose_petals.jpg
    - Rounded/heart-shaped tip with notch
    - Undulating edges
    - Organic curvature

    Control Points (2D):
    - CP1: Base left (narrow)
    - CP2: Lower left curve
    - CP3: Upper left curve (widest)
    - CP4: Tip left (before notch)
    - CP5: Tip center
    - CP6: Tip right (after notch)
    - CP7: Upper right curve (widest)
    - CP8: Lower right curve

    Features:
        - base_size: Overall rose size (2.0 - 8.0)
        - layer_index: 0 (inner), 1 (middle), 2 (outer)
        - petal_index: Position within layer (0-12)
        - opening_degree: 0.0 (closed) to 1.0 (fully open)

    Targets:
        - cp1_x, cp1_y through cp8_x, cp8_y
        - extrude_depth (ultra-thin)
    """
    data = []

    petals_per_layer = [5, 8, 13]  # Fibonacci

    for _ in range(n_samples):
        # Larger base_size range for bigger petals
        base_size = np.random.uniform(2.0, 8.0)
        opening_degree = np.random.uniform(0.0, 1.0)

        for layer_idx in range(3):  # 0, 1, 2 (0-based)
            n_petals = petals_per_layer[layer_idx]

            for petal_idx in range(n_petals):
                # === LAYER FACTOR ===
                # 0.8 + 0.1 * layer_idx gives [0.8, 0.9, 1.0]
                layer_factor = 0.8 + 0.1 * layer_idx

                # === WIDTH CALCULATIONS ===
                # Base spread at bottom
                base_spread = (
                    base_size * 0.35 * layer_factor *
                    (1 + opening_degree * 0.2)
                )

                # Mid width at 25% height
                mid_width = base_spread * 0.7

                # Upper width at 60% height
                upper_width = base_spread * 0.5

                # === PETAL HEIGHT ===
                petal_height = (
                    base_size * layer_factor *
                    (1.2 - opening_degree * 0.3)
                )

                # === TIP NOTCH (heart-shape) ===
                # Inner petals have deeper notch
                notch_depth = petal_height * 0.05 * (1 - layer_idx * 0.3)
                notch_width = base_spread * 0.15

                # Tip x offset (for asymmetry/tilt)
                tip_x_offset = base_size * 0.02 * layer_idx * opening_degree

                # === 8 CONTROL POINTS ===

                # CP1: Base left (narrow - 1/4 of spread)
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

                # CP7: Upper right curve (widest - symmetric)
                cp7_x = upper_width / 2
                cp7_y = petal_height * 0.6

                # CP8: Lower right curve (symmetric)
                cp8_x = mid_width / 2
                cp8_y = petal_height * 0.25

                # === ULTRA-THIN THICKNESS ===
                thickness_base = 0.005
                thickness = (
                    thickness_base * base_size *
                    (1 - layer_idx * 0.1) *  # Inner thicker
                    (1 - opening_degree * 0.3)  # Opening reduces
                )
                extrude_depth = max(0.001, thickness)

                # === ADD REALISTIC NOISE (3%) ===
                noise = 0.03

                # Apply noise to all control points
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
                    'base_size': round(base_size, 6),
                    'layer_index': layer_idx,
                    'petal_index': petal_idx,
                    'opening_degree': round(opening_degree, 6),

                    # Targets - 8 Control Points
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
                })

    return pd.DataFrame(data)


def main():
    """Generate petal spline v3 dataset."""
    print("=" * 60)
    print("Petal Spline V3 Dataset Generator")
    print("8 Control Points - Realistic Rose Petal Shape")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating petal spline v3 dataset...")
    df = generate_petal_spline_v3(n_samples=500)

    output_path = output_dir / "petal_spline_v3.csv"
    df.to_csv(output_path, index=False)

    print(f"\nGenerated {len(df)} samples")
    print(f"Output: {output_path}")

    # Statistics
    print("\nDataset Statistics:")
    print(f"  base_size: {df['base_size'].min():.2f} - {df['base_size'].max():.2f}")
    print(f"  petal_height (cp5_y): {df['cp5_y'].min():.4f} - {df['cp5_y'].max():.4f}")
    print(f"  extrude_depth: {df['extrude_depth'].min():.6f} - {df['extrude_depth'].max():.6f}")

    # Verify shape characteristics
    print("\nShape Characteristics:")

    # Upper width (at 60% height)
    upper_width = df['cp7_x'] - df['cp3_x']
    print(f"  Upper width (CP3 to CP7): {upper_width.mean():.4f}")

    # Lower width (at 25% height)
    lower_width = df['cp8_x'] - df['cp2_x']
    print(f"  Lower width (CP2 to CP8): {lower_width.mean():.4f}")

    # Base width
    base_width = -df['cp1_x'] * 2  # symmetric around 0
    print(f"  Base width: {base_width.mean():.4f}")

    # Notch depth
    notch_depth = df['cp5_y'] - df['cp4_y']
    print(f"  Notch depth: {notch_depth.mean():.4f}")

    # Width ratios
    upper_to_lower = upper_width / lower_width
    print(f"  Upper/Lower ratio: {upper_to_lower.mean():.2f}")

    print("\nSample data (first 3 rows):")
    print(df.head(3).to_string())

    print("\n" + "=" * 60)
    print("Petal Spline V3 Dataset Generation Complete!")
    print("=" * 60)

    print("\nShape Description (8 Control Points):")
    print("  CP1/CP8: Base (narrow) at ±1/4 spread")
    print("  CP2/CP8: Lower curve at 25% height")
    print("  CP3/CP7: Upper curve (widest) at 60% height")
    print("  CP4/CP6: Tip sides (before/after notch)")
    print("  CP5: Tip center (heart-shaped)")

    print("\nFeatures to capture from rose_petals.jpg:")
    print("  - Rounded/heart-shaped tip with notch")
    print("  - Undulating edges (multiple curve controls)")
    print("  - Organic curvature variation")


if __name__ == "__main__":
    main()
