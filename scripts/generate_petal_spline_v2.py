#!/usr/bin/env python3
"""
Generate petal spline dataset v2 with:
- Mid-curve widest shape (CP2/CP4 wider than base)
- Ultra-thin thickness (realistic 0.2-0.5mm)
- Larger petal size for bigger bones
- Continuous layer_factor formula
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_petal_spline_v2(n_samples: int = 500) -> pd.DataFrame:
    """
    Generate petal spline dataset with middle-wide shape.

    Shape: Base narrow → Middle widest → Tip pointed

    Control Points (2D):
    - CP1/CP5: Base (narrow) at ±base_spread/4
    - CP2/CP4: Mid-curve (WIDEST) at ±base_spread/2, 40% height
    - CP3: Tip (pointed center)

    Features:
        - base_size: Overall rose size (2.0 - 8.0, larger range)
        - layer_index: 0 (inner), 1 (middle), 2 (outer)
        - petal_index: Position within layer (0-12)
        - opening_degree: 0.0 (closed) to 1.0 (fully open)

    Targets:
        - cp1_x, cp1_y through cp5_x, cp5_y
        - extrude_depth (ultra-thin)
    """
    data = []

    petals_per_layer = [5, 8, 13]  # Fibonacci

    for _ in range(n_samples):
        # Larger base_size range for bigger petals → bigger bones
        base_size = np.random.uniform(2.0, 8.0)
        opening_degree = np.random.uniform(0.0, 1.0)

        for layer_idx in range(3):  # 0, 1, 2 (0-based)
            n_petals = petals_per_layer[layer_idx]

            for petal_idx in range(n_petals):
                # === CONTINUOUS LAYER FACTOR ===
                # 0.8 + 0.1 * layer_idx gives [0.8, 0.9, 1.0]
                layer_factor = 0.8 + 0.1 * layer_idx

                # === BASE SPREAD (total width at base) ===
                base_spread = (
                    base_size * 0.3 * layer_factor *
                    (1 + opening_degree * 0.2)
                )

                # === PETAL HEIGHT ===
                petal_height = (
                    base_size * layer_factor *
                    (1.2 - opening_degree * 0.3)
                )

                # === CONTROL POINTS (MIDDLE-WIDE SHAPE) ===
                # CP1: Base left (NARROW - 1/4 of spread)
                cp1_x = -base_spread / 4
                cp1_y = 0.0

                # CP2: Mid-left (WIDEST - 1/2 of spread)
                cp2_x = -base_spread / 2
                cp2_y = petal_height * 0.4

                # CP3: Tip (center, pointed)
                tip_x_offset = base_size * 0.02 * layer_idx * opening_degree
                cp3_x = tip_x_offset
                cp3_y = petal_height

                # CP4: Mid-right (WIDEST - symmetric)
                cp4_x = base_spread / 2
                cp4_y = petal_height * 0.4

                # CP5: Base right (NARROW - symmetric)
                cp5_x = base_spread / 4
                cp5_y = 0.0

                # === ULTRA-THIN THICKNESS ===
                # thickness_base = 0.005 (realistic 0.2-0.5mm)
                # Inner layers thicker, opening reduces thickness
                thickness_base = 0.005
                thickness = (
                    thickness_base * base_size *
                    (1 - layer_idx * 0.1) *  # Inner thicker
                    (1 - opening_degree * 0.3)  # Opening reduces
                )
                extrude_depth = max(0.001, thickness)

                # === ADD REALISTIC NOISE (3%) ===
                noise = 0.03
                cp1_x *= (1 + np.random.normal(0, noise))
                cp2_x *= (1 + np.random.normal(0, noise))
                cp2_y *= (1 + np.random.normal(0, noise))
                cp3_x += np.random.normal(0, noise * base_size * 0.05)
                cp3_y *= (1 + np.random.normal(0, noise))
                cp4_x *= (1 + np.random.normal(0, noise))
                cp4_y *= (1 + np.random.normal(0, noise))
                cp5_x *= (1 + np.random.normal(0, noise))
                extrude_depth *= (1 + np.random.normal(0, noise))
                extrude_depth = max(0.001, extrude_depth)

                data.append({
                    # Features
                    'base_size': round(base_size, 6),
                    'layer_index': layer_idx,
                    'petal_index': petal_idx,
                    'opening_degree': round(opening_degree, 6),

                    # Targets
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
                    'extrude_depth': round(extrude_depth, 6),
                })

    return pd.DataFrame(data)


def main():
    """Generate petal spline v2 dataset."""
    print("=" * 60)
    print("Petal Spline V2 Dataset Generator")
    print("Middle-Wide Shape + Ultra-Thin Thickness")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating petal spline v2 dataset...")
    df = generate_petal_spline_v2(n_samples=500)

    output_path = output_dir / "petal_spline_v2.csv"
    df.to_csv(output_path, index=False)

    print(f"\nGenerated {len(df)} samples")
    print(f"Output: {output_path}")

    # Statistics
    print("\nDataset Statistics:")
    print(f"  base_size: {df['base_size'].min():.2f} - {df['base_size'].max():.2f}")
    print(f"  petal_height (cp3_y): {df['cp3_y'].min():.4f} - {df['cp3_y'].max():.4f}")
    print(f"  mid_width (cp2_x to cp4_x): {(df['cp4_x'] - df['cp2_x']).min():.4f} - {(df['cp4_x'] - df['cp2_x']).max():.4f}")
    print(f"  base_width (cp5_x - cp1_x): {(df['cp5_x'] - df['cp1_x']).min():.4f} - {(df['cp5_x'] - df['cp1_x']).max():.4f}")
    print(f"  extrude_depth: {df['extrude_depth'].min():.6f} - {df['extrude_depth'].max():.6f}")

    # Verify middle-wide shape
    mid_width = df['cp4_x'] - df['cp2_x']
    base_width = df['cp5_x'] - df['cp1_x']
    ratio = mid_width / base_width
    print(f"\nMiddle-Wide Ratio (mid/base): {ratio.mean():.2f} (should be ~2.0)")

    # Verify ultra-thin
    thickness_to_height = df['extrude_depth'] / df['cp3_y']
    print(f"Thickness/Height Ratio: {thickness_to_height.mean():.4f} (ultra-thin)")

    print("\nSample data (first 3 rows):")
    print(df.head(3).to_string())

    print("\n" + "=" * 60)
    print("Petal Spline V2 Dataset Generation Complete!")
    print("=" * 60)

    print("\nShape Description:")
    print("  Base (narrow) → Middle (WIDEST) → Tip (pointed)")
    print("  CP1/CP5: ±1/4 base_spread (narrow)")
    print("  CP2/CP4: ±1/2 base_spread (widest)")
    print("  CP3: tip center")


if __name__ == "__main__":
    main()
