#!/usr/bin/env python3
"""
Generate petal spline dataset v3 - similar to rose_petals_ok.jpg:
- 15 Control Points
- Base NARROW like a point (5% height)
- Expands quickly from base
- WIDEST at 60-62% height (not 50%)
- Tapers to rounded tip
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_petal_spline(n_samples: int = 3000) -> pd.DataFrame:
    """
    Generate petal spline dataset with 15 control points.

    Shape: Similar to rose_petals_ok.jpg
    - Base (5%): NARROW like a point
    - Lower (25%): expands quickly
    - Mid-low (45%): continues to expand
    - Upper-mid (62%): WIDEST
    - Upper (78%): tapers
    - Near-tip (92%): narrows significantly
    - Tip (100%): rounded

    Control Points:
    - CP1: Base center
    - CP2/CP14: Base sides (5%) - NARROW
    - CP3/CP13: Lower (25%)
    - CP4/CP12: Mid-low (45%)
    - CP5/CP11: Upper-mid (62%) - WIDEST
    - CP6/CP10: Upper (78%)
    - CP7/CP9: Near-tip (92%)
    - CP8: Tip center
    - CP15: Close spline

    Features (2 dimensions):
        - base_size: Overall rose size (2.0 - 8.0)
        - opening_degree: 0.0 (closed) to 1.0 (fully open)

    Targets:
        - cp1_x, cp1_y through cp15_x, cp15_y
        - extrude_depth (ultra-thin)
    """
    data = []

    for _ in range(n_samples):
        base_size = np.random.uniform(2.0, 8.0)
        opening_degree = np.random.uniform(0.0, 1.0)

        # Single layer with 3 petals
        layer_idx = 0
        petal_idx = np.random.randint(0, 3)  # Random pick from 3 petals

        # === LAYER FACTOR ===
        layer_factor = 0.8 + 0.1 * layer_idx

        # === PETAL HEIGHT ===
        petal_height = (
            base_size * layer_factor *
            (1.2 - opening_degree * 0.3)
        )

        # === WIDTH CALCULATIONS - Similar to rose_petals_ok.jpg ===
        base_spread = (
            base_size * 0.30 * layer_factor *
            (1 + opening_degree * 0.2)
        )

        # Base NARROW like a point - BUT with random variation
        # Some petals have wider base, some narrower
        base_width_factor = np.random.uniform(0.3, 0.7)  # Random 30-70%
        base_width = base_spread * base_width_factor

        # Lower - expands quickly + RANDOM bulge
        # CP3/CP13 can bulge wider
        lower_width_factor = np.random.uniform(0.95, 1.15)  # 95-115%
        lower_width = base_spread * lower_width_factor

        # Mid-low - continues to expand
        mid_low_width = base_spread * 1.4  # Expands

        # Upper-mid - WIDEST (60-65% height, not 50%)
        upper_mid_width = base_spread * 1.6  # WIDEST

        # Upper - starts to taper
        upper_width = base_spread * 1.3    # Tapers back

        # Near-tip - narrows significantly
        near_tip_width = base_spread * 0.8  # Narrows

        # Tip x offset (for asymmetry/tilt)
        tip_x_offset = base_size * 0.02 * layer_idx * opening_degree

        # === 13 CONTROL POINTS (similar to rose_petals_ok.jpg) ===

        # CP1: Base center (starting point)
        cp1_x = 0.0
        cp1_y = 0.0

        # CP2: Base left - VERY NARROW (like a point) + RANDOM HEIGHT
        # Height can be from 3-8% (not fixed at 5%)
        base_left_height = np.random.uniform(0.03, 0.08)
        cp2_x = -base_width * 0.5
        cp2_y = petal_height * base_left_height

        # CP3: Lower left - expands quickly
        cp3_x = -lower_width * 0.5
        cp3_y = petal_height * 0.25  # 25% height

        # CP4: Mid-low left - continues to expand
        cp4_x = -mid_low_width * 0.5
        cp4_y = petal_height * 0.45  # 45% height

        # CP5: Upper-mid left - WIDEST (60-65%)
        cp5_x = -upper_mid_width * 0.5
        cp5_y = petal_height * 0.62  # 62% height

        # CP6: Upper left - tapers
        cp6_x = -upper_width * 0.5
        cp6_y = petal_height * 0.78  # 78% height

        # CP7: Near-tip left - narrows significantly
        cp7_x = -near_tip_width * 0.5
        cp7_y = petal_height * 0.92  # 92% height

        # CP8: Tip center (rounded)
        cp8_x = tip_x_offset
        cp8_y = petal_height

        # CP9: Near-tip right - symmetric
        cp9_x = near_tip_width * 0.5
        cp9_y = petal_height * 0.92

        # CP10: Upper right - symmetric
        cp10_x = upper_width * 0.5
        cp10_y = petal_height * 0.78

        # CP11: Upper-mid right - WIDEST symmetric
        cp11_x = upper_mid_width * 0.5
        cp11_y = petal_height * 0.62

        # CP12: Mid-low right - symmetric
        cp12_x = mid_low_width * 0.5
        cp12_y = petal_height * 0.45

        # CP13: Lower right - symmetric
        cp13_x = lower_width * 0.5
        cp13_y = petal_height * 0.25

        # CP14: Base right - NARROW + RANDOM HEIGHT (symmetric with CP2)
        cp14_x = base_width * 0.5
        cp14_y = petal_height * base_left_height  # Same height as CP2

        # CP15: Back to base center (closed spline)
        cp15_x = 0.0
        cp15_y = 0.0

        # === ULTRA-THIN THICKNESS ===
        thickness_base = 0.0005  # Ultra-thin like real rose petals (0.001-0.0015)
        thickness = (
            thickness_base * base_size *
            (1 - layer_idx * 0.1) *
            (1 - opening_degree * 0.3)
        )
        extrude_depth = max(0.001, thickness)

        # === ADD REALISTIC NOISE (3%) ===
        noise = 0.03

        # Apply noise (skip CP1 and CP15 - fixed at origin)
        cp2_x *= (1 + np.random.normal(0, noise))
        cp2_y *= (1 + np.random.normal(0, noise))
        cp3_x *= (1 + np.random.normal(0, noise))
        cp3_y *= (1 + np.random.normal(0, noise))
        cp4_x *= (1 + np.random.normal(0, noise))
        cp4_y *= (1 + np.random.normal(0, noise))
        cp5_x *= (1 + np.random.normal(0, noise))
        cp5_y *= (1 + np.random.normal(0, noise))
        cp6_x *= (1 + np.random.normal(0, noise))
        cp6_y *= (1 + np.random.normal(0, noise))
        cp7_x *= (1 + np.random.normal(0, noise))
        cp7_y *= (1 + np.random.normal(0, noise))
        cp8_x += np.random.normal(0, noise * base_size * 0.05)
        cp8_y *= (1 + np.random.normal(0, noise))
        cp9_x *= (1 + np.random.normal(0, noise))
        cp9_y *= (1 + np.random.normal(0, noise))
        cp10_x *= (1 + np.random.normal(0, noise))
        cp10_y *= (1 + np.random.normal(0, noise))
        cp11_x *= (1 + np.random.normal(0, noise))
        cp11_y *= (1 + np.random.normal(0, noise))
        cp12_x *= (1 + np.random.normal(0, noise))
        cp12_y *= (1 + np.random.normal(0, noise))
        cp13_x *= (1 + np.random.normal(0, noise))
        cp13_y *= (1 + np.random.normal(0, noise))
        cp14_x *= (1 + np.random.normal(0, noise))
        cp14_y *= (1 + np.random.normal(0, noise))
        extrude_depth *= (1 + np.random.normal(0, noise))
        extrude_depth = max(0.001, extrude_depth)

        data.append({
            # Features
            'base_size': round(base_size, 6),
            'opening_degree': round(opening_degree, 6),

            # Targets - 15 Control Points
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
            'cp9_x': round(cp9_x, 6),
            'cp9_y': round(cp9_y, 6),
            'cp10_x': round(cp10_x, 6),
            'cp10_y': round(cp10_y, 6),
            'cp11_x': round(cp11_x, 6),
            'cp11_y': round(cp11_y, 6),
            'cp12_x': round(cp12_x, 6),
            'cp12_y': round(cp12_y, 6),
            'cp13_x': round(cp13_x, 6),
            'cp13_y': round(cp13_y, 6),
            'cp14_x': round(cp14_x, 6),
            'cp14_y': round(cp14_y, 6),
            'cp15_x': round(cp15_x, 6),
            'cp15_y': round(cp15_y, 6),
            'extrude_depth': round(extrude_depth, 6),
        })

    return pd.DataFrame(data)


def main():
    """Generate petal spline v3 dataset."""
    print("=" * 60)
    print("Petal Spline V3 Dataset Generator")
    print("15 Control Points - similar to rose_petals_ok.jpg")
    print("Base NARROW + WIDEST at 60-62%")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating petal spline dataset...")
    df = generate_petal_spline(n_samples=3000)

    output_path = output_dir / "petal_spline.csv"
    df.to_csv(output_path, index=False)

    print(f"\nGenerated {len(df)} samples")
    print(f"Output: {output_path}")

    # Statistics
    print("\nDataset Statistics:")
    print(f"  base_size: {df['base_size'].min():.2f} - {df['base_size'].max():.2f}")
    print(f"  petal_height (cp8_y): {df['cp8_y'].min():.4f} - {df['cp8_y'].max():.4f}")
    print(f"  extrude_depth: {df['extrude_depth'].min():.6f} - {df['extrude_depth'].max():.6f}")

    # Verify shape
    print("\nShape Characteristics (similar to rose_petals_ok.jpg):")

    # Base width (5%, CP2 to CP14)
    base_width = df['cp14_x'] - df['cp2_x']
    print(f"  Base width (5%):       {base_width.mean():.4f} (NARROW)")

    # Lower width (25%, CP3 to CP13)
    lower_width = df['cp13_x'] - df['cp3_x']
    print(f"  Lower width (25%):     {lower_width.mean():.4f} - expands")

    # Mid-low width (45%, CP4 to CP12)
    mid_low_width = df['cp12_x'] - df['cp4_x']
    print(f"  Mid-low width (45%):   {mid_low_width.mean():.4f} - expands")

    # Upper-mid width (62%, CP5 to CP11) - WIDEST
    upper_mid_width = df['cp11_x'] - df['cp5_x']
    print(f"  Upper-mid width (62%): {upper_mid_width.mean():.4f} - WIDEST")

    # Upper width (78%, CP6 to CP10)
    upper_width = df['cp10_x'] - df['cp6_x']
    print(f"  Upper width (78%):     {upper_width.mean():.4f} - tapers")

    # Near-tip width (92%, CP7 to CP9)
    near_tip_width = df['cp9_x'] - df['cp7_x']
    print(f"  Near-tip width (92%):  {near_tip_width.mean():.4f} - tapers")

    # Width ratios
    print("\nWidth Ratios:")
    upper_mid_to_base = upper_mid_width / base_width
    upper_mid_to_lower = upper_mid_width / lower_width
    upper_mid_to_mid_low = upper_mid_width / mid_low_width
    print(f"  Upper-mid/Base:    {upper_mid_to_base.mean():.2f} (widest compared to base)")
    print(f"  Upper-mid/Lower:   {upper_mid_to_lower.mean():.2f}")
    print(f"  Upper-mid/Mid-low: {upper_mid_to_mid_low.mean():.2f}")

    print("\nSample data (first 3 rows):")
    print(df[['base_size', 'layer_index', 'petal_index', 'opening_degree',
              'cp2_x', 'cp5_x', 'cp8_y', 'cp11_x', 'cp14_x']].head(3).to_string())

    print("\n" + "=" * 60)
    print("Petal Spline V3 Dataset Generation Complete!")
    print("=" * 60)

    print("\nShape Description (15 Control Points):")
    print("  CP1: Base center")
    print("  CP2/CP14: Base (5%) - NARROW like a point")
    print("  CP3/CP13: Lower (25%) - expands quickly")
    print("  CP4/CP12: Mid-low (45%)")
    print("  CP5/CP11: Upper-mid (62%) - WIDEST")
    print("  CP6/CP10: Upper (78%)")
    print("  CP7/CP9: Near-tip (92%)")
    print("  CP8: Tip")
    print("  CP15: Close")

    print("\nSimilar to rose_petals_ok.jpg:")
    print("  - Base NARROW like a point")
    print("  - Expands quickly")
    print("  - WIDEST at 60-62% (not 50%)")
    print("  - Tapers to rounded tip")


if __name__ == "__main__":
    main()
