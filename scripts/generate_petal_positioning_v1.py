#!/usr/bin/env python3
"""
Generate petal positioning v1 dataset with MOVE + ROTATE approach.

This dataset trains a model to position petals in 3D space using:
- move <obj> <x> <y> <z>
- rotate <obj> <rx> <ry> <rz>

Approach: Cylindrical positioning with full rotation control
- Petals arranged in circular layers (Fibonacci-like: 3, 5, 8, 13 petals)
- Each petal positioned via (x, y, z) + rotation (rx, ry, rz)
- SR learns formulas for circular arrangement + natural variations

Key formulas to discover:
- angle = petal_index * 360 / num_petals
- pos_x = layer_radius * cos(angle)
- pos_y = layer_radius * sin(angle)
- rotate_z = angle + 90 (perpendicular to radius)
- rotate_y = base_tilt * (1 - opening_degree) (cup shape when closed)
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_petal_positioning_v1(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate dataset for petal positioning with MOVE + ROTATE commands.

    Features (6 dimensions):
        - layer_radius: Distance from flower center (3.0 - 10.0)
        - num_petals: Number of petals in this layer (3, 5, 8, 13)
        - petal_index: Index within layer (0 to num_petals-1)
        - opening_degree: 0.0=closed bud, 1.0=fully open
        - base_tilt_angle: Maximum outward tilt in degrees (0-45°)
        - z_variation: Random height variation (-0.5 to 0.5)

    Targets (6 dimensions):
        - pos_x: X position
        - pos_y: Y position
        - pos_z: Z position (height)
        - rotate_x: X-axis rotation (wobble)
        - rotate_y: Y-axis rotation (tilt outward/inward)
        - rotate_z: Z-axis rotation (facing direction)

    Returns:
        DataFrame with features and targets
    """
    np.random.seed(42)

    data = []

    # Petal count options (Fibonacci-like)
    petal_counts = [3, 5, 8, 13]

    for _ in range(n_samples):
        # Generate random features
        layer_radius = np.random.uniform(3.0, 10.0)
        num_petals = np.random.choice(petal_counts)
        petal_index = np.random.randint(0, num_petals)
        opening_degree = np.random.uniform(0.0, 1.0)
        base_tilt_angle = np.random.uniform(0, 45)
        z_variation = np.random.uniform(-0.5, 0.5)

        # Calculate positioning
        targets = calculate_petal_position(
            layer_radius=layer_radius,
            num_petals=num_petals,
            petal_index=petal_index,
            opening_degree=opening_degree,
            base_tilt_angle=base_tilt_angle,
            z_variation=z_variation
        )

        # Store features and targets
        data.append({
            # Features
            'layer_radius': layer_radius,
            'num_petals': num_petals,
            'petal_index': petal_index,
            'opening_degree': opening_degree,
            'base_tilt_angle': base_tilt_angle,
            'z_variation': z_variation,
            # Targets
            'pos_x': targets['pos_x'],
            'pos_y': targets['pos_y'],
            'pos_z': targets['pos_z'],
            'rotate_x': targets['rotate_x'],
            'rotate_y': targets['rotate_y'],
            'rotate_z': targets['rotate_z']
        })

    return pd.DataFrame(data)


def calculate_petal_position(layer_radius: float, num_petals: int,
                            petal_index: int, opening_degree: float,
                            base_tilt_angle: float, z_variation: float) -> dict:
    """
    Calculate position and rotation for a single petal in circular arrangement.

    Args:
        layer_radius: Distance from center
        num_petals: Total petals in this layer
        petal_index: Index of this petal (0 to num_petals-1)
        opening_degree: How open the flower is (0=closed, 1=open)
        base_tilt_angle: Maximum outward tilt
        z_variation: Random height offset

    Returns:
        Dictionary with pos_x, pos_y, pos_z, rotate_x, rotate_y, rotate_z
    """
    # 1. Calculate base angle (evenly distributed around circle)
    base_angle_deg = (petal_index * 360.0) / num_petals

    # 2. Add small random rotation noise for natural look (-10° to +10°)
    rotation_noise = np.random.uniform(-10, 10)
    angle_deg = base_angle_deg + rotation_noise
    angle_rad = np.radians(angle_deg)

    # 3. Position: circular arrangement
    pos_x = layer_radius * np.cos(angle_rad)
    pos_y = layer_radius * np.sin(angle_rad)
    pos_z = z_variation  # Random height variation

    # 4. Rotation X: slight wobble for asymmetry (-5° to +5°)
    rotate_x = np.random.uniform(-5, 5)

    # 5. Rotation Y: tilt angle depends on opening_degree
    # Closed flowers (opening_degree=0): petals tilt inward (cup shape)
    # Open flowers (opening_degree=1): petals tilt outward or flat
    # Formula: tilt = base_tilt * (1 - opening_degree)
    # When closed (0): full tilt, when open (1): no tilt
    tilt_factor = 1.0 - opening_degree
    rotate_y = base_tilt_angle * tilt_factor

    # Add small noise to tilt
    rotate_y += np.random.uniform(-3, 3)

    # 6. Rotation Z: face outward (perpendicular to radius)
    # For circular arrangement, each petal should face radially outward
    # This is 90° perpendicular to the radius direction
    rotate_z = angle_deg + 90

    # Normalize rotate_z to [0, 360) range
    rotate_z = rotate_z % 360

    return {
        'pos_x': pos_x,
        'pos_y': pos_y,
        'pos_z': pos_z,
        'rotate_x': rotate_x,
        'rotate_y': rotate_y,
        'rotate_z': rotate_z
    }


def generate_symmetric_test_case(num_petals: int = 3, layer_radius: float = 5.0,
                                 opening_degree: float = 0.8,
                                 base_tilt_angle: float = 15.0) -> pd.DataFrame:
    """
    Generate a perfectly symmetric test case for validation.

    This is useful to verify that petals are arranged symmetrically.
    For 3 petals, angles should be exactly 0°, 120°, 240°.

    Args:
        num_petals: Number of petals (default: 3)
        layer_radius: Radius of the layer (default: 5.0)
        opening_degree: Opening degree (default: 0.8)
        base_tilt_angle: Base tilt angle (default: 15.0)

    Returns:
        DataFrame with symmetric arrangement
    """
    data = []

    for petal_idx in range(num_petals):
        # Exact angles (no noise for perfect symmetry)
        angle_deg = (petal_idx * 360.0) / num_petals
        angle_rad = np.radians(angle_deg)

        # Position
        pos_x = layer_radius * np.cos(angle_rad)
        pos_y = layer_radius * np.sin(angle_rad)
        pos_z = 0.0  # No variation for test

        # Rotation
        rotate_x = 0.0  # No wobble for test
        tilt_factor = 1.0 - opening_degree
        rotate_y = base_tilt_angle * tilt_factor
        rotate_z = (angle_deg + 90) % 360

        data.append({
            # Features
            'layer_radius': layer_radius,
            'num_petals': num_petals,
            'petal_index': petal_idx,
            'opening_degree': opening_degree,
            'base_tilt_angle': base_tilt_angle,
            'z_variation': 0.0,
            # Targets
            'pos_x': pos_x,
            'pos_y': pos_y,
            'pos_z': pos_z,
            'rotate_x': rotate_x,
            'rotate_y': rotate_y,
            'rotate_z': rotate_z
        })

    return pd.DataFrame(data)


def validate_symmetry(df: pd.DataFrame, tolerance: float = 0.01) -> bool:
    """
    Validate that petals in a layer are symmetrically arranged.

    For a single layer with N petals:
    - All should be at same distance from origin
    - Angles should be evenly distributed

    Args:
        df: DataFrame with petal positions
        tolerance: Acceptable deviation

    Returns:
        True if symmetric, False otherwise
    """
    # Calculate distances from origin
    distances = np.sqrt(df['pos_x']**2 + df['pos_y']**2)

    # Check if all distances are equal (within tolerance)
    mean_distance = distances.mean()
    max_deviation = np.abs(distances - mean_distance).max()

    is_symmetric = max_deviation < tolerance

    print(f"\nSymmetry Validation:")
    print(f"  Mean distance from center: {mean_distance:.4f}")
    print(f"  Max deviation: {max_deviation:.6f}")
    print(f"  Symmetric: {'✓ YES' if is_symmetric else '✗ NO'}")

    # Calculate angles
    angles = np.degrees(np.arctan2(df['pos_y'], df['pos_x']))
    angles = (angles + 360) % 360  # Normalize to [0, 360)
    angles = angles.sort_values().values

    print(f"\nPetal angles:")
    for idx, angle in enumerate(angles):
        print(f"  Petal {idx}: {angle:.2f}°")

    # Check angle differences
    if len(angles) > 1:
        expected_diff = 360.0 / len(angles)
        angle_diffs = np.diff(np.append(angles, angles[0] + 360))
        print(f"\nAngle differences (expected: {expected_diff:.2f}°):")
        for idx, diff in enumerate(angle_diffs):
            print(f"  {idx} → {(idx+1)%len(angles)}: {diff:.2f}°")

    return is_symmetric


if __name__ == '__main__':
    print("=" * 70)
    print("Petal Positioning V1 Dataset Generator")
    print("Approach: MOVE + ROTATE (Cylindrical with Full Rotation Control)")
    print("=" * 70)

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'data' / 'generated'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate symmetric test case (3 petals)
    print("\n[1/3] Generating symmetric test case (3 petals)...")
    test_df = generate_symmetric_test_case(num_petals=3, layer_radius=5.0)
    test_file = output_dir / 'petal_positioning_v1_test.csv'
    test_df.to_csv(test_file, index=False)
    print(f"  Saved: {test_file}")
    print(f"  Samples: {len(test_df)}")

    # Validate symmetry
    validate_symmetry(test_df)

    # 2. Generate main training dataset
    print(f"\n[2/3] Generating main training dataset...")
    n_samples = 5000
    train_df = generate_petal_positioning_v1(n_samples=n_samples)
    train_file = output_dir / 'petal_positioning_v1.csv'
    train_df.to_csv(train_file, index=False)
    print(f"  Saved: {train_file}")
    print(f"  Samples: {len(train_df)}")

    # 3. Show statistics
    print(f"\n[3/3] Dataset Statistics:")
    print(f"\nFeatures:")
    print(f"  layer_radius: {train_df['layer_radius'].min():.2f} - {train_df['layer_radius'].max():.2f}")
    print(f"  num_petals: {train_df['num_petals'].unique()}")
    print(f"  petal_index: 0 - {train_df['petal_index'].max()}")
    print(f"  opening_degree: {train_df['opening_degree'].min():.2f} - {train_df['opening_degree'].max():.2f}")
    print(f"  base_tilt_angle: {train_df['base_tilt_angle'].min():.2f}° - {train_df['base_tilt_angle'].max():.2f}°")
    print(f"  z_variation: {train_df['z_variation'].min():.2f} - {train_df['z_variation'].max():.2f}")

    print(f"\nTargets:")
    print(f"  pos_x: {train_df['pos_x'].min():.2f} - {train_df['pos_x'].max():.2f}")
    print(f"  pos_y: {train_df['pos_y'].min():.2f} - {train_df['pos_y'].max():.2f}")
    print(f"  pos_z: {train_df['pos_z'].min():.2f} - {train_df['pos_z'].max():.2f}")
    print(f"  rotate_x: {train_df['rotate_x'].min():.2f}° - {train_df['rotate_x'].max():.2f}°")
    print(f"  rotate_y: {train_df['rotate_y'].min():.2f}° - {train_df['rotate_y'].max():.2f}°")
    print(f"  rotate_z: {train_df['rotate_z'].min():.2f}° - {train_df['rotate_z'].max():.2f}°")

    # Show sample data
    print(f"\n{'='*70}")
    print("Sample data (first 5 rows of test case):")
    print(f"{'='*70}")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.precision', 2)
    print(test_df.head())

    print(f"\n{'='*70}")
    print("✓ Dataset generation complete!")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"  1. Train SR model: python scripts/train_sr_positioning_v1.py")
    print(f"  2. Generate CLI commands: python scripts/generate_flower_cli_v1.py")
    print(f"  3. Visualize arrangement: python scripts/visualize_positioning.py")
