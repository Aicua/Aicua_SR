#!/usr/bin/env python3
"""
Generate bone rigging v7 dataset with vertical spine structure.

3 Connected Bones - Vertical Spine:
- bone_base (0% → 45%)      - vertical spine lower
- bone_mid (45% → 78%)      - vertical spine upper (connected to bone_base tail)
- bone_tip (78% → 100%)     - vertical tip (connected to bone_mid tail)

Key advantage: Connected spine creates continuous deformation chain.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_bone_rigging_v7(n_samples: int = 3000) -> pd.DataFrame:
    """
    Generate dataset for vertical spine bone rigging (3 bones).

    Features:
        - base_size: Overall petal size
        - layer_index: 0=inner, 1=middle, 2=outer
        - petal_index: Position within layer
        - opening_degree: 0.0=bud, 1.0=fully open
        - deformation_type: 0=straight, 1=s_curve, 2=c_curve, 3=wave
        - intensity: Deformation intensity (0.0-1.0)

    Targets (12 positions + 18 rotations = 30 total):
        - bone_base: start_x, start_y, end_x, end_y
        - bone_mid: start_x, start_y, end_x, end_y
        - bone_tip: start_x, start_y, end_x, end_y
    """
    np.random.seed(42)

    data = []

    for _ in range(n_samples):
        # Generate random features
        base_size = np.random.uniform(1.0, 4.0)
        layer_index = np.random.randint(0, 3)
        petal_index = np.random.randint(0, 13)
        opening_degree = np.random.uniform(0.0, 1.0)
        deformation_type = np.random.randint(0, 4)
        intensity = np.random.uniform(0.0, 1.0)

        # Calculate petal dimensions
        layer_factor = 0.8 + 0.1 * layer_index
        petal_height = base_size * layer_factor * (1.2 - opening_degree * 0.3)
        base_spread = base_size * 0.30 * layer_factor * (1 + opening_degree * 0.2)

        # Width at 62% height (widest point)
        width_at_62 = base_spread * 1.6

        # Height positions for vertical bones
        h_45 = petal_height * 0.45   # End of bone_base
        h_78 = petal_height * 0.78   # End of bone_mid
        h_100 = petal_height         # End of bone_tip

        # Initialize x-offsets for deformation
        offset_base = 0.0  # At 45%
        offset_mid = 0.0   # At 78%
        offset_tip = 0.0   # At 100%
        curl_left = 0.0    # How much left edge curls in
        curl_right = 0.0   # How much right edge curls in

        if deformation_type == 0:  # Straight
            pass

        elif deformation_type == 1:  # S-curve (vertical wave)
            offset_base = base_spread * 0.2 * intensity
            offset_mid = -base_spread * 0.15 * intensity
            offset_tip = base_spread * 0.1 * intensity
            # Edges curl slightly
            curl_left = width_at_62 * 0.1 * intensity
            curl_right = width_at_62 * 0.1 * intensity

        elif deformation_type == 2:  # C-curve (curl inward)
            offset_base = -base_spread * 0.05 * intensity
            offset_mid = -base_spread * 0.15 * intensity
            offset_tip = -base_spread * 0.1 * intensity
            # Edges curl inward significantly
            curl_left = width_at_62 * 0.35 * intensity
            curl_right = width_at_62 * 0.35 * intensity

        elif deformation_type == 3:  # Wave
            offset_base = base_spread * 0.15 * np.sin(0.45 * np.pi * 2) * intensity
            offset_mid = base_spread * 0.15 * np.sin(0.78 * np.pi * 2) * intensity
            offset_tip = base_spread * 0.15 * np.sin(1.0 * np.pi * 2) * intensity
            # Edges wave
            curl_left = width_at_62 * 0.2 * np.sin(0.62 * np.pi * 3) * intensity
            curl_right = width_at_62 * 0.2 * np.sin(0.62 * np.pi * 3 + np.pi) * intensity

        # Add small noise (2%)
        noise = lambda: np.random.uniform(-0.02, 0.02) * petal_height

        # Bone positions
        # bone_base: 0% → 45%
        bone_base_start_x = 0.0
        bone_base_start_y = 0.0
        bone_base_end_x = offset_base + noise()
        bone_base_end_y = h_45

        # bone_mid: 45% → 78% (connected - head at bone_base tail, tail at bone_tip head)
        bone_mid_start_x = offset_base + noise()
        bone_mid_start_y = h_45
        bone_mid_end_x = offset_mid
        bone_mid_end_y = h_78

        # bone_tip: 78% → 100%
        bone_tip_start_x = offset_mid
        bone_tip_start_y = h_78
        bone_tip_end_x = offset_tip + noise()
        bone_tip_end_y = h_100

        # Interpolate x position at 62% between base and mid



        # Calculate rotations for head and tail modes
        # head mode: head fixed, tail rotates around head
        # tail mode: tail fixed, head rotates around tail

        # Initialize rotations (in degrees)
        rot_noise = lambda: np.random.uniform(-2, 2)  # Small noise

        # Default rotations (straight)
        rotations = {
            'bone_base_head': [0, 0, 0],
            'bone_base_tail': [0, 0, 0],
            'bone_mid_head': [0, 0, 0],
            'bone_mid_tail': [0, 0, 0],
            'bone_tip_head': [0, 0, 0],
            'bone_tip_tail': [0, 0, 0],
        }

        if deformation_type == 1:  # S-curve
            # Vertical S: base right, mid left, tip right
            rotations['bone_base_head'] = [15 * intensity, 0, 0]
            rotations['bone_mid_head'] = [-25 * intensity, 0, 0]
            rotations['bone_tip_head'] = [15 * intensity, 0, 0]
            # Slight edge curl

        elif deformation_type == 2:  # C-curve (cup shape)
            # All curve forward/inward
            rotations['bone_base_head'] = [0, 20 * intensity, 0]
            rotations['bone_mid_head'] = [0, 30 * intensity, 0]
            rotations['bone_tip_head'] = [0, 25 * intensity, 0]
            # Edges curl inward significantly

        elif deformation_type == 3:  # Wave
            # Oscillating pattern
            rotations['bone_base_head'] = [10 * np.sin(0.45 * np.pi * 2) * intensity, 0, 0]
            rotations['bone_mid_head'] = [10 * np.sin(0.78 * np.pi * 2) * intensity, 0, 0]
            rotations['bone_tip_head'] = [10 * np.sin(1.0 * np.pi * 2) * intensity, 0, 0]

        row = {
            # Features
            'base_size': base_size,
            'layer_index': layer_index,
            'petal_index': petal_index,
            'opening_degree': opening_degree,
            'deformation_type': deformation_type,
            'intensity': intensity,
            # Targets - bone_base positions
            'bone_base_start_x': bone_base_start_x,
            'bone_base_start_y': bone_base_start_y,
            'bone_base_end_x': bone_base_end_x,
            'bone_base_end_y': bone_base_end_y,
            # Targets - bone_mid positions
            'bone_mid_start_x': bone_mid_start_x,
            'bone_mid_start_y': bone_mid_start_y,
            'bone_mid_end_x': bone_mid_end_x,
            'bone_mid_end_y': bone_mid_end_y,
            # Targets - bone_tip positions
            'bone_tip_start_x': bone_tip_start_x,
            'bone_tip_start_y': bone_tip_start_y,
            'bone_tip_end_x': bone_tip_end_x,
            'bone_tip_end_y': bone_tip_end_y,
            # Targets - bone_base rotations
            'bone_base_head_rx': rotations['bone_base_head'][0] + rot_noise(),
            'bone_base_head_ry': rotations['bone_base_head'][1] + rot_noise(),
            'bone_base_head_rz': rotations['bone_base_head'][2] + rot_noise(),
            'bone_base_tail_rx': rotations['bone_base_tail'][0] + rot_noise(),
            'bone_base_tail_ry': rotations['bone_base_tail'][1] + rot_noise(),
            'bone_base_tail_rz': rotations['bone_base_tail'][2] + rot_noise(),
            # Targets - bone_mid rotations
            'bone_mid_head_rx': rotations['bone_mid_head'][0] + rot_noise(),
            'bone_mid_head_ry': rotations['bone_mid_head'][1] + rot_noise(),
            'bone_mid_head_rz': rotations['bone_mid_head'][2] + rot_noise(),
            'bone_mid_tail_rx': rotations['bone_mid_tail'][0] + rot_noise(),
            'bone_mid_tail_ry': rotations['bone_mid_tail'][1] + rot_noise(),
            'bone_mid_tail_rz': rotations['bone_mid_tail'][2] + rot_noise(),
            # Targets - bone_tip rotations
            'bone_tip_head_rx': rotations['bone_tip_head'][0] + rot_noise(),
            'bone_tip_head_ry': rotations['bone_tip_head'][1] + rot_noise(),
            'bone_tip_head_rz': rotations['bone_tip_head'][2] + rot_noise(),
            'bone_tip_tail_rx': rotations['bone_tip_tail'][0] + rot_noise(),
            'bone_tip_tail_ry': rotations['bone_tip_tail'][1] + rot_noise(),
            'bone_tip_tail_rz': rotations['bone_tip_tail'][2] + rot_noise(),
        }

        data.append(row)

    return pd.DataFrame(data)


def main():
    """Generate and save bone rigging v7 dataset."""
    print("=" * 60)
    print("Generating Bone Rigging V7 Dataset (T-Shape)")
    print("=" * 60)

    # Generate dataset
    n_samples = 3000
    df = generate_bone_rigging_v7(n_samples)

    # Save to CSV
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bone_rigging_v7.csv"

    df.to_csv(output_path, index=False)

    print(f"\nDataset shape: {df.shape}")
    print(f"Features: base_size, layer_index, petal_index, opening_degree, deformation_type, intensity")
    print(f"Targets: 50 (5 bones × [4 coords + 6 head rot + 6 tail rot])")
    print(f"\nBone structure (T-shape):")
    print(f"  - bone_base (0% → 45%)   : vertical spine lower")
    print(f"  - bone_mid (78% → 45%)   : vertical spine upper (reversed)")
    print(f"  - bone_tip (78% → 100%)  : vertical tip")
    print(f"\nRotation modes:")
    print(f"  - head mode: head fixed, tail rotates (rx, ry, rz)")
    print(f"  - tail mode: tail fixed, head rotates (rx, ry, rz)")
    print(f"\nSaved to: {output_path}")

    # Show sample
    print(f"\nSample data:")
    print(df.head(3).to_string())

    # Statistics
    print(f"\nDeformation type distribution:")
    print(df['deformation_type'].value_counts().sort_index())

    print(f"\n✓ Bone rigging v7 dataset generation complete!")


if __name__ == "__main__":
    main()
