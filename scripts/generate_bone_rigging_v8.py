#!/usr/bin/env python3
"""
Generate bone rigging v8 dataset with 3-bone connected spine.

3 Vertical Bones - Connected Spine:
- bone_base (0% → 33%)      - lower spine
- bone_mid (33% → 67%)      - middle spine (connected to bone_base tail)
- bone_tip (67% → 100%)     - upper spine (connected to bone_mid tail)

Key advantage: Simple 3-bone spine with rotate_bone for cup/wave deformations.
Cup shape controlled by rx rotations (forward curve for bud).
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_bone_rigging_v8(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate dataset for 3-bone connected spine rigging.

    Features:
        - base_size: Overall petal size
        - layer_index: 0=inner, 1=middle, 2=outer
        - petal_index: Position within layer
        - opening_degree: 0.0=bud (cup), 1.0=fully open (flat)
        - deformation_type: 0=cup, 1=s_curve, 2=wave, 3=reverse_cup
        - intensity: Deformation intensity (0.0-1.0)

    Targets (30 values = 3 bones × [4 coords + 6 rotations]):
        Positions (12 values):
            - bone_base: start_x, start_y, end_x, end_y
            - bone_mid: start_x, start_y, end_x, end_y
            - bone_tip: start_x, start_y, end_x, end_y

        Rotations (18 values = 3 bones × 2 modes × 3 angles):
            - bone_base_head_rx, ry, rz
            - bone_base_tail_rx, ry, rz
            - bone_mid_head_rx, ry, rz
            - bone_mid_tail_rx, ry, rz
            - bone_tip_head_rx, ry, rz
            - bone_tip_tail_rx, ry, rz
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

        # Height positions for 3 vertical bones (equal thirds)
        h_33 = petal_height * 0.33   # End of bone_base
        h_67 = petal_height * 0.67   # End of bone_mid
        h_100 = petal_height          # End of bone_tip

        # Bone positions (straight spine, no x-offsets)
        bone_base_start_x = 0.0
        bone_base_start_y = 0.0
        bone_base_end_x = 0.0
        bone_base_end_y = h_33

        bone_mid_start_x = 0.0
        bone_mid_start_y = h_33  # Connected to bone_base tail
        bone_mid_end_x = 0.0
        bone_mid_end_y = h_67

        bone_tip_start_x = 0.0
        bone_tip_start_y = h_67  # Connected to bone_mid tail
        bone_tip_end_x = 0.0
        bone_tip_end_y = h_100

        # Calculate rotations for head and tail modes
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

        # Apply deformation based on type and opening_degree
        if deformation_type == 0:  # Cup shape (forward curve)
            # Cup intensity inversely proportional to opening_degree
            # Bud (opening_degree ~ 0) = strong cup
            # Open (opening_degree ~ 1) = weak cup
            cup_intensity = (1.0 - opening_degree) * intensity

            # Forward rotation (rx > 0) creates cup shape
            rotations['bone_base_head'] = [20 * cup_intensity + rot_noise(), 0, 0]
            rotations['bone_mid_head'] = [30 * cup_intensity + rot_noise(), 0, 0]
            rotations['bone_tip_head'] = [20 * cup_intensity + rot_noise(), 0, 0]

        elif deformation_type == 1:  # S-curve
            rotations['bone_base_head'] = [15 * intensity + rot_noise(), 0, 0]
            rotations['bone_mid_head'] = [-25 * intensity + rot_noise(), 0, 0]
            rotations['bone_tip_head'] = [15 * intensity + rot_noise(), 0, 0]

        elif deformation_type == 2:  # Wave
            rotations['bone_base_head'] = [10 * intensity + rot_noise(), 5 * intensity + rot_noise(), 0]
            rotations['bone_mid_head'] = [-15 * intensity + rot_noise(), -10 * intensity + rot_noise(), 0]
            rotations['bone_tip_head'] = [20 * intensity + rot_noise(), 8 * intensity + rot_noise(), 0]

        elif deformation_type == 3:  # Reverse cup (backward curve)
            # Reverse cup proportional to opening_degree
            # Open petal curves backward slightly
            reverse_intensity = opening_degree * intensity
            rotations['bone_base_head'] = [-10 * reverse_intensity + rot_noise(), 0, 0]
            rotations['bone_mid_head'] = [-15 * reverse_intensity + rot_noise(), 0, 0]
            rotations['bone_tip_head'] = [-10 * reverse_intensity + rot_noise(), 0, 0]

        # Build row
        row = {
            # Features
            'base_size': base_size,
            'layer_index': layer_index,
            'petal_index': petal_index,
            'opening_degree': opening_degree,
            'deformation_type': deformation_type,
            'intensity': intensity,

            # Bone positions (12 values)
            'bone_base_start_x': bone_base_start_x,
            'bone_base_start_y': bone_base_start_y,
            'bone_base_end_x': bone_base_end_x,
            'bone_base_end_y': bone_base_end_y,

            'bone_mid_start_x': bone_mid_start_x,
            'bone_mid_start_y': bone_mid_start_y,
            'bone_mid_end_x': bone_mid_end_x,
            'bone_mid_end_y': bone_mid_end_y,

            'bone_tip_start_x': bone_tip_start_x,
            'bone_tip_start_y': bone_tip_start_y,
            'bone_tip_end_x': bone_tip_end_x,
            'bone_tip_end_y': bone_tip_end_y,

            # Bone rotations (18 values)
            'bone_base_head_rx': rotations['bone_base_head'][0],
            'bone_base_head_ry': rotations['bone_base_head'][1],
            'bone_base_head_rz': rotations['bone_base_head'][2],
            'bone_base_tail_rx': rotations['bone_base_tail'][0],
            'bone_base_tail_ry': rotations['bone_base_tail'][1],
            'bone_base_tail_rz': rotations['bone_base_tail'][2],

            'bone_mid_head_rx': rotations['bone_mid_head'][0],
            'bone_mid_head_ry': rotations['bone_mid_head'][1],
            'bone_mid_head_rz': rotations['bone_mid_head'][2],
            'bone_mid_tail_rx': rotations['bone_mid_tail'][0],
            'bone_mid_tail_ry': rotations['bone_mid_tail'][1],
            'bone_mid_tail_rz': rotations['bone_mid_tail'][2],

            'bone_tip_head_rx': rotations['bone_tip_head'][0],
            'bone_tip_head_ry': rotations['bone_tip_head'][1],
            'bone_tip_head_rz': rotations['bone_tip_head'][2],
            'bone_tip_tail_rx': rotations['bone_tip_tail'][0],
            'bone_tip_tail_ry': rotations['bone_tip_tail'][1],
            'bone_tip_tail_rz': rotations['bone_tip_tail'][2],
        }

        data.append(row)

    df = pd.DataFrame(data)
    return df


def main():
    """Generate and save v8 bone rigging dataset."""
    print("Generating v8 bone rigging dataset (3 bones)...")

    df = generate_bone_rigging_v8(n_samples=1000)

    # Save
    output_dir = Path(__file__).parent.parent / "data" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "bone_rigging_v8_dataset.csv"
    df.to_csv(output_path, index=False)

    print(f"✓ Generated {len(df)} samples")
    print(f"✓ Saved to {output_path}")
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFeatures: 6")
    print(f"Targets: 30 (12 positions + 18 rotations)")
    print(f"Total columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
