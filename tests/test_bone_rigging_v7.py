#!/usr/bin/env python3
"""
Tests for bone rigging v7 (T-shape 5 independent bones).
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_bone_rigging_v7 import generate_bone_rigging_v7


class TestBoneRiggingV7Dataset:
    """Test bone rigging v7 dataset generation."""

    def test_generates_dataframe(self):
        """Test that generate_bone_rigging_v7 returns a DataFrame."""
        df = generate_bone_rigging_v7(n_samples=10)
        assert df is not None
        assert len(df) > 0

    def test_has_required_features(self):
        """Test that dataset has all required feature columns."""
        df = generate_bone_rigging_v7(n_samples=10)
        features = ['base_size', 'layer_index', 'petal_index',
                    'opening_degree', 'deformation_type', 'intensity']
        for col in features:
            assert col in df.columns, f"Missing feature: {col}"

    def test_has_required_targets(self):
        """Test that dataset has all 20 target columns (5 bones × 4 coords)."""
        df = generate_bone_rigging_v7(n_samples=10)

        # T-shape: 3 vertical + 2 horizontal
        bone_names = ['bone_base', 'bone_mid', 'bone_tip', 'bone_left', 'bone_right']
        coord_names = ['start_x', 'start_y', 'end_x', 'end_y']

        for bone in bone_names:
            for coord in coord_names:
                col = f"{bone}_{coord}"
                assert col in df.columns, f"Missing target: {col}"

    def test_total_columns(self):
        """Test that dataset has correct total columns (6 features + 20 targets)."""
        df = generate_bone_rigging_v7(n_samples=10)
        # 6 features + 20 targets = 26 columns
        assert len(df.columns) == 26

    def test_deformation_types(self):
        """Test that all deformation types (0-3) are present."""
        df = generate_bone_rigging_v7(n_samples=100)
        deformation_types = df['deformation_type'].unique()
        assert set(deformation_types) == {0, 1, 2, 3}

    def test_bone_base_starts_at_origin(self):
        """Test that bone_base always starts at origin (0, 0)."""
        df = generate_bone_rigging_v7(n_samples=50)
        assert (df['bone_base_start_x'] == 0.0).all()
        assert (df['bone_base_start_y'] == 0.0).all()

    def test_vertical_bone_positions(self):
        """Test that vertical bone positions are correct (bone_mid is reversed)."""
        df = generate_bone_rigging_v7(n_samples=50)

        # For each sample, check bone positions
        for idx in range(len(df)):
            base_end_y = df.iloc[idx]['bone_base_end_y']
            mid_start_y = df.iloc[idx]['bone_mid_start_y']
            mid_end_y = df.iloc[idx]['bone_mid_end_y']
            tip_end_y = df.iloc[idx]['bone_tip_end_y']

            assert base_end_y > 0, "bone_base_end_y should be positive"
            # bone_mid is reversed: starts at 78%, ends at 45%
            assert mid_start_y > mid_end_y, "bone_mid should point downward (reversed)"
            assert tip_end_y > mid_start_y, "bone_tip should be highest"

    def test_horizontal_bones_at_same_height(self):
        """Test that bone_left and bone_right are at 62% height."""
        df = generate_bone_rigging_v7(n_samples=50)

        for idx in range(len(df)):
            left_y = df.iloc[idx]['bone_left_start_y']
            right_y = df.iloc[idx]['bone_right_start_y']
            left_end_y = df.iloc[idx]['bone_left_end_y']
            right_end_y = df.iloc[idx]['bone_right_end_y']

            # Start and end y should be the same (horizontal bone)
            assert abs(left_y - left_end_y) < 0.001, "bone_left should be horizontal"
            assert abs(right_y - right_end_y) < 0.001, "bone_right should be horizontal"
            # Left and right should be at same height
            assert abs(left_y - right_y) < 0.001, "bone_left and bone_right should be at same height"

    def test_horizontal_bones_opposite_directions(self):
        """Test that bone_left goes left and bone_right goes right."""
        df = generate_bone_rigging_v7(n_samples=50)

        for idx in range(len(df)):
            left_end_x = df.iloc[idx]['bone_left_end_x']
            right_end_x = df.iloc[idx]['bone_right_end_x']

            # bone_left should have negative x (go left)
            assert left_end_x < 0, f"bone_left should go left, got {left_end_x}"
            # bone_right should have positive x (go right)
            assert right_end_x > 0, f"bone_right should go right, got {right_end_x}"

    def test_straight_deformation_has_zero_spine_offset(self):
        """Test that straight deformation (type 0) has minimal x offsets on spine."""
        df = generate_bone_rigging_v7(n_samples=100)
        straight = df[df['deformation_type'] == 0]

        # For straight deformation, spine x offsets should be ~0 (only noise)
        spine_x_cols = ['bone_base_end_x', 'bone_mid_end_x', 'bone_tip_end_x']

        for col in spine_x_cols:
            mean_abs = straight[col].abs().mean()
            # Should be very small (just noise ~2%)
            assert mean_abs < 0.1, f"Straight deformation {col} should be near 0, got {mean_abs}"

    def test_c_curve_deformation_curls_edges(self):
        """Test that c_curve deformation curls edges inward."""
        df = generate_bone_rigging_v7(n_samples=100)
        c_curve = df[(df['deformation_type'] == 2) & (df['intensity'] > 0.5)]

        if len(c_curve) > 0:
            # C-curve: edges should curl inward
            # bone_left goes less left (closer to 0)
            # bone_right goes less right (closer to 0)
            straight = df[(df['deformation_type'] == 0)]

            if len(straight) > 0:
                avg_left_c = c_curve['bone_left_end_x'].abs().mean()
                avg_left_s = straight['bone_left_end_x'].abs().mean()
                avg_right_c = c_curve['bone_right_end_x'].abs().mean()
                avg_right_s = straight['bone_right_end_x'].abs().mean()

                # C-curve should have smaller absolute x (curled in)
                assert avg_left_c < avg_left_s, "C-curve left edge should curl inward"
                assert avg_right_c < avg_right_s, "C-curve right edge should curl inward"

    def test_intensity_affects_deformation(self):
        """Test that higher intensity produces larger deformations."""
        df = generate_bone_rigging_v7(n_samples=200)

        # Compare s_curve with low vs high intensity
        low_intensity = df[(df['deformation_type'] == 1) & (df['intensity'] < 0.3)]
        high_intensity = df[(df['deformation_type'] == 1) & (df['intensity'] > 0.7)]

        if len(low_intensity) > 0 and len(high_intensity) > 0:
            # High intensity should have larger absolute x offsets on spine
            low_abs = low_intensity['bone_base_end_x'].abs().mean()
            high_abs = high_intensity['bone_base_end_x'].abs().mean()

            assert high_abs > low_abs, "Higher intensity should produce larger deformations"

    def test_layer_affects_petal_height(self):
        """Test that outer layers have taller petals."""
        df = generate_bone_rigging_v7(n_samples=100)

        # Group by layer and check average tip height
        avg_height_by_layer = df.groupby('layer_index')['bone_tip_end_y'].mean()

        # Layer 2 (outer) should be taller than layer 0 (inner)
        assert avg_height_by_layer[2] > avg_height_by_layer[0], \
            "Outer layer petals should be taller than inner layer"


class TestBoneRiggingV7CLI:
    """Test CLI generator for T-shape bones."""

    def test_cli_generator_import(self):
        """Test that CLI generator can be imported."""
        from scripts.generate_rose_cli_v2 import RoseCLIGeneratorV2
        generator = RoseCLIGeneratorV2()
        assert generator is not None

    def test_generate_single_petal(self):
        """Test generating CLI for a single petal."""
        from scripts.generate_rose_cli_v2 import RoseCLIGeneratorV2

        generator = RoseCLIGeneratorV2()
        petal_data = generator.generate_petal(
            layer_idx=1, petal_idx=0, base_size=2.0,
            opening_degree=0.8, deformation_type=0, intensity=0.5
        )

        assert 'geometry' in petal_data
        assert 'rigging' in petal_data
        assert len(petal_data['rigging']) > 0

    def test_cli_has_no_parent_bone(self):
        """Test that generated CLI has NO parent_bone commands."""
        from scripts.generate_rose_cli_v2 import RoseCLIGeneratorV2

        generator = RoseCLIGeneratorV2()
        cli = generator.generate_rose(base_size=2.0, opening_degree=0.8, n_layers=1)

        # Should NOT contain parent_bone
        assert 'parent_bone' not in cli, "CLI should NOT have parent_bone commands"

    def test_cli_has_5_bones_per_petal(self):
        """Test that CLI generates 5 bones per petal (T-shape)."""
        from scripts.generate_rose_cli_v2 import RoseCLIGeneratorV2

        generator = RoseCLIGeneratorV2()
        petal_data = generator.generate_petal(
            layer_idx=1, petal_idx=0, base_size=2.0,
            opening_degree=0.8, deformation_type=0, intensity=0.5
        )

        rigging = '\n'.join(petal_data['rigging'])

        # Check all 5 bone names are present (T-shape)
        bone_names = ['bone_base', 'bone_mid', 'bone_tip', 'bone_left', 'bone_right']
        for bone_name in bone_names:
            assert bone_name in rigging, f"Missing bone: {bone_name}"

    def test_generate_full_rose(self):
        """Test generating CLI for full rose."""
        from scripts.generate_rose_cli_v2 import RoseCLIGeneratorV2

        generator = RoseCLIGeneratorV2()
        cli = generator.generate_rose(
            base_size=2.0, opening_degree=0.8, n_layers=3,
            deformation_type=1, intensity=0.7
        )

        # Check basic structure
        assert '2d;' in cli
        assert 'exit;' in cli
        assert 'create_armature' in cli
        assert 'add_bone' in cli
        assert 'finalize_bones' in cli
        assert 'bind_armature' in cli

    def test_deformation_types_affect_positions(self):
        """Test that different deformation types produce different bone positions."""
        from scripts.generate_rose_cli_v2 import RoseCLIGeneratorV2

        generator = RoseCLIGeneratorV2()

        cli_straight = generator.generate_rose(
            base_size=2.0, opening_degree=0.8, n_layers=1,
            deformation_type=0, intensity=0.8
        )

        cli_s_curve = generator.generate_rose(
            base_size=2.0, opening_degree=0.8, n_layers=1,
            deformation_type=1, intensity=0.8
        )

        # CLIs should be different
        assert cli_straight != cli_s_curve, "Different deformation types should produce different CLIs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
