#!/usr/bin/env python3
"""
Tests for bone rigging v6 (5 independent bones).
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_bone_rigging_v6 import generate_bone_rigging_v6


class TestBoneRiggingV6Dataset:
    """Test bone rigging v6 dataset generation."""

    def test_generates_dataframe(self):
        """Test that generate_bone_rigging_v6 returns a DataFrame."""
        df = generate_bone_rigging_v6(n_samples=10)
        assert df is not None
        assert len(df) > 0

    def test_has_required_features(self):
        """Test that dataset has all required feature columns."""
        df = generate_bone_rigging_v6(n_samples=10)
        features = ['base_size', 'layer_index', 'petal_index',
                    'opening_degree', 'deformation_type', 'intensity']
        for col in features:
            assert col in df.columns, f"Missing feature: {col}"

    def test_has_required_targets(self):
        """Test that dataset has all 20 target columns (5 bones × 4 coords)."""
        df = generate_bone_rigging_v6(n_samples=10)

        bone_names = ['bone_base', 'bone_mid', 'bone_mid_upper', 'bone_upper', 'bone_tip']
        coord_names = ['start_x', 'start_y', 'end_x', 'end_y']

        for bone in bone_names:
            for coord in coord_names:
                col = f"{bone}_{coord}"
                assert col in df.columns, f"Missing target: {col}"

    def test_total_columns(self):
        """Test that dataset has correct total columns (6 features + 20 targets)."""
        df = generate_bone_rigging_v6(n_samples=10)
        # 6 features + 20 targets = 26 columns
        assert len(df.columns) == 26

    def test_deformation_types(self):
        """Test that all deformation types (0-3) are present."""
        df = generate_bone_rigging_v6(n_samples=100)
        deformation_types = df['deformation_type'].unique()
        assert set(deformation_types) == {0, 1, 2, 3}

    def test_bone_base_starts_at_origin(self):
        """Test that bone_base always starts at origin (0, 0)."""
        df = generate_bone_rigging_v6(n_samples=50)
        assert (df['bone_base_start_x'] == 0.0).all()
        assert (df['bone_base_start_y'] == 0.0).all()

    def test_bone_heights_increase(self):
        """Test that bone end_y positions increase along the spine."""
        df = generate_bone_rigging_v6(n_samples=50)

        # For each sample, check that end_y positions increase
        for idx in range(len(df)):
            base_end_y = df.iloc[idx]['bone_base_end_y']
            mid_end_y = df.iloc[idx]['bone_mid_end_y']
            mid_upper_end_y = df.iloc[idx]['bone_mid_upper_end_y']
            upper_end_y = df.iloc[idx]['bone_upper_end_y']
            tip_end_y = df.iloc[idx]['bone_tip_end_y']

            assert base_end_y > 0, "bone_base_end_y should be positive"
            assert mid_end_y > base_end_y, "bone_mid should be higher than bone_base"
            assert mid_upper_end_y > mid_end_y, "bone_mid_upper should be higher than bone_mid"
            assert upper_end_y > mid_upper_end_y, "bone_upper should be higher than bone_mid_upper"
            assert tip_end_y > upper_end_y, "bone_tip should be highest"

    def test_straight_deformation_has_zero_x_offset(self):
        """Test that straight deformation (type 0) has minimal x offsets."""
        df = generate_bone_rigging_v6(n_samples=100)
        straight = df[df['deformation_type'] == 0]

        # For straight deformation, x offsets should be ~0 (only noise)
        # Check that mean absolute x offset is small
        x_cols = ['bone_base_end_x', 'bone_mid_end_x', 'bone_mid_upper_end_x',
                  'bone_upper_end_x', 'bone_tip_end_x']

        for col in x_cols:
            mean_abs = straight[col].abs().mean()
            # Should be very small (just noise ~2%)
            assert mean_abs < 0.1, f"Straight deformation {col} should be near 0, got {mean_abs}"

    def test_s_curve_deformation_pattern(self):
        """Test that s_curve deformation shows expected pattern."""
        df = generate_bone_rigging_v6(n_samples=100)
        s_curve = df[(df['deformation_type'] == 1) & (df['intensity'] > 0.5)]

        if len(s_curve) > 0:
            # S-curve pattern: out → in → out
            # mid should be positive (out)
            # mid_upper should be negative (in)
            avg_mid = s_curve['bone_mid_end_x'].mean()
            avg_mid_upper = s_curve['bone_mid_upper_end_x'].mean()

            assert avg_mid > avg_mid_upper, "S-curve: mid should be more outward than mid_upper"

    def test_c_curve_deformation_inward(self):
        """Test that c_curve deformation shows inward curl."""
        df = generate_bone_rigging_v6(n_samples=100)
        c_curve = df[(df['deformation_type'] == 2) & (df['intensity'] > 0.5)]

        if len(c_curve) > 0:
            # C-curve: all offsets should be negative (inward)
            x_cols = ['bone_mid_end_x', 'bone_mid_upper_end_x', 'bone_upper_end_x']

            for col in x_cols:
                mean_val = c_curve[col].mean()
                assert mean_val < 0, f"C-curve {col} should be negative (inward), got {mean_val}"

    def test_intensity_affects_deformation(self):
        """Test that higher intensity produces larger deformations."""
        df = generate_bone_rigging_v6(n_samples=200)

        # Compare s_curve with low vs high intensity
        low_intensity = df[(df['deformation_type'] == 1) & (df['intensity'] < 0.3)]
        high_intensity = df[(df['deformation_type'] == 1) & (df['intensity'] > 0.7)]

        if len(low_intensity) > 0 and len(high_intensity) > 0:
            # High intensity should have larger absolute x offsets
            low_abs = low_intensity['bone_mid_end_x'].abs().mean()
            high_abs = high_intensity['bone_mid_end_x'].abs().mean()

            assert high_abs > low_abs, "Higher intensity should produce larger deformations"

    def test_layer_affects_petal_height(self):
        """Test that outer layers have taller petals."""
        df = generate_bone_rigging_v6(n_samples=100)

        # Group by layer and check average tip height
        avg_height_by_layer = df.groupby('layer_index')['bone_tip_end_y'].mean()

        # Layer 2 (outer) should be taller than layer 0 (inner)
        assert avg_height_by_layer[2] > avg_height_by_layer[0], \
            "Outer layer petals should be taller than inner layer"


class TestBoneRiggingV6CLI:
    """Test CLI generator for 5 independent bones."""

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
        """Test that CLI generates 5 bones per petal."""
        from scripts.generate_rose_cli_v2 import RoseCLIGeneratorV2

        generator = RoseCLIGeneratorV2()
        petal_data = generator.generate_petal(
            layer_idx=1, petal_idx=0, base_size=2.0,
            opening_degree=0.8, deformation_type=0, intensity=0.5
        )

        rigging = '\n'.join(petal_data['rigging'])

        # Check all 5 bone names are present
        bone_names = ['bone_base', 'bone_mid', 'bone_mid_upper', 'bone_upper', 'bone_tip']
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
