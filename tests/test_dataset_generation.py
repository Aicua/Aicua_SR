"""
Test dataset generation functionality.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from generate_rose_dataset import (
    generate_petal_geometry_dataset,
    generate_bone_rigging_dataset,
    generate_animation_dataset,
    generate_cli_examples,
)


class TestPetalGeometryDataset:
    """Test petal geometry dataset generation."""

    def test_generates_dataframe(self):
        """Test that function returns a DataFrame."""
        df = generate_petal_geometry_dataset(n_samples=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_has_required_columns(self):
        """Test that all required columns are present."""
        df = generate_petal_geometry_dataset(n_samples=10)

        required_features = ['base_size', 'layer_index', 'petal_index', 'opening_degree']
        required_targets = ['base_width', 'length', 'curvature', 'twist_angle', 'thickness']

        for col in required_features + required_targets:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_null_values(self):
        """Test that no null values are generated."""
        df = generate_petal_geometry_dataset(n_samples=10)
        assert df.isnull().sum().sum() == 0

    def test_value_ranges(self):
        """Test that values are within expected ranges."""
        df = generate_petal_geometry_dataset(n_samples=50)

        # Features
        assert (df['base_size'] >= 1.0).all() and (df['base_size'] <= 5.0).all()
        assert (df['layer_index'] >= 1).all() and (df['layer_index'] <= 3).all()
        assert (df['opening_degree'] >= 0.0).all() and (df['opening_degree'] <= 1.0).all()

        # Targets should be positive
        assert (df['base_width'] > 0).all()
        assert (df['length'] > 0).all()
        assert (df['curvature'] >= 0).all() and (df['curvature'] <= 1).all()
        assert (df['thickness'] > 0).all()

    def test_sample_size(self):
        """Test that correct number of samples is generated."""
        # Note: n_samples * layers * petals_per_layer
        df = generate_petal_geometry_dataset(n_samples=5)
        # 5 samples * (5 + 8 + 13) petals = 130 rows
        assert len(df) == 5 * (5 + 8 + 13)


class TestBoneRiggingDataset:
    """Test bone rigging dataset generation."""

    def test_generates_dataframe(self):
        """Test that function returns a DataFrame."""
        df = generate_bone_rigging_dataset(n_samples=10)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        """Test that all required columns are present."""
        df = generate_bone_rigging_dataset(n_samples=10)

        required = ['petal_length', 'petal_width', 'flexibility_factor', 'layer_index',
                    'bone_count', 'bone_length', 'joint_spacing', 'bind_weight']

        for col in required:
            assert col in df.columns

    def test_bone_count_positive(self):
        """Test that bone count is always positive."""
        df = generate_bone_rigging_dataset(n_samples=50)
        assert (df['bone_count'] >= 2).all()

    def test_flexibility_range(self):
        """Test flexibility is in valid range."""
        df = generate_bone_rigging_dataset(n_samples=50)
        assert (df['flexibility_factor'] >= 0.3).all()
        assert (df['flexibility_factor'] <= 1.0).all()


class TestAnimationDataset:
    """Test animation parameters dataset generation."""

    def test_generates_dataframe(self):
        """Test that function returns a DataFrame."""
        df = generate_animation_dataset(n_samples=10)
        assert isinstance(df, pd.DataFrame)

    def test_frequency_positive(self):
        """Test that frequency is positive."""
        df = generate_animation_dataset(n_samples=50)
        assert (df['frequency'] > 0).all()

    def test_amplitude_positive(self):
        """Test that amplitude is positive."""
        df = generate_animation_dataset(n_samples=50)
        assert (df['amplitude'] > 0).all()

    def test_damping_reasonable(self):
        """Test damping factor is in reasonable range."""
        df = generate_animation_dataset(n_samples=50)
        assert (df['damping_factor'] > 0).all()
        assert (df['damping_factor'] < 1).all()


class TestCLIExamples:
    """Test CLI example generation."""

    def test_generates_examples(self):
        """Test that examples are generated."""
        examples = generate_cli_examples()
        assert len(examples) > 0
        assert all(isinstance(ex, str) for ex in examples)

    def test_examples_contain_commands(self):
        """Test that examples contain expected commands."""
        examples = generate_cli_examples()

        for example in examples:
            assert 'obj' in example or 'create_armature' in example
            assert 'add_bone' in example
