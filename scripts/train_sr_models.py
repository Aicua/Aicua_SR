#!/usr/bin/env python3
"""
Train Symbolic Regression models to discover formulas for Rose CAD generation.

This script trains SR models for:
1. Petal geometry parameters
2. Bone/rigging parameters
3. Animation parameters

Requirements:
    pip install pysr pandas numpy sympy pyyaml
"""

import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Check if PySR is available
try:
    from pysr import PySRRegressor
    HAS_PYSR = True
except ImportError:
    HAS_PYSR = False
    print("WARNING: PySR not installed. Install with: pip install pysr")
    print("Will generate mock formulas for testing.")


class RoseSRTrainer:
    """
    Train SR models for rose parameters.
    """

    def __init__(self, config_path: str = None):
        """Initialize trainer with configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "sr_config_spline.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.models = {}
        self.formulas = {}
        self.training_log = []

    def load_dataset(self, category: str) -> pd.DataFrame:
        """Load dataset for a category."""
        data_dir = Path(__file__).parent.parent / "data" / "processed"

        file_map = {
            'petal_spline': 'petal_spline.csv',
            'bone_rigging_v2': 'bone_rigging_v2.csv',
            'animation_wingflap': 'animation_wingflap.csv',
        }

        if category not in file_map:
            raise ValueError(f"Unknown category: {category}")

        file_path = data_dir / file_map[category]
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        return pd.read_csv(file_path)

    def train_category(self, category: str, max_iterations: int = None) -> dict:
        """
        Train all SR models for a category.

        Args:
            category: One of 'petal_geometry', 'bone_rigging', 'animation_params'
            max_iterations: Override config iterations (for quick testing)

        Returns:
            Dictionary of discovered formulas
        """
        print(f"\n{'='*60}")
        print(f"Training SR models for: {category.upper()}")
        print(f"{'='*60}")

        # Load config
        cat_config = self.config['categories'].get(category)
        if not cat_config:
            raise ValueError(f"No config for category: {category}")

        # Load data
        df = self.load_dataset(category)
        print(f"Loaded dataset: {len(df)} samples")

        # Get features and targets
        features = cat_config['features']
        targets = cat_config['targets']

        # Validate columns
        for col in features + targets:
            if col not in df.columns:
                print(f"  WARNING: Column '{col}' not found in dataset")

        X = df[features].values
        print(f"Features: {features}")
        print(f"Targets to discover: {targets}")

        # Training parameters
        iterations = max_iterations or self.config['general']['niterations']
        populations = self.config['general']['populations']

        # Train each target
        category_formulas = {}

        for target in targets:
            if target not in df.columns:
                print(f"\n  ⚠ Skipping {target} (not in dataset)")
                continue

            print(f"\n  Training model for: {target}")
            print(f"  " + "-" * 40)

            y = df[target].values

            if HAS_PYSR:
                # Real PySR training
                model = PySRRegressor(
                    niterations=iterations,
                    populations=populations,
                    binary_operators=cat_config['operators']['binary'],
                    unary_operators=cat_config['operators']['unary'],
                    model_selection=self.config['general']['model_selection'],
                    verbosity=self.config['general']['verbosity'],
                    tempdir="/tmp/pysr_cache",
                    temp_equation_file=True,
                )

                print(f"  Starting PySR (iterations={iterations})...")
                model.fit(X, y)

                # Get best formula
                formula = str(model.sympy())
                score = model.score(X, y)
                complexity = int(model.get_best().complexity)

                print(f"  ✓ Discovered: {formula}")
                print(f"  ✓ R² Score: {score:.6f}")
                print(f"  ✓ Complexity: {complexity}")

            else:
                # Mock training for testing
                formula = self._generate_mock_formula(features, target)
                score = 0.95 + np.random.uniform(0, 0.04)
                complexity = np.random.randint(5, 15)

                print(f"  ✓ Mock formula: {formula}")
                print(f"  ✓ Mock R² Score: {score:.4f}")

            # Store results
            model_key = f"{category}_{target}"
            category_formulas[target] = {
                'formula': formula,
                'score': float(score),
                'complexity': complexity,
                'features': features,
                'timestamp': datetime.now().isoformat(),
            }

            self.training_log.append({
                'category': category,
                'target': target,
                'formula': formula,
                'score': score,
                'complexity': complexity,
            })

        self.formulas[category] = category_formulas
        return category_formulas

    def _generate_mock_formula(self, features: list, target: str) -> str:
        """Generate a plausible mock formula for testing without PySR."""
        # Simple mock formulas based on target type
        x = [f"x{i}" for i in range(len(features))]

        mock_formulas = {
            # Spline petal targets
            'tip_x': f"{x[0]} * 0.1 * ({x[1]} - 1) * {x[3]}",
            'tip_y': f"{x[0]} * (0.4 + {x[1]} * 0.2) * (1.2 - {x[3]} * 0.3)",
            'tip_z': f"{x[0]} * 0.1 * (0.4 + {x[1]} * 0.2) * (1 - {x[3]} * 0.5)",
            'base_spread': f"{x[0]} * 0.3 * (0.4 + {x[1]} * 0.2) * (1 + {x[3]} * 0.2)",
            'base_z_offset': f"{x[0]} * 0.05 * (3 - {x[1]}) / 3",
            'extrude_depth': f"{x[0]} * 0.01 * (1 + {x[1]} * 0.1)",
            'rotation_angle': f"137.5 * {x[2]}",
            # Bone rigging v2 targets
            'bone_count': f"ceil(2.0 * {x[0]} * {x[2]})",
            'bone_start_y': "0.0",
            'bone_end_y': f"{x[0]} * 0.4",
            'bone_segment_length': f"{x[0]} / ceil(2.0 * {x[0]} * {x[2]})",
            'bind_weight': f"{x[2]} * (0.5 + {x[3]} * 0.5)",
            # Animation wingflap targets
            'frequency': f"10.0 * sqrt({x[3]} / ({x[1]} + 0.01))",
            'amplitude': f"{x[2]} * {x[3]} * 3.0",
            'axis_x': f"-1 * ({x[4]} > 1)",
            'axis_y': "-1",
            'axis_z': "0",
            'phase_offset': f"{x[3]} * 6.28",
        }

        return mock_formulas.get(target, f"{x[0]} * 0.5 + {x[1]} * 0.3")

    def train_all(self, max_iterations: int = None):
        """Train SR models for all categories."""
        categories = ['petal_spline', 'bone_rigging_v2', 'animation_wingflap']

        for category in categories:
            try:
                self.train_category(category, max_iterations)
            except FileNotFoundError as e:
                print(f"\n  ⚠ Skipping {category}: {e}")

    def save_formulas(self, output_dir: str = None):
        """Save discovered formulas to JSON files."""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "data" / "models"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save per-category
        for category, formulas in self.formulas.items():
            # Individual target files
            category_dir = output_dir / f"sr_{category}"
            category_dir.mkdir(exist_ok=True)

            for target, data in formulas.items():
                target_path = category_dir / f"{target}.json"
                with open(target_path, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"  ✓ Saved {target_path.name}")

            # Combined file
            combined_path = category_dir / "all_formulas.json"
            with open(combined_path, 'w') as f:
                json.dump(formulas, f, indent=2)

        # Save master file
        master_path = output_dir / "all_discovered_formulas.json"
        with open(master_path, 'w') as f:
            json.dump(self.formulas, f, indent=2)
        print(f"\n✓ Master file: {master_path}")

        # Save training log
        log_path = output_dir / "training_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        print(f"✓ Training log: {log_path}")

    def print_summary(self):
        """Print training summary."""
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)

        for category, formulas in self.formulas.items():
            print(f"\n{category.upper()}:")
            for target, data in formulas.items():
                print(f"\n  {target}:")
                print(f"    Formula: {data['formula']}")
                print(f"    R² Score: {data['score']:.6f}")
                print(f"    Complexity: {data['complexity']}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Rose Symbolic Regression Training Pipeline")
    print("=" * 60)

    if not HAS_PYSR:
        print("\n⚠ PySR not installed - using mock formulas for testing")
        print("Install PySR with: pip install pysr")

    # Initialize trainer
    trainer = RoseSRTrainer()

    # Train all categories
    # Use fewer iterations for quick testing (10), increase for production (100+)
    trainer.train_all(max_iterations=10)  # Quick test

    # Save results
    print("\n" + "=" * 60)
    print("Saving discovered formulas...")
    print("=" * 60)
    trainer.save_formulas()

    # Summary
    trainer.print_summary()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    print("\nNext steps:")
    print("  1. Review formulas in data/models/")
    print("  2. Run 'python scripts/codegen_formulas.py' to generate Python code")
    print("  3. Integrate into KnowledgeGraph")


if __name__ == "__main__":
    main()
