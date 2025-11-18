#!/usr/bin/env python3
"""
Train Symbolic Regression models for Self-Awareness prediction using PySR.

Discovers symbolic equations that predict self-awareness metrics
WITHOUT running CoT reasoning (100x faster for production).

Usage:
    python train_self_awareness_models.py [--use-pysr] [--iterations 100]

    --use-pysr: Use actual PySR (requires pysr package)
    --iterations: Number of SR iterations (default: 100)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pickle
import argparse
import json

# Try to import PySR
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
    print("Warning: PySR not installed. Using fallback linear regression.")
    print("Install with: pip install pysr")


class SymbolicRegressionModel:
    """
    Symbolic Regression model that discovers equations from data.

    Uses PySR when available, falls back to linear regression otherwise.
    """

    def __init__(self, name: str, use_pysr: bool = True, iterations: int = 100):
        self.name = name
        self.use_pysr = use_pysr and PYSR_AVAILABLE
        self.iterations = iterations
        self.equation = None
        self.equation_latex = None
        self.feature_names = None
        self.model = None
        self.r2_score = None
        self.complexity = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train SR to find symbolic equation."""
        self.feature_names = list(X.columns)

        if self.use_pysr:
            self._fit_pysr(X, y)
        else:
            self._fit_linear(X, y)

    def _fit_pysr(self, X: pd.DataFrame, y: pd.Series):
        """Train using PySR to discover symbolic equations."""
        print(f"  Training {self.name} with PySR ({self.iterations} iterations)...")

        # Configure PySR
        self.model = PySRRegressor(
            niterations=self.iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "exp", "log", "sin", "cos"],
            populations=15,
            population_size=33,
            ncyclesperiteration=550,
            maxsize=25,
            maxdepth=10,
            parsimony=0.0032,
            model_selection="best",
            loss="loss(x, y) = (x - y)^2",
            progress=False,
            verbosity=0,
            random_state=42,
            deterministic=True,
            procs=0,  # Single process for reproducibility
            multithreading=False,
        )

        # Fit model
        self.model.fit(X.values, y.values, variable_names=self.feature_names)

        # Get best equation
        best_idx = self.model.equations_.query("loss == loss.min()").index[0]
        self.equation = str(self.model.sympy(best_idx))
        self.equation_latex = self.model.latex(best_idx)
        self.complexity = int(self.model.equations_.loc[best_idx, "complexity"])

        # Calculate R² score
        y_pred = self.model.predict(X.values)
        ss_res = np.sum((y.values - y_pred) ** 2)
        ss_tot = np.sum((y.values - y.mean()) ** 2)
        self.r2_score = 1 - (ss_res / ss_tot)

        print(f"    Equation: {self.equation}")
        print(f"    R² Score: {self.r2_score:.4f}")
        print(f"    Complexity: {self.complexity}")

    def _fit_linear(self, X: pd.DataFrame, y: pd.Series):
        """Fallback to linear regression when PySR not available."""
        print(f"  Training {self.name} with linear regression (fallback)...")

        # Simple linear regression
        X_with_intercept = np.column_stack([np.ones(len(X)), X.values])
        coeffs = np.linalg.lstsq(X_with_intercept, y.values, rcond=None)[0]

        intercept = coeffs[0]
        feature_coeffs = coeffs[1:]

        # Build equation string
        terms = [f"{intercept:.4f}"]
        for i, feat in enumerate(self.feature_names):
            if abs(feature_coeffs[i]) > 0.001:
                sign = "+" if feature_coeffs[i] > 0 else "-"
                terms.append(f"{sign} {abs(feature_coeffs[i]):.4f}*{feat}")

        self.equation = " ".join(terms)
        self.equation_latex = self.equation  # No LaTeX for fallback

        # Store for prediction
        self._intercept = intercept
        self._coeffs = feature_coeffs

        # Calculate R² score
        y_pred = intercept + X.values @ feature_coeffs
        ss_res = np.sum((y.values - y_pred) ** 2)
        ss_tot = np.sum((y.values - y.mean()) ** 2)
        self.r2_score = 1 - (ss_res / ss_tot)
        self.complexity = len(self.feature_names) + 1

        print(f"    Equation: {self.equation}")
        print(f"    R² Score: {self.r2_score:.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using discovered equation."""
        if self.use_pysr and self.model is not None:
            return self.model.predict(X.values)
        else:
            return self._intercept + X.values @ self._coeffs

    def to_dict(self) -> Dict[str, Any]:
        """Export model info as dictionary."""
        return {
            "name": self.name,
            "equation": self.equation,
            "equation_latex": self.equation_latex,
            "r2_score": self.r2_score,
            "complexity": self.complexity,
            "feature_names": self.feature_names,
            "use_pysr": self.use_pysr,
        }


def train_genesis_models(
    data_path: Path, use_pysr: bool = True, iterations: int = 100
) -> Dict[str, SymbolicRegressionModel]:
    """
    Train SR models for Genesis reasoning.

    Targets:
    - cp_count: Predict control points from dimensions
    - self_understanding: Predict self-awareness score
    - structural_confidence: Predict structural confidence
    """
    print("\n" + "=" * 60)
    print("Training Genesis Models")
    print("=" * 60)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")

    # Feature columns
    feature_cols = [
        "layer",
        "width",
        "height",
        "aspect_ratio",
        "detail_code",
        "opening_degree",
    ]
    X = df[feature_cols]

    models = {}

    # Model 1: CP Count
    print("\n1. Training cp_count model...")
    model_cp = SymbolicRegressionModel("cp_count", use_pysr, iterations)
    model_cp.fit(X, df["cp_count"])
    models["cp_count"] = model_cp

    # Model 2: Self Understanding
    print("\n2. Training self_understanding model...")
    model_self = SymbolicRegressionModel("self_understanding", use_pysr, iterations)
    model_self.fit(X, df["self_understanding"])
    models["self_understanding"] = model_self

    # Model 3: Structural Confidence
    print("\n3. Training structural_confidence model...")
    model_struct = SymbolicRegressionModel("structural_confidence", use_pysr, iterations)
    model_struct.fit(X, df["structural_confidence"])
    models["structural_confidence"] = model_struct

    return models


def train_transformation_models(
    data_path: Path, use_pysr: bool = True, iterations: int = 100
) -> Dict[str, SymbolicRegressionModel]:
    """
    Train SR models for Transformation reasoning.

    Targets:
    - morph_confidence: Predict morphing ability
    - avg_risk_level: Predict transformation risk
    - structural_stability: Predict structural stability
    """
    print("\n" + "=" * 60)
    print("Training Transformation Models")
    print("=" * 60)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")

    feature_cols = [
        "layer",
        "width",
        "height",
        "opening_degree",
        "cp_count",
        "self_understanding",
    ]
    X = df[feature_cols]

    models = {}

    # Model 1: Morph Confidence
    print("\n1. Training morph_confidence model...")
    model_morph = SymbolicRegressionModel("morph_confidence", use_pysr, iterations)
    model_morph.fit(X, df["morph_confidence"])
    models["morph_confidence"] = model_morph

    # Model 2: Risk Level
    print("\n2. Training avg_risk_level model...")
    model_risk = SymbolicRegressionModel("avg_risk_level", use_pysr, iterations)
    model_risk.fit(X, df["avg_risk_level"])
    models["avg_risk_level"] = model_risk

    # Model 3: Structural Stability
    print("\n3. Training structural_stability model...")
    model_stability = SymbolicRegressionModel("structural_stability", use_pysr, iterations)
    model_stability.fit(X, df["structural_stability"])
    models["structural_stability"] = model_stability

    return models


def train_composition_models(
    data_path: Path, use_pysr: bool = True, iterations: int = 100
) -> Dict[str, SymbolicRegressionModel]:
    """
    Train SR models for Composition reasoning.

    Targets:
    - harmony_score: Predict group harmony
    - cooperation_confidence: Predict cooperation ability
    """
    print("\n" + "=" * 60)
    print("Training Composition Models")
    print("=" * 60)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")

    feature_cols = [
        "layer",
        "group_size",
        "same_layer_count",
        "adjacent_layer_count",
        "self_understanding",
    ]
    X = df[feature_cols]

    models = {}

    # Model 1: Harmony Score
    print("\n1. Training harmony_score model...")
    model_harmony = SymbolicRegressionModel("harmony_score", use_pysr, iterations)
    model_harmony.fit(X, df["harmony_score"])
    models["harmony_score"] = model_harmony

    # Model 2: Cooperation Confidence
    print("\n2. Training cooperation_confidence model...")
    model_coop = SymbolicRegressionModel("cooperation_confidence", use_pysr, iterations)
    model_coop.fit(X, df["cooperation_confidence"])
    models["cooperation_confidence"] = model_coop

    return models


def save_models(
    models: Dict[str, Dict[str, SymbolicRegressionModel]], output_dir: Path
):
    """Save trained models to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save pickle files
    for category, category_models in models.items():
        model_path = output_dir / f"{category}_models.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(category_models, f)
        print(f"  Saved {category} models to {model_path}")

    # Save equations as JSON for easy viewing
    equations = {}
    for category, category_models in models.items():
        equations[category] = {}
        for name, model in category_models.items():
            equations[category][name] = model.to_dict()

    json_path = output_dir / "discovered_equations.json"
    with open(json_path, "w") as f:
        json.dump(equations, f, indent=2)
    print(f"  Saved equations to {json_path}")


def generate_python_formulas(
    models: Dict[str, Dict[str, SymbolicRegressionModel]], output_path: Path
):
    """Generate Python file with discovered formulas for fast inference."""

    lines = [
        '"""',
        "Auto-generated Self-Awareness Prediction Formulas",
        "",
        "These formulas were discovered using Symbolic Regression.",
        "Use these for fast inference without CoT reasoning.",
        '"""',
        "",
        "import math",
        "",
    ]

    for category, category_models in models.items():
        lines.append(f"# {'='*60}")
        lines.append(f"# {category.upper()} FORMULAS")
        lines.append(f"# {'='*60}")
        lines.append("")

        for name, model in category_models.items():
            func_name = f"predict_{category}_{name}"
            params = ", ".join(model.feature_names)

            lines.append(f"def {func_name}({params}):")
            lines.append(f'    """')
            lines.append(f"    Predict {name}.")
            lines.append(f"    ")
            lines.append(f"    Discovered Formula: {model.equation}")
            lines.append(f"    R² Score: {model.r2_score:.6f}")
            lines.append(f"    Complexity: {model.complexity}")
            lines.append(f'    """')
            lines.append(f"    try:")

            # Convert equation to Python code
            eq = model.equation
            # Replace common math functions
            eq = eq.replace("sqrt", "math.sqrt")
            eq = eq.replace("exp", "math.exp")
            eq = eq.replace("log", "math.log")
            eq = eq.replace("sin", "math.sin")
            eq = eq.replace("cos", "math.cos")

            lines.append(f"        result = {eq}")
            lines.append(f"        return float(result)")
            lines.append(f"    except (ValueError, ZeroDivisionError) as e:")
            lines.append(f'        print(f"Warning: Error computing {name}: {{e}}")')
            lines.append(f"        return 0.0")
            lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"  Generated Python formulas at {output_path}")


def main():
    """Train all self-awareness models."""
    parser = argparse.ArgumentParser(
        description="Train Self-Awareness Symbolic Regression Models"
    )
    parser.add_argument(
        "--use-pysr",
        action="store_true",
        default=False,
        help="Use PySR for real symbolic regression (requires pysr package)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of PySR iterations (default: 100)",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data" / "generated"
    model_dir = Path(__file__).parent.parent / "data" / "models" / "self_awareness"

    print("=" * 60)
    print("Self-Awareness Symbolic Regression Training")
    print("=" * 60)

    if args.use_pysr:
        if PYSR_AVAILABLE:
            print("Using PySR for symbolic regression")
        else:
            print("PySR requested but not available. Using linear regression fallback.")
            args.use_pysr = False
    else:
        print("Using linear regression (for fast demo)")
        print("Use --use-pysr for real symbolic regression")

    print(f"Iterations: {args.iterations}")

    # Check datasets exist
    genesis_path = data_dir / "sa_genesis_dataset.csv"
    transform_path = data_dir / "sa_transformation_dataset.csv"
    composition_path = data_dir / "sa_composition_dataset.csv"

    if not genesis_path.exists():
        print(f"\nError: Dataset not found at {genesis_path}")
        print("Run: python scripts/generate_self_awareness_dataset.py")
        return

    all_models = {}

    # Train Genesis models
    all_models["genesis"] = train_genesis_models(
        genesis_path, args.use_pysr, args.iterations
    )

    # Train Transformation models
    all_models["transformation"] = train_transformation_models(
        transform_path, args.use_pysr, args.iterations
    )

    # Train Composition models
    all_models["composition"] = train_composition_models(
        composition_path, args.use_pysr, args.iterations
    )

    # Save all models
    print("\n" + "=" * 60)
    print("Saving Models")
    print("=" * 60)
    save_models(all_models, model_dir)

    # Generate Python formulas
    formula_path = data_dir / "self_awareness_formulas.py"
    generate_python_formulas(all_models, formula_path)

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    print("\nDiscovered Equations Summary:")
    for category, category_models in all_models.items():
        print(f"\n{category.upper()}:")
        for name, model in category_models.items():
            print(f"  {name}:")
            print(f"    R² = {model.r2_score:.4f}")
            print(f"    {model.equation[:60]}..." if len(model.equation) > 60 else f"    {model.equation}")

    print(f"\nModels saved to: {model_dir}/")
    print(f"Python formulas: {formula_path}")

    print("\nNext steps:")
    print("  1. Import formulas: from data.generated.self_awareness_formulas import *")
    print("  2. Use in awareness_cli_executor.py for fast prediction")
    print("  3. Compare SR predictions vs CoT reasoning accuracy")

    if not args.use_pysr:
        print("\nNote: Run with --use-pysr for better equations (requires: pip install pysr)")


if __name__ == "__main__":
    main()
