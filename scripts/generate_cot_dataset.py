#!/usr/bin/env python3
"""
Generate dataset for CoT (Chain-of-Thought) learning.

SR learns to predict optimal CP count from shape features.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cot_reasoning import CoTReasoner


def generate_cot_dataset(n_samples: int = 500) -> pd.DataFrame:
    """
    Generate dataset for learning CP count prediction.

    Features:
        - object_type_code: Encoded object type (0-9)
        - width: Object width
        - height: Object height
        - symmetry_code: 0=none, 1=bilateral, 2=radial
        - smooth_curves: 0 or 1
        - detail_code: 0=low, 1=medium, 2=high

    Target:
        - optimal_cp_count: Predicted by CoT reasoner
        - confidence: Reasoner confidence score
    """

    reasoner = CoTReasoner()

    # Object types with codes
    object_types = [
        ("triangle", 0),
        ("rectangle", 1),
        ("pentagon", 2),
        ("wing", 3),
        ("petal", 4),
        ("leaf", 5),
        ("heart", 6),
        ("circle", 7),
        ("ellipse", 8),
        ("custom", 9),
    ]

    detail_levels = [("low", 0), ("medium", 1), ("high", 2)]

    data = []
    np.random.seed(42)

    for i in range(n_samples):
        # Random selection
        obj_name, obj_code = object_types[np.random.randint(len(object_types))]
        detail_name, detail_code = detail_levels[np.random.randint(len(detail_levels))]

        # Random dimensions
        width = np.random.uniform(0.2, 2.0)
        height = np.random.uniform(0.5, 3.0)

        # Random properties
        symmetry_required = np.random.choice([True, False], p=[0.8, 0.2])
        smooth_curves = np.random.choice([True, False], p=[0.7, 0.3])

        # Use CoT reasoning
        result = reasoner.reason_and_generate(
            object_name=f"{obj_name}_{i}",
            width=width,
            height=height,
            symmetry_required=symmetry_required,
            smooth_curves=smooth_curves,
            detail_level=detail_name,
            verbose=False,
        )

        # Encode symmetry
        if result["analysis"].symmetry == "radial":
            symmetry_code = 2
        elif result["analysis"].symmetry == "bilateral":
            symmetry_code = 1
        else:
            symmetry_code = 0

        data.append(
            {
                "object_type_code": obj_code,
                "width": round(width, 4),
                "height": round(height, 4),
                "aspect_ratio": round(height / width, 4),
                "symmetry_code": symmetry_code,
                "smooth_curves": int(smooth_curves),
                "detail_code": detail_code,
                "optimal_cp_count": result["cp_count"],
                "confidence": round(result["decision"].confidence, 4),
            }
        )

    return pd.DataFrame(data)


def generate_shape_type_dataset(n_samples: int = 200) -> pd.DataFrame:
    """
    Generate dataset for shape type classification.

    SR learns to predict shape type from geometric properties.
    """

    data = []
    np.random.seed(43)

    # Shape characteristics
    # (name, typical_aspect_ratio, has_curves, is_organic, is_symmetric)
    shapes = [
        ("triangle", 1.0, False, False, True),
        ("rectangle", 0.7, False, False, True),
        ("wing", 2.5, True, True, True),
        ("petal", 3.0, True, True, True),
        ("leaf", 4.0, True, True, True),
        ("heart", 1.2, True, True, True),
        ("circle", 1.0, True, False, True),
    ]

    for _ in range(n_samples):
        for i, (name, base_ar, has_curves, is_organic, is_symmetric) in enumerate(
            shapes
        ):
            # Add noise to aspect ratio
            aspect_ratio = base_ar * np.random.uniform(0.8, 1.2)
            width = np.random.uniform(0.3, 2.0)
            height = width * aspect_ratio

            data.append(
                {
                    "width": round(width, 4),
                    "height": round(height, 4),
                    "aspect_ratio": round(aspect_ratio, 4),
                    "has_curves": int(has_curves),
                    "is_organic": int(is_organic),
                    "is_symmetric": int(is_symmetric),
                    "shape_type_code": i,
                }
            )

    return pd.DataFrame(data)


def main():
    """Generate all CoT learning datasets."""

    print("=" * 60)
    print("Generating CoT Learning Datasets")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "data" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. CP Count Prediction Dataset
    print("\n1. CP Count Prediction Dataset...")
    df_cp = generate_cot_dataset(500)
    cp_path = output_dir / "cot_cp_count_dataset.csv"
    df_cp.to_csv(cp_path, index=False)
    print(f"   ✓ Saved {len(df_cp)} samples to {cp_path}")
    print(f"   Features: {list(df_cp.columns[:-2])}")
    print(f"   Targets: optimal_cp_count, confidence")
    print(f"   CP count distribution:")
    print(df_cp["optimal_cp_count"].value_counts().sort_index().to_string())

    # 2. Shape Type Classification Dataset
    print("\n2. Shape Type Classification Dataset...")
    df_shape = generate_shape_type_dataset(200)
    shape_path = output_dir / "cot_shape_type_dataset.csv"
    df_shape.to_csv(shape_path, index=False)
    print(f"   ✓ Saved {len(df_shape)} samples to {shape_path}")
    print(f"   Features: {list(df_shape.columns[:-1])}")
    print(f"   Target: shape_type_code")

    # 3. Summary statistics
    print("\n" + "=" * 60)
    print("Dataset Summary:")
    print("=" * 60)
    print(f"CP Count Dataset: {len(df_cp)} samples")
    print(f"  - Mean CP count: {df_cp['optimal_cp_count'].mean():.2f}")
    print(f"  - Min CP count: {df_cp['optimal_cp_count'].min()}")
    print(f"  - Max CP count: {df_cp['optimal_cp_count'].max()}")
    print(f"  - Mean confidence: {df_cp['confidence'].mean():.2%}")

    print(f"\nShape Type Dataset: {len(df_shape)} samples")
    print(f"  - {len(df_shape['shape_type_code'].unique())} shape types")

    print("\n✓ CoT dataset generation complete!")


if __name__ == "__main__":
    main()
