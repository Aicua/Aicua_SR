#!/usr/bin/env python3
"""
Generate datasets for Self-Awareness (SA) learning.

SR learns to predict petal self-awareness metrics from features:
- Genesis: How petal was created (CP count, layer, position)
- Transformation: What changes are possible (bloom, scale, deform)
- Composition: How petal relates to others (harmony, cooperation)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from petal_self_awareness import (
    GenesisReasoner,
    TransformationReasoner,
    CompositionReasoner,
    SelfAwarePetal,
)
from cot_reasoning import CoTReasoner


def generate_genesis_dataset(n_samples: int = 3000) -> pd.DataFrame:
    """
    Generate dataset for Genesis reasoning.

    Features:
        - layer: Petal layer (1-5)
        - petal_idx: Index within layer (0-based)
        - width: Petal width
        - height: Petal height
        - aspect_ratio: height/width
        - detail_code: 0=low, 1=medium, 2=high
        - opening_degree: Current opening angle

    Targets:
        - cp_count: Number of control points
        - self_understanding: Confidence in self-knowledge (0-1)
        - structural_confidence: Confidence in structure (0-1)
        - is_inner_layer: 1 if inner (protective), 0 if outer
        - is_outer_layer: 1 if outer (exposed), 0 otherwise
    """

    genesis_reasoner = GenesisReasoner()

    detail_levels = [("low", 0), ("medium", 1), ("high", 2)]

    data = []
    np.random.seed(100)

    for i in range(n_samples):
        # Random petal configuration
        layer = np.random.randint(1, 6)  # 1-5
        petal_idx = np.random.randint(0, 10)  # 0-9

        # Dimensions
        width = np.random.uniform(0.3, 0.8)
        height = np.random.uniform(0.8, 2.0)

        # Properties
        detail_name, detail_code = detail_levels[np.random.randint(len(detail_levels))]
        opening_degree = np.random.uniform(0.0, 90.0)

        # Create petal ID
        petal_id = f"petal_L{layer}_P{petal_idx}"

        # Create SelfAwarePetal object
        petal = SelfAwarePetal(
            name=petal_id,
            layer=layer,
            position_in_layer=petal_idx,
            width=width,
            height=height,
            opening_degree=opening_degree,
            detail_level=detail_name,
        )

        # Get Genesis reasoning
        genesis_knowledge = genesis_reasoner.reason_about_genesis(petal)

        # Encode features
        is_inner = 1 if layer <= 2 else 0
        is_outer = 1 if layer >= 4 else 0

        data.append({
            "layer": layer,
            "petal_idx": petal_idx,
            "width": round(width, 4),
            "height": round(height, 4),
            "aspect_ratio": round(height / width, 4),
            "detail_code": detail_code,
            "opening_degree": round(opening_degree, 4),
            "cp_count": genesis_knowledge.cp_count,
            "self_understanding": round(genesis_knowledge.self_understanding, 4),
            "structural_confidence": round(genesis_knowledge.structural_confidence, 4),
            "is_inner_layer": is_inner,
            "is_outer_layer": is_outer,
        })

    return pd.DataFrame(data)


def generate_transformation_dataset(n_samples: int = 3000) -> pd.DataFrame:
    """
    Generate dataset for Transformation reasoning.

    Features:
        - layer: Petal layer (1-5)
        - width: Petal width
        - height: Petal height
        - opening_degree: Current opening angle
        - cp_count: Number of control points
        - self_understanding: From genesis (0-1)

    Targets:
        - morph_confidence: Confidence in morphing ability (0-1)
        - structural_stability: Structural stability (0-1)
        - num_capabilities: Number of transformation capabilities
        - avg_risk_level: Average risk across all capabilities
        - num_reversible: Number of reversible transformations
    """

    transform_reasoner = TransformationReasoner()

    detail_levels = ["low", "medium", "high"]

    data = []
    np.random.seed(101)

    for i in range(n_samples):
        # Random petal configuration
        layer = np.random.randint(1, 6)
        petal_idx = np.random.randint(0, 10)

        # Dimensions
        width = np.random.uniform(0.3, 0.8)
        height = np.random.uniform(0.8, 2.0)

        # Properties
        detail_name = detail_levels[np.random.randint(len(detail_levels))]
        opening_degree = np.random.uniform(0.0, 90.0)

        petal_id = f"petal_L{layer}_P{petal_idx}"

        # Create SelfAwarePetal object
        petal = SelfAwarePetal(
            name=petal_id,
            layer=layer,
            position_in_layer=petal_idx,
            width=width,
            height=height,
            opening_degree=opening_degree,
            detail_level=detail_name,
        )

        # Get Transformation knowledge (will auto-generate genesis)
        transform_knowledge = transform_reasoner.reason_about_transformations(petal)

        # Calculate aggregate metrics
        num_caps = len(transform_knowledge.capabilities)
        avg_risk = sum(c.risk_level for c in transform_knowledge.capabilities) / max(num_caps, 1)
        num_reversible = sum(1 for c in transform_knowledge.capabilities if c.reversible)

        data.append({
            "layer": layer,
            "width": round(width, 4),
            "height": round(height, 4),
            "opening_degree": round(opening_degree, 4),
            "cp_count": petal.genesis.cp_count,
            "self_understanding": round(petal.genesis.self_understanding, 4),
            "morph_confidence": round(transform_knowledge.morph_confidence, 4),
            "structural_stability": round(transform_knowledge.structural_stability, 4),
            "num_capabilities": num_caps,
            "avg_risk_level": round(avg_risk, 4),
            "num_reversible": num_reversible,
        })

    return pd.DataFrame(data)


def generate_composition_dataset(n_samples: int = 3000) -> pd.DataFrame:
    """
    Generate dataset for Composition reasoning.

    Features:
        - layer: Petal layer (1-5)
        - petal_idx: Index within layer
        - group_size: Number of petals in group
        - same_layer_count: How many siblings in same layer
        - adjacent_layer_count: How many cousins in adjacent layers
        - self_understanding: From genesis (0-1)

    Targets:
        - harmony_score: Group harmony (0-1)
        - cooperation_confidence: Cooperation ability (0-1)
        - sibling_count: Number of sibling relationships
        - cousin_count: Number of cousin relationships
    """

    genesis_reasoner = GenesisReasoner()
    composition_reasoner = CompositionReasoner()

    detail_levels = ["low", "medium", "high"]

    data = []
    np.random.seed(102)

    for i in range(n_samples):
        # Create a group of petals
        group_size = np.random.randint(3, 12)  # 3-11 petals

        # Generate group composition
        petals = []
        layers_used = {}

        for p in range(group_size):
            layer = np.random.randint(1, 6)
            if layer not in layers_used:
                layers_used[layer] = 0
            petal_idx = layers_used[layer]
            layers_used[layer] += 1

            width = np.random.uniform(0.3, 0.8)
            height = np.random.uniform(0.8, 2.0)
            detail_name = detail_levels[np.random.randint(len(detail_levels))]
            opening_degree = np.random.uniform(0.0, 90.0)

            petal_id = f"petal_L{layer}_P{petal_idx}"

            # Create SelfAwarePetal
            petal = SelfAwarePetal(
                name=petal_id,
                layer=layer,
                position_in_layer=petal_idx,
                width=width,
                height=height,
                opening_degree=opening_degree,
                detail_level=detail_name,
            )

            # Generate genesis knowledge
            petal.genesis = genesis_reasoner.reason_about_genesis(petal)

            petals.append(petal)

        # Analyze composition for each petal in the group
        for petal in petals:
            other_petals = [p for p in petals if p.name != petal.name]

            composition_knowledge = composition_reasoner.reason_about_composition(
                petal=petal,
                other_petals=other_petals,
            )

            # Count relationship types
            sibling_count = sum(1 for r in composition_knowledge.relationships if r.relationship_type == "sibling")
            cousin_count = sum(1 for r in composition_knowledge.relationships if r.relationship_type == "cousin")

            # Count same layer and adjacent layer petals
            same_layer = sum(1 for p in petals if p.layer == petal.layer and p.name != petal.name)
            adjacent_layers = sum(1 for p in petals if abs(p.layer - petal.layer) == 1)

            data.append({
                "layer": petal.layer,
                "petal_idx": petal.position_in_layer,
                "group_size": group_size,
                "same_layer_count": same_layer,
                "adjacent_layer_count": adjacent_layers,
                "self_understanding": round(petal.genesis.self_understanding, 4),
                "harmony_score": round(composition_knowledge.harmony_score, 4),
                "cooperation_confidence": round(composition_knowledge.cooperation_confidence, 4),
                "sibling_count": sibling_count,
                "cousin_count": cousin_count,
            })

    return pd.DataFrame(data)


def main():
    """Generate all Self-Awareness learning datasets."""

    print("=" * 60)
    print("Generating Self-Awareness Learning Datasets")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "data" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Genesis Dataset
    print("\n1. Genesis Reasoning Dataset...")
    df_genesis = generate_genesis_dataset(500)
    genesis_path = output_dir / "sa_genesis_dataset.csv"
    df_genesis.to_csv(genesis_path, index=False)
    print(f"   Saved {len(df_genesis)} samples to {genesis_path}")
    print(f"   Features: {list(df_genesis.columns[:7])}")
    print(f"   Targets: {list(df_genesis.columns[7:])}")
    print(f"   Mean self_understanding: {df_genesis['self_understanding'].mean():.2%}")
    print(f"   Mean structural_confidence: {df_genesis['structural_confidence'].mean():.2%}")

    # 2. Transformation Dataset
    print("\n2. Transformation Reasoning Dataset...")
    df_transform = generate_transformation_dataset(500)
    transform_path = output_dir / "sa_transformation_dataset.csv"
    df_transform.to_csv(transform_path, index=False)
    print(f"   Saved {len(df_transform)} samples to {transform_path}")
    print(f"   Features: {list(df_transform.columns[:6])}")
    print(f"   Targets: {list(df_transform.columns[6:])}")
    print(f"   Risk level distribution:")
    print(f"     Low (<0.3): {(df_transform['avg_risk_level'] < 0.3).sum()} samples")
    print(f"     Medium (0.3-0.6): {((df_transform['avg_risk_level'] >= 0.3) & (df_transform['avg_risk_level'] < 0.6)).sum()} samples")
    print(f"     High (>=0.6): {(df_transform['avg_risk_level'] >= 0.6).sum()} samples")

    # 3. Composition Dataset
    print("\n3. Composition Reasoning Dataset...")
    df_composition = generate_composition_dataset(300)
    composition_path = output_dir / "sa_composition_dataset.csv"
    df_composition.to_csv(composition_path, index=False)
    print(f"   Saved {len(df_composition)} samples to {composition_path}")
    print(f"   Features: {list(df_composition.columns[:6])}")
    print(f"   Targets: {list(df_composition.columns[6:])}")
    print(f"   Mean harmony_score: {df_composition['harmony_score'].mean():.2%}")
    print(f"   Mean cooperation_confidence: {df_composition['cooperation_confidence'].mean():.2%}")

    # Summary
    print("\n" + "=" * 60)
    print("Dataset Summary:")
    print("=" * 60)

    print(f"\nGenesis Dataset: {len(df_genesis)} samples")
    print(f"  - CP count range: {df_genesis['cp_count'].min()}-{df_genesis['cp_count'].max()}")
    print(f"  - Self understanding: {df_genesis['self_understanding'].min():.2%} - {df_genesis['self_understanding'].max():.2%}")

    print(f"\nTransformation Dataset: {len(df_transform)} samples")
    print(f"  - Mean morph confidence: {df_transform['morph_confidence'].mean():.2%}")
    print(f"  - Mean avg risk level: {df_transform['avg_risk_level'].mean():.4f}")
    print(f"  - Mean num_capabilities: {df_transform['num_capabilities'].mean():.1f}")

    print(f"\nComposition Dataset: {len(df_composition)} samples")
    print(f"  - Group sizes: {df_composition['group_size'].min()}-{df_composition['group_size'].max()}")
    print(f"  - Total sibling relationships: {df_composition['sibling_count'].sum()}")
    print(f"  - Total cousin relationships: {df_composition['cousin_count'].sum()}")

    print("\nSelf-Awareness dataset generation complete!")


if __name__ == "__main__":
    main()
