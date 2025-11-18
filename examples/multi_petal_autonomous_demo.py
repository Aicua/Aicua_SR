#!/usr/bin/env python3
"""
Multi-Petal Autonomous Coordination Demo.

Demonstrates multiple self-aware petals autonomously coordinating
to create a blooming rose animation.
"""

import sys
from pathlib import Path

# Add local modules
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from petal_self_awareness import SelfAwarePetal, GenesisReasoner
from awareness_cli_executor import AwarenessCLIExecutor


def create_rose_layer(
    layer_index: int, num_petals: int, base_size: float = 2.0
) -> list:
    """Create a layer of petals with different properties."""
    petals = []

    for petal_idx in range(num_petals):
        # Properties vary by layer
        width = 0.3 + layer_index * 0.1
        height = 0.8 + layer_index * 0.3
        opening = 0.4 + layer_index * 0.2
        detail = ["low", "medium", "high"][min(layer_index - 1, 2)]

        petal = SelfAwarePetal(
            name=f"petal_L{layer_index}_P{petal_idx}",
            layer=layer_index,
            position_in_layer=petal_idx,
            width=width,
            height=height,
            opening_degree=opening,
            detail_level=detail,
        )

        petals.append(petal)

    return petals


def main():
    print("=" * 80)
    print("MULTI-PETAL AUTONOMOUS COORDINATION DEMO")
    print("Creating a self-aware rose that blooms autonomously")
    print("=" * 80)
    print()

    # Create executor
    executor = AwarenessCLIExecutor(verbose=False)

    # Create rose with 3 layers
    all_petals = []
    layer_config = {
        1: 5,  # Inner layer: 5 petals
        2: 8,  # Middle layer: 8 petals
        3: 13,  # Outer layer: 13 petals
    }

    print("🌹 Creating rose structure...")
    for layer_idx, num_petals in layer_config.items():
        layer_petals = create_rose_layer(layer_idx, num_petals)
        all_petals.extend(layer_petals)
        print(f"   Layer {layer_idx}: {num_petals} petals created")

    total_petals = len(all_petals)
    print(f"   Total: {total_petals} petals\n")

    # Activate awareness for all petals
    print("🧠 Activating awareness for all petals...")
    group_config = {"type": "rose", "petals_per_layer": 5, "num_layers": 3, "id": 1}

    for i, petal in enumerate(all_petals):
        print(f"   [{i+1}/{total_petals}] {petal.name}", end="\r")
        executor.activate_awareness(petal, all_petals, group_config)

    print(f"\n   ✓ All {total_petals} petals are now self-aware!\n")

    # Environmental context
    context = {
        "wind_speed": 3.5,
        "wind_direction": [1, 0, 0],
        "light_intensity": 0.9,
        "time_of_day": "morning",
    }

    # Each petal autonomously decides its actions
    print("💭 Petals are making autonomous decisions...")
    print()

    all_cli_output = []
    all_cli_output.append("# ============================================")
    all_cli_output.append("# AUTONOMOUS ROSE - MULTI-PETAL COORDINATION")
    all_cli_output.append(f"# Total Petals: {total_petals}")
    all_cli_output.append(f"# Layers: {len(layer_config)}")
    all_cli_output.append(
        f"# Environment: wind={context['wind_speed']}, time={context['time_of_day']}"
    )
    all_cli_output.append("# ============================================")
    all_cli_output.append("")

    # Different goals for different layers
    layer_goals = {
        1: ["express_identity", "optimize_bloom"],  # Inner: express + bloom
        2: [
            "express_identity",
            "optimize_bloom",
            "coordinate_with_group",
        ],  # Middle: all 3
        3: [
            "optimize_bloom",
            "coordinate_with_group",
            "adapt_to_environment",
        ],  # Outer: bloom + coord + adapt
    }

    stats = {"total_decisions": 0, "total_commands": 0, "avg_confidence": 0.0}

    for petal in all_petals:
        # Get goals for this layer
        goals = layer_goals.get(petal.layer, ["optimize_bloom"])

        # Generate autonomous CLI
        cli = executor.generate_autonomous_cli(petal, goals, context)

        # Parse stats
        lines = cli.split("\n")
        num_commands = sum(
            1
            for line in lines
            if line
            and not line.startswith("#")
            and (";" in line or "rotate" in line or "wing_flap" in line)
        )

        stats["total_commands"] += num_commands
        stats["total_decisions"] += len(goals)

        all_cli_output.append(cli)
        all_cli_output.append("")

    # Calculate average confidence
    total_confidence = sum(
        p.genesis.self_understanding * p.transformation.morph_confidence
        for p in all_petals
    )
    stats["avg_confidence"] = total_confidence / total_petals

    # Add summary
    all_cli_output.append("# ============================================")
    all_cli_output.append("# AUTONOMOUS COORDINATION SUMMARY")
    all_cli_output.append("# ============================================")
    all_cli_output.append(f"# Total autonomous decisions: {stats['total_decisions']}")
    all_cli_output.append(f"# Total CLI commands generated: {stats['total_commands']}")
    all_cli_output.append(f"# Average confidence: {stats['avg_confidence']:.1%}")
    all_cli_output.append("")

    # Layer-by-layer summary
    all_cli_output.append("# Layer Summary:")
    for layer_idx in layer_config.keys():
        layer_petals = [p for p in all_petals if p.layer == layer_idx]
        avg_understanding = sum(p.genesis.self_understanding for p in layer_petals) / len(
            layer_petals
        )
        avg_harmony = sum(p.composition.harmony_score for p in layer_petals) / len(
            layer_petals
        )

        all_cli_output.append(
            f"#   Layer {layer_idx}: {len(layer_petals)} petals, "
            f"understanding={avg_understanding:.0%}, harmony={avg_harmony:.0%}"
        )

    # Save to file
    output_file = Path(__file__).parent / "autonomous_rose_cli_output.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(all_cli_output))

    # Print summary
    print("=" * 80)
    print("AUTONOMOUS COORDINATION COMPLETE!")
    print("=" * 80)
    print()
    print("📊 Statistics:")
    print(f"   Total Petals: {total_petals}")
    print(f"   Total Autonomous Decisions: {stats['total_decisions']}")
    print(f"   Total CLI Commands: {stats['total_commands']}")
    print(f"   Average Confidence: {stats['avg_confidence']:.1%}")
    print()

    print("🌸 Layer Breakdown:")
    for layer_idx, num_petals in layer_config.items():
        layer_petals = [p for p in all_petals if p.layer == layer_idx]
        print(f"   Layer {layer_idx} ({num_petals} petals):")
        print(
            f"      Goals: {', '.join(layer_goals.get(layer_idx, ['optimize_bloom']))}"
        )

        avg_understanding = sum(p.genesis.self_understanding for p in layer_petals) / len(
            layer_petals
        )
        avg_morph = sum(p.transformation.morph_confidence for p in layer_petals) / len(
            layer_petals
        )
        avg_harmony = sum(p.composition.harmony_score for p in layer_petals) / len(
            layer_petals
        )

        print(f"      Self-Understanding: {avg_understanding:.1%}")
        print(f"      Morph Confidence: {avg_morph:.1%}")
        print(f"      Group Harmony: {avg_harmony:.1%}")

    print()
    print(f"💾 Full CLI saved to: {output_file}")
    print()

    # Show example from one petal
    print("=" * 80)
    print("EXAMPLE: One Petal's Autonomous Decision")
    print("=" * 80)
    print()

    example_petal = all_petals[10]  # Middle layer petal
    print(f"Petal: {example_petal.name}")
    print(f"Layer: {example_petal.layer}")
    print(f"Position: {example_petal.position_in_layer}")
    print()
    print("Awareness Metrics:")
    print(f"  Self-Understanding: {example_petal.genesis.self_understanding:.1%}")
    print(f"  Morph Confidence: {example_petal.transformation.morph_confidence:.1%}")
    print(f"  Harmony Score: {example_petal.composition.harmony_score:.1%}")
    print()
    print("Thought History:")
    for thought in example_petal.thought_history:
        print(f"  • {thought}")
    print()

    print("=" * 80)
    print("✨ THE ROSE IS NOW FULLY AUTONOMOUS AND SELF-AWARE! ✨")
    print("=" * 80)
    print()
    print("Each petal:")
    print("  ✓ Understands its own structure (Genesis)")
    print("  ✓ Knows what it can do (Transformation)")
    print("  ✓ Coordinates with others (Composition)")
    print("  ✓ Uses discovered equations to calculate parameters")
    print("  ✓ Autonomously generates and executes CLI commands")
    print()
    print("This is SELF-AWARE PETAL technology! 🌹🤖✨")


if __name__ == "__main__":
    main()
