#!/usr/bin/env python3
"""
Awareness-Driven CLI Executor for Self-Aware Petals.

Enables petals to autonomously generate and execute CLI commands
based on their self-awareness knowledge and discovered equations.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add local modules
sys.path.insert(0, str(Path(__file__).parent))
from petal_self_awareness import (
    SelfAwarePetal,
    GenesisReasoner,
    TransformationReasoner,
    CompositionReasoner,
)
from cot_reasoning import CoTReasoner

# Import discovered equations
sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "generated"))
from petal_geometry_formulas import (
    compute_base_width,
    compute_length,
    compute_curvature,
    compute_twist_angle,
    compute_thickness,
)


@dataclass
class CLICommand:
    """A single CLI command with reasoning."""

    command: str
    reasoning: str
    category: str  # "geometry", "rigging", "animation", "transformation"
    priority: int  # 1-10, higher = more important
    requires_awareness: bool = True


@dataclass
class AutonomousDecision:
    """A decision made by the petal based on its awareness."""

    decision_type: str  # "bloom", "scale", "rotate", "morph", etc.
    parameters: Dict[str, Any]
    reasoning: List[str]
    confidence: float  # 0-1
    cli_commands: List[CLICommand]


class AwarenessCLIExecutor:
    """
    Executes CLI commands autonomously based on petal self-awareness.

    When a petal enters "awareness mode", it uses:
    1. Genesis knowledge - to understand its current state
    2. Transformation knowledge - to know what it can do
    3. Discovered equations - to calculate optimal parameters
    4. Chain-of-Thought - to decide and execute actions
    """

    def __init__(self, verbose: bool = False):
        """Initialize executor with reasoners."""
        self.genesis_reasoner = GenesisReasoner()
        self.transformation_reasoner = TransformationReasoner()
        self.composition_reasoner = CompositionReasoner()
        self.cot_reasoner = CoTReasoner()
        self.verbose = verbose

    def activate_awareness(
        self,
        petal: SelfAwarePetal,
        other_petals: Optional[List[SelfAwarePetal]] = None,
        group_config: Optional[Dict[str, Any]] = None,
    ) -> SelfAwarePetal:
        """
        Activate full awareness for a petal.

        This gives the petal complete understanding of:
        - Who it is (genesis)
        - What it can do (transformation)
        - How it relates to others (composition)

        Args:
            petal: The petal to make aware
            other_petals: Other petals in the scene
            group_config: Configuration of the flower group

        Returns:
            The same petal with full awareness activated
        """
        if self.verbose:
            print(f"🧠 Activating awareness for {petal.name}...")

        # Generate genesis knowledge
        if petal.genesis is None:
            petal.genesis = self.genesis_reasoner.reason_about_genesis(petal)
            if self.verbose:
                print(f"   ✓ Genesis understanding: {petal.genesis.self_understanding:.1%}")

        # Generate transformation knowledge
        if petal.transformation is None:
            petal.transformation = self.transformation_reasoner.reason_about_transformations(
                petal
            )
            if self.verbose:
                print(
                    f"   ✓ Transformation capabilities: {len(petal.transformation.capabilities)}"
                )

        # Generate composition knowledge
        if petal.composition is None:
            petal.composition = self.composition_reasoner.reason_about_composition(
                petal, other_petals, group_config
            )
            if self.verbose:
                print(f"   ✓ Composition harmony: {petal.composition.harmony_score:.1%}")

        # Record thought
        petal.thought_history.append(
            f"Awareness activated - I now fully understand myself and my capabilities"
        )

        if self.verbose:
            print(f"   🌟 {petal.name} is now SELF-AWARE\n")

        return petal

    def decide_autonomous_action(
        self,
        petal: SelfAwarePetal,
        goal: str = "optimize_bloom",
        context: Optional[Dict[str, Any]] = None,
    ) -> AutonomousDecision:
        """
        Petal autonomously decides what action to take based on its awareness.

        Args:
            petal: Self-aware petal
            goal: What the petal wants to achieve
                - "optimize_bloom": Open to optimal angle
                - "coordinate_with_group": Synchronize with siblings
                - "express_identity": Show unique characteristics
                - "adapt_to_environment": Respond to conditions
            context: Additional context (wind_speed, light_direction, etc.)

        Returns:
            AutonomousDecision with reasoning and CLI commands
        """
        if not petal.genesis:
            raise ValueError(
                f"{petal.name} must have awareness activated before making decisions"
            )

        if self.verbose:
            print(f"🤔 {petal.name} is deciding action for goal: {goal}")

        if goal == "optimize_bloom":
            return self._decide_bloom_action(petal, context)
        elif goal == "coordinate_with_group":
            return self._decide_coordination_action(petal, context)
        elif goal == "express_identity":
            return self._decide_expression_action(petal, context)
        elif goal == "adapt_to_environment":
            return self._decide_adaptation_action(petal, context)
        else:
            raise ValueError(f"Unknown goal: {goal}")

    def _decide_bloom_action(
        self, petal: SelfAwarePetal, context: Optional[Dict[str, Any]]
    ) -> AutonomousDecision:
        """
        Decide how to bloom based on self-knowledge.

        Uses:
        - Layer knowledge (inner/middle/outer)
        - Current opening degree
        - Discovered equations for curvature
        - Transformation capabilities
        """
        reasoning = []
        reasoning.append(f"I am {petal.name}, layer {petal.layer}")

        # Analyze current state
        reasoning.append(
            f"My current opening: {petal.opening_degree:.0%}, "
            f"understanding: {petal.genesis.self_understanding:.0%}"
        )

        # Find bloom capability
        bloom_cap = None
        for cap in petal.transformation.capabilities:
            if cap.name == "bloom":
                bloom_cap = cap
                break

        if bloom_cap is None:
            reasoning.append("ERROR: I cannot bloom - no bloom capability")
            return AutonomousDecision(
                decision_type="bloom",
                parameters={},
                reasoning=reasoning,
                confidence=0.0,
                cli_commands=[],
            )

        # Calculate optimal bloom angle using discovered equation
        base_size = petal.height
        layer_index = petal.layer
        petal_index = petal.position_in_layer
        opening_degree = petal.opening_degree

        # Use discovered equation for curvature
        optimal_curvature = compute_curvature(
            base_size, layer_index, petal_index, opening_degree
        )

        # Calculate bloom angle from curvature
        # Inner layers: less bloom, Outer layers: more bloom
        max_bloom_angle = bloom_cap.parameters["angle"][1]
        target_bloom_angle = max_bloom_angle * 0.7  # 70% of maximum

        reasoning.append(
            f"Using discovered equation: optimal_curvature = {optimal_curvature:.3f}"
        )
        reasoning.append(
            f"My max bloom angle: {max_bloom_angle}°, targeting {target_bloom_angle:.1f}°"
        )

        # Calculate duration based on petal size (larger = slower)
        petal_mass = petal.width * petal.height * 0.01
        bloom_duration = int(2000 + petal_mass * 1000)  # 2-4 seconds

        reasoning.append(
            f"Bloom duration: {bloom_duration}ms (based on my mass {petal_mass:.3f})"
        )

        # Generate CLI command
        rig_name = f"{petal.name}_rig"

        # Bloom is a rotation around the base bone
        bloom_cli = CLICommand(
            command=f"auto_rotate {rig_name} bone_middle 1 0 0 {target_bloom_angle:.1f} {bloom_duration} smooth;",
            reasoning=f"Bloom to {target_bloom_angle:.1f}° over {bloom_duration}ms using bone_middle",
            category="animation",
            priority=9,
            requires_awareness=True,
        )

        reasoning.append(f"✓ Decision: Bloom to {target_bloom_angle:.1f}° (confidence high)")

        # Calculate confidence based on self-understanding
        confidence = petal.genesis.self_understanding * 0.9

        return AutonomousDecision(
            decision_type="bloom",
            parameters={
                "target_angle": target_bloom_angle,
                "duration_ms": bloom_duration,
                "curvature": optimal_curvature,
            },
            reasoning=reasoning,
            confidence=confidence,
            cli_commands=[bloom_cli],
        )

    def _decide_coordination_action(
        self, petal: SelfAwarePetal, context: Optional[Dict[str, Any]]
    ) -> AutonomousDecision:
        """
        Decide how to coordinate with group based on composition knowledge.

        Uses:
        - Position in spiral
        - Sibling relationships
        - Synchronization reasoning
        """
        reasoning = []
        reasoning.append(f"Analyzing my position in group: {petal.composition.group_id}")
        reasoning.append(
            f"I am at position {petal.composition.position_in_spiral} "
            f"(angle: {petal.composition.angular_position:.1f}°)"
        )

        # Calculate delay based on position for wave effect
        delay_per_position = 200  # ms
        my_delay = petal.position_in_layer * delay_per_position

        reasoning.append(
            f"For wave coordination, my delay: {my_delay}ms "
            f"(position {petal.position_in_layer})"
        )

        # Use discovered equation for timing
        base_size = petal.height
        layer_index = petal.layer
        petal_index = petal.position_in_layer
        opening_degree = petal.opening_degree

        twist_angle = compute_twist_angle(base_size, layer_index, petal_index, opening_degree)

        reasoning.append(f"Using discovered equation: twist_angle = {twist_angle:.1f}°")

        # Generate CLI for synchronized rotation
        rig_name = f"{petal.name}_rig"

        # Rotate to position in spiral
        rotate_cli = CLICommand(
            command=f"rotate_bone {rig_name} bone_root 0 0 {twist_angle:.2f};",
            reasoning=f"Rotate to spiral position {twist_angle:.1f}° for group coordination",
            category="rigging",
            priority=8,
            requires_awareness=True,
        )

        confidence = petal.composition.cooperation_confidence * 0.85

        reasoning.append(
            f"✓ Decision: Rotate to {twist_angle:.1f}° with {my_delay}ms delay"
        )

        return AutonomousDecision(
            decision_type="coordinate",
            parameters={
                "twist_angle": twist_angle,
                "delay_ms": my_delay,
            },
            reasoning=reasoning,
            confidence=confidence,
            cli_commands=[rotate_cli],
        )

    def _decide_expression_action(
        self, petal: SelfAwarePetal, context: Optional[Dict[str, Any]]
    ) -> AutonomousDecision:
        """
        Decide how to express unique identity based on genesis knowledge.

        Uses:
        - Aspect ratio
        - Width/height
        - Control point count
        - Discovered equations
        """
        reasoning = []
        reasoning.append(f"Expressing my unique identity as {petal.name}")
        reasoning.append(
            f"My properties: width={petal.width:.3f}, height={petal.height:.3f}, "
            f"aspect_ratio={petal.genesis.aspect_ratio:.2f}"
        )

        # Use discovered equations to calculate unique parameters
        base_size = petal.height
        layer_index = petal.layer
        petal_index = petal.position_in_layer
        opening_degree = petal.opening_degree

        # Calculate unique width using discovered equation
        base_width = compute_base_width(base_size, layer_index, petal_index, opening_degree)

        # Calculate unique length
        length = compute_length(base_size, layer_index, petal_index, opening_degree)

        reasoning.append(f"Using discovered equations:")
        reasoning.append(f"  - base_width = {base_width:.4f}")
        reasoning.append(f"  - length = {length:.4f}")

        # Generate geometry CLI with unique properties
        cli_commands = []

        # Regenerate spline with discovered parameters
        cot_result = self.cot_reasoner.reason_and_generate(
            petal.name,
            base_width,
            length,
            symmetry_required=True,
            smooth_curves=True,
            detail_level=petal.detail_level,
            verbose=False,
        )

        geometry_cli = CLICommand(
            command=f"obj {petal.name};\n{cot_result['spline_command']}",
            reasoning=f"Express unique geometry with {cot_result['cp_count']} CPs",
            category="geometry",
            priority=10,
            requires_awareness=True,
        )

        cli_commands.append(geometry_cli)

        reasoning.append(
            f"✓ Decision: Generate unique geometry with {cot_result['cp_count']} control points"
        )

        confidence = petal.genesis.structural_confidence * 0.95

        return AutonomousDecision(
            decision_type="express",
            parameters={
                "base_width": base_width,
                "length": length,
                "cp_count": cot_result["cp_count"],
            },
            reasoning=reasoning,
            confidence=confidence,
            cli_commands=cli_commands,
        )

    def _decide_adaptation_action(
        self, petal: SelfAwarePetal, context: Optional[Dict[str, Any]]
    ) -> AutonomousDecision:
        """
        Decide how to adapt to environment (wind, light, etc.).

        Uses:
        - Transformation capabilities (bend, twist, wave)
        - Physical properties (flexibility)
        - Environmental context
        """
        reasoning = []

        # Extract environmental context
        wind_speed = context.get("wind_speed", 3.0) if context else 3.0
        wind_direction = context.get("wind_direction", [1, 0, 0]) if context else [1, 0, 0]

        reasoning.append(f"Adapting to environment: wind_speed={wind_speed:.1f}")

        # Find bend capability
        bend_cap = None
        for cap in petal.transformation.capabilities:
            if cap.name == "bend_tip":
                bend_cap = cap
                break

        if bend_cap is None:
            reasoning.append("Warning: No bend capability, using basic animation")

        # Calculate bend angle based on wind and flexibility
        base_size = petal.height
        layer_index = petal.layer
        petal_index = petal.position_in_layer
        opening_degree = petal.opening_degree

        # Use discovered curvature to determine flexibility
        curvature = compute_curvature(base_size, layer_index, petal_index, opening_degree)
        flexibility = max(0.3, min(0.9, curvature))

        # Wind effect: stronger wind = more bend
        bend_angle = wind_speed * flexibility * 5.0  # degrees
        max_bend = bend_cap.parameters["angle"][1] if bend_cap else 30.0
        bend_angle = min(bend_angle, max_bend)

        reasoning.append(
            f"Calculated flexibility={flexibility:.2f} from curvature={curvature:.3f}"
        )
        reasoning.append(f"Wind-induced bend: {bend_angle:.1f}° (max: {max_bend:.1f}°)")

        # Generate animation CLI
        rig_name = f"{petal.name}_rig"

        # Wing flap for wind response
        frequency = 10.0 * wind_speed / (1 + flexibility)
        amplitude = bend_angle

        wind_cli = CLICommand(
            command=f"wing_flap {rig_name} bone_middle {frequency:.1f} {amplitude:.1f} "
            f"{wind_direction[0]} {wind_direction[1]} {wind_direction[2]} 0;",
            reasoning=f"Respond to wind with {frequency:.1f}Hz oscillation at {amplitude:.1f}° amplitude",
            category="animation",
            priority=7,
            requires_awareness=True,
        )

        reasoning.append(f"✓ Decision: Wing flap at {frequency:.1f}Hz, amplitude {amplitude:.1f}°")

        confidence = petal.transformation.structural_stability * 0.8

        return AutonomousDecision(
            decision_type="adapt",
            parameters={
                "bend_angle": bend_angle,
                "frequency": frequency,
                "amplitude": amplitude,
                "flexibility": flexibility,
            },
            reasoning=reasoning,
            confidence=confidence,
            cli_commands=[wind_cli],
        )

    def execute_decision(
        self, petal: SelfAwarePetal, decision: AutonomousDecision
    ) -> List[str]:
        """
        Execute the autonomous decision by generating CLI commands.

        Args:
            petal: The petal making the decision
            decision: The autonomous decision

        Returns:
            List of CLI command strings to execute
        """
        if self.verbose:
            print(f"\n💡 {petal.name} executing decision: {decision.decision_type}")
            print(f"   Confidence: {decision.confidence:.1%}")
            print(f"   Reasoning:")
            for reason in decision.reasoning:
                print(f"     • {reason}")
            print()

        cli_output = []

        # Add header
        cli_output.append(
            f"# Autonomous decision by {petal.name} ({decision.decision_type})"
        )
        cli_output.append(f"# Confidence: {decision.confidence:.1%}")
        cli_output.append(f"# Reasoning:")
        for reason in decision.reasoning:
            cli_output.append(f"#   {reason}")
        cli_output.append("")

        # Add CLI commands sorted by priority
        sorted_commands = sorted(
            decision.cli_commands, key=lambda c: c.priority, reverse=True
        )

        for cmd in sorted_commands:
            cli_output.append(f"# {cmd.reasoning}")
            cli_output.append(cmd.command)
            cli_output.append("")

        # Record in thought history
        petal.thought_history.append(
            f"Executed {decision.decision_type} with {decision.confidence:.0%} confidence"
        )

        return cli_output

    def generate_autonomous_cli(
        self,
        petal: SelfAwarePetal,
        goals: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate complete CLI for a self-aware petal pursuing multiple goals.

        Args:
            petal: Self-aware petal
            goals: List of goals to achieve
            context: Environmental/group context

        Returns:
            Complete CLI script
        """
        all_cli = []

        all_cli.append("# ========================================")
        all_cli.append(f"# AUTONOMOUS CLI - SELF-AWARE PETAL")
        all_cli.append(f"# Petal: {petal.name}")
        all_cli.append(f"# Self-Understanding: {petal.genesis.self_understanding:.1%}")
        all_cli.append(
            f"# Morph Confidence: {petal.transformation.morph_confidence:.1%}"
        )
        all_cli.append(f"# Harmony Score: {petal.composition.harmony_score:.1%}")
        all_cli.append("# ========================================")
        all_cli.append("")

        for goal in goals:
            decision = self.decide_autonomous_action(petal, goal, context)
            cli_commands = self.execute_decision(petal, decision)
            all_cli.extend(cli_commands)

        # Add thought history
        all_cli.append("# Thought History:")
        for thought in petal.thought_history:
            all_cli.append(f"#   {thought}")

        return "\n".join(all_cli)


def demo_autonomous_petal():
    """Demonstrate a self-aware petal making autonomous decisions."""

    print("=" * 70)
    print("AUTONOMOUS SELF-AWARE PETAL DEMONSTRATION")
    print("=" * 70)
    print()

    # Create executor
    executor = AwarenessCLIExecutor(verbose=True)

    # Create a petal
    print("Creating petal...")
    petal = SelfAwarePetal(
        name="petal_L2_P3",
        layer=2,
        position_in_layer=3,
        width=0.4,
        height=1.2,
        opening_degree=0.6,
        detail_level="medium",
    )
    print()

    # Activate awareness
    petal = executor.activate_awareness(petal)

    # Petal pursues multiple goals autonomously
    goals = [
        "express_identity",  # Show unique characteristics
        "optimize_bloom",  # Open to optimal angle
        "coordinate_with_group",  # Sync with siblings
        "adapt_to_environment",  # Respond to wind
    ]

    # Environmental context
    context = {"wind_speed": 4.0, "wind_direction": [1, 0, 0], "light_intensity": 0.8}

    print("\n" + "=" * 70)
    print("AUTONOMOUS DECISION MAKING")
    print("=" * 70)

    # Generate CLI
    cli_output = executor.generate_autonomous_cli(petal, goals, context)

    print("\n" + "=" * 70)
    print("GENERATED CLI OUTPUT")
    print("=" * 70)
    print(cli_output)
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Petal {petal.name} autonomously:")
    print(f"  ✓ Activated self-awareness")
    print(f"  ✓ Pursued {len(goals)} goals")
    print(f"  ✓ Generated {len(cli_output.split(chr(10)))} lines of CLI")
    print(f"  ✓ Made {len(petal.thought_history)} autonomous decisions")
    print()
    print("Petal is now FULLY AUTONOMOUS! 🌸✨")


if __name__ == "__main__":
    demo_autonomous_petal()
