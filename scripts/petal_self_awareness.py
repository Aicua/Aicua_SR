#!/usr/bin/env python3
"""
Petal Self-Awareness Module - Genesis Reasoning.

Enables petals to understand their own creation, structure, and identity
through Chain-of-Thought reasoning.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from cot_reasoning import CoTReasoner, ShapeAnalysis, CPDecision


@dataclass
class GenesisKnowledge:
    """Knowledge about how the petal was created."""

    # Identity
    object_id: str
    object_type: str
    creation_timestamp: float

    # Structural understanding
    cp_count: int
    control_points: List[Tuple[float, float]]
    width: float
    height: float
    aspect_ratio: float

    # Why these properties?
    genesis_reasoning: List[str]
    structural_reasoning: List[str]
    identity_reasoning: List[str]

    # Self-knowledge scores
    self_understanding: float  # 0-1
    structural_confidence: float  # 0-1

    # Constraints learned
    min_viable_cps: int
    max_viable_cps: int
    deformation_limits: Dict[str, Tuple[float, float]]


@dataclass
class SelfAwarePetal:
    """A petal that understands itself."""

    # Core identity
    name: str
    layer: int
    position_in_layer: int

    # Physical properties
    width: float
    height: float
    opening_degree: float
    detail_level: str

    # Genesis knowledge
    genesis: Optional[GenesisKnowledge] = None

    # Reasoning history
    thought_history: List[str] = field(default_factory=list)


class GenesisReasoner:
    """
    Reasons about the creation and identity of a petal.

    Enables petals to understand:
    - Why they were created with specific parameters
    - What their structural properties mean
    - How they relate to their purpose
    """

    def __init__(self):
        """Initialize genesis reasoner."""
        self.cot_reasoner = CoTReasoner()

        # Knowledge base about petal properties
        self.property_meanings = {
            "width": {
                "small": (0.0, 0.3, "compact, inner layer petal"),
                "medium": (0.3, 0.6, "standard petal form"),
                "large": (0.6, 1.0, "prominent, outer layer petal"),
            },
            "height": {
                "short": (0.0, 0.8, "young/budding petal"),
                "medium": (0.8, 1.5, "mature petal"),
                "tall": (1.5, 3.0, "elongated, decorative petal"),
            },
            "aspect_ratio": {
                "wide": (0.0, 0.3, "broad, covering petal"),
                "balanced": (0.3, 0.5, "natural petal proportions"),
                "narrow": (0.5, 1.0, "slender, elegant petal"),
            },
            "cp_count": {
                "minimal": (3, 4, "simple silhouette"),
                "standard": (5, 6, "smooth curves with detail"),
                "complex": (7, 8, "intricate organic form"),
            },
        }

        # Purpose knowledge
        self.layer_purposes = {
            1: "inner layer - protects center, small and curved inward",
            2: "middle layer - provides volume and color display",
            3: "outer layer - largest, most visible, attracts attention",
        }

        self.position_purposes = {
            "first": "leading petal in arrangement",
            "middle": "supporting petal in cluster",
            "last": "completing the circular pattern",
        }

    def reason_about_genesis(self, petal: SelfAwarePetal) -> GenesisKnowledge:
        """
        Generate complete genesis knowledge for a petal.

        This is the petal "understanding itself".
        """
        import time

        genesis_reasoning = []
        structural_reasoning = []
        identity_reasoning = []

        # === PHASE 1: IDENTITY REASONING ===
        identity_reasoning.append(
            f"I am '{petal.name}', a petal in layer {petal.layer}, "
            f"position {petal.position_in_layer}"
        )

        # Understand layer purpose
        layer_purpose = self.layer_purposes.get(
            petal.layer, "custom layer with special function"
        )
        identity_reasoning.append(
            f"My layer purpose: {layer_purpose}"
        )

        # Understand position
        if petal.position_in_layer == 0:
            pos_desc = "first"
        elif petal.position_in_layer >= 4:
            pos_desc = "last"
        else:
            pos_desc = "middle"

        pos_purpose = self.position_purposes[pos_desc]
        identity_reasoning.append(
            f"My position role: {pos_purpose}"
        )

        # === PHASE 2: STRUCTURAL REASONING ===
        aspect_ratio = petal.height / petal.width if petal.width > 0 else 1.0

        # Reason about width
        width_category = self._categorize_property("width", petal.width)
        structural_reasoning.append(
            f"My width is {petal.width:.3f} ({width_category[0]}): {width_category[1]}"
        )

        # Reason about height
        height_category = self._categorize_property("height", petal.height)
        structural_reasoning.append(
            f"My height is {petal.height:.3f} ({height_category[0]}): {height_category[1]}"
        )

        # Reason about aspect ratio
        ar_category = self._categorize_property("aspect_ratio", 1/aspect_ratio)
        structural_reasoning.append(
            f"My aspect ratio is {aspect_ratio:.2f} ({ar_category[0]}): {ar_category[1]}"
        )

        # === PHASE 3: GENESIS REASONING (How I was created) ===

        # Use CoT reasoner to understand CP decision
        analysis = self.cot_reasoner.analyze_shape(
            petal.name,
            petal.width,
            petal.height,
            symmetry_required=True,
            smooth_curves=True,
            detail_level=petal.detail_level,
        )

        cp_decision = self.cot_reasoner.decide_cp_count(analysis)
        control_points = self.cot_reasoner.generate_cp_positions(
            cp_decision.cp_count,
            petal.width,
            petal.height,
            analysis.symmetry,
        )

        genesis_reasoning.append(
            f"Step 1 - Type Detection: I am recognized as '{analysis.object_type}' "
            f"because my name contains 'petal'"
        )

        genesis_reasoning.append(
            f"Step 2 - Symmetry Analysis: I have {analysis.symmetry} symmetry "
            f"(organic shapes need bilateral balance)"
        )

        genesis_reasoning.append(
            f"Step 3 - Curvature Determination: My curvature is {analysis.curvature} "
            f"(detail_level={petal.detail_level} affects this)"
        )

        genesis_reasoning.append(
            f"Step 4 - Organic Classification: I am organic={analysis.organic} "
            f"(petals are natural, living forms)"
        )

        genesis_reasoning.append(
            f"Step 5 - CP Count Decision: I need {cp_decision.cp_count} control points "
            f"(confidence: {cp_decision.confidence:.1%})"
        )

        # Explain each CP
        for i, (x, y) in enumerate(control_points):
            cp_role = self._explain_cp_role(i, len(control_points), x, y, petal.width, petal.height)
            genesis_reasoning.append(f"  CP{i+1} at ({x:.4f}, {y:.4f}): {cp_role}")

        # === PHASE 4: CALCULATE SELF-UNDERSTANDING ===

        # Higher understanding if reasoning is consistent
        self_understanding = cp_decision.confidence

        # Adjust based on how well properties align with purpose
        if petal.layer == 1 and petal.width < 0.4:
            self_understanding += 0.05  # Inner layer should be small
        if petal.layer == 3 and petal.width > 0.5:
            self_understanding += 0.05  # Outer layer should be large

        self_understanding = min(1.0, self_understanding)

        # Structural confidence based on CP distribution
        structural_confidence = self._calculate_structural_confidence(control_points)

        # === PHASE 5: LEARN CONSTRAINTS ===

        min_cp, max_cp, _ = self.cot_reasoner.shape_patterns["petal"]

        # Deformation limits based on current state
        deformation_limits = {
            "width": (petal.width * 0.5, petal.width * 2.0),
            "height": (petal.height * 0.6, petal.height * 1.5),
            "opening": (0.0, 1.0),  # Can close completely or open fully
            "rotation": (-180.0, 180.0),  # Full rotation possible
        }

        # Create genesis knowledge
        genesis = GenesisKnowledge(
            object_id=petal.name,
            object_type="petal",
            creation_timestamp=time.time(),
            cp_count=cp_decision.cp_count,
            control_points=control_points,
            width=petal.width,
            height=petal.height,
            aspect_ratio=aspect_ratio,
            genesis_reasoning=genesis_reasoning,
            structural_reasoning=structural_reasoning,
            identity_reasoning=identity_reasoning,
            self_understanding=self_understanding,
            structural_confidence=structural_confidence,
            min_viable_cps=min_cp,
            max_viable_cps=max_cp,
            deformation_limits=deformation_limits,
        )

        return genesis

    def _categorize_property(self, prop_name: str, value: float) -> Tuple[str, str]:
        """Categorize a property value and return its meaning."""
        categories = self.property_meanings.get(prop_name, {})

        for category_name, (min_val, max_val, meaning) in categories.items():
            if min_val <= value < max_val:
                return (category_name, meaning)

        # Default if not found
        return ("custom", "specialized configuration")

    def _explain_cp_role(
        self,
        cp_index: int,
        total_cps: int,
        x: float,
        y: float,
        width: float,
        height: float
    ) -> str:
        """Explain the role of a specific control point."""

        # Normalize position
        norm_x = x / (width / 2) if width > 0 else 0
        norm_y = y / height if height > 0 else 0

        # Determine role based on position
        if norm_y < 0.1:
            # Base region
            if norm_x < -0.3:
                return "left base anchor - provides foundation"
            elif norm_x > 0.3:
                return "right base anchor - provides foundation"
            else:
                return "center base - attachment point"
        elif norm_y > 0.9:
            return "tip - defines petal apex and direction"
        elif norm_y > 0.5:
            # Upper curve
            if norm_x < 0:
                return "left upper curve - shapes top silhouette"
            else:
                return "right upper curve - shapes top silhouette"
        else:
            # Middle curve (widest part)
            if norm_x < 0:
                return "left widest point - defines petal breadth"
            else:
                return "right widest point - defines petal breadth"

    def _calculate_structural_confidence(
        self,
        control_points: List[Tuple[float, float]]
    ) -> float:
        """Calculate confidence in structural integrity."""

        if len(control_points) < 3:
            return 0.5

        # Check for symmetry
        mid_idx = len(control_points) // 2
        symmetry_score = 0.0

        for i in range(mid_idx):
            left_cp = control_points[i]
            right_cp = control_points[-(i+1)]

            # Check Y similarity (should be close for bilateral)
            y_diff = abs(left_cp[1] - right_cp[1])
            # Check X mirror (should be opposite)
            x_sum = abs(left_cp[0] + right_cp[0])

            if y_diff < 0.01 and x_sum < 0.01:
                symmetry_score += 1.0

        symmetry_score /= max(1, mid_idx)

        # Check for smooth progression
        smoothness_score = 1.0
        for i in range(1, len(control_points) - 1):
            prev_cp = control_points[i-1]
            curr_cp = control_points[i]
            next_cp = control_points[i+1]

            # Large sudden changes reduce smoothness
            angle_change = self._calculate_angle_change(prev_cp, curr_cp, next_cp)
            if angle_change > 90:
                smoothness_score -= 0.1

        smoothness_score = max(0.0, smoothness_score)

        # Combined confidence
        return (symmetry_score * 0.6 + smoothness_score * 0.4)

    def _calculate_angle_change(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float]
    ) -> float:
        """Calculate angle change at p2."""
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # Dot product
        dot = v1[0] * v2[0] + v1[1] * v2[1]

        # Magnitudes
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if mag1 == 0 or mag2 == 0:
            return 0

        # Angle in degrees
        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp
        angle = math.degrees(math.acos(cos_angle))

        return angle

    def generate_self_description(self, petal: SelfAwarePetal) -> str:
        """Generate a natural language self-description."""

        if petal.genesis is None:
            petal.genesis = self.reason_about_genesis(petal)

        genesis = petal.genesis

        description = []
        description.append("=== PETAL SELF-AWARENESS REPORT ===\n")

        # Identity section
        description.append("WHO AM I:")
        for line in genesis.identity_reasoning:
            description.append(f"  • {line}")
        description.append("")

        # Structure section
        description.append("MY STRUCTURE:")
        for line in genesis.structural_reasoning:
            description.append(f"  • {line}")
        description.append("")

        # Genesis section
        description.append("HOW I WAS CREATED:")
        for line in genesis.genesis_reasoning:
            description.append(f"  • {line}")
        description.append("")

        # Self-understanding scores
        description.append("MY SELF-UNDERSTANDING:")
        description.append(f"  • Overall understanding: {genesis.self_understanding:.1%}")
        description.append(f"  • Structural confidence: {genesis.structural_confidence:.1%}")
        description.append("")

        # Constraints
        description.append("MY LIMITATIONS:")
        description.append(f"  • CP count must stay between {genesis.min_viable_cps}-{genesis.max_viable_cps}")
        for prop, (min_val, max_val) in genesis.deformation_limits.items():
            description.append(f"  • {prop}: {min_val:.3f} to {max_val:.3f}")

        return "\n".join(description)

    def create_aware_petal(
        self,
        name: str,
        layer: int = 1,
        position: int = 0,
        width: float = 0.4,
        height: float = 1.2,
        opening: float = 0.8,
        detail: str = "medium"
    ) -> SelfAwarePetal:
        """Create a petal with self-awareness."""

        petal = SelfAwarePetal(
            name=name,
            layer=layer,
            position_in_layer=position,
            width=width,
            height=height,
            opening_degree=opening,
            detail_level=detail,
        )

        # Generate genesis knowledge
        petal.genesis = self.reason_about_genesis(petal)

        # Record initial thought
        petal.thought_history.append(
            f"I have been created with {petal.genesis.cp_count} control points "
            f"and {petal.genesis.self_understanding:.1%} self-understanding"
        )

        return petal


def demo_genesis_reasoning():
    """Demonstrate genesis reasoning for different petals."""

    reasoner = GenesisReasoner()

    # Create petals with different configurations
    test_cases = [
        ("petal_L1_P0", 1, 0, 0.3, 0.9, 0.6, "low"),
        ("petal_L2_P1", 2, 1, 0.4, 1.2, 0.8, "medium"),
        ("petal_L3_P2", 3, 2, 0.6, 1.5, 0.9, "high"),
    ]

    for name, layer, pos, w, h, opening, detail in test_cases:
        print("=" * 70)
        petal = reasoner.create_aware_petal(name, layer, pos, w, h, opening, detail)
        description = reasoner.generate_self_description(petal)
        print(description)
        print()


if __name__ == "__main__":
    demo_genesis_reasoning()
