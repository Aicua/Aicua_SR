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
class TransformationCapability:
    """A single transformation the petal can perform."""
    name: str
    description: str
    parameters: Dict[str, Tuple[float, float]]  # param -> (min, max)
    risk_level: float  # 0-1 (0=safe, 1=destructive)
    reversible: bool
    reasoning: List[str]


@dataclass
class TransformationKnowledge:
    """Knowledge about how the petal can transform."""

    # Available transformations
    capabilities: List[TransformationCapability]

    # Current state awareness
    current_state: Dict[str, float]

    # Transformation history
    transformation_history: List[Dict[str, Any]]

    # Morphing understanding
    morph_reasoning: List[str]
    scale_reasoning: List[str]
    deform_reasoning: List[str]
    animate_reasoning: List[str]

    # Confidence in transformation ability
    morph_confidence: float
    structural_stability: float  # How stable after transformation


@dataclass
class RelationshipInfo:
    """Information about relationship with another object."""
    other_id: str
    other_type: str  # "petal", "stem", "leaf", "center"
    relationship_type: str  # "sibling", "neighbor", "parent", "child"
    spatial_relation: str  # "adjacent", "above", "below", "inside", "outside"
    reasoning: List[str]


@dataclass
class CompositionKnowledge:
    """Knowledge about how the petal composes with other objects."""

    # Relationships with other objects
    relationships: List[RelationshipInfo]

    # Group membership
    group_id: str  # e.g., "rose_flower_1"
    group_role: str  # e.g., "petal_layer_2"

    # Spatial awareness
    position_in_spiral: int
    angular_position: float  # degrees from reference
    radial_distance: float  # distance from center

    # Coordination capabilities
    coordination_reasoning: List[str]
    synchronization_reasoning: List[str]
    hierarchy_reasoning: List[str]

    # Composition patterns
    available_patterns: List[str]  # ["spiral", "radial", "layered", "clustered"]
    current_pattern: str

    # Interaction constraints
    collision_avoidance: Dict[str, float]  # min distance to other objects
    overlap_rules: Dict[str, bool]  # can overlap with object types

    # Group harmony
    harmony_score: float  # 0-1, how well petal fits in composition
    cooperation_confidence: float  # 0-1, confidence in group coordination


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

    # Transformation knowledge
    transformation: Optional[TransformationKnowledge] = None

    # Composition knowledge
    composition: Optional[CompositionKnowledge] = None

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
                "complex": (7, 8, "intricate organic form with heart-shaped tip"),
                "detailed": (9, 10, "highly detailed with undulating edges"),
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
        elif norm_y > 0.95:
            # Tip center
            return "tip center - defines petal apex"
        elif norm_y > 0.85:
            # Tip sides (heart-shape notch area)
            if norm_x < 0:
                return "left tip side - creates heart-shaped notch"
            else:
                return "right tip side - creates heart-shaped notch"
        elif norm_y > 0.5:
            # Upper curve
            if norm_x < 0:
                return "left upper curve - shapes organic silhouette"
            else:
                return "right upper curve - shapes organic silhouette"
        elif norm_y > 0.2:
            # Lower curve
            if norm_x < 0:
                return "left lower curve - defines petal breadth"
            else:
                return "right lower curve - defines petal breadth"
        else:
            # Base area
            if norm_x < 0:
                return "left base - narrow attachment point"
            else:
                return "right base - narrow attachment point"

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


class TransformationReasoner:
    """
    Reasons about transformation capabilities of a petal.

    Enables petals to understand:
    - What transformations they can perform
    - How each transformation affects their structure
    - Risks and reversibility of transformations
    """

    def __init__(self):
        """Initialize transformation reasoner."""
        self.genesis_reasoner = GenesisReasoner()

        # Define transformation templates
        self.transformation_templates = {
            "bloom": {
                "description": "Open petal by rotating around base bone",
                "params": {"angle": (0, 90), "duration_ms": (500, 5000)},
                "risk": 0.1,
                "reversible": True,
            },
            "wilt": {
                "description": "Close/droop petal by negative rotation",
                "params": {"angle": (-45, 0), "droop_factor": (0.5, 1.0)},
                "risk": 0.2,
                "reversible": True,
            },
            "scale_uniform": {
                "description": "Scale entire petal uniformly",
                "params": {"factor": (0.5, 2.0)},
                "risk": 0.3,
                "reversible": True,
            },
            "scale_width": {
                "description": "Scale petal width only (change aspect ratio)",
                "params": {"factor": (0.5, 2.0)},
                "risk": 0.4,
                "reversible": True,
            },
            "scale_height": {
                "description": "Scale petal height only",
                "params": {"factor": (0.6, 1.5)},
                "risk": 0.3,
                "reversible": True,
            },
            "bend_tip": {
                "description": "Bend petal tip in direction",
                "params": {"angle": (-30, 30), "curvature": (0.1, 0.9)},
                "risk": 0.5,
                "reversible": True,
            },
            "twist": {
                "description": "Twist petal around vertical axis",
                "params": {"degrees": (-180, 180)},
                "risk": 0.4,
                "reversible": True,
            },
            "wave": {
                "description": "Add wave deformation to edges",
                "params": {"amplitude": (0.01, 0.1), "frequency": (1, 5)},
                "risk": 0.6,
                "reversible": True,
            },
            "add_cp": {
                "description": "Add control point for more detail",
                "params": {"position": (0.0, 1.0)},
                "risk": 0.7,
                "reversible": False,  # Structural change
            },
            "remove_cp": {
                "description": "Remove control point to simplify",
                "params": {"cp_index": (1, 7)},
                "risk": 0.8,
                "reversible": False,  # Structural change
            },
            "morph_to_leaf": {
                "description": "Transform petal shape towards leaf",
                "params": {"morph_factor": (0.0, 1.0)},
                "risk": 0.9,
                "reversible": False,  # Identity change
            },
        }

    def reason_about_transformations(
        self, petal: SelfAwarePetal
    ) -> TransformationKnowledge:
        """
        Generate transformation knowledge for a petal.

        Petal learns what it can do and what risks are involved.
        """
        if petal.genesis is None:
            petal.genesis = self.genesis_reasoner.reason_about_genesis(petal)

        capabilities = []
        morph_reasoning = []
        scale_reasoning = []
        deform_reasoning = []
        animate_reasoning = []

        # === ANALYZE BLOOM CAPABILITY ===
        bloom_cap = self._reason_bloom_capability(petal)
        capabilities.append(bloom_cap)
        animate_reasoning.extend(bloom_cap.reasoning)

        # === ANALYZE WILT CAPABILITY ===
        wilt_cap = self._reason_wilt_capability(petal)
        capabilities.append(wilt_cap)
        animate_reasoning.extend(wilt_cap.reasoning)

        # === ANALYZE SCALE CAPABILITIES ===
        scale_caps = self._reason_scale_capabilities(petal)
        capabilities.extend(scale_caps)
        for cap in scale_caps:
            scale_reasoning.extend(cap.reasoning)

        # === ANALYZE DEFORMATION CAPABILITIES ===
        deform_caps = self._reason_deform_capabilities(petal)
        capabilities.extend(deform_caps)
        for cap in deform_caps:
            deform_reasoning.extend(cap.reasoning)

        # === ANALYZE MORPHING CAPABILITIES ===
        morph_caps = self._reason_morph_capabilities(petal)
        capabilities.extend(morph_caps)
        for cap in morph_caps:
            morph_reasoning.extend(cap.reasoning)

        # Calculate confidence scores
        morph_confidence = self._calculate_morph_confidence(petal, capabilities)
        structural_stability = petal.genesis.structural_confidence

        # Current state
        current_state = {
            "width": petal.width,
            "height": petal.height,
            "opening": petal.opening_degree,
            "cp_count": petal.genesis.cp_count,
            "rotation": 0.0,
            "scale": 1.0,
        }

        return TransformationKnowledge(
            capabilities=capabilities,
            current_state=current_state,
            transformation_history=[],
            morph_reasoning=morph_reasoning,
            scale_reasoning=scale_reasoning,
            deform_reasoning=deform_reasoning,
            animate_reasoning=animate_reasoning,
            morph_confidence=morph_confidence,
            structural_stability=structural_stability,
        )

    def _reason_bloom_capability(
        self, petal: SelfAwarePetal
    ) -> TransformationCapability:
        """Reason about bloom (opening) capability."""
        reasoning = []

        # Base capability from template
        template = self.transformation_templates["bloom"]

        # Adjust based on layer
        if petal.layer == 1:
            max_angle = 45  # Inner petals open less
            reasoning.append(
                f"As inner layer petal, my maximum bloom angle is {max_angle}° "
                "(limited by structural role)"
            )
        elif petal.layer == 2:
            max_angle = 60
            reasoning.append(
                f"As middle layer petal, I can bloom up to {max_angle}° "
                "(moderate opening)"
            )
        else:
            max_angle = 90
            reasoning.append(
                f"As outer layer petal, I can bloom fully to {max_angle}° "
                "(maximum visibility)"
            )

        # Adjust based on opening degree
        current_bloom = petal.opening_degree * max_angle
        remaining = max_angle - current_bloom
        reasoning.append(
            f"Currently at {petal.opening_degree:.0%} bloom ({current_bloom:.1f}°), "
            f"can open {remaining:.1f}° more"
        )

        # Duration based on size
        base_duration = 1000 + (petal.height * 500)
        reasoning.append(
            f"Bloom animation should take {base_duration:.0f}ms "
            f"(based on my height {petal.height:.2f})"
        )

        return TransformationCapability(
            name="bloom",
            description=f"Open petal up to {max_angle}°",
            parameters={
                "angle": (0, max_angle),
                "duration_ms": (500, base_duration * 2),
            },
            risk_level=0.1,
            reversible=True,
            reasoning=reasoning,
        )

    def _reason_wilt_capability(
        self, petal: SelfAwarePetal
    ) -> TransformationCapability:
        """Reason about wilting capability."""
        reasoning = []

        # Wilting depends on current state
        if petal.opening_degree > 0.5:
            wilt_range = (-45, 0)
            reasoning.append(
                "I am currently open, can wilt by closing and drooping"
            )
        else:
            wilt_range = (-30, 0)
            reasoning.append(
                "I am already partially closed, limited wilting possible"
            )

        # Droop factor based on organic nature
        droop_factor = 0.8 if petal.genesis.cp_count >= 5 else 0.6
        reasoning.append(
            f"With {petal.genesis.cp_count} CPs, my droop factor is {droop_factor:.1f} "
            "(more CPs = smoother droop)"
        )

        return TransformationCapability(
            name="wilt",
            description="Close and droop petal",
            parameters={
                "angle": wilt_range,
                "droop_factor": (0.5, droop_factor),
            },
            risk_level=0.2,
            reversible=True,
            reasoning=reasoning,
        )

    def _reason_scale_capabilities(
        self, petal: SelfAwarePetal
    ) -> List[TransformationCapability]:
        """Reason about scaling capabilities."""
        capabilities = []

        # UNIFORM SCALE
        uniform_reasoning = []
        min_scale = 0.5
        max_scale = 2.0

        # Adjust based on layer constraints
        if petal.layer == 1:
            max_scale = 1.5  # Inner shouldn't grow too big
            uniform_reasoning.append(
                f"As inner layer, max uniform scale is {max_scale}x "
                "(must stay smaller than middle layer)"
            )
        elif petal.layer == 3:
            min_scale = 0.7  # Outer shouldn't shrink too much
            uniform_reasoning.append(
                f"As outer layer, min scale is {min_scale}x "
                "(must maintain visibility)"
            )
        else:
            uniform_reasoning.append(
                f"Middle layer allows full scaling range {min_scale}x to {max_scale}x"
            )

        capabilities.append(
            TransformationCapability(
                name="scale_uniform",
                description=f"Scale uniformly between {min_scale}x and {max_scale}x",
                parameters={"factor": (min_scale, max_scale)},
                risk_level=0.3,
                reversible=True,
                reasoning=uniform_reasoning,
            )
        )

        # WIDTH SCALE
        width_reasoning = []
        width_min = petal.genesis.deformation_limits["width"][0] / petal.width
        width_max = petal.genesis.deformation_limits["width"][1] / petal.width

        width_reasoning.append(
            f"My width can scale from {width_min:.2f}x to {width_max:.2f}x "
            f"(current width: {petal.width:.3f})"
        )

        if petal.genesis.aspect_ratio > 2.5:
            width_reasoning.append(
                "Warning: I am already narrow, reducing width may make me unstable"
            )

        capabilities.append(
            TransformationCapability(
                name="scale_width",
                description="Adjust petal width (changes aspect ratio)",
                parameters={"factor": (width_min, width_max)},
                risk_level=0.4,
                reversible=True,
                reasoning=width_reasoning,
            )
        )

        # HEIGHT SCALE
        height_reasoning = []
        height_min = petal.genesis.deformation_limits["height"][0] / petal.height
        height_max = petal.genesis.deformation_limits["height"][1] / petal.height

        height_reasoning.append(
            f"My height can scale from {height_min:.2f}x to {height_max:.2f}x "
            f"(current height: {petal.height:.3f})"
        )

        capabilities.append(
            TransformationCapability(
                name="scale_height",
                description="Adjust petal height",
                parameters={"factor": (height_min, height_max)},
                risk_level=0.3,
                reversible=True,
                reasoning=height_reasoning,
            )
        )

        return capabilities

    def _reason_deform_capabilities(
        self, petal: SelfAwarePetal
    ) -> List[TransformationCapability]:
        """Reason about deformation capabilities."""
        capabilities = []

        # BEND TIP
        bend_reasoning = []
        if petal.genesis.cp_count >= 5:
            max_bend = 30
            bend_reasoning.append(
                f"With {petal.genesis.cp_count} CPs, I can bend my tip up to {max_bend}°"
            )
        else:
            max_bend = 15
            bend_reasoning.append(
                f"With only {petal.genesis.cp_count} CPs, tip bending limited to {max_bend}°"
            )

        bend_reasoning.append(
            "Bending changes my tip direction while maintaining base position"
        )

        capabilities.append(
            TransformationCapability(
                name="bend_tip",
                description=f"Bend tip up to {max_bend}°",
                parameters={
                    "angle": (-max_bend, max_bend),
                    "curvature": (0.1, 0.9),
                },
                risk_level=0.5,
                reversible=True,
                reasoning=bend_reasoning,
            )
        )

        # TWIST
        twist_reasoning = []
        twist_reasoning.append(
            "I can twist around my vertical axis for 3D effect"
        )
        if petal.detail_level == "high":
            twist_reasoning.append(
                "High detail level allows full 360° twist capability"
            )
            max_twist = 180
        else:
            twist_reasoning.append(
                f"{petal.detail_level} detail limits twist to avoid distortion"
            )
            max_twist = 90

        capabilities.append(
            TransformationCapability(
                name="twist",
                description=f"Twist around axis (±{max_twist}°)",
                parameters={"degrees": (-max_twist, max_twist)},
                risk_level=0.4,
                reversible=True,
                reasoning=twist_reasoning,
            )
        )

        # WAVE (edge deformation)
        wave_reasoning = []
        if petal.genesis.cp_count >= 6:
            max_amp = 0.1
            wave_reasoning.append(
                f"With {petal.genesis.cp_count} CPs, I can create edge waves "
                f"(amplitude up to {max_amp})"
            )
        else:
            max_amp = 0.05
            wave_reasoning.append(
                f"Limited CPs restrict wave amplitude to {max_amp}"
            )

        capabilities.append(
            TransformationCapability(
                name="wave",
                description="Add wave pattern to edges",
                parameters={
                    "amplitude": (0.01, max_amp),
                    "frequency": (1, 5),
                },
                risk_level=0.6,
                reversible=True,
                reasoning=wave_reasoning,
            )
        )

        return capabilities

    def _reason_morph_capabilities(
        self, petal: SelfAwarePetal
    ) -> List[TransformationCapability]:
        """Reason about morphing (shape change) capabilities."""
        capabilities = []

        # ADD CP
        add_cp_reasoning = []
        if petal.genesis.cp_count < petal.genesis.max_viable_cps:
            can_add = petal.genesis.max_viable_cps - petal.genesis.cp_count
            add_cp_reasoning.append(
                f"I can add up to {can_add} more CPs "
                f"(current: {petal.genesis.cp_count}, max: {petal.genesis.max_viable_cps})"
            )
            add_cp_reasoning.append(
                "Adding CPs increases my detail but changes my identity"
            )
            add_cp_reasoning.append(
                "WARNING: This is irreversible - I will become a different petal"
            )

            capabilities.append(
                TransformationCapability(
                    name="add_cp",
                    description=f"Add control point (can add {can_add} more)",
                    parameters={"position": (0.0, 1.0)},
                    risk_level=0.7,
                    reversible=False,
                    reasoning=add_cp_reasoning,
                )
            )

        # REMOVE CP
        remove_cp_reasoning = []
        if petal.genesis.cp_count > petal.genesis.min_viable_cps:
            can_remove = petal.genesis.cp_count - petal.genesis.min_viable_cps
            remove_cp_reasoning.append(
                f"I can remove up to {can_remove} CPs "
                f"(current: {petal.genesis.cp_count}, min: {petal.genesis.min_viable_cps})"
            )
            remove_cp_reasoning.append(
                "Removing CPs simplifies my form but loses detail"
            )
            remove_cp_reasoning.append(
                "WARNING: This permanently changes my structure"
            )

            capabilities.append(
                TransformationCapability(
                    name="remove_cp",
                    description=f"Remove control point (can remove {can_remove})",
                    parameters={"cp_index": (1, petal.genesis.cp_count - 1)},
                    risk_level=0.8,
                    reversible=False,
                    reasoning=remove_cp_reasoning,
                )
            )

        # MORPH TO LEAF
        morph_reasoning = []
        morph_reasoning.append(
            "I can morph towards leaf shape by adjusting my proportions"
        )
        morph_reasoning.append(
            "Morphing factor: 0.0 = pure petal, 1.0 = pure leaf"
        )

        # Check compatibility
        if petal.genesis.aspect_ratio > 2.0:
            morph_reasoning.append(
                f"My aspect ratio ({petal.genesis.aspect_ratio:.2f}) is compatible "
                "with leaf morphing"
            )
            risk = 0.7
        else:
            morph_reasoning.append(
                f"My aspect ratio ({petal.genesis.aspect_ratio:.2f}) requires "
                "significant change for leaf form"
            )
            risk = 0.9

        morph_reasoning.append(
            "WARNING: Full morph changes my identity from petal to leaf"
        )

        capabilities.append(
            TransformationCapability(
                name="morph_to_leaf",
                description="Transform shape towards leaf",
                parameters={"morph_factor": (0.0, 1.0)},
                risk_level=risk,
                reversible=False,
                reasoning=morph_reasoning,
            )
        )

        return capabilities

    def _calculate_morph_confidence(
        self,
        petal: SelfAwarePetal,
        capabilities: List[TransformationCapability]
    ) -> float:
        """Calculate overall confidence in morphing abilities."""
        if not capabilities:
            return 0.5

        # Average of (1 - risk) for all capabilities
        total_safety = sum(1 - cap.risk_level for cap in capabilities)
        avg_safety = total_safety / len(capabilities)

        # Bonus for structural confidence
        structural_bonus = petal.genesis.structural_confidence * 0.1

        return min(1.0, avg_safety + structural_bonus)

    def generate_transformation_report(self, petal: SelfAwarePetal) -> str:
        """Generate a report of transformation capabilities."""
        if petal.transformation is None:
            petal.transformation = self.reason_about_transformations(petal)

        trans = petal.transformation
        report = []

        report.append("=== TRANSFORMATION CAPABILITIES ===\n")

        # Summary
        report.append("WHAT I CAN DO:")
        for cap in trans.capabilities:
            risk_desc = "safe" if cap.risk_level < 0.3 else (
                "moderate" if cap.risk_level < 0.6 else "risky"
            )
            reversible_desc = "reversible" if cap.reversible else "PERMANENT"
            report.append(
                f"  • {cap.name}: {cap.description} [{risk_desc}, {reversible_desc}]"
            )
        report.append("")

        # Animation reasoning
        report.append("HOW I CAN ANIMATE:")
        for line in trans.animate_reasoning:
            report.append(f"  • {line}")
        report.append("")

        # Scale reasoning
        report.append("HOW I CAN SCALE:")
        for line in trans.scale_reasoning:
            report.append(f"  • {line}")
        report.append("")

        # Deform reasoning
        report.append("HOW I CAN DEFORM:")
        for line in trans.deform_reasoning:
            report.append(f"  • {line}")
        report.append("")

        # Morph reasoning
        report.append("HOW I CAN MORPH (IDENTITY CHANGE):")
        for line in trans.morph_reasoning:
            report.append(f"  • {line}")
        report.append("")

        # Confidence
        report.append("MY TRANSFORMATION CONFIDENCE:")
        report.append(f"  • Morph confidence: {trans.morph_confidence:.1%}")
        report.append(f"  • Structural stability: {trans.structural_stability:.1%}")

        return "\n".join(report)


class CompositionReasoner:
    """
    Reasons about how petals compose with other objects.

    Enables petals to understand:
    - Their relationships with other petals
    - How they fit into larger structures (flowers)
    - Coordination and synchronization with neighbors
    """

    def __init__(self):
        """Initialize composition reasoner."""
        self.genesis_reasoner = GenesisReasoner()
        self.golden_angle = 137.5  # Fibonacci spiral angle

        # Object type knowledge
        self.object_relationships = {
            "petal": {
                "same_layer": "sibling",
                "different_layer": "cousin",
                "adjacent": "neighbor",
            },
            "stem": "parent",
            "leaf": "sibling",
            "center": "reference_point",
        }

        # Spatial rules
        self.layer_spacing = {
            1: 0.2,  # Inner layer close to center
            2: 0.5,  # Middle layer
            3: 0.8,  # Outer layer far from center
        }

    def reason_about_composition(
        self,
        petal: SelfAwarePetal,
        other_petals: List[SelfAwarePetal] = None,
        group_config: Dict[str, Any] = None,
    ) -> CompositionKnowledge:
        """
        Generate composition knowledge for a petal.

        Args:
            petal: The petal to reason about
            other_petals: Other petals in the composition
            group_config: Configuration of the flower/group
        """
        if petal.genesis is None:
            petal.genesis = self.genesis_reasoner.reason_about_genesis(petal)

        if other_petals is None:
            other_petals = []

        if group_config is None:
            group_config = {
                "type": "rose",
                "petals_per_layer": 5,
                "num_layers": 3,
            }

        relationships = []
        coordination_reasoning = []
        synchronization_reasoning = []
        hierarchy_reasoning = []

        # === SPATIAL POSITIONING ===
        position_in_spiral = petal.layer * group_config["petals_per_layer"] + petal.position_in_layer
        angular_position = petal.position_in_layer * self.golden_angle
        radial_distance = self.layer_spacing.get(petal.layer, 0.5)

        # === RELATIONSHIP REASONING ===

        # Reason about center relationship
        center_rel = self._reason_center_relationship(petal, radial_distance)
        relationships.append(center_rel)
        hierarchy_reasoning.extend(center_rel.reasoning)

        # Reason about stem relationship
        stem_rel = self._reason_stem_relationship(petal)
        relationships.append(stem_rel)
        hierarchy_reasoning.extend(stem_rel.reasoning)

        # Reason about relationships with other petals
        for other in other_petals:
            if other.name != petal.name:
                petal_rel = self._reason_petal_relationship(petal, other)
                relationships.append(petal_rel)
                coordination_reasoning.extend(petal_rel.reasoning)

        # If no other petals provided, reason about hypothetical neighbors
        if not other_petals:
            hypothetical_rels = self._reason_hypothetical_neighbors(petal, group_config)
            relationships.extend(hypothetical_rels)
            for rel in hypothetical_rels:
                coordination_reasoning.extend(rel.reasoning)

        # === SYNCHRONIZATION REASONING ===
        sync_reasoning = self._reason_synchronization(petal, other_petals, group_config)
        synchronization_reasoning.extend(sync_reasoning)

        # === PATTERN REASONING ===
        available_patterns = ["spiral", "radial", "layered", "clustered"]
        current_pattern = "spiral"  # Default for rose

        if petal.layer == 1:
            coordination_reasoning.append(
                "As inner layer, I form the protective core of the spiral"
            )
        elif petal.layer == 2:
            coordination_reasoning.append(
                "As middle layer, I bridge inner and outer petals in the spiral"
            )
        else:
            coordination_reasoning.append(
                "As outer layer, I complete the visual spiral pattern"
            )

        # === COLLISION AVOIDANCE ===
        collision_avoidance = self._calculate_collision_avoidance(petal, group_config)

        # === OVERLAP RULES ===
        overlap_rules = {
            "petal_same_layer": True,  # Can partially overlap
            "petal_different_layer": True,  # Natural layering
            "stem": False,  # Should not overlap
            "center": False,  # Should not overlap
            "leaf": True,  # Can overlap slightly
        }

        # === HARMONY SCORE ===
        harmony_score = self._calculate_harmony_score(petal, group_config)
        cooperation_confidence = self._calculate_cooperation_confidence(
            petal, relationships
        )

        # === GROUP IDENTITY ===
        group_id = f"rose_flower_{group_config.get('id', 1)}"
        group_role = f"petal_layer_{petal.layer}_pos_{petal.position_in_layer}"

        return CompositionKnowledge(
            relationships=relationships,
            group_id=group_id,
            group_role=group_role,
            position_in_spiral=position_in_spiral,
            angular_position=angular_position,
            radial_distance=radial_distance,
            coordination_reasoning=coordination_reasoning,
            synchronization_reasoning=synchronization_reasoning,
            hierarchy_reasoning=hierarchy_reasoning,
            available_patterns=available_patterns,
            current_pattern=current_pattern,
            collision_avoidance=collision_avoidance,
            overlap_rules=overlap_rules,
            harmony_score=harmony_score,
            cooperation_confidence=cooperation_confidence,
        )

    def _reason_center_relationship(
        self, petal: SelfAwarePetal, radial_distance: float
    ) -> RelationshipInfo:
        """Reason about relationship to flower center."""
        reasoning = []

        reasoning.append(
            f"I am positioned {radial_distance:.2f} units from the center"
        )

        if petal.layer == 1:
            spatial = "inside"
            reasoning.append(
                "As inner layer, I am closest to and protect the center"
            )
        elif petal.layer == 3:
            spatial = "outside"
            reasoning.append(
                "As outer layer, I am farthest from center, providing visibility"
            )
        else:
            spatial = "adjacent"
            reasoning.append(
                "As middle layer, I surround the inner petals"
            )

        return RelationshipInfo(
            other_id="flower_center",
            other_type="center",
            relationship_type="child",
            spatial_relation=spatial,
            reasoning=reasoning,
        )

    def _reason_stem_relationship(self, petal: SelfAwarePetal) -> RelationshipInfo:
        """Reason about relationship to stem."""
        reasoning = []

        reasoning.append(
            "The stem is my structural parent - it provides support"
        )
        reasoning.append(
            f"My base attaches to the stem's top via bone hierarchy"
        )

        if petal.layer == 1:
            reasoning.append(
                "As inner layer, my attachment is most secure and direct"
            )
        else:
            reasoning.append(
                f"As layer {petal.layer}, I attach through the bone chain"
            )

        return RelationshipInfo(
            other_id="main_stem",
            other_type="stem",
            relationship_type="parent",
            spatial_relation="above",
            reasoning=reasoning,
        )

    def _reason_petal_relationship(
        self, petal: SelfAwarePetal, other: SelfAwarePetal
    ) -> RelationshipInfo:
        """Reason about relationship to another petal."""
        reasoning = []

        if petal.layer == other.layer:
            relationship = "sibling"
            reasoning.append(
                f"'{other.name}' is my sibling in layer {petal.layer}"
            )

            # Position difference
            pos_diff = abs(petal.position_in_layer - other.position_in_layer)
            if pos_diff == 1 or pos_diff >= 4:  # Adjacent in circular arrangement
                spatial = "adjacent"
                reasoning.append(
                    f"We are adjacent in the spiral (position difference: {pos_diff})"
                )
            else:
                spatial = "same_layer"
                reasoning.append(
                    f"We are separated by {pos_diff-1} petals in our layer"
                )

            # Synchronization potential
            reasoning.append(
                "We should bloom and animate in coordination for visual harmony"
            )

        else:
            relationship = "cousin"
            if petal.layer < other.layer:
                spatial = "inside"
                reasoning.append(
                    f"'{other.name}' is in outer layer {other.layer}, I am inner"
                )
                reasoning.append(
                    "I may be partially covered by outer layer petals"
                )
            else:
                spatial = "outside"
                reasoning.append(
                    f"'{other.name}' is in inner layer {other.layer}, I am outer"
                )
                reasoning.append(
                    "I provide visual coverage for inner layer petals"
                )

        return RelationshipInfo(
            other_id=other.name,
            other_type="petal",
            relationship_type=relationship,
            spatial_relation=spatial,
            reasoning=reasoning,
        )

    def _reason_hypothetical_neighbors(
        self, petal: SelfAwarePetal, group_config: Dict[str, Any]
    ) -> List[RelationshipInfo]:
        """Reason about neighbors that should exist."""
        relationships = []
        petals_per_layer = group_config.get("petals_per_layer", 5)

        # Previous neighbor in same layer
        prev_pos = (petal.position_in_layer - 1) % petals_per_layer
        prev_name = f"petal_L{petal.layer}_P{prev_pos}"

        prev_reasoning = [
            f"'{prev_name}' should be my previous sibling in layer {petal.layer}",
            f"It is positioned at {prev_pos * self.golden_angle:.1f}° in the spiral",
            "We should coordinate our bloom timing for smooth progression",
        ]

        relationships.append(
            RelationshipInfo(
                other_id=prev_name,
                other_type="petal",
                relationship_type="sibling",
                spatial_relation="adjacent",
                reasoning=prev_reasoning,
            )
        )

        # Next neighbor in same layer
        next_pos = (petal.position_in_layer + 1) % petals_per_layer
        next_name = f"petal_L{petal.layer}_P{next_pos}"

        next_reasoning = [
            f"'{next_name}' should be my next sibling in layer {petal.layer}",
            f"It is positioned at {next_pos * self.golden_angle:.1f}° in the spiral",
            "Our tips should not collide during bloom animation",
        ]

        relationships.append(
            RelationshipInfo(
                other_id=next_name,
                other_type="petal",
                relationship_type="sibling",
                spatial_relation="adjacent",
                reasoning=next_reasoning,
            )
        )

        # Petal in different layer (if exists)
        if petal.layer > 1:
            inner_name = f"petal_L{petal.layer-1}_P{petal.position_in_layer}"
            inner_reasoning = [
                f"'{inner_name}' is my inner layer counterpart",
                "I should cover part of it when fully bloomed",
                "Our bloom timings should be staggered for natural appearance",
            ]

            relationships.append(
                RelationshipInfo(
                    other_id=inner_name,
                    other_type="petal",
                    relationship_type="cousin",
                    spatial_relation="outside",
                    reasoning=inner_reasoning,
                )
            )

        if petal.layer < group_config.get("num_layers", 3):
            outer_name = f"petal_L{petal.layer+1}_P{petal.position_in_layer}"
            outer_reasoning = [
                f"'{outer_name}' is my outer layer counterpart",
                "It will partially cover me when fully bloomed",
                "I should complete my bloom before it starts",
            ]

            relationships.append(
                RelationshipInfo(
                    other_id=outer_name,
                    other_type="petal",
                    relationship_type="cousin",
                    spatial_relation="inside",
                    reasoning=outer_reasoning,
                )
            )

        return relationships

    def _reason_synchronization(
        self,
        petal: SelfAwarePetal,
        other_petals: List[SelfAwarePetal],
        group_config: Dict[str, Any],
    ) -> List[str]:
        """Reason about synchronization with group."""
        reasoning = []

        # Bloom synchronization
        if petal.layer == 1:
            reasoning.append(
                "I should bloom first (inner layer) to protect the center"
            )
        elif petal.layer == 2:
            reasoning.append(
                "I should bloom after inner layer but before outer"
            )
        else:
            reasoning.append(
                "I should bloom last (outer layer) for maximum visual effect"
            )

        # Position-based timing
        delay_ms = petal.position_in_layer * 200  # 200ms delay per position
        reasoning.append(
            f"My bloom should start after {delay_ms}ms delay "
            f"(position {petal.position_in_layer} in layer)"
        )

        # Wind animation coordination
        reasoning.append(
            "Wind animation should have slight phase offset from neighbors"
        )
        phase_offset = petal.position_in_layer * 0.2  # 0.2 radians per position
        reasoning.append(
            f"My wind phase offset: {phase_offset:.2f} radians"
        )

        # Group coordination
        reasoning.append(
            f"Total petals in my layer: {group_config.get('petals_per_layer', 5)}"
        )
        reasoning.append(
            f"Total layers in flower: {group_config.get('num_layers', 3)}"
        )

        return reasoning

    def _calculate_collision_avoidance(
        self, petal: SelfAwarePetal, group_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate minimum distances to avoid collisions."""
        collision_avoidance = {}

        # Same layer petals - based on width
        same_layer_dist = petal.width * 0.3  # Allow 30% overlap
        collision_avoidance["petal_same_layer"] = same_layer_dist

        # Different layer petals - minimal distance (natural overlap)
        collision_avoidance["petal_different_layer"] = petal.width * 0.1

        # Stem - must not touch
        collision_avoidance["stem"] = petal.width * 0.5

        # Center - based on layer
        if petal.layer == 1:
            collision_avoidance["center"] = petal.height * 0.2
        else:
            collision_avoidance["center"] = petal.height * 0.5

        return collision_avoidance

    def _calculate_harmony_score(
        self, petal: SelfAwarePetal, group_config: Dict[str, Any]
    ) -> float:
        """Calculate how well petal fits in composition."""
        score = 0.7  # Base score

        # Layer-appropriate size bonus
        if petal.layer == 1 and petal.width < 0.4:
            score += 0.1  # Inner should be small
        elif petal.layer == 3 and petal.width > 0.5:
            score += 0.1  # Outer should be large

        # Position alignment bonus
        if petal.position_in_layer < group_config.get("petals_per_layer", 5):
            score += 0.05  # Valid position

        # Opening degree harmony
        if petal.layer == 1 and petal.opening_degree < 0.7:
            score += 0.05  # Inner less open
        elif petal.layer == 3 and petal.opening_degree > 0.7:
            score += 0.05  # Outer more open

        return min(1.0, score)

    def _calculate_cooperation_confidence(
        self,
        petal: SelfAwarePetal,
        relationships: List[RelationshipInfo],
    ) -> float:
        """Calculate confidence in group cooperation."""
        if not relationships:
            return 0.5

        # More relationships = better understanding
        relationship_bonus = min(0.3, len(relationships) * 0.05)

        # Genesis understanding contributes
        if petal.genesis:
            genesis_bonus = petal.genesis.self_understanding * 0.2
        else:
            genesis_bonus = 0.0

        # Base confidence
        base_confidence = 0.5

        return min(1.0, base_confidence + relationship_bonus + genesis_bonus)

    def generate_composition_report(
        self,
        petal: SelfAwarePetal,
        other_petals: List[SelfAwarePetal] = None,
        group_config: Dict[str, Any] = None,
    ) -> str:
        """Generate a report of composition knowledge."""
        if petal.composition is None:
            petal.composition = self.reason_about_composition(
                petal, other_petals, group_config
            )

        comp = petal.composition
        report = []

        report.append("=== COMPOSITION KNOWLEDGE ===\n")

        # Group identity
        report.append("MY PLACE IN THE FLOWER:")
        report.append(f"  • Group: {comp.group_id}")
        report.append(f"  • Role: {comp.group_role}")
        report.append(f"  • Position in spiral: #{comp.position_in_spiral}")
        report.append(f"  • Angular position: {comp.angular_position:.1f}°")
        report.append(f"  • Distance from center: {comp.radial_distance:.2f}")
        report.append("")

        # Relationships
        report.append("MY RELATIONSHIPS:")
        for rel in comp.relationships:
            report.append(
                f"  • {rel.other_id} ({rel.other_type}): "
                f"{rel.relationship_type}, {rel.spatial_relation}"
            )
            for reason in rel.reasoning[:2]:  # Show first 2 reasons
                report.append(f"      - {reason}")
        report.append("")

        # Coordination
        report.append("HOW I COORDINATE:")
        for line in comp.coordination_reasoning[:5]:
            report.append(f"  • {line}")
        report.append("")

        # Synchronization
        report.append("HOW I SYNCHRONIZE:")
        for line in comp.synchronization_reasoning[:4]:
            report.append(f"  • {line}")
        report.append("")

        # Hierarchy
        report.append("MY HIERARCHY:")
        for line in comp.hierarchy_reasoning[:4]:
            report.append(f"  • {line}")
        report.append("")

        # Collision rules
        report.append("COLLISION AVOIDANCE:")
        for obj_type, min_dist in comp.collision_avoidance.items():
            report.append(f"  • Min distance to {obj_type}: {min_dist:.3f}")
        report.append("")

        # Harmony
        report.append("GROUP HARMONY:")
        report.append(f"  • Harmony score: {comp.harmony_score:.1%}")
        report.append(f"  • Cooperation confidence: {comp.cooperation_confidence:.1%}")
        report.append(f"  • Current pattern: {comp.current_pattern}")

        return "\n".join(report)


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


def demo_transformation_reasoning():
    """Demonstrate transformation reasoning for different petals."""

    genesis_reasoner = GenesisReasoner()
    trans_reasoner = TransformationReasoner()

    # Create petals with different configurations
    test_cases = [
        ("petal_L1_P0", 1, 0, 0.3, 0.9, 0.6, "low"),
        ("petal_L2_P1", 2, 1, 0.4, 1.2, 0.8, "medium"),
        ("petal_L3_P2", 3, 2, 0.6, 1.5, 0.9, "high"),
    ]

    for name, layer, pos, w, h, opening, detail in test_cases:
        print("=" * 70)
        petal = genesis_reasoner.create_aware_petal(name, layer, pos, w, h, opening, detail)
        trans_report = trans_reasoner.generate_transformation_report(petal)
        print(f"PETAL: {name}")
        print(trans_report)
        print()


def demo_composition_reasoning():
    """Demonstrate composition reasoning for petals."""

    genesis_reasoner = GenesisReasoner()
    comp_reasoner = CompositionReasoner()

    # Create petals with different configurations
    test_cases = [
        ("petal_L1_P0", 1, 0, 0.3, 0.9, 0.6, "low"),
        ("petal_L2_P2", 2, 2, 0.4, 1.2, 0.8, "medium"),
        ("petal_L3_P4", 3, 4, 0.6, 1.5, 0.9, "high"),
    ]

    for name, layer, pos, w, h, opening, detail in test_cases:
        print("=" * 70)
        petal = genesis_reasoner.create_aware_petal(name, layer, pos, w, h, opening, detail)
        comp_report = comp_reasoner.generate_composition_report(petal)
        print(f"PETAL: {name}")
        print(comp_report)
        print()


def demo_multi_petal_composition():
    """Demonstrate composition with multiple petals interacting."""

    genesis_reasoner = GenesisReasoner()
    comp_reasoner = CompositionReasoner()

    # Create multiple petals
    petals = []
    for layer in range(1, 4):
        for pos in range(3):  # 3 petals per layer for demo
            petal = genesis_reasoner.create_aware_petal(
                f"petal_L{layer}_P{pos}",
                layer,
                pos,
                0.3 + layer * 0.1,
                0.8 + layer * 0.3,
                0.5 + layer * 0.15,
                ["low", "medium", "high"][layer - 1],
            )
            petals.append(petal)

    print("=" * 70)
    print("MULTI-PETAL COMPOSITION DEMONSTRATION")
    print("=" * 70)
    print(f"Total petals created: {len(petals)}")
    print()

    # Show composition for middle layer petal with awareness of others
    target_petal = petals[4]  # Layer 2, position 1
    print(f"Analyzing: {target_petal.name}")
    print()

    comp_report = comp_reasoner.generate_composition_report(
        target_petal,
        other_petals=petals,
        group_config={
            "type": "rose",
            "petals_per_layer": 3,
            "num_layers": 3,
            "id": 1,
        },
    )
    print(comp_report)


def demo_full_self_awareness():
    """Demonstrate complete self-awareness (genesis + transformation + composition)."""

    genesis_reasoner = GenesisReasoner()
    trans_reasoner = TransformationReasoner()
    comp_reasoner = CompositionReasoner()

    # Create a single petal with full awareness
    petal = genesis_reasoner.create_aware_petal(
        "petal_L2_P0", 2, 0, 0.4, 1.2, 0.8, "medium"
    )

    print("=" * 70)
    print("COMPLETE PETAL SELF-AWARENESS DEMONSTRATION")
    print("=" * 70)
    print()

    # Genesis report
    genesis_report = genesis_reasoner.generate_self_description(petal)
    print(genesis_report)
    print()

    # Transformation report
    trans_report = trans_reasoner.generate_transformation_report(petal)
    print(trans_report)
    print()

    # Composition report
    comp_report = comp_reasoner.generate_composition_report(petal)
    print(comp_report)
    print()

    # Summary
    print("=" * 70)
    print("COMPLETE SELF-AWARENESS SUMMARY")
    print("=" * 70)
    print(f"Petal: {petal.name}")
    print()
    print("GENESIS:")
    print(f"  • Self Understanding: {petal.genesis.self_understanding:.1%}")
    print(f"  • Structural Confidence: {petal.genesis.structural_confidence:.1%}")
    print(f"  • Control Points: {petal.genesis.cp_count}")
    print()
    print("TRANSFORMATION:")
    print(f"  • Morph Confidence: {petal.transformation.morph_confidence:.1%}")
    print(f"  • Total Capabilities: {len(petal.transformation.capabilities)}")
    print()
    print("COMPOSITION:")
    print(f"  • Harmony Score: {petal.composition.harmony_score:.1%}")
    print(f"  • Cooperation Confidence: {petal.composition.cooperation_confidence:.1%}")
    print(f"  • Relationships: {len(petal.composition.relationships)}")
    print(f"  • Pattern: {petal.composition.current_pattern}")
    print()

    # Overall awareness score
    overall_awareness = (
        petal.genesis.self_understanding * 0.3
        + petal.transformation.morph_confidence * 0.3
        + petal.composition.harmony_score * 0.2
        + petal.composition.cooperation_confidence * 0.2
    )
    print(f"OVERALL SELF-AWARENESS: {overall_awareness:.1%}")
    print()

    # Thought history
    print("THOUGHT HISTORY:")
    for thought in petal.thought_history:
        print(f"  • {thought}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "genesis":
            demo_genesis_reasoning()
        elif sys.argv[1] == "transform":
            demo_transformation_reasoning()
        elif sys.argv[1] == "compose":
            demo_composition_reasoning()
        elif sys.argv[1] == "multi":
            demo_multi_petal_composition()
        elif sys.argv[1] == "full":
            demo_full_self_awareness()
        else:
            print("Usage: python petal_self_awareness.py [genesis|transform|compose|multi|full]")
    else:
        # Default: show full demo
        demo_full_self_awareness()
