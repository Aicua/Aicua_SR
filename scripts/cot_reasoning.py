#!/usr/bin/env python3
"""
Chain-of-Thought (CoT) Reasoning for Dynamic Control Point Decision.

Analyzes shape requirements and determines optimal number of control points
for spline generation.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


@dataclass
class ShapeAnalysis:
    """Result of shape analysis."""
    object_type: str
    symmetry: str  # "none", "bilateral", "radial"
    curvature: str  # "angular", "smooth", "complex"
    detail_level: str  # "low", "medium", "high"
    organic: bool
    closed: bool


@dataclass
class CPDecision:
    """Control point decision with reasoning."""
    cp_count: int
    reasoning: List[str]
    confidence: float


class CoTReasoner:
    """Chain-of-Thought reasoner for spline control points."""

    def __init__(self):
        """Initialize reasoning rules."""
        self.shape_patterns = {
            # Object type -> (min_cp, max_cp, preferred_cp)
            "triangle": (3, 4, 3),
            "rectangle": (4, 6, 4),
            "pentagon": (5, 7, 5),
            "hexagon": (6, 8, 6),
            "circle": (8, 16, 12),
            "ellipse": (6, 12, 8),
            "wing": (3, 6, 4),
            "petal": (13, 15, 15),  # Updated: 15 CPs for realistic rose petal shape (V3)
            "leaf": (5, 9, 6),
            "heart": (6, 10, 7),
            "star": (10, 20, 10),
            "custom": (3, 20, 5),
        }

    def analyze_shape(
        self,
        object_name: str,
        width: float,
        height: float,
        symmetry_required: bool = True,
        smooth_curves: bool = True,
        detail_level: str = "medium",
    ) -> ShapeAnalysis:
        """
        Analyze shape requirements from parameters.

        Args:
            object_name: Name/type of object
            width: Object width
            height: Object height
            symmetry_required: Whether object should be symmetric
            smooth_curves: Whether curves should be smooth
            detail_level: "low", "medium", "high"

        Returns:
            ShapeAnalysis with detected properties
        """
        # Detect object type from name
        object_type = "custom"
        name_lower = object_name.lower()

        for pattern in self.shape_patterns.keys():
            if pattern in name_lower:
                object_type = pattern
                break

        # Determine symmetry
        if symmetry_required:
            if "radial" in name_lower or object_type in ["circle", "star"]:
                symmetry = "radial"
            else:
                symmetry = "bilateral"
        else:
            symmetry = "none"

        # Determine curvature
        if smooth_curves:
            if detail_level == "high":
                curvature = "complex"
            else:
                curvature = "smooth"
        else:
            curvature = "angular"

        # Organic detection
        organic = object_type in ["petal", "leaf", "wing", "heart"]

        return ShapeAnalysis(
            object_type=object_type,
            symmetry=symmetry,
            curvature=curvature,
            detail_level=detail_level,
            organic=organic,
            closed=True,  # All our splines are closed
        )

    def decide_cp_count(self, analysis: ShapeAnalysis) -> CPDecision:
        """
        Decide optimal control point count based on shape analysis.

        Uses Chain-of-Thought reasoning to explain decision.

        Args:
            analysis: ShapeAnalysis result

        Returns:
            CPDecision with count, reasoning, and confidence
        """
        reasoning = []
        cp_count = 5  # Default
        confidence = 0.8

        # Step 1: Base count from object type
        min_cp, max_cp, preferred_cp = self.shape_patterns.get(
            analysis.object_type, (3, 20, 5)
        )

        reasoning.append(
            f"Step 1: Object type '{analysis.object_type}' suggests {preferred_cp} CPs (range: {min_cp}-{max_cp})"
        )
        cp_count = preferred_cp

        # Step 2: Adjust for symmetry
        if analysis.symmetry == "bilateral":
            # Bilateral symmetry benefits from odd number of CPs (center point)
            if cp_count % 2 == 0:
                cp_count += 1
            reasoning.append(
                f"Step 2: Bilateral symmetry → prefer odd CP count → {cp_count} CPs"
            )
        elif analysis.symmetry == "radial":
            # Radial symmetry benefits from higher CP count
            cp_count = max(cp_count, 8)
            reasoning.append(
                f"Step 2: Radial symmetry → need more CPs → {cp_count} CPs"
            )
        else:
            reasoning.append(f"Step 2: No symmetry constraint → keep {cp_count} CPs")

        # Step 3: Adjust for curvature
        if analysis.curvature == "angular":
            # Angular shapes need fewer CPs
            cp_count = max(min_cp, cp_count - 2)
            reasoning.append(
                f"Step 3: Angular curvature → reduce CPs → {cp_count} CPs"
            )
        elif analysis.curvature == "complex":
            # Complex curves need more CPs
            cp_count = min(max_cp, cp_count + 2)
            reasoning.append(
                f"Step 3: Complex curvature → increase CPs → {cp_count} CPs"
            )
        else:
            reasoning.append(f"Step 3: Smooth curvature → keep {cp_count} CPs")

        # Step 4: Adjust for detail level
        if analysis.detail_level == "low":
            cp_count = max(min_cp, cp_count - 1)
            confidence *= 0.9
            reasoning.append(
                f"Step 4: Low detail → simplify → {cp_count} CPs"
            )
        elif analysis.detail_level == "high":
            cp_count = min(max_cp, cp_count + 1)
            reasoning.append(
                f"Step 4: High detail → add precision → {cp_count} CPs"
            )
        else:
            reasoning.append(f"Step 4: Medium detail → keep {cp_count} CPs")

        # Step 5: Organic adjustment
        if analysis.organic:
            # Organic shapes benefit from curve control points
            if cp_count < 5:
                cp_count = 5
            reasoning.append(
                f"Step 5: Organic shape → ensure curve control → {cp_count} CPs"
            )
        else:
            reasoning.append(f"Step 5: Non-organic → keep {cp_count} CPs")

        # Final validation
        cp_count = max(min_cp, min(max_cp, cp_count))
        reasoning.append(
            f"Final: Optimal CP count = {cp_count} (+ 1 close point = {cp_count + 1} total values)"
        )

        return CPDecision(
            cp_count=cp_count, reasoning=reasoning, confidence=confidence
        )

    def generate_cp_positions(
        self,
        cp_count: int,
        width: float,
        height: float,
        symmetry: str = "bilateral",
        tip_offset: float = 0.0,
    ) -> List[Tuple[float, float]]:
        """
        Generate control point positions for given CP count.

        Args:
            cp_count: Number of control points (not including close point)
            width: Total width
            height: Total height
            symmetry: "bilateral" or "radial" or "none"
            tip_offset: X offset for tip (for asymmetric petals)

        Returns:
            List of (x, y) tuples for control points
        """
        if symmetry == "bilateral" and cp_count >= 3:
            return self._generate_bilateral_cps(cp_count, width, height, tip_offset)
        elif symmetry == "radial":
            return self._generate_radial_cps(cp_count, width, height)
        else:
            return self._generate_custom_cps(cp_count, width, height)

    def _generate_bilateral_cps(
        self, cp_count: int, width: float, height: float, tip_offset: float = 0.0
    ) -> List[Tuple[float, float]]:
        """Generate bilaterally symmetric control points."""
        cps = []
        half_width = width / 2

        if cp_count == 3:
            # Triangle: base_left, tip, base_right
            cps = [
                (-half_width, 0.0),
                (tip_offset, height),
                (half_width, 0.0),
            ]
        elif cp_count == 4:
            # Quadrilateral with peak
            cps = [
                (-half_width, 0.0),
                (-half_width * 0.3, height * 0.6),
                (tip_offset, height),
                (half_width, 0.0),
            ]
        elif cp_count == 5:
            # Simple petal shape
            cps = [
                (-half_width / 2, 0.0),  # base_left (narrow)
                (-half_width, height * 0.4),  # left_curve (WIDEST)
                (tip_offset, height),  # tip (pointed)
                (half_width, height * 0.4),  # right_curve (WIDEST)
                (half_width / 2, 0.0),  # base_right (narrow)
            ]
        elif cp_count == 6:
            # More detailed petal/leaf
            cps = [
                (-half_width, 0.0),
                (-half_width * 0.8, height * 0.25),
                (-half_width * 0.4, height * 0.7),
                (tip_offset, height),
                (half_width * 0.4, height * 0.7),
                (half_width, 0.0),
            ]
        elif cp_count == 7:
            # Complex organic shape with rounded tip
            cps = [
                (-half_width / 2, 0.0),  # base_left
                (-half_width * 0.7, height * 0.25),  # lower_left
                (-half_width * 0.5, height * 0.6),  # upper_left
                (tip_offset, height),  # tip
                (half_width * 0.5, height * 0.6),  # upper_right
                (half_width * 0.7, height * 0.25),  # lower_right
                (half_width / 2, 0.0),  # base_right
            ]
        elif cp_count == 8:
            # Realistic petal V3 - heart-shaped tip with notch
            # Based on rose_petals.jpg reference
            notch_depth = height * 0.05
            notch_width = half_width * 0.3
            cps = [
                (-half_width / 2, 0.0),  # CP1: base_left
                (-half_width * 0.7, height * 0.25),  # CP2: lower_left
                (-half_width * 0.5, height * 0.6),  # CP3: upper_left
                (-notch_width, height - notch_depth),  # CP4: tip_left
                (tip_offset, height),  # CP5: tip_center
                (notch_width, height - notch_depth),  # CP6: tip_right
                (half_width * 0.5, height * 0.6),  # CP7: upper_right
                (half_width * 0.7, height * 0.25),  # CP8: lower_right
            ]
        elif cp_count == 9:
            # Extra detailed petal with more edge control
            notch_depth = height * 0.05
            notch_width = half_width * 0.25
            cps = [
                (-half_width / 2, 0.0),  # base_left
                (-half_width * 0.65, height * 0.15),  # lower_left_1
                (-half_width * 0.7, height * 0.35),  # lower_left_2
                (-half_width * 0.5, height * 0.65),  # upper_left
                (tip_offset, height),  # tip
                (half_width * 0.5, height * 0.65),  # upper_right
                (half_width * 0.7, height * 0.35),  # lower_right_2
                (half_width * 0.65, height * 0.15),  # lower_right_1
                (half_width / 2, 0.0),  # base_right
            ]
        elif cp_count == 10:
            # Maximum detail petal
            notch_depth = height * 0.05
            notch_width = half_width * 0.2
            cps = [
                (-half_width / 2, 0.0),  # base_left
                (-half_width * 0.6, height * 0.12),  # lower_left_1
                (-half_width * 0.7, height * 0.3),  # lower_left_2
                (-half_width * 0.55, height * 0.55),  # mid_left
                (-notch_width, height - notch_depth),  # tip_left
                (tip_offset, height),  # tip_center
                (notch_width, height - notch_depth),  # tip_right
                (half_width * 0.55, height * 0.55),  # mid_right
                (half_width * 0.7, height * 0.3),  # lower_right_2
                (half_width / 2, 0.0),  # base_right
            ]
        elif cp_count == 11:
            # Hemispherical rose petal - V3 dome-like distribution
            # Y values follow sin curve for realistic rose petal shape
            # Angles: 0°, 15°, 35°, 60°, 80°, 90° (and symmetric)
            angles = [0, 15, 35, 60, 80, 90, 80, 60, 35, 15, 0]
            y_factors = [math.sin(math.radians(a)) for a in angles]

            # X spread: narrow at base, widest at middle height, narrow at tip
            x_factors = [0.25, 0.45, 0.65, 0.85, 0.6, 0.0, 0.6, 0.85, 0.65, 0.45, 0.25]

            cps = [
                (-half_width * x_factors[0], height * y_factors[0]),   # base_left
                (-half_width * x_factors[1], height * y_factors[1]),   # lower_left
                (-half_width * x_factors[2], height * y_factors[2]),   # mid_lower_left
                (-half_width * x_factors[3], height * y_factors[3]),   # mid_upper_left
                (-half_width * x_factors[4] + tip_offset, height * y_factors[4]),  # near_tip_left
                (tip_offset, height * y_factors[5]),                   # tip
                (half_width * x_factors[6] + tip_offset, height * y_factors[6]),   # near_tip_right
                (half_width * x_factors[7], height * y_factors[7]),    # mid_upper_right
                (half_width * x_factors[8], height * y_factors[8]),    # mid_lower_right
                (half_width * x_factors[9], height * y_factors[9]),    # lower_right
                (half_width * x_factors[10], height * y_factors[10]),  # base_right
            ]
        elif cp_count in [13, 14, 15]:
            # 15 CPs for realistic rose petal shape (V3)
            # Y positions (percentage of petal height)
            # cp1: 0%, cp2: 5%, cp3: 25%, cp4: 45%, cp5: 62% (WIDEST), cp6: 78%, cp7: 92%, cp8: 100% (tip)
            # cp9-cp14: mirror, cp15: close
            y_positions = [0.0, 0.05, 0.25, 0.45, 0.62, 0.78, 0.92, 1.0, 0.92, 0.78, 0.62, 0.45, 0.25, 0.05, 0.0]

            # X width factors (base narrow, widest at 62%, tip narrow)
            x_factors = [0.0, 0.15, 0.45, 0.70, 0.85, 0.65, 0.30, 0.0, 0.30, 0.65, 0.85, 0.70, 0.45, 0.15, 0.0]

            cps = [
                (0.0, height * y_positions[0]),                                      # CP1: base center
                (-half_width * x_factors[1], height * y_positions[1]),               # CP2: base left (5%)
                (-half_width * x_factors[2], height * y_positions[2]),               # CP3: lower left (25%)
                (-half_width * x_factors[3], height * y_positions[3]),               # CP4: mid-low left (45%)
                (-half_width * x_factors[4], height * y_positions[4]),               # CP5: upper-mid left (62% - WIDEST)
                (-half_width * x_factors[5] + tip_offset, height * y_positions[5]),  # CP6: upper left (78%)
                (-half_width * x_factors[6] + tip_offset, height * y_positions[6]),  # CP7: near-tip left (92%)
                (tip_offset, height * y_positions[7]),                               # CP8: tip center (100%)
                (half_width * x_factors[8] + tip_offset, height * y_positions[8]),   # CP9: near-tip right (92%)
                (half_width * x_factors[9] + tip_offset, height * y_positions[9]),   # CP10: upper right (78%)
                (half_width * x_factors[10], height * y_positions[10]),              # CP11: upper-mid right (62% - WIDEST)
                (half_width * x_factors[11], height * y_positions[11]),              # CP12: mid-low right (45%)
                (half_width * x_factors[12], height * y_positions[12]),              # CP13: lower right (25%)
                (half_width * x_factors[13], height * y_positions[13]),              # CP14: base right (5%)
                (0.0, height * y_positions[14]),                                     # CP15: close spline (back to base)
            ]
        else:
            # General case: distribute points along curve
            cps = []
            for i in range(cp_count):
                t = i / (cp_count - 1)  # 0 to 1
                if t <= 0.5:
                    # Left side going up
                    x = -half_width * (1 - 2 * t)
                    y = height * (2 * t)
                else:
                    # Right side going down
                    x = half_width * (2 * t - 1)
                    y = height * (2 - 2 * t)
                cps.append((x + tip_offset * (y / height), y))

        return cps

    def _generate_radial_cps(
        self, cp_count: int, width: float, height: float
    ) -> List[Tuple[float, float]]:
        """Generate radially symmetric control points (circle/ellipse)."""
        cps = []
        radius_x = width / 2
        radius_y = height / 2

        for i in range(cp_count):
            angle = 2 * math.pi * i / cp_count
            x = radius_x * math.cos(angle)
            y = radius_y * math.sin(angle)
            cps.append((x, y))

        return cps

    def _generate_custom_cps(
        self, cp_count: int, width: float, height: float
    ) -> List[Tuple[float, float]]:
        """Generate custom control points."""
        # Default to bilateral for now
        return self._generate_bilateral_cps(cp_count, width, height)

    def reason_and_generate(
        self,
        object_name: str,
        width: float,
        height: float,
        symmetry_required: bool = True,
        smooth_curves: bool = True,
        detail_level: str = "medium",
        tip_offset: float = 0.0,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Complete CoT reasoning: analyze → decide → generate.

        Args:
            object_name: Name of object
            width: Object width
            height: Object height
            symmetry_required: Require symmetry
            smooth_curves: Use smooth curves
            detail_level: "low", "medium", "high"
            tip_offset: X offset for tip
            verbose: Print reasoning steps

        Returns:
            Dict with analysis, decision, and control points
        """
        # Step 1: Analyze
        analysis = self.analyze_shape(
            object_name, width, height, symmetry_required, smooth_curves, detail_level
        )

        if verbose:
            print(f"Shape Analysis:")
            print(f"  Object Type: {analysis.object_type}")
            print(f"  Symmetry: {analysis.symmetry}")
            print(f"  Curvature: {analysis.curvature}")
            print(f"  Detail: {analysis.detail_level}")
            print(f"  Organic: {analysis.organic}")
            print()

        # Step 2: Decide CP count
        decision = self.decide_cp_count(analysis)

        if verbose:
            print("Reasoning Chain:")
            for step in decision.reasoning:
                print(f"  {step}")
            print(f"  Confidence: {decision.confidence:.1%}")
            print()

        # Step 3: Generate positions
        cps = self.generate_cp_positions(
            decision.cp_count, width, height, analysis.symmetry, tip_offset
        )

        if verbose:
            print(f"Generated {len(cps)} Control Points:")
            for i, (x, y) in enumerate(cps):
                print(f"  CP{i+1}: ({x:.4f}, {y:.4f})")
            print()

        # Build spline command
        spline_values = []
        for x, y in cps:
            spline_values.extend([f"{x:.4f}", f"{y:.4f}"])
        # Add close point (= first point)
        spline_values.extend([f"{cps[0][0]:.4f}", f"{cps[0][1]:.4f}"])

        spline_cmd = "spline " + " ".join(spline_values) + ";"

        if verbose:
            print(f"Spline Command:")
            print(f"  {spline_cmd}")

        return {
            "analysis": analysis,
            "decision": decision,
            "control_points": cps,
            "spline_command": spline_cmd,
            "cp_count": decision.cp_count,
        }


def demo_cot_reasoning():
    """Demonstrate CoT reasoning for different shapes."""
    reasoner = CoTReasoner()

    test_cases = [
        # (name, width, height, symmetry, smooth, detail)
        ("wing_left", 0.4, 1.0, True, True, "low"),
        ("petal_L1_P0", 0.4, 1.2, True, True, "medium"),
        ("leaf_main", 0.3, 1.5, True, True, "high"),
        ("triangle_simple", 1.0, 1.0, False, False, "low"),
        ("circle_base", 2.0, 2.0, True, True, "medium"),
        ("heart_shape", 1.0, 1.2, True, True, "high"),
    ]

    for name, w, h, sym, smooth, detail in test_cases:
        print("=" * 70)
        print(f"Object: {name} (width={w}, height={h})")
        print("=" * 70)

        result = reasoner.reason_and_generate(
            name, w, h, sym, smooth, detail, verbose=True
        )
        print()


if __name__ == "__main__":
    demo_cot_reasoning()
