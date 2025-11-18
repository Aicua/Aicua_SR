#!/usr/bin/env python3
"""
Awareness CLI Executor using Symbolic Regression (SR) Formulas.

Uses SR-discovered mathematical formulas instead of Chain-of-Thought reasoning
for computing petal geometry, bone rigging, and animation parameters.

SR formulas are trained using PySR (see notebooks/kaggle_sr_training.ipynb).
"""

import argparse
import math
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


# =============================================================================
# SR FORMULAS - Petal Spline Control Points (V2 Middle-Wide Shape)
# =============================================================================

def sr_compute_layer_factor(layer_index: int) -> float:
    """SR formula for layer factor.

    Discovered: 0.8 + 0.1 * layer_index
    Gives: [0.8, 0.9, 1.0] for layers [0, 1, 2]
    """
    return 0.8 + 0.1 * layer_index


def sr_compute_base_spread(base_size: float, layer_index: int, opening_degree: float) -> float:
    """SR formula for base spread (total width at base).

    Discovered: base_size * 0.3 * layer_factor * (1 + opening_degree * 0.2)
    """
    layer_factor = sr_compute_layer_factor(layer_index)
    return base_size * 0.3 * layer_factor * (1 + opening_degree * 0.2)


def sr_compute_petal_height(base_size: float, layer_index: int, opening_degree: float) -> float:
    """SR formula for petal height.

    Discovered: base_size * layer_factor * (1.2 - opening_degree * 0.3)
    """
    layer_factor = sr_compute_layer_factor(layer_index)
    return base_size * layer_factor * (1.2 - opening_degree * 0.3)


def sr_compute_cp1_x(base_size: float, layer_index: int, petal_index: int, opening_degree: float) -> float:
    """SR formula for CP1 x (base left - narrow).

    Discovered: -base_spread / 4
    """
    base_spread = sr_compute_base_spread(base_size, layer_index, opening_degree)
    return -base_spread / 4


def sr_compute_cp1_y(base_size: float, layer_index: int, petal_index: int, opening_degree: float) -> float:
    """SR formula for CP1 y (base left).

    Discovered: 0.0
    """
    return 0.0


def sr_compute_cp2_x(base_size: float, layer_index: int, petal_index: int, opening_degree: float) -> float:
    """SR formula for CP2 x (mid-left - WIDEST).

    Discovered: -base_spread / 2
    """
    base_spread = sr_compute_base_spread(base_size, layer_index, opening_degree)
    return -base_spread / 2


def sr_compute_cp2_y(base_size: float, layer_index: int, petal_index: int, opening_degree: float) -> float:
    """SR formula for CP2 y (mid-left height).

    Discovered: petal_height * 0.4
    """
    petal_height = sr_compute_petal_height(base_size, layer_index, opening_degree)
    return petal_height * 0.4


def sr_compute_cp3_x(base_size: float, layer_index: int, petal_index: int, opening_degree: float) -> float:
    """SR formula for CP3 x (tip offset).

    Discovered: base_size * 0.02 * layer_index * opening_degree
    """
    return base_size * 0.02 * layer_index * opening_degree


def sr_compute_cp3_y(base_size: float, layer_index: int, petal_index: int, opening_degree: float) -> float:
    """SR formula for CP3 y (petal height/tip).

    Discovered: base_size * layer_factor * (1.2 - opening_degree * 0.3)
    """
    return sr_compute_petal_height(base_size, layer_index, opening_degree)


def sr_compute_cp4_x(base_size: float, layer_index: int, petal_index: int, opening_degree: float) -> float:
    """SR formula for CP4 x (mid-right - WIDEST, symmetric).

    Discovered: base_spread / 2
    """
    base_spread = sr_compute_base_spread(base_size, layer_index, opening_degree)
    return base_spread / 2


def sr_compute_cp4_y(base_size: float, layer_index: int, petal_index: int, opening_degree: float) -> float:
    """SR formula for CP4 y (mid-right height, symmetric).

    Discovered: petal_height * 0.4
    """
    return sr_compute_cp2_y(base_size, layer_index, petal_index, opening_degree)


def sr_compute_cp5_x(base_size: float, layer_index: int, petal_index: int, opening_degree: float) -> float:
    """SR formula for CP5 x (base right - narrow, symmetric).

    Discovered: base_spread / 4
    """
    base_spread = sr_compute_base_spread(base_size, layer_index, opening_degree)
    return base_spread / 4


def sr_compute_cp5_y(base_size: float, layer_index: int, petal_index: int, opening_degree: float) -> float:
    """SR formula for CP5 y (base right).

    Discovered: 0.0
    """
    return 0.0


def sr_compute_extrude_depth(base_size: float, layer_index: int, petal_index: int, opening_degree: float) -> float:
    """SR formula for extrude depth (ultra-thin thickness).

    Discovered: max(0.001, 0.005 * base_size * (1 - layer_index * 0.1) * (1 - opening_degree * 0.3))
    """
    thickness = 0.005 * base_size * (1 - layer_index * 0.1) * (1 - opening_degree * 0.3)
    return max(0.001, thickness)


# =============================================================================
# SR FORMULAS - Bone Rigging (V4 Branching Structure)
# =============================================================================

def sr_compute_bone_root_end_y(petal_height: float, petal_width: float, opening_degree: float,
                               layer_index: int, curvature_intensity: float) -> float:
    """SR formula for bone root end Y position.

    Discovered: petal_height * 0.3 * layer_factor
    Root bone goes from 0 to 30% of petal height.
    """
    layer_factor = sr_compute_layer_factor(layer_index)
    return petal_height * 0.3 * layer_factor


def sr_compute_bone_middle_end_y(petal_height: float, petal_width: float, opening_degree: float,
                                  layer_index: int, curvature_intensity: float) -> float:
    """SR formula for bone middle end Y position.

    Discovered: petal_height * 0.65 * layer_factor
    Middle bone goes from root to 65% of petal height.
    """
    layer_factor = sr_compute_layer_factor(layer_index)
    return petal_height * 0.65 * layer_factor


def sr_compute_bone_left_end_x(petal_height: float, petal_width: float, opening_degree: float,
                                layer_index: int, curvature_intensity: float) -> float:
    """SR formula for bone left end X position.

    Discovered: -petal_width * 0.5 * (0.5 + opening_degree * 0.5) * (1 + curvature_intensity * 0.1)
    Left branch spreads outward.
    """
    left_spread = petal_width * 0.5 * (0.5 + opening_degree * 0.5)
    curvature_factor = curvature_intensity * 0.1
    return -left_spread * (1 + curvature_factor)


def sr_compute_bone_left_end_y(petal_height: float, petal_width: float, opening_degree: float,
                                layer_index: int, curvature_intensity: float) -> float:
    """SR formula for bone left end Y position.

    Discovered: petal_height * 0.9 * layer_factor
    Branch ends at 90% of petal height.
    """
    layer_factor = sr_compute_layer_factor(layer_index)
    return petal_height * 0.9 * layer_factor


def sr_compute_bone_right_end_x(petal_height: float, petal_width: float, opening_degree: float,
                                 layer_index: int, curvature_intensity: float) -> float:
    """SR formula for bone right end X position (symmetric to left).

    Discovered: petal_width * 0.5 * (0.5 + opening_degree * 0.5) * (1 + curvature_intensity * 0.1)
    """
    left_spread = petal_width * 0.5 * (0.5 + opening_degree * 0.5)
    curvature_factor = curvature_intensity * 0.1
    return left_spread * (1 + curvature_factor)


def sr_compute_bind_weight(layer_index: int, flexibility: float) -> float:
    """SR formula for armature bind weight.

    Discovered: flexibility * (0.5 + layer_index * 0.5)
    """
    return flexibility * (0.5 + layer_index * 0.5)


def sr_compute_flexibility(layer_index: int) -> float:
    """SR formula for petal flexibility.

    Discovered: 0.5 + (2 - layer_index) * 0.15
    Inner layers are more flexible.
    """
    return 0.5 + (2 - layer_index) * 0.15


# =============================================================================
# SR FORMULAS - Animation Parameters
# =============================================================================

def sr_compute_frequency(petal_mass: float, flexibility: float) -> float:
    """SR formula for wind animation frequency.

    Discovered: clamp(10.0 * sqrt(flexibility / (petal_mass + 0.01)), 5.0, 30.0)
    """
    frequency = 10.0 * math.sqrt(flexibility / (petal_mass + 0.01))
    return max(5.0, min(30.0, frequency))


def sr_compute_amplitude(wind_speed: float, flexibility: float) -> float:
    """SR formula for wind animation amplitude.

    Discovered: clamp(wind_speed * flexibility * 3.0, 10.0, 60.0)
    """
    amplitude = wind_speed * flexibility * 3.0
    return max(10.0, min(60.0, amplitude))


def sr_compute_bloom_angle(layer_index: int) -> float:
    """SR formula for bloom animation angle.

    Discovered: 15.0 + layer_index * 10.0
    Outer layers open more.
    """
    return 15.0 + layer_index * 10.0


def sr_compute_rotation_angle(petal_index: int) -> float:
    """SR formula for spiral arrangement rotation.

    Discovered: (petal_index * 137.5) % 360
    Golden angle spiral.
    """
    return (petal_index * 137.5) % 360


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SRPetalGeometry:
    """Petal geometry computed using SR formulas."""
    cp1: Tuple[float, float]
    cp2: Tuple[float, float]
    cp3: Tuple[float, float]
    cp4: Tuple[float, float]
    cp5: Tuple[float, float]
    extrude_depth: float
    petal_height: float
    petal_width: float


@dataclass
class SRBoneRigging:
    """Bone rigging computed using SR formulas."""
    root_end_y: float
    middle_end_y: float
    left_end_x: float
    left_end_y: float
    right_end_x: float
    bind_weight: float


@dataclass
class SRAnimationParams:
    """Animation parameters computed using SR formulas."""
    frequency: float
    amplitude: float
    bloom_angle: float
    rotation_angle: float


# =============================================================================
# SR AWARENESS EXECUTOR
# =============================================================================

class SRAwarenessExecutor:
    """
    Execute petal geometry generation using SR-discovered formulas.

    Replaces Chain-of-Thought (CoT) reasoning with direct mathematical computation
    for faster and more consistent results.
    """

    def __init__(self, verbose: bool = False):
        """Initialize SR executor."""
        self.verbose = verbose
        self.formula_calls = 0

    def compute_petal_geometry(
        self,
        base_size: float,
        layer_index: int,
        petal_index: int,
        opening_degree: float
    ) -> SRPetalGeometry:
        """
        Compute petal geometry using SR formulas.

        Args:
            base_size: Overall rose size
            layer_index: 0 (inner), 1 (middle), 2 (outer)
            petal_index: Position within layer
            opening_degree: 0.0 (closed) to 1.0 (fully open)

        Returns:
            SRPetalGeometry with all control points and dimensions
        """
        self.formula_calls += 1

        # Compute all control points using SR formulas
        cp1 = (
            sr_compute_cp1_x(base_size, layer_index, petal_index, opening_degree),
            sr_compute_cp1_y(base_size, layer_index, petal_index, opening_degree)
        )
        cp2 = (
            sr_compute_cp2_x(base_size, layer_index, petal_index, opening_degree),
            sr_compute_cp2_y(base_size, layer_index, petal_index, opening_degree)
        )
        cp3 = (
            sr_compute_cp3_x(base_size, layer_index, petal_index, opening_degree),
            sr_compute_cp3_y(base_size, layer_index, petal_index, opening_degree)
        )
        cp4 = (
            sr_compute_cp4_x(base_size, layer_index, petal_index, opening_degree),
            sr_compute_cp4_y(base_size, layer_index, petal_index, opening_degree)
        )
        cp5 = (
            sr_compute_cp5_x(base_size, layer_index, petal_index, opening_degree),
            sr_compute_cp5_y(base_size, layer_index, petal_index, opening_degree)
        )

        extrude_depth = sr_compute_extrude_depth(base_size, layer_index, petal_index, opening_degree)

        # Compute dimensions
        petal_height = cp3[1]
        petal_width = cp4[0] - cp2[0]

        if self.verbose:
            print(f"  SR Geometry: 5 CPs computed, height={petal_height:.4f}, width={petal_width:.4f}")

        return SRPetalGeometry(
            cp1=cp1, cp2=cp2, cp3=cp3, cp4=cp4, cp5=cp5,
            extrude_depth=extrude_depth,
            petal_height=petal_height,
            petal_width=petal_width
        )

    def compute_bone_rigging(
        self,
        petal_height: float,
        petal_width: float,
        opening_degree: float,
        layer_index: int,
        curvature_intensity: float = 1.0
    ) -> SRBoneRigging:
        """
        Compute bone rigging using SR formulas.

        Args:
            petal_height: Height of the petal
            petal_width: Width of the petal
            opening_degree: 0.0 (closed) to 1.0 (fully open)
            layer_index: 0 (inner), 1 (middle), 2 (outer)
            curvature_intensity: Curvature control (default 1.0)

        Returns:
            SRBoneRigging with all bone positions
        """
        self.formula_calls += 1

        flexibility = sr_compute_flexibility(layer_index)

        rigging = SRBoneRigging(
            root_end_y=sr_compute_bone_root_end_y(petal_height, petal_width, opening_degree, layer_index, curvature_intensity),
            middle_end_y=sr_compute_bone_middle_end_y(petal_height, petal_width, opening_degree, layer_index, curvature_intensity),
            left_end_x=sr_compute_bone_left_end_x(petal_height, petal_width, opening_degree, layer_index, curvature_intensity),
            left_end_y=sr_compute_bone_left_end_y(petal_height, petal_width, opening_degree, layer_index, curvature_intensity),
            right_end_x=sr_compute_bone_right_end_x(petal_height, petal_width, opening_degree, layer_index, curvature_intensity),
            bind_weight=sr_compute_bind_weight(layer_index, flexibility)
        )

        if self.verbose:
            print(f"  SR Rigging: 4 bones computed, bind_weight={rigging.bind_weight:.4f}")

        return rigging

    def compute_animation_params(
        self,
        base_size: float,
        petal_width: float,
        petal_height: float,
        layer_index: int,
        petal_index: int
    ) -> SRAnimationParams:
        """
        Compute animation parameters using SR formulas.

        Args:
            base_size: Overall rose size
            petal_width: Width of the petal
            petal_height: Height of the petal
            layer_index: 0 (inner), 1 (middle), 2 (outer)
            petal_index: Position within layer

        Returns:
            SRAnimationParams with all animation values
        """
        self.formula_calls += 1

        flexibility = sr_compute_flexibility(layer_index)
        petal_mass = base_size * petal_width * petal_height * 0.01
        wind_speed = 3.0

        params = SRAnimationParams(
            frequency=sr_compute_frequency(petal_mass, flexibility),
            amplitude=sr_compute_amplitude(wind_speed, flexibility),
            bloom_angle=sr_compute_bloom_angle(layer_index),
            rotation_angle=sr_compute_rotation_angle(petal_index)
        )

        if self.verbose:
            print(f"  SR Animation: freq={params.frequency:.1f}Hz, amp={params.amplitude:.1f}deg")

        return params

    def generate_petal_cli(
        self,
        layer_idx: int,
        petal_idx: int,
        base_size: float,
        opening_degree: float,
        bloom_animation: bool = False,
        bloom_duration: int = 3000
    ) -> Dict[str, Any]:
        """
        Generate complete petal CLI using SR formulas.

        Args:
            layer_idx: Layer number (1-based)
            petal_idx: Petal position in layer
            base_size: Overall rose size
            opening_degree: 0.0 (closed) to 1.0 (fully open)
            bloom_animation: Enable bloom animation
            bloom_duration: Duration of bloom in ms

        Returns:
            Dict with geometry, rigging, animation CLI lists
        """
        petal_name = f"petal_L{layer_idx}_P{petal_idx}"
        layer_index_0based = layer_idx - 1  # Convert to 0-based for formulas

        if self.verbose:
            print(f"\nGenerating {petal_name} with SR formulas:")

        # === COMPUTE GEOMETRY ===
        geometry = self.compute_petal_geometry(
            base_size, layer_index_0based, petal_idx, opening_degree
        )

        # Build control points list
        control_points = [geometry.cp1, geometry.cp2, geometry.cp3, geometry.cp4, geometry.cp5]

        # Generate spline command
        spline_values = []
        for x, y in control_points:
            spline_values.extend([f"{x:.4f}", f"{y:.4f}"])
        # Close spline
        spline_values.extend([f"{control_points[0][0]:.4f}", f"{control_points[0][1]:.4f}"])
        spline_cmd = "spline " + " ".join(spline_values) + ";"

        # Geometry CLI
        geometry_cli = [
            f"# {petal_name} - Layer {layer_idx}, Petal {petal_idx}",
            f"# SR Formula: 5 CPs (direct computation, no reasoning)",
            f"2d;",
            f"obj {petal_name};",
            spline_cmd,
            f"exit;",
            f"sketch_extrude {petal_name} {geometry.extrude_depth:.4f};",
        ]

        # === COMPUTE RIGGING ===
        rigging = self.compute_bone_rigging(
            geometry.petal_height, geometry.petal_width, opening_degree, layer_index_0based
        )

        rig_name = f"{petal_name}_rig"
        rigging_cli = [
            f"",
            f"# Rigging for {petal_name} (SR v4 branching structure)",
            f"create_armature {rig_name};",
            f"add_bone {rig_name} bone_root 0 0 0 0 {rigging.root_end_y:.4f} 0;",
            f"add_bone {rig_name} bone_middle 0 {rigging.root_end_y:.4f} 0 0 {rigging.middle_end_y:.4f} 0;",
            f"add_bone {rig_name} bone_left 0 {rigging.middle_end_y:.4f} 0 {rigging.left_end_x:.4f} {rigging.left_end_y:.4f} 0;",
            f"add_bone {rig_name} bone_right 0 {rigging.middle_end_y:.4f} 0 {rigging.right_end_x:.4f} {rigging.left_end_y:.4f} 0;",
            f"parent_bone {rig_name} bone_middle bone_root;",
            f"parent_bone {rig_name} bone_left bone_middle;",
            f"parent_bone {rig_name} bone_right bone_middle;",
            f"finalize_bones {rig_name};",
            f"bind_armature {rig_name} {petal_name} {rigging.bind_weight:.4f};",
        ]

        # === COMPUTE ANIMATION ===
        anim_params = self.compute_animation_params(
            base_size, geometry.petal_width, geometry.petal_height, layer_index_0based, petal_idx
        )

        # Rotate petal into position
        if anim_params.rotation_angle > 0:
            rigging_cli.append(f"")
            rigging_cli.append(f"# Position petal in spiral arrangement")
            rigging_cli.append(f"rotate_bone {rig_name} bone_root 0 0 {anim_params.rotation_angle:.2f};")

        animation_cli = [
            f"",
            f"# Animation for {petal_name} (SR computed)",
            f"wing_flap {rig_name} bone_middle {anim_params.frequency:.0f} {anim_params.amplitude:.1f} 0 -1 0 0;",
            f"wing_flap {rig_name} bone_left {anim_params.frequency:.0f} {anim_params.amplitude * 0.5:.1f} -1 0 0 0.25;",
            f"wing_flap {rig_name} bone_right {anim_params.frequency:.0f} {anim_params.amplitude * 0.5:.1f} 1 0 0 0.25;",
        ]

        # Bloom animation
        if bloom_animation:
            animation_cli.append(f"")
            animation_cli.append(f"# Bloom animation (SR computed angle)")
            animation_cli.append(
                f"auto_rotate {rig_name} bone_middle 1 0 0 {anim_params.bloom_angle:.1f} {bloom_duration} smooth;"
            )

        return {
            "geometry": geometry_cli,
            "rigging": rigging_cli,
            "animation": animation_cli,
            "sr_data": {
                "geometry": geometry,
                "rigging": rigging,
                "animation": anim_params,
            }
        }

    def generate_rose(
        self,
        base_size: float = 2.0,
        opening_degree: float = 0.8,
        n_layers: int = 3,
        bloom_animation: bool = False,
        bloom_duration: int = 3000
    ) -> str:
        """
        Generate complete rose CLI using SR formulas.

        Args:
            base_size: Overall rose size
            opening_degree: 0.0 (closed) to 1.0 (fully open)
            n_layers: Number of petal layers (1-3)
            bloom_animation: Enable bloom animation
            bloom_duration: Duration of bloom in ms

        Returns:
            Complete CLI as string
        """
        petals_per_layer = [5, 8, 13]  # Fibonacci sequence

        all_cli = [
            "# Rose CAD Generation - SR (Symbolic Regression) VERSION",
            f"# Base Size: {base_size}",
            f"# Opening Degree: {opening_degree}",
            f"# Layers: {n_layers}",
            f"# Bloom Animation: {bloom_animation}",
            f"# Direct formula computation (no CoT reasoning)",
            "",
        ]

        total_petals = 0

        for layer_idx in range(1, n_layers + 1):
            n_petals = petals_per_layer[layer_idx - 1]

            all_cli.append(f"# ====== LAYER {layer_idx} ({n_petals} petals) ======")
            all_cli.append("")

            for petal_idx in range(n_petals):
                petal_data = self.generate_petal_cli(
                    layer_idx, petal_idx, base_size, opening_degree,
                    bloom_animation, bloom_duration
                )

                all_cli.extend(petal_data["geometry"])
                all_cli.extend(petal_data["rigging"])
                all_cli.extend(petal_data["animation"])
                all_cli.append("")

                total_petals += 1

        # Summary
        all_cli.append(f"# Total petals: {total_petals}")
        all_cli.append(f"# SR formula calls: {self.formula_calls}")
        all_cli.append(f"# Method: Direct SR formula computation")
        all_cli.append(f"# CP count: 5 (fixed, middle-wide shape)")

        return "\n".join(all_cli)

    def save_cli(self, cli_text: str, output_path: str):
        """Save CLI to file."""
        with open(output_path, "w") as f:
            f.write(cli_text)
        print(f"Saved CLI to {output_path}")


# =============================================================================
# COMPARISON: SR vs CoT
# =============================================================================

def compare_sr_vs_cot():
    """Compare SR formula execution vs CoT reasoning."""
    import time

    print("=" * 60)
    print("Comparison: SR Formulas vs CoT Reasoning")
    print("=" * 60)

    # Test parameters
    base_size = 5.0
    layer_index = 1
    petal_index = 3
    opening_degree = 0.8

    # SR computation
    print("\nSR Formula Computation:")
    start = time.time()

    cp1_x = sr_compute_cp1_x(base_size, layer_index, petal_index, opening_degree)
    cp1_y = sr_compute_cp1_y(base_size, layer_index, petal_index, opening_degree)
    cp2_x = sr_compute_cp2_x(base_size, layer_index, petal_index, opening_degree)
    cp2_y = sr_compute_cp2_y(base_size, layer_index, petal_index, opening_degree)
    cp3_x = sr_compute_cp3_x(base_size, layer_index, petal_index, opening_degree)
    cp3_y = sr_compute_cp3_y(base_size, layer_index, petal_index, opening_degree)
    cp4_x = sr_compute_cp4_x(base_size, layer_index, petal_index, opening_degree)
    cp4_y = sr_compute_cp4_y(base_size, layer_index, petal_index, opening_degree)
    cp5_x = sr_compute_cp5_x(base_size, layer_index, petal_index, opening_degree)
    cp5_y = sr_compute_cp5_y(base_size, layer_index, petal_index, opening_degree)

    sr_time = time.time() - start

    print(f"  CP1: ({cp1_x:.4f}, {cp1_y:.4f})")
    print(f"  CP2: ({cp2_x:.4f}, {cp2_y:.4f})")
    print(f"  CP3: ({cp3_x:.4f}, {cp3_y:.4f})")
    print(f"  CP4: ({cp4_x:.4f}, {cp4_y:.4f})")
    print(f"  CP5: ({cp5_x:.4f}, {cp5_y:.4f})")
    print(f"  Time: {sr_time*1000:.3f}ms")
    print(f"  Method: Direct mathematical formulas")
    print(f"  Reasoning: None (formulas encode the knowledge)")

    print("\n" + "=" * 60)
    print("SR Advantages:")
    print("  - Faster: Direct computation vs step-by-step reasoning")
    print("  - Consistent: Same inputs always produce same outputs")
    print("  - Compact: Formulas discovered by PySR are optimized")
    print("  - Interpretable: Mathematical expressions are explicit")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Rose CAD CLI using SR (Symbolic Regression) Formulas"
    )

    parser.add_argument("--size", type=float, default=2.0, help="Base size")
    parser.add_argument("--opening", type=float, default=0.8, help="Opening degree 0-1")
    parser.add_argument("--layers", type=int, default=3, help="Number of layers 1-3")
    parser.add_argument("--bloom", action="store_true", help="Enable bloom animation")
    parser.add_argument("--bloom-duration", type=int, default=3000, help="Bloom duration in ms")
    parser.add_argument("--output", type=str, default=None, help="Output file")
    parser.add_argument("--verbose", action="store_true", help="Show SR formula details")
    parser.add_argument("--compare", action="store_true", help="Compare SR vs CoT")

    args = parser.parse_args()

    if args.compare:
        compare_sr_vs_cot()
        return

    print("=" * 60)
    print("Rose CLI Generator - SR (Symbolic Regression) VERSION")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  Base Size: {args.size}")
    print(f"  Opening Degree: {args.opening}")
    print(f"  Layers: {args.layers}")
    print(f"  Bloom Animation: {args.bloom}")
    if args.bloom:
        print(f"  Bloom Duration: {args.bloom_duration}ms")
    print()

    executor = SRAwarenessExecutor(verbose=args.verbose)
    cli = executor.generate_rose(
        base_size=args.size,
        opening_degree=args.opening,
        n_layers=args.layers,
        bloom_animation=args.bloom,
        bloom_duration=args.bloom_duration,
    )

    if args.output:
        executor.save_cli(cli, args.output)
    else:
        print("\nGenerated CLI:")
        print("-" * 60)
        print(cli)

    print(f"\nSR formula calls: {executor.formula_calls}")
    print("SR-based Rose CLI generation complete!")


if __name__ == "__main__":
    main()
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
