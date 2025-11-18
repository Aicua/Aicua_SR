#!/usr/bin/env python3
"""
Generate Rose CAD CLI using Spline geometry and SR-discovered formulas.

Output format matches:
- Spline for petal shape
- Bones for rigging
- wing_flap for animation
"""

import argparse
import math
import sys
from pathlib import Path

# Add generated module path
sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "generated"))


class SplineRoseCLIGenerator:
    """Generate spline-based rose CLI with bones and animation."""

    def __init__(self):
        """Initialize with formula modules."""
        self.use_sr_petal = False
        self.use_sr_bone = False
        self.use_sr_anim = False

        try:
            import petal_spline_formulas as petal
            self.petal_mod = petal
            self.use_sr_petal = True
        except ImportError:
            print("Petal SR formulas not found. Using fallback.")

        try:
            import bone_rigging_v5_formulas as bone
            self.bone_mod = bone
            self.use_sr_bone = True
        except ImportError:
            print("Bone rigging v5 SR formulas not found. Using fallback.")

        try:
            import animation_wingflap_formulas as anim
            self.anim_mod = anim
            self.use_sr_anim = True
        except ImportError:
            print("Animation SR formulas not found. Using fallback.")

        # Legacy compatibility
        self.use_sr = self.use_sr_petal and self.use_sr_bone and self.use_sr_anim

    def compute_spline_params(self, base_size, layer_idx, petal_idx, opening_degree):
        """Compute 2D spline control points using SR formulas or fallback."""
        if self.use_sr_petal:
            return {
                # 15 Control Points matching config
                'cp1_x': self.petal_mod.compute_cp1_x(base_size, layer_idx, petal_idx, opening_degree),
                'cp1_y': self.petal_mod.compute_cp1_y(base_size, layer_idx, petal_idx, opening_degree),
                'cp2_x': self.petal_mod.compute_cp2_x(base_size, layer_idx, petal_idx, opening_degree),
                'cp2_y': self.petal_mod.compute_cp2_y(base_size, layer_idx, petal_idx, opening_degree),
                'cp3_x': self.petal_mod.compute_cp3_x(base_size, layer_idx, petal_idx, opening_degree),
                'cp3_y': self.petal_mod.compute_cp3_y(base_size, layer_idx, petal_idx, opening_degree),
                'cp4_x': self.petal_mod.compute_cp4_x(base_size, layer_idx, petal_idx, opening_degree),
                'cp4_y': self.petal_mod.compute_cp4_y(base_size, layer_idx, petal_idx, opening_degree),
                'cp5_x': self.petal_mod.compute_cp5_x(base_size, layer_idx, petal_idx, opening_degree),
                'cp5_y': self.petal_mod.compute_cp5_y(base_size, layer_idx, petal_idx, opening_degree),
                'cp6_x': self.petal_mod.compute_cp6_x(base_size, layer_idx, petal_idx, opening_degree),
                'cp6_y': self.petal_mod.compute_cp6_y(base_size, layer_idx, petal_idx, opening_degree),
                'cp7_x': self.petal_mod.compute_cp7_x(base_size, layer_idx, petal_idx, opening_degree),
                'cp7_y': self.petal_mod.compute_cp7_y(base_size, layer_idx, petal_idx, opening_degree),
                'cp8_x': self.petal_mod.compute_cp8_x(base_size, layer_idx, petal_idx, opening_degree),
                'cp8_y': self.petal_mod.compute_cp8_y(base_size, layer_idx, petal_idx, opening_degree),
                'cp9_x': self.petal_mod.compute_cp9_x(base_size, layer_idx, petal_idx, opening_degree),
                'cp9_y': self.petal_mod.compute_cp9_y(base_size, layer_idx, petal_idx, opening_degree),
                'cp10_x': self.petal_mod.compute_cp10_x(base_size, layer_idx, petal_idx, opening_degree),
                'cp10_y': self.petal_mod.compute_cp10_y(base_size, layer_idx, petal_idx, opening_degree),
                'cp11_x': self.petal_mod.compute_cp11_x(base_size, layer_idx, petal_idx, opening_degree),
                'cp11_y': self.petal_mod.compute_cp11_y(base_size, layer_idx, petal_idx, opening_degree),
                'cp12_x': self.petal_mod.compute_cp12_x(base_size, layer_idx, petal_idx, opening_degree),
                'cp12_y': self.petal_mod.compute_cp12_y(base_size, layer_idx, petal_idx, opening_degree),
                'cp13_x': self.petal_mod.compute_cp13_x(base_size, layer_idx, petal_idx, opening_degree),
                'cp13_y': self.petal_mod.compute_cp13_y(base_size, layer_idx, petal_idx, opening_degree),
                'cp14_x': self.petal_mod.compute_cp14_x(base_size, layer_idx, petal_idx, opening_degree),
                'cp14_y': self.petal_mod.compute_cp14_y(base_size, layer_idx, petal_idx, opening_degree),
                'cp15_x': self.petal_mod.compute_cp15_x(base_size, layer_idx, petal_idx, opening_degree),
                'cp15_y': self.petal_mod.compute_cp15_y(base_size, layer_idx, petal_idx, opening_degree),
                'extrude_depth': self.petal_mod.compute_extrude_depth(base_size, layer_idx, petal_idx, opening_degree),
            }
        else:
            # Fallback formulas (2D spline control points) - V3 with 15 CPs
            # Matching config: sr_config_spline_v3.yaml
            layer_factor = 0.8 + 0.1 * (layer_idx - 1)  # [0.8, 0.9, 1.0]

            base_spread = base_size * 0.3 * layer_factor * (1 + opening_degree * 0.2)
            petal_height = base_size * layer_factor * (1.2 - opening_degree * 0.3)
            tip_x_offset = base_size * 0.02 * (layer_idx - 1) * opening_degree

            # 15 CPs with Y positions from config:
            # cp1: 0%, cp2: 5%, cp3: 25%, cp4: 45%, cp5: 62% (WIDEST), cp6: 78%, cp7: 92%, cp8: 100% (tip)
            # cp9-cp14: mirror, cp15: close
            import math

            # Y positions (percentage of petal height)
            y_positions = [0.0, 0.05, 0.25, 0.45, 0.62, 0.78, 0.92, 1.0, 0.92, 0.78, 0.62, 0.45, 0.25, 0.05, 0.0]

            # X width factors (base narrow, widest at 62%, tip narrow)
            # Follows rose petal shape: point at base, bulge at middle, point at tip
            x_factors = [0.0, 0.15, 0.45, 0.70, 0.85, 0.65, 0.30, 0.0, 0.30, 0.65, 0.85, 0.70, 0.45, 0.15, 0.0]

            return {
                # Left side (cp1-cp7): negative X
                'cp1_x': 0.0,                                      # base center
                'cp1_y': petal_height * y_positions[0],
                'cp2_x': -base_spread * x_factors[1],              # base left (5%)
                'cp2_y': petal_height * y_positions[1],
                'cp3_x': -base_spread * x_factors[2],              # lower left (25%)
                'cp3_y': petal_height * y_positions[2],
                'cp4_x': -base_spread * x_factors[3],              # mid-low left (45%)
                'cp4_y': petal_height * y_positions[3],
                'cp5_x': -base_spread * x_factors[4],              # upper-mid left (62% - WIDEST)
                'cp5_y': petal_height * y_positions[4],
                'cp6_x': -base_spread * x_factors[5] + tip_x_offset,  # upper left (78%)
                'cp6_y': petal_height * y_positions[5],
                'cp7_x': -base_spread * x_factors[6] + tip_x_offset,  # near-tip left (92%)
                'cp7_y': petal_height * y_positions[6],
                # Tip (cp8)
                'cp8_x': tip_x_offset,                             # tip center (100%)
                'cp8_y': petal_height * y_positions[7],
                # Right side (cp9-cp14): positive X (mirror of left)
                'cp9_x': base_spread * x_factors[8] + tip_x_offset,   # near-tip right (92%)
                'cp9_y': petal_height * y_positions[8],
                'cp10_x': base_spread * x_factors[9] + tip_x_offset,  # upper right (78%)
                'cp10_y': petal_height * y_positions[9],
                'cp11_x': base_spread * x_factors[10],             # upper-mid right (62% - WIDEST)
                'cp11_y': petal_height * y_positions[10],
                'cp12_x': base_spread * x_factors[11],             # mid-low right (45%)
                'cp12_y': petal_height * y_positions[11],
                'cp13_x': base_spread * x_factors[12],             # lower right (25%)
                'cp13_y': petal_height * y_positions[12],
                'cp14_x': base_spread * x_factors[13],             # base right (5%)
                'cp14_y': petal_height * y_positions[13],
                'cp15_x': 0.0,                                     # close spline (back to base)
                'cp15_y': petal_height * y_positions[14],
                # ULTRA-THIN THICKNESS
                'extrude_depth': max(0.001, base_size * 0.005 * (1 - (layer_idx - 1) * 0.1) * (1 - opening_degree * 0.3)),
            }

    def compute_bone_params_v5(self, petal_height, petal_width, opening_degree, layer_idx, curvature_intensity=1.0):
        """Compute bone rigging parameters for v5 fishbone structure (12 bones)."""
        if self.use_sr_bone:
            return {
                # Central Spine (4 bones)
                'bone_root_start_x': self.bone_mod.compute_bone_root_start_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_root_start_y': self.bone_mod.compute_bone_root_start_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_root_end_x': self.bone_mod.compute_bone_root_end_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_root_end_y': self.bone_mod.compute_bone_root_end_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_lower_mid_start_x': self.bone_mod.compute_bone_lower_mid_start_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_lower_mid_start_y': self.bone_mod.compute_bone_lower_mid_start_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_lower_mid_end_x': self.bone_mod.compute_bone_lower_mid_end_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_lower_mid_end_y': self.bone_mod.compute_bone_lower_mid_end_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_upper_mid_start_x': self.bone_mod.compute_bone_upper_mid_start_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_upper_mid_start_y': self.bone_mod.compute_bone_upper_mid_start_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_upper_mid_end_x': self.bone_mod.compute_bone_upper_mid_end_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_upper_mid_end_y': self.bone_mod.compute_bone_upper_mid_end_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_tip_start_x': self.bone_mod.compute_bone_tip_start_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_tip_start_y': self.bone_mod.compute_bone_tip_start_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_tip_end_x': self.bone_mod.compute_bone_tip_end_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_tip_end_y': self.bone_mod.compute_bone_tip_end_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                # Left Ribs (4 bones)
                'bone_left_lower_start_x': self.bone_mod.compute_bone_left_lower_start_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_left_lower_start_y': self.bone_mod.compute_bone_left_lower_start_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_left_lower_end_x': self.bone_mod.compute_bone_left_lower_end_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_left_lower_end_y': self.bone_mod.compute_bone_left_lower_end_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_left_mid_lower_start_x': self.bone_mod.compute_bone_left_mid_lower_start_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_left_mid_lower_start_y': self.bone_mod.compute_bone_left_mid_lower_start_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_left_mid_lower_end_x': self.bone_mod.compute_bone_left_mid_lower_end_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_left_mid_lower_end_y': self.bone_mod.compute_bone_left_mid_lower_end_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_left_mid_upper_start_x': self.bone_mod.compute_bone_left_mid_upper_start_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_left_mid_upper_start_y': self.bone_mod.compute_bone_left_mid_upper_start_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_left_mid_upper_end_x': self.bone_mod.compute_bone_left_mid_upper_end_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_left_mid_upper_end_y': self.bone_mod.compute_bone_left_mid_upper_end_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_left_upper_start_x': self.bone_mod.compute_bone_left_upper_start_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_left_upper_start_y': self.bone_mod.compute_bone_left_upper_start_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_left_upper_end_x': self.bone_mod.compute_bone_left_upper_end_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_left_upper_end_y': self.bone_mod.compute_bone_left_upper_end_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                # Right Ribs (4 bones)
                'bone_right_lower_start_x': self.bone_mod.compute_bone_right_lower_start_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_right_lower_start_y': self.bone_mod.compute_bone_right_lower_start_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_right_lower_end_x': self.bone_mod.compute_bone_right_lower_end_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_right_lower_end_y': self.bone_mod.compute_bone_right_lower_end_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_right_mid_lower_start_x': self.bone_mod.compute_bone_right_mid_lower_start_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_right_mid_lower_start_y': self.bone_mod.compute_bone_right_mid_lower_start_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_right_mid_lower_end_x': self.bone_mod.compute_bone_right_mid_lower_end_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_right_mid_lower_end_y': self.bone_mod.compute_bone_right_mid_lower_end_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_right_mid_upper_start_x': self.bone_mod.compute_bone_right_mid_upper_start_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_right_mid_upper_start_y': self.bone_mod.compute_bone_right_mid_upper_start_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_right_mid_upper_end_x': self.bone_mod.compute_bone_right_mid_upper_end_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_right_mid_upper_end_y': self.bone_mod.compute_bone_right_mid_upper_end_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_right_upper_start_x': self.bone_mod.compute_bone_right_upper_start_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_right_upper_start_y': self.bone_mod.compute_bone_right_upper_start_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_right_upper_end_x': self.bone_mod.compute_bone_right_upper_end_x(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
                'bone_right_upper_end_y': self.bone_mod.compute_bone_right_upper_end_y(petal_height, petal_width, opening_degree, layer_idx, curvature_intensity),
            }
        else:
            # Fallback formulas for v5 fishbone structure (12 bones)
            layer_factor = [0.8, 0.9, 1.0][layer_idx]

            # Central Spine heights (matching config)
            # bone_root: 0% → 25%
            root_end_y = petal_height * 0.25 * layer_factor
            # bone_lower_mid: 25% → 45%
            lower_mid_end_y = petal_height * 0.45 * layer_factor
            # bone_upper_mid: 45% → 62%
            upper_mid_end_y = petal_height * 0.62 * layer_factor
            # bone_tip: 62% → 100%
            tip_end_y = petal_height * layer_factor

            # Apply curvature
            curvature_factor = curvature_intensity * 0.1

            # Rib widths at different heights (matching config width_factors)
            lower_width = petal_width * 0.525 * (0.5 + opening_degree * 0.5)  # 25% height, factor 1.05
            mid_low_width = petal_width * 0.7 * (0.5 + opening_degree * 0.5)  # 45% height, factor 1.4
            mid_up_width = petal_width * 0.8 * (0.5 + opening_degree * 0.5)   # 62% height, factor 1.6 (WIDEST)
            upper_width = petal_width * 0.65 * (0.5 + opening_degree * 0.5)   # 78% height, factor 1.3

            return {
                # Central Spine (4 bones)
                'bone_root_start_x': 0.0,
                'bone_root_start_y': 0.0,
                'bone_root_end_x': 0.0,
                'bone_root_end_y': root_end_y,
                'bone_lower_mid_start_x': 0.0,
                'bone_lower_mid_start_y': root_end_y,
                'bone_lower_mid_end_x': 0.0,
                'bone_lower_mid_end_y': lower_mid_end_y,
                'bone_upper_mid_start_x': 0.0,
                'bone_upper_mid_start_y': lower_mid_end_y,
                'bone_upper_mid_end_x': 0.0,
                'bone_upper_mid_end_y': upper_mid_end_y,
                'bone_tip_start_x': 0.0,
                'bone_tip_start_y': upper_mid_end_y,
                'bone_tip_end_x': 0.0,
                'bone_tip_end_y': tip_end_y,
                # Left Ribs (4 bones)
                'bone_left_lower_start_x': 0.0,
                'bone_left_lower_start_y': root_end_y,
                'bone_left_lower_end_x': -lower_width * (1 + curvature_factor),
                'bone_left_lower_end_y': root_end_y,
                'bone_left_mid_lower_start_x': 0.0,
                'bone_left_mid_lower_start_y': lower_mid_end_y,
                'bone_left_mid_lower_end_x': -mid_low_width * (1 + curvature_factor),
                'bone_left_mid_lower_end_y': lower_mid_end_y,
                'bone_left_mid_upper_start_x': 0.0,
                'bone_left_mid_upper_start_y': upper_mid_end_y,
                'bone_left_mid_upper_end_x': -mid_up_width * (1 + curvature_factor),
                'bone_left_mid_upper_end_y': upper_mid_end_y,
                'bone_left_upper_start_x': 0.0,
                'bone_left_upper_start_y': petal_height * 0.78 * layer_factor,
                'bone_left_upper_end_x': -upper_width * (1 + curvature_factor),
                'bone_left_upper_end_y': petal_height * 0.78 * layer_factor,
                # Right Ribs (4 bones)
                'bone_right_lower_start_x': 0.0,
                'bone_right_lower_start_y': root_end_y,
                'bone_right_lower_end_x': lower_width * (1 + curvature_factor),
                'bone_right_lower_end_y': root_end_y,
                'bone_right_mid_lower_start_x': 0.0,
                'bone_right_mid_lower_start_y': lower_mid_end_y,
                'bone_right_mid_lower_end_x': mid_low_width * (1 + curvature_factor),
                'bone_right_mid_lower_end_y': lower_mid_end_y,
                'bone_right_mid_upper_start_x': 0.0,
                'bone_right_mid_upper_start_y': upper_mid_end_y,
                'bone_right_mid_upper_end_x': mid_up_width * (1 + curvature_factor),
                'bone_right_mid_upper_end_y': upper_mid_end_y,
                'bone_right_upper_start_x': 0.0,
                'bone_right_upper_start_y': petal_height * 0.78 * layer_factor,
                'bone_right_upper_end_x': upper_width * (1 + curvature_factor),
                'bone_right_upper_end_y': petal_height * 0.78 * layer_factor,
            }

    def compute_anim_params(self, base_size, petal_mass, wind_speed, flexibility, layer_idx):
        """Compute animation parameters for wing_flap."""
        if self.use_sr_anim:
            return {
                'frequency': self.anim_mod.compute_frequency(base_size, petal_mass, wind_speed, flexibility, layer_idx),
                'amplitude': self.anim_mod.compute_amplitude(base_size, petal_mass, wind_speed, flexibility, layer_idx),
                'axis_x': int(self.anim_mod.compute_axis_x(base_size, petal_mass, wind_speed, flexibility, layer_idx)),
                'axis_y': int(self.anim_mod.compute_axis_y(base_size, petal_mass, wind_speed, flexibility, layer_idx)),
                'axis_z': int(self.anim_mod.compute_axis_z(base_size, petal_mass, wind_speed, flexibility, layer_idx)),
            }
        else:
            frequency = 10.0 * math.sqrt(flexibility / (petal_mass + 0.01))
            frequency = max(5.0, min(30.0, frequency))
            amplitude = wind_speed * flexibility * 3.0
            amplitude = max(10.0, min(60.0, amplitude))
            return {
                'frequency': frequency,
                'amplitude': amplitude,
                'axis_x': -1 if layer_idx > 1 else 0,
                'axis_y': -1,
                'axis_z': 0,
            }

    def generate_petal_spline(self, layer_idx, petal_idx, base_size, opening_degree):
        """Generate CLI for single petal using 2D spline."""

        petal_name = f"petal_L{layer_idx}_P{petal_idx}"

        # Get 2D spline control points
        sp = self.compute_spline_params(base_size, layer_idx, petal_idx, opening_degree)

        # Calculate rotation angle for spiral arrangement
        golden_angle = 137.5
        rotation_angle = (petal_idx * golden_angle) % 360

        # Generate geometry CLI with 2D spline (x y pairs)
        # Format: spline x1 y1 x2 y2 ... (closed curve)
        # V3: 15 CPs for realistic rose petal shape
        geometry_cli = [
            f"# {petal_name} - Layer {layer_idx}, Petal {petal_idx}",
            f"2d;",
            f"obj {petal_name};",
            f"spline {sp['cp1_x']:.4f} {sp['cp1_y']:.4f} {sp['cp2_x']:.4f} {sp['cp2_y']:.4f} {sp['cp3_x']:.4f} {sp['cp3_y']:.4f} {sp['cp4_x']:.4f} {sp['cp4_y']:.4f} {sp['cp5_x']:.4f} {sp['cp5_y']:.4f} {sp['cp6_x']:.4f} {sp['cp6_y']:.4f} {sp['cp7_x']:.4f} {sp['cp7_y']:.4f} {sp['cp8_x']:.4f} {sp['cp8_y']:.4f} {sp['cp9_x']:.4f} {sp['cp9_y']:.4f} {sp['cp10_x']:.4f} {sp['cp10_y']:.4f} {sp['cp11_x']:.4f} {sp['cp11_y']:.4f} {sp['cp12_x']:.4f} {sp['cp12_y']:.4f} {sp['cp13_x']:.4f} {sp['cp13_y']:.4f} {sp['cp14_x']:.4f} {sp['cp14_y']:.4f} {sp['cp15_x']:.4f} {sp['cp15_y']:.4f};",
            f"exit;",
            f"sketch_extrude {petal_name} {sp['extrude_depth']:.4f};",
        ]

        # Generate bone rigging with v5 fishbone structure (12 bones)
        # Use cp8_y as petal height (tip at 100%), use WIDEST width at cp5/cp11 (62% height)
        petal_height = sp['cp8_y']
        # For V3 shape: cp11_x - cp5_x is the widest part (at 62% height)
        petal_width = sp['cp11_x'] - sp['cp5_x']  # Width at 62% (WIDEST)
        curvature_intensity = 1.0  # Default curvature

        # layer_idx is 1-based, convert to 0-based for v5
        layer_idx_0based = layer_idx - 1
        bp = self.compute_bone_params_v5(petal_height, petal_width, opening_degree, layer_idx_0based, curvature_intensity)

        rig_name = f"{petal_name}_rig"

        rigging_cli = [
            f"",
            f"# Rigging for {petal_name} (v5 fishbone structure - 12 bones)",
            f"create_armature {rig_name};",
        ]

        # Generate 12 bones with v5 fishbone structure (2D coords, z=0)
        # Central spine: root → lower_mid → upper_mid → tip
        rigging_cli.append(
            f"add_bone {rig_name} bone_root {bp['bone_root_start_x']:.4f} {bp['bone_root_start_y']:.4f} 0 {bp['bone_root_end_x']:.4f} {bp['bone_root_end_y']:.4f} 0;"
        )
        rigging_cli.append(
            f"add_bone {rig_name} bone_lower_mid {bp['bone_lower_mid_start_x']:.4f} {bp['bone_lower_mid_start_y']:.4f} 0 {bp['bone_lower_mid_end_x']:.4f} {bp['bone_lower_mid_end_y']:.4f} 0;"
        )
        rigging_cli.append(
            f"add_bone {rig_name} bone_upper_mid {bp['bone_upper_mid_start_x']:.4f} {bp['bone_upper_mid_start_y']:.4f} 0 {bp['bone_upper_mid_end_x']:.4f} {bp['bone_upper_mid_end_y']:.4f} 0;"
        )
        rigging_cli.append(
            f"add_bone {rig_name} bone_tip {bp['bone_tip_start_x']:.4f} {bp['bone_tip_start_y']:.4f} 0 {bp['bone_tip_end_x']:.4f} {bp['bone_tip_end_y']:.4f} 0;"
        )
        # Left ribs (4 bones): from spine at different heights
        rigging_cli.append(
            f"add_bone {rig_name} bone_left_lower {bp['bone_left_lower_start_x']:.4f} {bp['bone_left_lower_start_y']:.4f} 0 {bp['bone_left_lower_end_x']:.4f} {bp['bone_left_lower_end_y']:.4f} 0;"
        )
        rigging_cli.append(
            f"add_bone {rig_name} bone_left_mid_lower {bp['bone_left_mid_lower_start_x']:.4f} {bp['bone_left_mid_lower_start_y']:.4f} 0 {bp['bone_left_mid_lower_end_x']:.4f} {bp['bone_left_mid_lower_end_y']:.4f} 0;"
        )
        rigging_cli.append(
            f"add_bone {rig_name} bone_left_mid_upper {bp['bone_left_mid_upper_start_x']:.4f} {bp['bone_left_mid_upper_start_y']:.4f} 0 {bp['bone_left_mid_upper_end_x']:.4f} {bp['bone_left_mid_upper_end_y']:.4f} 0;"
        )
        rigging_cli.append(
            f"add_bone {rig_name} bone_left_upper {bp['bone_left_upper_start_x']:.4f} {bp['bone_left_upper_start_y']:.4f} 0 {bp['bone_left_upper_end_x']:.4f} {bp['bone_left_upper_end_y']:.4f} 0;"
        )
        # Right ribs (4 bones): symmetric to left
        rigging_cli.append(
            f"add_bone {rig_name} bone_right_lower {bp['bone_right_lower_start_x']:.4f} {bp['bone_right_lower_start_y']:.4f} 0 {bp['bone_right_lower_end_x']:.4f} {bp['bone_right_lower_end_y']:.4f} 0;"
        )
        rigging_cli.append(
            f"add_bone {rig_name} bone_right_mid_lower {bp['bone_right_mid_lower_start_x']:.4f} {bp['bone_right_mid_lower_start_y']:.4f} 0 {bp['bone_right_mid_lower_end_x']:.4f} {bp['bone_right_mid_lower_end_y']:.4f} 0;"
        )
        rigging_cli.append(
            f"add_bone {rig_name} bone_right_mid_upper {bp['bone_right_mid_upper_start_x']:.4f} {bp['bone_right_mid_upper_start_y']:.4f} 0 {bp['bone_right_mid_upper_end_x']:.4f} {bp['bone_right_mid_upper_end_y']:.4f} 0;"
        )
        rigging_cli.append(
            f"add_bone {rig_name} bone_right_upper {bp['bone_right_upper_start_x']:.4f} {bp['bone_right_upper_start_y']:.4f} 0 {bp['bone_right_upper_end_x']:.4f} {bp['bone_right_upper_end_y']:.4f} 0;"
        )

        # Parent bones in v5 fishbone structure
        # Spine hierarchy
        rigging_cli.append(f"parent_bone {rig_name} bone_lower_mid bone_root;")
        rigging_cli.append(f"parent_bone {rig_name} bone_upper_mid bone_lower_mid;")
        rigging_cli.append(f"parent_bone {rig_name} bone_tip bone_upper_mid;")
        # Left ribs parent to spine
        rigging_cli.append(f"parent_bone {rig_name} bone_left_lower bone_root;")
        rigging_cli.append(f"parent_bone {rig_name} bone_left_mid_lower bone_lower_mid;")
        rigging_cli.append(f"parent_bone {rig_name} bone_left_mid_upper bone_upper_mid;")
        rigging_cli.append(f"parent_bone {rig_name} bone_left_upper bone_upper_mid;")
        # Right ribs parent to spine
        rigging_cli.append(f"parent_bone {rig_name} bone_right_lower bone_root;")
        rigging_cli.append(f"parent_bone {rig_name} bone_right_mid_lower bone_lower_mid;")
        rigging_cli.append(f"parent_bone {rig_name} bone_right_mid_upper bone_upper_mid;")
        rigging_cli.append(f"parent_bone {rig_name} bone_right_upper bone_upper_mid;")

        # Calculate bind weight based on flexibility
        flexibility = 0.5 + (3 - layer_idx) * 0.15
        bind_weight = flexibility * [1.0, 1.5, 2.0][layer_idx - 1]

        rigging_cli.append(f"finalize_bones {rig_name};")
        rigging_cli.append(f"bind_armature {rig_name} {petal_name} {bind_weight:.4f};")

        # Rotate petal into position using root bone
        if rotation_angle > 0:
            rigging_cli.append(f"")
            rigging_cli.append(f"# Position petal in spiral arrangement (rotate root bone)")
            rigging_cli.append(f"rotate_bone {rig_name} bone_root 0 0 {rotation_angle:.2f};")

        # Generate animation (wing_flap style)
        petal_mass = base_size * petal_width * petal_height * 0.01
        wind_speed = 3.0  # Default

        ap = self.compute_anim_params(base_size, petal_mass, wind_speed, flexibility, layer_idx)

        animation_cli = [
            f"",
            f"# Animation for {petal_name} (v5 - 12 bones)",
            f"# Spine bones control overall petal movement",
            f"wing_flap {rig_name} bone_lower_mid {ap['frequency']:.0f} {ap['amplitude'] * 0.5:.1f} 0 -1 0 0;",
            f"wing_flap {rig_name} bone_upper_mid {ap['frequency']:.0f} {ap['amplitude']:.1f} 0 -1 0 0.05;",
            f"wing_flap {rig_name} bone_tip {ap['frequency'] * 1.5:.0f} {ap['amplitude'] * 0.3:.1f} 0 -1 0 0.1;",
            f"# Lower ribs: subtle base movement (25% height)",
            f"wing_flap {rig_name} bone_left_lower {ap['frequency']:.0f} {ap['amplitude'] * 0.2:.1f} -1 0 0 0.15;",
            f"wing_flap {rig_name} bone_right_lower {ap['frequency']:.0f} {ap['amplitude'] * 0.2:.1f} 1 0 0 0.15;",
            f"# Mid-lower ribs: moderate movement (45% height)",
            f"wing_flap {rig_name} bone_left_mid_lower {ap['frequency']:.0f} {ap['amplitude'] * 0.4:.1f} -1 0 0 0.2;",
            f"wing_flap {rig_name} bone_right_mid_lower {ap['frequency']:.0f} {ap['amplitude'] * 0.4:.1f} 1 0 0 0.2;",
            f"# Mid-upper ribs: main opening (62% height - WIDEST)",
            f"wing_flap {rig_name} bone_left_mid_upper {ap['frequency']:.0f} {ap['amplitude'] * 0.6:.1f} -1 0 0 0.25;",
            f"wing_flap {rig_name} bone_right_mid_upper {ap['frequency']:.0f} {ap['amplitude'] * 0.6:.1f} 1 0 0 0.25;",
            f"# Upper ribs: tip area movement (78% height)",
            f"wing_flap {rig_name} bone_left_upper {ap['frequency']:.0f} {ap['amplitude'] * 0.4:.1f} -1 0 0 0.3;",
            f"wing_flap {rig_name} bone_right_upper {ap['frequency']:.0f} {ap['amplitude'] * 0.4:.1f} 1 0 0 0.3;",
        ]

        return {
            'geometry': geometry_cli,
            'rigging': rigging_cli,
            'animation': animation_cli,
            'params': sp,
        }

    def generate_rose(self, base_size=2.0, opening_degree=0.8, n_layers=3):
        """Generate complete rose CLI with splines."""

        petals_per_layer = [5, 8, 13]

        all_cli = [
            "# Rose CAD Generation - SPLINE VERSION",
            f"# Base Size: {base_size}",
            f"# Opening Degree: {opening_degree}",
            f"# Layers: {n_layers}",
            f"# Using spline geometry with bones and wing_flap animation",
            "",
        ]

        total_petals = 0

        for layer_idx in range(1, n_layers + 1):
            n_petals = petals_per_layer[layer_idx - 1]

            all_cli.append(f"# ====== LAYER {layer_idx} ({n_petals} petals) ======")
            all_cli.append("")

            for petal_idx in range(n_petals):
                petal_data = self.generate_petal_spline(
                    layer_idx, petal_idx, base_size, opening_degree
                )

                all_cli.extend(petal_data['geometry'])
                all_cli.extend(petal_data['rigging'])
                all_cli.extend(petal_data['animation'])
                all_cli.append("")

                total_petals += 1

        # Summary
        all_cli.append(f"# Total petals: {total_petals}")
        all_cli.append(f"# Geometry: Spline-based 3D curves")
        all_cli.append(f"# Animation: wing_flap style oscillation")

        return '\n'.join(all_cli)

    def save_cli(self, cli_text, output_path):
        """Save CLI to file."""
        with open(output_path, 'w') as f:
            f.write(cli_text)
        print(f"✓ Saved CLI to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Rose CAD CLI using Spline geometry"
    )

    parser.add_argument('--size', type=float, default=2.0, help='Base size')
    parser.add_argument('--opening', type=float, default=0.8, help='Opening degree 0-1')
    parser.add_argument('--layers', type=int, default=3, help='Number of layers 1-3')
    parser.add_argument('--output', type=str, default=None, help='Output file')
    parser.add_argument('--verbose', action='store_true', help='Show details')

    args = parser.parse_args()

    print("=" * 60)
    print("Rose CLI Generator - SPLINE VERSION")
    print("=" * 60)

    if args.verbose:
        print(f"Parameters:")
        print(f"  Base Size: {args.size}")
        print(f"  Opening Degree: {args.opening}")
        print(f"  Layers: {args.layers}")
        print()

    generator = SplineRoseCLIGenerator()
    cli = generator.generate_rose(
        base_size=args.size,
        opening_degree=args.opening,
        n_layers=args.layers
    )

    if args.output:
        generator.save_cli(cli, args.output)
    else:
        print("\nGenerated CLI:")
        print("-" * 60)
        print(cli)

    print("\n✓ Spline-based Rose CLI generation complete!")


if __name__ == "__main__":
    main()
