# Technical Implementation Guide: 4-Bone Petal Deformation System

## Overview

This guide provides concrete code examples and implementation details for building the 4-bone deformation system based on the comprehensive analysis.

---

## Section 1: Data Structures

### 1.1 Bone Definition

```python
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import numpy as np

@dataclass
class BoneSegment:
    """Represents a single bone in the petal skeleton"""
    name: str
    start: Tuple[float, float]      # (x, y) in petal space
    end: Tuple[float, float]        # (x, y) in petal space
    height_range: Tuple[float, float]  # (start_percent, end_percent)
    cp_influences: List[int]        # Which CPs this bone influences
    
    @property
    def length(self) -> float:
        """Calculate bone length"""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return np.sqrt(dx**2 + dy**2)
    
    @property
    def angle(self) -> float:
        """Calculate bone rotation angle in radians"""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return np.arctan2(dy, dx)
    
    def rotate(self, angle_rad: float, pivot: Optional[Tuple[float, float]] = None):
        """Rotate bone around a pivot point"""
        if pivot is None:
            pivot = self.start
        
        # Translate
        dx = self.end[0] - pivot[0]
        dy = self.end[1] - pivot[1]
        
        # Rotate
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        new_dx = dx * cos_a - dy * sin_a
        new_dy = dx * sin_a + dy * cos_a
        
        # Translate back
        self.end = (pivot[0] + new_dx, pivot[1] + new_dy)


@dataclass
class BoneSkeleton:
    """Complete skeleton for a petal"""
    petal_height: float
    bones: Dict[str, BoneSegment]
    
    @classmethod
    def create_default(cls, petal_height: float) -> 'BoneSkeleton':
        """Create default straight skeleton"""
        return cls(
            petal_height=petal_height,
            bones={
                'bone_1_base': BoneSegment(
                    name='bone_1_base',
                    start=(0.0, 0.0),
                    end=(0.0, petal_height * 0.25),
                    height_range=(0.0, 0.25),
                    cp_influences=[0, 1, 13]  # CP1, CP2, CP14
                ),
                'bone_2_mid': BoneSegment(
                    name='bone_2_mid',
                    start=(0.0, petal_height * 0.25),
                    end=(0.0, petal_height * 0.45),
                    height_range=(0.25, 0.45),
                    cp_influences=[2, 11]  # CP3, CP13, CP4, CP12
                ),
                'bone_3_mid_upper': BoneSegment(
                    name='bone_3_mid_upper',
                    start=(0.0, petal_height * 0.45),
                    end=(0.0, petal_height * 0.62),
                    height_range=(0.45, 0.62),
                    cp_influences=[3, 4, 10, 11]  # CP4, CP12, CP5, CP11, CP6, CP10
                ),
                'bone_4_tip': BoneSegment(
                    name='bone_4_tip',
                    start=(0.0, petal_height * 0.62),
                    end=(0.0, petal_height),
                    height_range=(0.62, 1.0),
                    cp_influences=[4, 5, 9, 10, 11, 12, 7]  # CP5, CP11, CP6, CP10, CP7, CP9, CP8
                ),
            }
        )
```

### 1.2 Control Point Definition

```python
@dataclass
class ControlPoint:
    """Represents a spline control point"""
    index: int              # 0-14
    x: float
    y: float
    height_percent: float   # 0.0 to 1.0
    is_symmetric: bool      # Left/right symmetry
    is_primary: bool        # Critical for shape
    
    def __repr__(self):
        return f"CP{self.index+1}(x={self.x:.3f}, y={self.y:.3f})"


class ControlPointSet:
    """Container for all 15 control points"""
    
    def __init__(self, cp_list: List[ControlPoint]):
        self.cps = cp_list
    
    def get_by_index(self, idx: int) -> ControlPoint:
        return self.cps[idx]
    
    def get_by_height(self, height_percent: float) -> List[ControlPoint]:
        """Get CPs at specific height"""
        return [cp for cp in self.cps if abs(cp.height_percent - height_percent) < 0.01]
    
    def get_width_at_height(self, height_percent: float) -> float:
        """Get petal width at height"""
        cps = self.get_by_height(height_percent)
        if len(cps) == 0:
            return 0.0
        
        # Width = distance between left and right CPs
        left_x = min(cp.x for cp in cps)
        right_x = max(cp.x for cp in cps)
        return right_x - left_x
    
    def to_array(self) -> np.ndarray:
        """Convert to flat array for training"""
        return np.array([[cp.x, cp.y] for cp in self.cps]).flatten()
```

---

## Section 2: Transformation Functions

### 2.1 Bone-to-CP Transformation

```python
class BoneToControlPointTransformer:
    """Transform bone positions to control point positions"""
    
    def __init__(self, petal_height: float, base_spread: float):
        self.petal_height = petal_height
        self.base_spread = base_spread
        self.width_profile = self._compute_width_profile()
    
    def _compute_width_profile(self) -> Dict[float, float]:
        """Pre-compute width at each height"""
        return {
            0.00: self.base_spread * 0.20,
            0.05: self.base_spread * 0.30,
            0.25: self.base_spread * 1.05,
            0.45: self.base_spread * 1.40,
            0.62: self.base_spread * 1.60,  # WIDEST
            0.78: self.base_spread * 1.30,
            0.92: self.base_spread * 0.80,
            1.00: self.base_spread * 0.10,
        }
    
    def interpolate_width(self, height_percent: float) -> float:
        """Interpolate width at arbitrary height"""
        heights = sorted(self.width_profile.keys())
        
        for i in range(len(heights) - 1):
            h1, h2 = heights[i], heights[i+1]
            if h1 <= height_percent <= h2:
                w1 = self.width_profile[h1]
                w2 = self.width_profile[h2]
                alpha = (height_percent - h1) / (h2 - h1)
                return w1 + alpha * (w2 - w1)
        
        return 0.0
    
    def transform_bone_to_cps(self, skeleton: BoneSkeleton) -> ControlPointSet:
        """
        Convert bone positions to control point positions
        
        Key logic:
        - CP x-position influenced by bone end x
        - CP y-position = height_percent * petal_height
        - Width at height constrains CP movement
        """
        cps = []
        
        # Define CP specifications (index, height, symmetric, primary)
        cp_specs = [
            (0, 0.00, False, False),    # CP1: base center
            (1, 0.05, True, True),      # CP2/14: base edges
            (2, 0.25, True, True),      # CP3/13: lower
            (3, 0.45, True, True),      # CP4/12: mid-low
            (4, 0.62, True, True),      # CP5/11: widest
            (5, 0.78, True, True),      # CP6/10: upper
            (6, 0.92, True, True),      # CP7/9: near-tip
            (7, 1.00, False, True),     # CP8: tip
            (8, 1.00, False, True),     # CP9: near-tip right
            (9, 0.78, True, False),     # CP10: upper right
            (10, 0.62, True, False),    # CP11: widest right
            (11, 0.45, True, False),    # CP12: mid-low right
            (12, 0.25, True, False),    # CP13: lower right
            (13, 0.05, True, False),    # CP14: base right
            (14, 0.00, False, False),   # CP15: close
        ]
        
        for idx, height, is_sym, is_primary in cp_specs:
            # Get bone influence at this height
            bone_x = self._get_bone_x_at_height(skeleton, height)
            
            # Get width constraint
            width = self.interpolate_width(height)
            
            # Apply constraints
            cp_x = self._constrain_cp_x(bone_x, width, is_sym)
            cp_y = height * self.petal_height
            
            cp = ControlPoint(
                index=idx,
                x=cp_x,
                y=cp_y,
                height_percent=height,
                is_symmetric=is_sym,
                is_primary=is_primary
            )
            cps.append(cp)
        
        return ControlPointSet(cps)
    
    def _get_bone_x_at_height(self, skeleton: BoneSkeleton, 
                              height: float) -> float:
        """Interpolate bone x-position at given height"""
        # Find which bone(s) span this height
        for bone in skeleton.bones.values():
            h1, h2 = bone.height_range
            if h1 <= height <= h2:
                # Interpolate along bone
                alpha = (height - h1) / (h2 - h1)
                bone_x = bone.start[0] + alpha * (bone.end[0] - bone.start[0])
                return bone_x
        
        return 0.0
    
    def _constrain_cp_x(self, bone_x: float, width: float, 
                        is_symmetric: bool) -> float:
        """Apply width constraint to CP x-position"""
        if is_symmetric:
            # Left side: constrain to ≤ -width/2
            # Right side: constrain to ≥ +width/2
            # For now, return absolute value
            return abs(bone_x)
        else:
            return bone_x
```

### 2.2 CP-to-Bone Transformation (Inverse)

```python
class ControlPointToBoneTransformer:
    """Transform control point positions to bone positions"""
    
    @staticmethod
    def transform_cps_to_bones(cps: ControlPointSet, 
                               petal_height: float) -> BoneSkeleton:
        """
        Compute bone positions from target control points
        
        Strategy:
        - Group CPs by height ranges
        - Bone end = average of CPs at that height
        """
        
        # Group CPs by height ranges
        cp_groups = {
            (0.00, 0.25): [],
            (0.25, 0.45): [],
            (0.45, 0.62): [],
            (0.62, 1.00): [],
        }
        
        for cp in cps.cps:
            for h_range in cp_groups:
                h1, h2 = h_range
                if h1 <= cp.height_percent <= h2:
                    cp_groups[h_range].append(cp)
        
        # Compute bone end positions
        bones = {}
        bone_names = ['bone_1_base', 'bone_2_mid', 'bone_3_mid_upper', 'bone_4_tip']
        height_ranges = [(0.0, 0.25), (0.25, 0.45), (0.45, 0.62), (0.62, 1.0)]
        
        prev_bone = None
        
        for name, h_range in zip(bone_names, height_ranges):
            group = cp_groups[h_range]
            
            if group:
                # Compute bone end from CP average
                end_x = np.median([cp.x for cp in group])
                end_y = petal_height * h_range[1]
            else:
                end_x = 0.0
                end_y = petal_height * h_range[1]
            
            # Start = previous bone end
            if prev_bone:
                start_x, start_y = prev_bone.end
            else:
                start_x, start_y = 0.0, 0.0
            
            bone = BoneSegment(
                name=name,
                start=(start_x, start_y),
                end=(end_x, end_y),
                height_range=h_range,
                cp_influences=[]  # Would be populated from CP indices
            )
            
            bones[name] = bone
            prev_bone = bone
        
        return BoneSkeleton(petal_height=petal_height, bones=bones)
```

---

## Section 3: Deformation Algorithms

### 3.1 S-Curve Deformation

```python
class SDeformationController:
    """Apply S-curve deformation to petal"""
    
    def __init__(self, skeleton: BoneSkeleton, base_spread: float):
        self.skeleton = skeleton
        self.base_spread = base_spread
    
    def apply(self, intensity: float = 0.5) -> BoneSkeleton:
        """
        Apply S-curve deformation
        
        intensity: 0.0 (no curve) to 1.0 (strong S)
        
        Pattern:
        - bone_2 (25%-45%): Outward
        - bone_3 (45%-62%): Inward
        - bone_4 (62%-100%): Outward
        """
        
        # Width factors at each height
        widths = {
            0.25: self.base_spread * 1.05,
            0.45: self.base_spread * 1.40,
            0.62: self.base_spread * 1.60,
        }
        
        # Offsets
        offsets = {
            0.25: widths[0.25] * 0.30 * intensity,
            0.45: widths[0.45] * 0.30 * intensity,
            0.62: -widths[0.62] * 0.20 * intensity,
        }
        
        # Apply to bones
        self.skeleton.bones['bone_2_mid'].end = (
            offsets[0.25],
            self.skeleton.bones['bone_2_mid'].end[1]
        )
        
        self.skeleton.bones['bone_3_mid_upper'].end = (
            offsets[0.45],
            self.skeleton.bones['bone_3_mid_upper'].end[1]
        )
        
        self.skeleton.bones['bone_4_tip'].end = (
            offsets[0.62],
            self.skeleton.bones['bone_4_tip'].end[1]
        )
        
        return self.skeleton


class CDeformationController:
    """Apply C-curve (inward curl) deformation"""
    
    def __init__(self, skeleton: BoneSkeleton, base_spread: float):
        self.skeleton = skeleton
        self.base_spread = base_spread
    
    def apply(self, intensity: float = 0.5) -> BoneSkeleton:
        """
        Apply C-curve deformation
        
        All bones curve inward, stronger at widest region
        """
        
        widths = {
            0.25: self.base_spread * 1.05,
            0.45: self.base_spread * 1.40,
            0.62: self.base_spread * 1.60,
        }
        
        # Negative offsets = inward
        offsets = {
            0.25: -widths[0.25] * 0.15 * intensity,
            0.45: -widths[0.45] * 0.35 * intensity,
            0.62: -widths[0.62] * 0.35 * intensity,
        }
        
        # Apply
        self.skeleton.bones['bone_2_mid'].end = (
            offsets[0.25],
            self.skeleton.bones['bone_2_mid'].end[1]
        )
        
        self.skeleton.bones['bone_3_mid_upper'].end = (
            offsets[0.45],
            self.skeleton.bones['bone_3_mid_upper'].end[1]
        )
        
        self.skeleton.bones['bone_4_tip'].end = (
            offsets[0.62],
            self.skeleton.bones['bone_4_tip'].end[1]
        )
        
        return self.skeleton
```

### 3.2 Deformation Factory

```python
class DeformationFactory:
    """Factory for creating deformations"""
    
    @staticmethod
    def create(deformation_type: str, skeleton: BoneSkeleton, 
               base_spread: float, intensity: float) -> BoneSkeleton:
        """
        Create and apply deformation
        
        deformation_type: "straight", "s_curve", "c_curve", "wave", "twist"
        """
        
        if deformation_type == "straight":
            return skeleton  # No deformation
        
        elif deformation_type == "s_curve":
            controller = SDeformationController(skeleton, base_spread)
            return controller.apply(intensity)
        
        elif deformation_type == "c_curve":
            controller = CDeformationController(skeleton, base_spread)
            return controller.apply(intensity)
        
        elif deformation_type == "wave":
            return DeformationFactory._apply_wave(skeleton, base_spread, intensity)
        
        elif deformation_type == "twist":
            return DeformationFactory._apply_twist(skeleton, base_spread, intensity)
        
        else:
            raise ValueError(f"Unknown deformation type: {deformation_type}")
    
    @staticmethod
    def _apply_wave(skeleton: BoneSkeleton, base_spread: float, 
                    intensity: float) -> BoneSkeleton:
        """Wave deformation (sinusoidal)"""
        
        widths = {
            0.25: base_spread * 1.05,
            0.45: base_spread * 1.40,
            0.62: base_spread * 1.60,
        }
        
        # Sinusoidal wave
        offsets = {
            0.25: widths[0.25] * 0.25 * np.sin(0.0) * intensity,
            0.45: widths[0.45] * 0.25 * np.sin(np.pi/2) * intensity,
            0.62: widths[0.62] * 0.25 * np.sin(np.pi) * intensity,
        }
        
        skeleton.bones['bone_2_mid'].end = (
            offsets[0.25],
            skeleton.bones['bone_2_mid'].end[1]
        )
        skeleton.bones['bone_3_mid_upper'].end = (
            offsets[0.45],
            skeleton.bones['bone_3_mid_upper'].end[1]
        )
        skeleton.bones['bone_4_tip'].end = (
            offsets[0.62],
            skeleton.bones['bone_4_tip'].end[1]
        )
        
        return skeleton
    
    @staticmethod
    def _apply_twist(skeleton: BoneSkeleton, base_spread: float, 
                     intensity: float) -> BoneSkeleton:
        """Rotational twist deformation"""
        
        # Apply rotation to each bone
        angles = [0.0, 0.2 * intensity, 0.4 * intensity, 0.3 * intensity]
        
        for i, (bone_name, angle) in enumerate(
            zip(['bone_1_base', 'bone_2_mid', 'bone_3_mid_upper', 'bone_4_tip'], 
                angles)):
            bone = skeleton.bones[bone_name]
            bone.rotate(angle, pivot=bone.start)
        
        return skeleton
```

---

## Section 4: Dataset Generation

### 4.1 4-Bone Training Dataset Generator

```python
class BoneDatasetGenerator:
    """Generate training data for 4-bone system"""
    
    def __init__(self):
        self.deformation_types = ["straight", "s_curve", "c_curve", "wave"]
    
    def generate_sample(self, 
                       base_size: float,
                       layer_index: int,
                       petal_index: int,
                       opening_degree: float,
                       deformation_type: str,
                       intensity: float) -> Dict:
        """Generate single training sample"""
        
        # Compute petal dimensions
        layer_factor = 0.8 + 0.1 * layer_index
        petal_height = (
            base_size * layer_factor * 
            (1.2 - opening_degree * 0.3)
        )
        base_spread = (
            base_size * 0.30 * layer_factor * 
            (1 + opening_degree * 0.2)
        )
        
        # Create base skeleton
        skeleton = BoneSkeleton.create_default(petal_height)
        
        # Apply deformation
        skeleton = DeformationFactory.create(
            deformation_type,
            skeleton,
            base_spread,
            intensity
        )
        
        # Extract targets (bone positions)
        targets = {}
        for bone_name, bone in skeleton.bones.items():
            targets[f'{bone_name}_end_x'] = bone.end[0]
            targets[f'{bone_name}_end_y'] = bone.end[1]
        
        return {
            'features': {
                'base_size': base_size,
                'layer_index': layer_index,
                'petal_index': petal_index,
                'opening_degree': opening_degree,
                'deformation_type': deformation_type,
                'intensity': intensity,
            },
            'targets': targets,
        }
    
    def generate_dataset(self, n_samples: int = 500) -> pd.DataFrame:
        """Generate full dataset"""
        
        data = []
        petals_per_layer = [5, 8, 13]
        
        for _ in range(n_samples):
            base_size = np.random.uniform(2.0, 8.0)
            opening_degree = np.random.uniform(0.0, 1.0)
            deformation_type = np.random.choice(self.deformation_types)
            intensity = np.random.uniform(0.0, 1.0)
            
            for layer_idx in range(3):
                for petal_idx in range(petals_per_layer[layer_idx]):
                    sample = self.generate_sample(
                        base_size,
                        layer_idx,
                        petal_idx,
                        opening_degree,
                        deformation_type,
                        intensity
                    )
                    
                    # Flatten into single row
                    row = sample['features'].copy()
                    row.update(sample['targets'])
                    data.append(row)
        
        return pd.DataFrame(data)
```

---

## Section 5: Integration with SR Training

### 5.1 SR Configuration (YAML)

```yaml
# configs/sr_config_bone_deformation_v4.yaml

general:
  niterations: 100
  populations: 20
  population_size: 33
  model_selection: "best"
  verbosity: 1

categories:
  bone_deformation_control:
    description: "4-bone deformation system for petal shapes"
    features:
      - base_size
      - layer_index
      - petal_index
      - opening_degree
      - deformation_type_encoded  # 0=straight, 1=s_curve, 2=c_curve, 3=wave
      - intensity
    targets:
      # 4 bones × 2 coordinates (end_x, end_y)
      - bone_1_base_end_x
      - bone_1_base_end_y
      - bone_2_mid_end_x
      - bone_2_mid_end_y
      - bone_3_mid_upper_end_x
      - bone_3_mid_upper_end_y
      - bone_4_tip_end_x
      - bone_4_tip_end_y
    operators:
      binary: ["+", "-", "*", "/"]
      unary: ["sqrt", "log", "sin", "cos"]
    constraints:
      maxsize: 20

quality:
  min_r2_score: 0.90
  max_complexity: 20
  validation_split: 0.2
```

### 5.2 Discovery Function Template

```python
# Expected SR output (code generation)

def compute_bone_positions_discovered(
    base_size: float,
    layer_index: int,
    petal_index: int,
    opening_degree: float,
    deformation_type: int,  # 0=straight, 1=s_curve, 2=c_curve, 3=wave
    intensity: float
) -> Dict[str, Tuple[float, float]]:
    """
    Discovered formula for bone positions
    
    This function would be automatically generated by PySR
    """
    
    # SR would discover patterns like:
    layer_factor = 0.8 + 0.1 * layer_index
    petal_height = base_size * layer_factor * (1.2 - opening_degree * 0.3)
    base_spread = base_size * 0.30 * layer_factor * (1 + opening_degree * 0.2)
    
    # Deformation-specific formulas
    if deformation_type == 0:  # straight
        bone_1_end_x = 0.0
        bone_2_end_x = 0.0
        bone_3_end_x = 0.0
        bone_4_end_x = 0.0
        
    elif deformation_type == 1:  # s_curve
        bone_1_end_x = 0.0
        bone_2_end_x = base_spread * 1.05 * 0.30 * intensity
        bone_3_end_x = -base_spread * 1.60 * 0.20 * intensity
        bone_4_end_x = base_spread * 0.80 * 0.25 * intensity
        
    elif deformation_type == 2:  # c_curve
        bone_1_end_x = 0.0
        bone_2_end_x = -base_spread * 1.05 * 0.15 * intensity
        bone_3_end_x = -base_spread * 1.60 * 0.35 * intensity
        bone_4_end_x = -base_spread * 0.80 * 0.25 * intensity
    
    else:  # wave
        bone_1_end_x = 0.0
        bone_2_end_x = base_spread * 1.05 * 0.25 * np.sin(0) * intensity
        bone_3_end_x = base_spread * 1.60 * 0.25 * np.sin(np.pi/2) * intensity
        bone_4_end_x = base_spread * 0.80 * 0.25 * np.sin(np.pi) * intensity
    
    return {
        'bone_1_base': (bone_1_end_x, petal_height * 0.25),
        'bone_2_mid': (bone_2_end_x, petal_height * 0.45),
        'bone_3_mid_upper': (bone_3_end_x, petal_height * 0.62),
        'bone_4_tip': (bone_4_end_x, petal_height * 1.0),
    }
```

---

## Section 6: Validation & Testing

### 6.1 Reconstruction Accuracy Test

```python
def test_bone_cp_transformation_accuracy(n_tests: int = 100):
    """Verify bone-to-CP transformation preserves information"""
    
    for _ in range(n_tests):
        # Generate random skeleton
        petal_height = np.random.uniform(2.0, 8.0)
        base_spread = np.random.uniform(0.2, 1.0)
        
        skeleton = BoneSkeleton.create_default(petal_height)
        
        # Transform to CPs
        transformer = BoneToControlPointTransformer(petal_height, base_spread)
        cps = transformer.transform_bone_to_cps(skeleton)
        
        # Transform back to bones
        inverse_transformer = ControlPointToBoneTransformer()
        skeleton_reconstructed = inverse_transformer.transform_cps_to_bones(
            cps, 
            petal_height
        )
        
        # Verify accuracy
        for bone_name in skeleton.bones:
            original = skeleton.bones[bone_name].end
            reconstructed = skeleton_reconstructed.bones[bone_name].end
            
            error = np.linalg.norm(
                np.array(original) - np.array(reconstructed)
            )
            
            assert error < 0.01, f"Reconstruction error > 1% for {bone_name}"
    
    print(f"✓ All {n_tests} reconstruction tests passed")


def test_deformation_patterns(deformation_type: str):
    """Verify deformation produces expected patterns"""
    
    petal_height = 5.0
    base_spread = 1.0
    
    skeleton = BoneSkeleton.create_default(petal_height)
    
    # Apply deformation at full intensity
    skeleton_deformed = DeformationFactory.create(
        deformation_type,
        skeleton,
        base_spread,
        intensity=1.0
    )
    
    # Check patterns
    if deformation_type == "s_curve":
        # bone_2 outward, bone_3 inward, bone_4 outward
        assert skeleton_deformed.bones['bone_2_mid'].end[0] > 0
        assert skeleton_deformed.bones['bone_3_mid_upper'].end[0] < 0
        assert skeleton_deformed.bones['bone_4_tip'].end[0] > 0
        
    elif deformation_type == "c_curve":
        # All inward
        assert skeleton_deformed.bones['bone_2_mid'].end[0] < 0
        assert skeleton_deformed.bones['bone_3_mid_upper'].end[0] < 0
        assert skeleton_deformed.bones['bone_4_tip'].end[0] < 0
    
    print(f"✓ {deformation_type} pattern verified")
```

---

## Section 7: CLI Integration Example

```python
def generate_petal_with_deformation(
    base_size: float = 2.0,
    layer_index: int = 0,
    deformation_type: str = "s_curve",
    intensity: float = 0.5,
    use_discovered_formula: bool = True
) -> str:
    """
    Generate CLI for petal with specific deformation
    
    Returns: CLI command string
    """
    
    petal_height = base_size * (0.8 + 0.1 * layer_index) * (1.2 - 0.3)
    base_spread = base_size * 0.30 * (0.8 + 0.1 * layer_index)
    
    if use_discovered_formula:
        bone_positions = compute_bone_positions_discovered(
            base_size,
            layer_index,
            0,  # petal_index
            0.8,  # opening_degree
            {'straight': 0, 's_curve': 1, 'c_curve': 2, 'wave': 3}[deformation_type],
            intensity
        )
    else:
        skeleton = BoneSkeleton.create_default(petal_height)
        skeleton = DeformationFactory.create(
            deformation_type,
            skeleton,
            base_spread,
            intensity
        )
        bone_positions = {
            f'bone_{i+1}_base': skeleton.bones[f'bone_{i+1}_base'].end
            for i in range(4)
        }
    
    # Generate CLI
    cli_lines = [
        f"# Petal L{layer_index} - {deformation_type} (intensity={intensity})",
        "2d;",
        f"obj petal_L{layer_index};",
        f"# Bones: {', '.join(f'{k}={v}' for k, v in bone_positions.items())}",
        "# [CAD generation commands would go here]",
        "exit;",
    ]
    
    return '\n'.join(cli_lines)


if __name__ == "__main__":
    # Example usage
    cli = generate_petal_with_deformation(
        base_size=3.0,
        layer_index=1,
        deformation_type="c_curve",
        intensity=0.7,
        use_discovered_formula=False
    )
    print(cli)
```

---

## Summary

This implementation guide provides:

1. **Data structures** for bones and control points
2. **Transformation functions** between bone space and CP space
3. **Deformation algorithms** for S-curve, C-curve, wave, twist
4. **Dataset generation** for SR training
5. **SR configuration** and expected formula template
6. **Validation tests** for accuracy verification
7. **CLI integration** examples

The system is designed to be modular, allowing incremental development and testing at each stage.

