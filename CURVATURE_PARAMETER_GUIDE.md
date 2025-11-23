# Curvature Parameter Guide

## Overview

The `curvature` parameter (0.0 - 1.0) controls bone deformation style **independently** from `opening_degree`.

This decoupling allows precise artistic control:
- **opening_degree**: Controls petal geometry (width/height)
- **curvature**: Controls bone bend (S vs C shape)

## Curvature Ranges

### C-Shape (0.0 - 0.4)
**Uniform bend in same direction**
- All bones bend downward (negative rx)
- Creates smooth, closed petal curve
- Best for tight, closed petals
- Intensity increases from 0.0 (no bend) to 0.4 (maximum C-bend)

### S-Shape (0.5 - 1.0)
**Alternating bend (down → up → down)**
- Base: Bends down (negative rx)
- Mid: Bends up (positive rx) ← KEY DIFFERENCE
- Tip: Bends down (negative rx)
- Creates dramatic bloomed petal
- Intensity increases from 0.5 (slight S) to 1.0 (maximum S-bend)

## Usage

### Basic Examples

```bash
# Petal geometry only (no bones)
python scripts/cli_from_sr_formulas.py -b 3.0 -o 0.8

# Petal geometry + C-shape bones
python scripts/cli_from_sr_formulas.py -b 3.0 -o 0.8 -c 0.3

# Petal geometry + S-shape bones
python scripts/cli_from_sr_formulas.py -b 3.0 -o 0.8 -c 0.7

# Bones only (skip petal geometry)
python scripts/cli_from_sr_formulas.py -b 3.0 -o 0.8 -c 0.7 --bones-only
```

### Complete Example

```bash
python scripts/cli_from_sr_formulas.py -b 4.0 -o 0.6 -c 0.7 -l 1 -p 1
```

**Output:**
```
============================================================
Petal Geometry CLI (base_size=4.0, opening=0.6)
============================================================
2d;
obj petal_L1_P1;
spline 0.0000 0.0000 -0.2664 0.1842 ... 0.0000 0.0000;
exit;
sketch_extrude petal_L1_P1 0.0015;

============================================================
Bone Rotation CLI (curvature=0.70, S-shape (alternating bend, intensity=0.4))
============================================================
rotate_bone petal_L1_P1_rig bone_base -14.0000 0.0 0.0 head;
rotate_bone petal_L1_P1_rig bone_base -22.4000 0.0 0.0 tail;
rotate_bone petal_L1_P1_rig bone_mid 11.6000 0.0 0.0 head;   # POSITIVE = bend up
rotate_bone petal_L1_P1_rig bone_mid 24.0000 0.0 0.0 tail;   # POSITIVE = bend up
rotate_bone petal_L1_P1_rig bone_tip -10.0000 0.0 0.0 head;
rotate_bone petal_L1_P1_rig bone_tip -3.6000 0.0 0.0 tail;
```

## Curvature Effect Comparison

**Base Size: 3.0, Opening Degree: 0.8**

| Curvature | Style | bone_base_tail | bone_mid_tail | bone_tip_head | Description |
|-----------|-------|----------------|---------------|---------------|-------------|
| 0.2 | C-shape | -6.0° | -7.2° | -4.8° | Gentle uniform bend |
| 0.4 | C-shape | -12.0° | -14.4° | -9.6° | Strong uniform bend |
| 0.6 | S-shape | -14.4° | **+15.0°** | -6.0° | Slight alternating |
| 0.8 | S-shape | -19.2° | **+21.0°** | -9.0° | Dramatic bloom |

**Notice:** S-shape has **positive** bone_mid values (bends up)

## Size Effect

Larger petals need more rotation for visible curves:

```bash
# Small petal (base_size=3.0)
python scripts/cli_from_sr_formulas.py -b 3.0 -o 0.8 -c 0.8 --bones-only
# bone_mid_tail: +21.0°

# Large petal (base_size=5.0)
python scripts/cli_from_sr_formulas.py -b 5.0 -o 0.8 -c 0.8 --bones-only
# bone_mid_tail: +35.0°
```

**Formula:** `rotation_intensity = base_size * 5.0 * curvature_intensity`

## Artistic Recommendations

### Tight Closed Petals
```bash
# Narrow geometry + gentle C-curve
-b 3.0 -o 0.2 -c 0.2
```

### Natural Rose Petals
```bash
# Medium geometry + slight S-curve
-b 4.0 -o 0.5 -c 0.6
```

### Dramatic Bloomed Petals
```bash
# Wide geometry + strong S-curve
-b 5.0 -o 0.8 -c 0.9
```

### Experimental: Wide but Tight Curve
```bash
# Wide geometry + C-curve (unusual but interesting)
-b 4.0 -o 0.8 -c 0.3
```

## Command Line Options

```
-b, --base_size FLOAT       Overall rose size (2.0 - 8.0) [REQUIRED]
-o, --opening_degree FLOAT  Opening degree 0.0-1.0 [REQUIRED]
-c, --curvature FLOAT       Bone curvature 0.0-1.0 [OPTIONAL]
                            0.0-0.4: C-shape (uniform)
                            0.5-0.7: Natural S-shape
                            0.8-1.0: Dramatic S-shape
-l, --layer INT             Layer number (default: 1)
-p, --petal INT             Petal number (default: 1)
--show-coords               Show control point coordinates
--bones-only                Generate only bone CLI (skip geometry)
```

## Implementation Details

### Rotation Formula (Simplified)

```python
base_intensity = base_size * 5.0

if curvature < 0.5:  # C-shape
    intensity = curvature * 2.0  # Map 0-0.5 → 0-1
    bone_base_tail_rx = -base_intensity * intensity * 1.0
    bone_mid_tail_rx = -base_intensity * intensity * 1.2
    bone_tip_head_rx = -base_intensity * intensity * 0.8
else:  # S-shape
    intensity = (curvature - 0.5) * 2.0  # Map 0.5-1.0 → 0-1
    bone_base_tail_rx = -base_intensity * (0.8 + intensity * 0.8)
    bone_mid_tail_rx = +base_intensity * (0.8 + intensity * 1.0)  # POSITIVE!
    bone_tip_head_rx = -base_intensity * (0.3 + intensity * 0.5)
```

**Key difference:** S-shape flips bone_mid to **positive** rotation (bend up)

## Future Work: Training SR on Curvature

To train symbolic regression on the curvature parameter:

1. **Update `generate_bone_rigging.py`:**
   - Add `curvature` as a feature (0.0 - 1.0)
   - Remove random noise from rotation formulas
   - Use deterministic formulas (like above)

2. **Generate new dataset:**
   ```bash
   python scripts/generate_bone_rigging.py
   ```

3. **Train on Kaggle:**
   - Features: `[base_size, curvature]` (2 features, not 3!)
   - Targets: 6 bone rotations (head + tail for 3 bones)
   - Expected R² > 0.99 (deterministic formulas)

4. **Replace rule-based formulas with SR formulas:**
   - Update `compute_bone_rotations()` with discovered formulas

## Why Separate Curvature from opening_degree?

**Problem:** No physical relationship between petal width and bone bend

**Old approach (incorrect):**
- opening < 0.5 → C-shape
- opening >= 0.5 → S-shape
- **Issue:** Why should wide petals always have S-curves?

**New approach (correct):**
- `opening_degree` controls geometry (width/height)
- `curvature` controls bone deformation (S vs C)
- **Benefit:** Artistic freedom - can have wide C-curves or narrow S-curves

## Examples for Different Use Cases

### Generate 3 Petals with Different Curves

```bash
# Layer 1, varying curvature
python scripts/cli_from_sr_formulas.py -b 4.0 -o 0.6 -c 0.3 -l 1 -p 1
python scripts/cli_from_sr_formulas.py -b 4.0 -o 0.6 -c 0.6 -l 1 -p 2
python scripts/cli_from_sr_formulas.py -b 4.0 -o 0.6 -c 0.9 -l 1 -p 3
```

### Generate Full Rose Layer (26 Petals)

```bash
#!/bin/bash
# generate_layer.sh
layer=1
base_size=4.0
opening=0.6

for petal in {1..26}; do
  # Vary curvature by petal position
  angle=$(echo "scale=4; $petal * 2 * 3.14159 / 26" | bc)
  curvature=$(echo "scale=2; 0.6 + 0.3 * c($angle)" | bc -l)

  python scripts/cli_from_sr_formulas.py \
    -b $base_size -o $opening -c $curvature \
    -l $layer -p $petal >> layer_${layer}.cli
done
```

## Summary

✅ **Decoupled parameters**: opening_degree (geometry) ≠ curvature (bones)
✅ **Precise control**: 0-1 numeric scale for continuous variation
✅ **Two distinct styles**: C-shape (uniform) vs S-shape (alternating)
✅ **Size-aware**: Larger petals get proportionally larger rotations
✅ **Trainable**: Can use SR to discover formulas from curvature parameter
✅ **Flexible CLI**: Can generate geometry only, bones only, or both

**Next step:** Update `generate_bone_rigging.py` to include curvature parameter and remove random noise for clean SR training.
