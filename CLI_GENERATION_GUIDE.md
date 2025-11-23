# CLI Generation from SR Formulas - Usage Guide

This guide explains how to generate CLI output from symbolic regression (SR) discovered formulas.

## Overview

After training PySR models on Kaggle, you can use the discovered formulas to generate CLI commands for petal geometry. Two scripts are provided:

1. **`generate_cli_from_formulas.py`** - Uses original mathematical formulas
2. **`cli_from_sr_formulas.py`** - Uses SR-discovered formulas (template)

## Quick Start

### Using Original Formulas

```bash
# Generate CLI for a single petal
python scripts/generate_cli_from_formulas.py --base_size 3.0 --opening_degree 0.8

# Show control point coordinates
python scripts/generate_cli_from_formulas.py -b 5.0 -o 0.5 --show-coords

# Specify layer and petal numbers for naming
python scripts/generate_cli_from_formulas.py -b 4.0 -o 0.6 -l 2 -p 3
```

### Using SR Formulas

```bash
# After updating sr_* functions with your Kaggle formulas:
python scripts/cli_from_sr_formulas.py --base_size 3.0 --opening_degree 0.8
```

## Expected CLI Output Format

```
2d;
obj petal_L1_P1;
spline 0.0000 0.0000 -0.2349 0.1426 -0.4933 0.6480 ... 0.0000 0.0000;
exit;
sketch_extrude petal_L1_P1 0.010;
```

## Integrating SR Formulas from Kaggle Training

### Step 1: Train on Kaggle

Run the notebook `notebooks/kaggle_sr_training.ipynb` to discover formulas.

After training completes, you'll see output like:

```
cp1_x = 0.0 (R²=1.0000)
cp1_y = 0.0 (R²=1.0000)
cp2_x = -0.12*base_size*(1.08 + 0.06*opening_degree) (R²=0.9982)
cp2_y = 0.066*base_size*(1.2 - 0.3*opening_degree) (R²=0.9756)
...
extrude_depth = 0.005*base_size*(1.0 - 0.3*opening_degree) (R²=0.9912)
```

### Step 2: Update SR Functions

Open `scripts/cli_from_sr_formulas.py` and replace each `sr_*` function with the actual formula:

**Example - Before (placeholder):**
```python
def sr_cp2_x(base_size, opening_degree):
    """SR formula for cp2_x - REPLACE WITH ACTUAL FORMULA"""
    return -0.12 * base_size * (1.08 + 0.06 * opening_degree)
```

**After (with actual SR formula from Kaggle):**
```python
def sr_cp2_x(base_size, opening_degree):
    """SR formula: -0.11973*base_size*(1.0812 + 0.0588*opening_degree)"""
    return -0.11973 * base_size * (1.0812 + 0.0588 * opening_degree)
```

### Step 3: Handle Complex Formulas

If SR discovers complex formulas with `sqrt`, `square`, etc.:

```python
def sr_cp5_x(base_size, opening_degree):
    """SR formula: -0.4*base_size*sqrt(1.08 + 0.06*opening_degree**2)"""
    return -0.4 * base_size * math.sqrt(1.08 + 0.06 * opening_degree**2)
```

### Step 4: Test Your Formulas

```bash
python scripts/cli_from_sr_formulas.py -b 3.0 -o 0.8 --show-coords
```

Compare the output with your training data to verify correctness.

## Parameters

### Input Parameters

- **`base_size`** (required): Overall rose size (typical range: 2.0 - 8.0)
  - Smaller values → smaller petals
  - Larger values → larger petals

- **`opening_degree`** (required): Petal opening (0.0 - 1.0)
  - `0.0` → Closed, tight petal (taller, narrower)
  - `1.0` → Fully open petal (shorter, wider)

- **`--layer`** (optional): Layer number for naming (default: 1)

- **`--petal`** (optional): Petal number for naming (default: 1)

- **`--show-coords`** (optional): Display control point coordinates

### Output

The script generates CLI commands in the format:

```
2d;                           # Enter 2D mode
obj petal_L{layer}_P{petal};  # Create object
spline [15 control points];   # Define spline (30 coords)
exit;                         # Exit object definition
sketch_extrude petal_L{layer}_P{petal} {depth};  # Extrude
```

## Examples

### Example 1: Small Closed Petal

```bash
python scripts/generate_cli_from_formulas.py -b 2.5 -o 0.2
```

**Output:**
```
2d;
obj petal_L1_P1;
spline 0.0000 0.0000 -0.1727 0.1724 -0.3627 0.7830 ...;
exit;
sketch_extrude petal_L1_P1 0.011;
```

### Example 2: Large Open Petal

```bash
python scripts/generate_cli_from_formulas.py -b 7.0 -o 0.9
```

**Output:**
```
2d;
obj petal_L1_P1;
spline 0.0000 0.0000 -0.4488 0.2564 -0.9424 1.1655 ...;
exit;
sketch_extrude petal_L1_P1 0.023;
```

### Example 3: Multiple Petals in a Layer

```bash
# Generate 3 petals in layer 1 with different opening degrees
for i in {1..3}; do
  opening=$(echo "scale=1; $i * 0.3" | bc)
  python scripts/generate_cli_from_formulas.py -b 4.0 -o $opening -l 1 -p $i
done
```

## Understanding the Output

### Control Points (15 total)

The spline has 15 control points defining the petal shape:

- **CP1**: Base center (0, 0)
- **CP2/CP14**: Base sides (5% height) - NARROW like a point
- **CP3/CP13**: Lower (25% height)
- **CP4/CP12**: Mid-low (45% height)
- **CP5/CP11**: Upper-mid (62% height) - **WIDEST**
- **CP6/CP10**: Upper (78% height)
- **CP7/CP9**: Near-tip (92% height)
- **CP8**: Tip center (100% height)
- **CP15**: Close spline (back to 0, 0)

### Petal Shape Characteristics

- Base: NARROW like a point (~5% height)
- Expansion: Quick from base to 25%
- Widest: At 60-62% height (NOT 50%)
- Taper: Gradual from 62% to rounded tip at 100%

### Extrude Depth

Ultra-thin thickness (typically 0.001 - 0.025):
- Smaller with larger `opening_degree`
- Proportional to `base_size`

## Formula Quality Metrics

When reviewing SR training results:

### Good R² Scores
- **R² > 0.99**: Excellent - formula captures relationship very well
- **R² > 0.95**: Good - acceptable for most targets
- **R² > 0.90**: Fair - may need more iterations or different operators

### Poor R² Scores
- **R² < 0.90**: Poor - investigate:
  - Is there randomness in the generation code?
  - Are there enough samples?
  - Do you need different operators (sin, cos, exp)?
  - Should you skip training this target (if it's constant)?

### Typical R² Results (After Fixes)

From the latest training with 2 features (base_size, opening_degree):

```
✓ cp1_x, cp1_y: R² = 1.0 (constant 0)
✓ cp2_y, cp8_y, cp14_y: R² > 0.99 (height-based)
✓ cp3-cp7, cp9-cp13 (x,y): R² > 0.99 (good formulas)
⚠ cp2_x, cp14_x: R² ~0.64 (random base_width_factor in generation)
⚠ cp8_x: R² ~0.005 (tip_x_offset = 0 when layer_idx=0)
✓ extrude_depth: R² > 0.99
```

## Troubleshooting

### Problem: Formulas with division by zero

**Solution:** Use try-except in SR functions:

```python
def sr_cp3_x(base_size, opening_degree):
    """SR formula with division"""
    try:
        return -0.26 * base_size / (1.0 - 0.05 * opening_degree)
    except ZeroDivisionError:
        return -0.26 * base_size
```

### Problem: R² score is very low (< 0.5)

**Causes:**
1. Random variance in generation code (e.g., `np.random.uniform`)
2. Fixed values with no variance (e.g., `layer_idx=0` always)
3. Not enough training iterations

**Solutions:**
1. Reduce randomness in generation code
2. Skip training for constant targets
3. Increase `niterations` in PySR (100 → 200 → 500)
4. Try different operators (`sin`, `cos`, `exp`, `log`)

### Problem: Asymmetric petals (left ≠ right)

**Cause:** Formulas for symmetric points (e.g., CP3 and CP13) are different

**Solution:** After training, enforce symmetry:

```python
def sr_cp13_x(base_size, opening_degree):
    """Symmetric to cp3_x"""
    return -sr_cp3_x(base_size, opening_degree)  # Negate for symmetry
```

## Advanced Usage

### Batch Generation

Create a script to generate multiple petals:

```bash
#!/bin/bash
# generate_layer.sh - Generate all petals in a layer

layer=1
base_size=4.0

for petal in {1..26}; do
  angle=$(echo "scale=4; $petal * 2 * 3.14159 / 26" | bc)
  opening=$(echo "scale=2; 0.5 + 0.3 * c($angle)" | bc -l)

  python scripts/cli_from_sr_formulas.py \
    -b $base_size \
    -o $opening \
    -l $layer \
    -p $petal >> layer_${layer}.cli
done
```

### Validation

Compare SR output with original data:

```python
import pandas as pd

# Load training data
df = pd.read_csv('data/processed/petal_spline.csv')

# Test SR formulas
for _, row in df.head(10).iterrows():
    base_size = row['base_size']
    opening = row['opening_degree']

    # Compute using SR
    sr_cp2_x_val = sr_cp2_x(base_size, opening)

    # Compare with actual
    actual = row['cp2_x']
    diff = abs(sr_cp2_x_val - actual)

    print(f"base={base_size:.2f}, open={opening:.2f}: "
          f"SR={sr_cp2_x_val:.4f}, actual={actual:.4f}, diff={diff:.4f}")
```

## Next Steps

1. ✅ Train PySR models on Kaggle
2. ✅ Review R² scores and formulas
3. ✅ Update `cli_from_sr_formulas.py` with actual formulas
4. ✅ Test CLI generation with various inputs
5. ✅ Validate against training data
6. 🔄 Generate full rose geometry (3 layers × 26 petals)
7. 🔄 Integrate with bone rigging and positioning formulas

## Reference

- **Training notebook**: `notebooks/kaggle_sr_training.ipynb`
- **Generation scripts**: `scripts/generate_petal_spline.py`
- **CLI generators**:
  - `scripts/generate_cli_from_formulas.py` (original formulas)
  - `scripts/cli_from_sr_formulas.py` (SR formulas)

---

**Note**: Always verify the R² scores from your Kaggle training before using the formulas in production. Formulas with R² < 0.95 may need additional tuning.
