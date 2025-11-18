# Rose Petals V3 - Realistic Petal Shape Design

## Reference Image Analysis

Based on `rose_petals.jpg`, the realistic rose petal shapes have these characteristics:

### Shape Characteristics
1. **Rounded/Heart-shaped tip** - Not pointed, often with slight indentation
2. **Undulating edges** - Multiple curves along the sides
3. **Variable widths** - Organic variation from narrow base to wide middle
4. **Soft curvature** - No sharp angles, smooth transitions
5. **Asymmetric variations** - Natural imperfections

## New 8-CP Petal Shape (V3)

### Control Point Layout

```
         CP4 (tip_left)    CP5 (tip_right)
              \   notch   /
               \   CP5   /
                \       /
         CP3 ----\     /---- CP6
        (upper    \   /    upper
         left)     \ /     right)
                    |
         CP2 ------   ------ CP7
        (mid        |       mid
         left)      |       right)
                    |
         CP1 ------   ------ CP8
        (base       |       base
         left)    notch     right)
              \           /
               \---------/
                  close
```

### 8 Control Points Definition

| CP | Name | Position | Description |
|----|------|----------|-------------|
| CP1 | base_left | (-w/4, 0) | Narrow base left |
| CP2 | lower_left | (-w/2, h*0.25) | Lower curve control |
| CP3 | upper_left | (-w/2.5, h*0.6) | Upper curve widest |
| CP4 | tip_left | (-w/6, h*0.95) | Left of tip notch |
| CP5 | tip_center | (0, h) | Tip with optional notch |
| CP6 | tip_right | (w/6, h*0.95) | Right of tip notch |
| CP7 | upper_right | (w/2.5, h*0.6) | Upper curve widest |
| CP8 | lower_right | (w/2, h*0.25) | Lower curve control |
| (close) | base_right | (w/4, 0) | Back to base (implicit) |

### Shape Variations

#### 1. Classic Petal (8 CP)
- Heart-shaped tip with subtle notch
- Smooth undulating edges
- Widest at 60% height

#### 2. Round Petal (7 CP)
- Fully rounded tip without notch
- Remove CP4 and CP6, merge into CP5
- Simpler curves

#### 3. Complex Petal (9-10 CP)
- Additional edge control points
- More detailed undulations
- For high-detail rendering

### Mathematical Formulas

```python
# Base parameters
layer_factor = 0.8 + 0.1 * layer_idx  # [0.8, 0.9, 1.0]

# Width calculations
base_spread = base_size * 0.35 * layer_factor * (1 + opening_degree * 0.2)
mid_width = base_spread * 0.7   # at 25% height
upper_width = base_spread * 0.5  # at 60% height

# Height
petal_height = base_size * layer_factor * (1.2 - opening_degree * 0.3)

# Tip notch depth
notch_depth = petal_height * 0.05 * (1 - layer_idx * 0.3)
notch_width = base_spread * 0.15

# 8 Control Points (x, y)
cp1 = (-base_spread/4, 0)                           # base_left
cp2 = (-mid_width/2, petal_height * 0.25)           # lower_left
cp3 = (-upper_width/2, petal_height * 0.6)          # upper_left
cp4 = (-notch_width, petal_height - notch_depth)    # tip_left
cp5 = (tip_offset, petal_height)                    # tip_center
cp6 = (notch_width, petal_height - notch_depth)     # tip_right
cp7 = (upper_width/2, petal_height * 0.6)           # upper_right
cp8 = (mid_width/2, petal_height * 0.25)            # lower_right
close = (base_spread/4, 0)                          # back to base
```

### Dataset Structure

#### Features (Input)
- `base_size`: Overall rose size (2.0 - 8.0)
- `layer_index`: 0 (inner), 1 (middle), 2 (outer)
- `petal_index`: Position within layer (0-12)
- `opening_degree`: 0.0 (closed) to 1.0 (fully open)

#### Targets (Output) - 17 values
- `cp1_x, cp1_y`: Base left
- `cp2_x, cp2_y`: Lower left curve
- `cp3_x, cp3_y`: Upper left curve
- `cp4_x, cp4_y`: Tip left
- `cp5_x, cp5_y`: Tip center
- `cp6_x, cp6_y`: Tip right
- `cp7_x, cp7_y`: Upper right curve
- `cp8_x, cp8_y`: Lower right curve
- `extrude_depth`: Thickness

### CoT Reasoning Update

For petal shapes:
```python
shape_patterns = {
    "petal": (7, 10, 8),  # Updated: min=7, max=10, preferred=8
    # ...
}
```

### Benefits of 8-CP Design

1. **Realistic tip** - Heart-shape with notch option
2. **Better curvature** - More control over edge waves
3. **Organic feel** - Natural imperfections possible
4. **Flexible** - Easy to reduce to 7 or extend to 9-10
5. **Animation ready** - More bones can attach to curve segments

### Migration from V2 (5-CP)

| V2 (5-CP) | V3 (8-CP) |
|-----------|-----------|
| CP1 base_left | CP1 base_left |
| CP2 mid_left | CP2 lower_left + CP3 upper_left |
| CP3 tip | CP4 tip_left + CP5 tip + CP6 tip_right |
| CP4 mid_right | CP7 upper_right + CP8 lower_right |
| CP5 base_right | close (implicit) |

### CLI Format

```
# 8-CP Petal Spline (closed curve = 9 points)
spline cp1_x cp1_y cp2_x cp2_y cp3_x cp3_y cp4_x cp4_y cp5_x cp5_y cp6_x cp6_y cp7_x cp7_y cp8_x cp8_y cp1_x cp1_y;
```

### Example Output

```
# Rose Petal L1_P0 - 8 Control Points
2d;
obj petal_L1_P0;

# Spline: base_left → lower_left → upper_left → tip_left → tip → tip_right → upper_right → lower_right → close
spline -0.1 0.0 -0.2 0.3 -0.16 0.72 -0.05 1.14 0.0 1.2 0.05 1.14 0.16 0.72 0.2 0.3 -0.1 0.0;
exit;

sketch_extrude petal_L1_P0 0.015;
```

## Implementation Files

1. `scripts/generate_petal_spline_v3.py` - 8-CP generator
2. `scripts/generate_spline_dataset.py` - Updated dataset
3. `scripts/cot_reasoning.py` - Updated petal range (7-10)
4. `configs/sr_config_spline_v3.yaml` - Training config
