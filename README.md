# Rose Symbolic Regression (Aicua_SR)

Symbolic Regression for CAD Rose Generation - Automatically discover optimal formulas for 3D rose flower creation.

## Overview

This project uses **Symbolic Regression (PySR)** to discover mathematical formulas for generating 3D rose flowers. Instead of hardcoding parameters, SR learns optimal relationships from data.

### Components:
- **Petals**: Geometry generation with curvature, twist, and layering
- **Bones/Rigging**: Skeletal structure for animation
- **Animation**: Wind effects, blooming sequences

## Architecture

```
User Input: "rose"
    ↓
Knowledge Graph: Define rose structure
    ↓
SR-Discovered Formulas:
  - petal_curvature = f(base_size, layer_index)
  - petal_width = g(base_size, position)
  - bone_length = h(segment_size)
    ↓
CLI Generation: 3D CAD commands
    ↓
Animated Rose Model
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample dataset
python scripts/generate_rose_dataset.py

# Train SR models (local)
python scripts/train_sr_models.py

# Generate rose CLI
python scripts/generate_rose_cli.py --size 2.0 --petals 5
```

## Project Structure

```
Aicua_SR/
├── data/
│   ├── raw/                    # Raw CLI outputs
│   ├── processed/              # Training datasets
│   │   ├── petal_geometry.csv
│   │   ├── bone_rigging.csv
│   │   └── animation_params.csv
│   ├── features/               # Feature definitions
│   ├── models/                 # Discovered SR formulas
│   └── generated/              # Auto-generated Python code
│
├── scripts/
│   ├── generate_petal_spline.py
│   ├── generate_bone_rigging.py
│   ├── generate_petal_positioning.py
│   ├── train_sr_models.py
│   └── generate_flower_cli.py
│
├── notebooks/
│   └── kaggle_sr_training.ipynb  # Main training notebook (Kaggle GPU)
│
├── tests/
│   └── test_discovered_formulas.py
│
└── examples/
    └── rose_samples/
```

## Workflow

### 1. Data Generation (On Kaggle)
- Generate synthetic datasets using mathematical formulas
- Create training data for petal splines, bone rigging, and positioning
- 3000 samples per dataset for fast testing

### 2. Symbolic Regression Training (Kaggle GPU)
- Discover formulas for petal geometry
- Learn bone/rigging relationships
- Find animation parameters

### 3. Integration
- Auto-generate Python functions
- Use discovered formulas in KnowledgeGraph
- Continuous improvement loop

## Key Concepts

### Petal Geometry
```
Petal parameters discovered by SR:
- curvature = f(layer, base_size)
- width = g(position, opening_degree)
- twist = h(layer, radial_position)
```

### Bone Rigging
```
Skeletal structure for animation:
- bone_length = f(petal_size, flexibility)
- joint_angle = g(petal_curvature, wind_force)
```

### Animation
```
Dynamic motion:
- frequency = f(petal_mass, wind_speed)
- amplitude = g(bone_length, flexibility)
```

## Integration with Kaggle

1. Upload dataset to Kaggle
2. Run PySR on GPU instances
3. Download discovered formulas
4. Auto-integrate into codebase

## License

MIT License
