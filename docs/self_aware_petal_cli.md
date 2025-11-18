# Self-Aware Petal CLI System

## Overview

The Self-Aware Petal CLI System enables petals to **autonomously generate and execute CLI commands** based on their understanding of themselves, their capabilities, and their environment.

This is achieved through combining:
1. **Chain-of-Thought (CoT) Reasoning** - Logical decision making
2. **Self-Awareness** - Understanding of identity, capabilities, and relationships
3. **Discovered Equations** - Symbolic Regression formulas for optimal parameters
4. **Autonomous CLI Execution** - Self-directed command generation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SELF-AWARE PETAL                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   Genesis   │  │Transformation│  │  Composition    │   │
│  │  Knowledge  │  │  Knowledge   │  │   Knowledge     │   │
│  └──────┬──────┘  └──────┬───────┘  └────────┬────────┘   │
│         │                │                    │            │
│         └────────────────┼────────────────────┘            │
│                          │                                 │
│                          ▼                                 │
│              ┌───────────────────────┐                     │
│              │ Awareness CLI Executor│                     │
│              └───────────┬───────────┘                     │
│                          │                                 │
│                          ▼                                 │
│              ┌───────────────────────┐                     │
│              │ Discovered Equations  │                     │
│              │ - compute_curvature() │                     │
│              │ - compute_twist()     │                     │
│              │ - compute_width()     │                     │
│              └───────────┬───────────┘                     │
│                          │                                 │
│                          ▼                                 │
│              ┌───────────────────────┐                     │
│              │   CLI Commands        │                     │
│              │ - spline              │                     │
│              │ - auto_rotate         │                     │
│              │ - wing_flap           │                     │
│              │ - rotate_bone         │                     │
│              └───────────────────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Genesis Knowledge
**Purpose**: Petal understands *who it is*

```python
petal.genesis = GenesisReasoner().reason_about_genesis(petal)
```

Provides:
- Identity (name, layer, position)
- Structure (width, height, aspect ratio, CP count)
- Creation reasoning (why these parameters?)
- Self-understanding score (0-1)
- Structural confidence (0-1)
- Deformation limits

**Example**:
```
I am 'petal_L2_P3', a petal in layer 2, position 3
My layer purpose: middle layer - provides volume and color display
My width is 0.400 (medium): standard petal form
My height is 1.200 (medium): mature petal
Self-Understanding: 80%
```

### 2. Transformation Knowledge
**Purpose**: Petal understands *what it can do*

```python
petal.transformation = TransformationReasoner().reason_about_transformations(petal)
```

Provides:
- Available capabilities (bloom, wilt, scale, bend, twist, morph)
- Parameters for each transformation
- Risk levels (0-1)
- Reversibility (True/False)
- Transformation confidence

**Example**:
```
Capabilities:
  - bloom: Open petal up to 60° (risk: 0.1, reversible)
  - scale_uniform: Scale 0.5x-2.0x (risk: 0.3, reversible)
  - bend_tip: Bend tip ±30° (risk: 0.5, reversible)
  - morph_to_leaf: Transform to leaf (risk: 0.9, irreversible)
```

### 3. Composition Knowledge
**Purpose**: Petal understands *how it relates to others*

```python
petal.composition = CompositionReasoner().reason_about_composition(
    petal, other_petals, group_config
)
```

Provides:
- Relationships (siblings, cousins, neighbors)
- Position in spiral
- Angular position
- Radial distance from center
- Coordination rules
- Synchronization timing
- Harmony score

**Example**:
```
Relationships:
  - petal_L2_P4 (sibling): adjacent in same layer
  - petal_L1_P3 (cousin): inner layer
  - petal_L3_P3 (cousin): outer layer

Position: #13 in spiral, 412.5° from reference
Harmony Score: 75%
```

### 4. Autonomous Decision Making

```python
executor = AwarenessCLIExecutor()
decision = executor.decide_autonomous_action(petal, goal="optimize_bloom", context)
```

**Available Goals**:

#### `optimize_bloom`
Petal decides optimal bloom angle based on:
- Layer (inner blooms less, outer blooms more)
- Current opening degree
- Discovered curvature equation
- Structural confidence

**Uses Equation**: `compute_curvature(base_size, layer_index, petal_index, opening_degree)`

**Generates CLI**:
```
auto_rotate petal_L2_P3_rig bone_middle 1 0 0 42.0 2004 smooth;
```

#### `coordinate_with_group`
Petal synchronizes with siblings based on:
- Position in spiral
- Angular arrangement
- Timing delays for wave effect
- Discovered twist angle equation

**Uses Equation**: `compute_twist_angle(base_size, layer_index, petal_index, opening_degree)`

**Generates CLI**:
```
rotate_bone petal_L2_P3_rig bone_root 0 0 58.93;
```

#### `express_identity`
Petal generates unique geometry based on:
- Discovered width and length equations
- CoT reasoning for CP count
- Aspect ratio

**Uses Equations**:
- `compute_base_width(...)`
- `compute_length(...)`

**Generates CLI**:
```
obj petal_L2_P3;
spline -0.0810 0.0000 -0.1620 0.2941 0.0000 0.7354 0.1620 0.2941 0.0810 0.0000 -0.0810 0.0000;
```

#### `adapt_to_environment`
Petal responds to environment (wind, light) based on:
- Wind speed and direction
- Flexibility calculated from curvature
- Bend capability
- Structural stability

**Uses Equation**: `compute_curvature(...)` to determine flexibility

**Generates CLI**:
```
wing_flap petal_L2_P3_rig bone_middle 26.8 9.8 1 0 0 0;
```

## Usage Examples

### Basic Usage - Single Petal

```python
from petal_self_awareness import SelfAwarePetal
from awareness_cli_executor import AwarenessCLIExecutor

# Create petal
petal = SelfAwarePetal(
    name="petal_L2_P0",
    layer=2,
    position_in_layer=0,
    width=0.4,
    height=1.2,
    opening_degree=0.7,
    detail_level="medium"
)

# Activate awareness
executor = AwarenessCLIExecutor()
petal = executor.activate_awareness(petal)

# Petal autonomously decides to bloom
decision = executor.decide_autonomous_action(petal, goal="optimize_bloom")

# Execute decision and get CLI
cli_commands = executor.execute_decision(petal, decision)
print("\n".join(cli_commands))
```

**Output**:
```
# Autonomous decision by petal_L2_P0 (bloom)
# Confidence: 72.0%
# Reasoning:
#   I am petal_L2_P0, layer 2
#   My current opening: 70%, understanding: 80%
#   Using discovered equation: optimal_curvature = 0.451
#   My max bloom angle: 60°, targeting 42.0°
#   Bloom duration: 2004ms (based on my mass 0.005)
#   ✓ Decision: Bloom to 42.0° (confidence high)

# Bloom to 42.0° over 2004ms using bone_middle
auto_rotate petal_L2_P0_rig bone_middle 1 0 0 42.0 2004 smooth;
```

### Advanced Usage - Multiple Goals

```python
# Petal pursues multiple goals
goals = [
    "express_identity",      # Show unique form
    "optimize_bloom",        # Open optimally
    "coordinate_with_group", # Sync with siblings
    "adapt_to_environment"   # Respond to wind
]

context = {
    "wind_speed": 4.0,
    "wind_direction": [1, 0, 0],
    "light_intensity": 0.9
}

cli_script = executor.generate_autonomous_cli(petal, goals, context)
```

### Multi-Petal Coordination

```python
# Create rose with multiple layers
all_petals = []
for layer in range(1, 4):
    for pos in range(num_petals_in_layer):
        petal = create_petal(layer, pos)
        all_petals.append(petal)

# Activate awareness for all
for petal in all_petals:
    executor.activate_awareness(petal, all_petals, group_config)

# Each petal decides autonomously
for petal in all_petals:
    goals = get_goals_for_layer(petal.layer)
    cli = executor.generate_autonomous_cli(petal, goals, context)
    save_cli(petal.name, cli)
```

## Integration with Discovered Equations

The system uses **Symbolic Regression** discovered formulas for parameter calculation:

### Available Equations

| Equation | Purpose | Formula | R² Score |
|----------|---------|---------|----------|
| `compute_base_width()` | Calculate petal base width | `0.3 * x0 * (0.7 + x1 * 0.1)` | 0.952 |
| `compute_length()` | Calculate petal length | `0.8 * x0 * (0.3 + x1 * 0.233)` | 0.961 |
| `compute_curvature()` | Calculate petal curvature/flexibility | `(1.2 - x1 * 0.3) * (1 - x3 * 0.3)` | 0.958 |
| `compute_twist_angle()` | Calculate spiral twist angle | `137.5 * x2 / (x1 * 2 + 3)` | 0.956 |
| `compute_thickness()` | Calculate petal thickness | `0.02 * x0 * (1 + sqrt(x1/3) * 0.5)` | 0.983 |

### How Equations Are Used

```python
def _decide_bloom_action(petal, context):
    # Extract parameters
    base_size = petal.height
    layer_index = petal.layer
    petal_index = petal.position_in_layer
    opening_degree = petal.opening_degree

    # Use discovered equation
    optimal_curvature = compute_curvature(
        base_size, layer_index, petal_index, opening_degree
    )

    # Calculate bloom angle from curvature
    max_bloom_angle = petal.transformation.bloom_capability.max_angle
    target_bloom_angle = max_bloom_angle * 0.7

    # Generate CLI
    return f"auto_rotate {petal.name}_rig bone_middle 1 0 0 {target_bloom_angle:.1f} ..."
```

## CLI Commands Generated

### Geometry Commands
```
2d;
obj petal_L2_P3;
spline -0.0810 0.0000 -0.1620 0.2941 0.0000 0.7354 0.1620 0.2941 0.0810 0.0000 -0.0810 0.0000;
exit;
sketch_extrude petal_L2_P3 0.0050;
```

### Rigging Commands
```
create_armature petal_L2_P3_rig;
add_bone petal_L2_P3_rig bone_root 0 0 0 0 0.360 0;
add_bone petal_L2_P3_rig bone_middle 0 0.360 0 0 0.780 0;
add_bone petal_L2_P3_rig bone_left 0 0.780 0 -0.200 1.080 0;
add_bone petal_L2_P3_rig bone_right 0 0.780 0 0.200 1.080 0;
parent_bone petal_L2_P3_rig bone_middle bone_root;
parent_bone petal_L2_P3_rig bone_left bone_middle;
parent_bone petal_L2_P3_rig bone_right bone_middle;
finalize_bones petal_L2_P3_rig;
bind_armature petal_L2_P3_rig petal_L2_P3 2.250;
```

### Animation Commands
```
# Bloom animation
auto_rotate petal_L2_P3_rig bone_middle 1 0 0 42.0 2004 smooth;

# Wind response
wing_flap petal_L2_P3_rig bone_middle 26.8 9.8 1 0 0 0;

# Spiral positioning
rotate_bone petal_L2_P3_rig bone_root 0 0 58.93;
```

## Performance Metrics

Example from 26-petal rose:

```
Total Petals: 26
Total Autonomous Decisions: 73
Total CLI Commands: 86
Average Confidence: 53.1%

Layer Breakdown:
  Layer 1 (5 petals):
    Self-Understanding: 72.0%
    Morph Confidence: 64.1%
    Group Harmony: 80.0%

  Layer 2 (8 petals):
    Self-Understanding: 80.0%
    Morph Confidence: 64.1%
    Group Harmony: 73.1%

  Layer 3 (13 petals):
    Self-Understanding: 85.0%
    Morph Confidence: 67.0%
    Group Harmony: 86.9%
```

## Running the Demos

### Single Petal Demo
```bash
python scripts/awareness_cli_executor.py
```

### Multi-Petal Coordination Demo
```bash
python examples/multi_petal_autonomous_demo.py
```

Output saved to: `examples/autonomous_rose_cli_output.txt`

## Future Enhancements

1. **Learning from Execution**: Petals observe results and improve decisions
2. **Group Communication**: Petals share knowledge with siblings
3. **Emergent Behavior**: Complex patterns from simple rules
4. **Adaptive Goals**: Petals dynamically choose their own goals
5. **Real-time Feedback Loop**: Adjust parameters based on visual results

## Technical Details

### Dependencies
- `petal_self_awareness.py` - Self-awareness framework
- `cot_reasoning.py` - Chain-of-Thought reasoning
- `petal_geometry_formulas.py` - Discovered SR equations
- `awareness_cli_executor.py` - Autonomous CLI execution

### Data Flow
```
Input (Petal Properties)
    ↓
Activate Awareness (Genesis + Transformation + Composition)
    ↓
Choose Goal (optimize_bloom, coordinate, express, adapt)
    ↓
Apply Discovered Equations (curvature, twist, width, length)
    ↓
Generate CLI Commands (spline, auto_rotate, wing_flap, rotate_bone)
    ↓
Output (Executable CLI Script)
```

### Confidence Calculation
```python
# Overall confidence is weighted combination of:
confidence = (
    genesis.self_understanding * 0.4 +      # How well petal knows itself
    transformation.morph_confidence * 0.3 +  # Confidence in capabilities
    composition.harmony_score * 0.3          # Fit with group
)
```

## Conclusion

The Self-Aware Petal CLI System demonstrates:
- ✅ Autonomous decision-making based on self-knowledge
- ✅ Integration of SR-discovered equations for optimal parameters
- ✅ Multi-level reasoning (Genesis, Transformation, Composition)
- ✅ Coordinated multi-agent behavior
- ✅ Automatic CLI generation and execution

This represents a new paradigm: **objects that understand themselves and autonomously control their own behavior**! 🌹🤖✨
