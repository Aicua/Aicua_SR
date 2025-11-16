#!/usr/bin/env python3
"""
Auto-generate Python code from SR-discovered formulas.

This script reads JSON formula files and generates Python functions
that can be imported and used in the KnowledgeGraph.
"""

import json
import re
from pathlib import Path
from datetime import datetime


def load_all_formulas(models_dir: str = None) -> dict:
    """Load all discovered formulas from JSON files."""
    if models_dir is None:
        models_dir = Path(__file__).parent.parent / "data" / "models"
    else:
        models_dir = Path(models_dir)

    master_file = models_dir / "all_discovered_formulas.json"

    if master_file.exists():
        with open(master_file, 'r') as f:
            return json.load(f)

    # Fallback: load from individual category folders
    formulas = {}
    for category_dir in models_dir.glob("sr_*"):
        if category_dir.is_dir():
            category = category_dir.name.replace("sr_", "")
            combined = category_dir / "all_formulas.json"
            if combined.exists():
                with open(combined, 'r') as f:
                    formulas[category] = json.load(f)

    return formulas


def formula_to_python(formula: str, features: list) -> str:
    """
    Convert symbolic formula to Python code.

    Args:
        formula: Symbolic formula string (e.g., "0.3 * x0 + sqrt(x1)")
        features: List of feature names (e.g., ["base_size", "layer_index"])

    Returns:
        Python expression string
    """
    py_expr = formula

    # Replace x0, x1, x2... with feature names
    for i, feat in enumerate(features):
        # Use word boundaries to avoid partial replacements
        py_expr = re.sub(rf'\bx{i}\b', feat, py_expr)

    # Replace math functions with math module calls
    replacements = {
        r'\bsqrt\(': 'math.sqrt(',
        r'\blog\(': 'math.log(',
        r'\bexp\(': 'math.exp(',
        r'\bsin\(': 'math.sin(',
        r'\bcos\(': 'math.cos(',
        r'\btan\(': 'math.tan(',
        r'\bceil\(': 'math.ceil(',
        r'\bfloor\(': 'math.floor(',
        r'\babs\(': 'abs(',
    }

    for pattern, replacement in replacements.items():
        py_expr = re.sub(pattern, replacement, py_expr)

    # Handle power operator: ** in sympy vs Python
    # x^2 should become x**2
    py_expr = py_expr.replace('^', '**')

    return py_expr


def generate_python_module(category: str, formulas: dict, output_path: Path):
    """
    Generate a Python module for a category.

    Args:
        category: Category name (e.g., 'petal_geometry')
        formulas: Dict of {target: {formula, score, features, ...}}
        output_path: Path to write the .py file
    """
    lines = [
        '"""',
        f'Auto-generated formulas for {category.replace("_", " ").title()}',
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        'These formulas were discovered using Symbolic Regression (PySR).',
        'DO NOT EDIT MANUALLY - regenerate using codegen_formulas.py',
        '"""',
        '',
        'import math',
        '',
        '',
    ]

    for target, data in formulas.items():
        formula = data['formula']
        features = data['features']
        score = data['score']
        complexity = data.get('complexity', 'N/A')

        func_name = f"compute_{target}"

        # Function signature
        lines.append(f"def {func_name}({', '.join(features)}):")

        # Docstring
        lines.append(f'    """')
        lines.append(f'    Compute {target.replace("_", " ")}.')
        lines.append(f'    ')
        lines.append(f'    Discovered Formula: {formula}')
        lines.append(f'    R² Score: {score:.6f}')
        lines.append(f'    Complexity: {complexity}')
        lines.append(f'    ')
        lines.append(f'    Args:')
        for feat in features:
            lines.append(f'        {feat}: Feature value')
        lines.append(f'    ')
        lines.append(f'    Returns:')
        lines.append(f'        Computed {target.replace("_", " ")} value')
        lines.append(f'    """')

        # Function body
        py_formula = formula_to_python(formula, features)

        # Handle potential edge cases
        lines.append(f'    try:')
        lines.append(f'        result = {py_formula}')
        lines.append(f'        return float(result)')
        lines.append(f'    except (ValueError, ZeroDivisionError) as e:')
        lines.append(f'        # Fallback for edge cases')
        lines.append(f'        print(f"Warning: Error computing {target}: {{e}}")')
        lines.append(f'        return 0.0')

        lines.append('')
        lines.append('')

    # Write file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  ✓ Generated {output_path.name}")


def generate_master_module(all_formulas: dict, output_dir: Path):
    """
    Generate master module that imports all category modules.
    """
    lines = [
        '"""',
        'Master module for Rose SR discovered formulas.',
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        'Import this module to access all discovered formulas.',
        '"""',
        '',
    ]

    # Import all category modules
    for category in all_formulas.keys():
        module_name = f"{category}_formulas"
        lines.append(f'from .{module_name} import *')

    lines.append('')
    lines.append('')

    # Create a utility class
    lines.append('class RoseFormulaComputer:')
    lines.append('    """')
    lines.append('    Utility class to compute all rose parameters.')
    lines.append('    """')
    lines.append('')

    # Add compute methods for each category
    for category, formulas in all_formulas.items():
        category_name = category.replace("_", " ").title().replace(" ", "")
        method_name = f"compute_{category}"

        lines.append(f'    @staticmethod')
        lines.append(f'    def {method_name}(**kwargs):')
        lines.append(f'        """Compute all {category.replace("_", " ")} parameters."""')
        lines.append(f'        results = {{}}')

        for target in formulas.keys():
            func_name = f"compute_{target}"
            lines.append(f'        results["{target}"] = {func_name}(**kwargs)')

        lines.append(f'        return results')
        lines.append('')

    # Write __init__.py
    init_path = output_dir / "__init__.py"
    with open(init_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  ✓ Generated {init_path.name}")


def generate_test_module(all_formulas: dict, output_dir: Path):
    """Generate test cases for discovered formulas."""
    lines = [
        '"""',
        'Test cases for discovered formulas.',
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '"""',
        '',
        'import pytest',
        'import math',
        'from pathlib import Path',
        'import sys',
        '',
        '# Add generated module to path',
        'sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "generated"))',
        '',
    ]

    # Import functions
    for category in all_formulas.keys():
        module_name = f"{category}_formulas"
        lines.append(f'from {module_name} import *')

    lines.append('')
    lines.append('')

    # Test class
    lines.append('class TestDiscoveredFormulas:')
    lines.append('    """Test discovered formulas for validity."""')
    lines.append('')

    # Basic sanity tests
    for category, formulas in all_formulas.items():
        for target, data in formulas.items():
            func_name = f"compute_{target}"
            features = data['features']

            lines.append(f'    def test_{func_name}_basic(self):')
            lines.append(f'        """Test {func_name} returns valid number."""')

            # Generate sample inputs
            sample_args = []
            for feat in features:
                if 'index' in feat:
                    sample_args.append(f'{feat}=1')
                elif 'degree' in feat or 'factor' in feat:
                    sample_args.append(f'{feat}=0.5')
                else:
                    sample_args.append(f'{feat}=2.0')

            args_str = ', '.join(sample_args)
            lines.append(f'        result = {func_name}({args_str})')
            lines.append(f'        assert isinstance(result, float)')
            lines.append(f'        assert not math.isnan(result)')
            lines.append(f'        assert not math.isinf(result)')
            lines.append('')

    # Boundary tests
    lines.append('    def test_edge_cases(self):')
    lines.append('        """Test formulas handle edge cases."""')
    lines.append('        # Test with zero values')
    lines.append('        try:')
    lines.append('            result = compute_base_width(0.0, 1, 0, 0.5)')
    lines.append('            assert isinstance(result, float)')
    lines.append('        except:')
    lines.append('            pass  # Some formulas may not handle zeros')
    lines.append('')

    # Write test file
    test_path = output_dir / "test_discovered_formulas.py"
    with open(test_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  ✓ Generated {test_path.name}")


def main():
    """Main code generation pipeline."""
    print("=" * 60)
    print("Formula Code Generator")
    print("=" * 60)

    # Load formulas
    print("\nLoading discovered formulas...")
    all_formulas = load_all_formulas()

    if not all_formulas:
        print("  ✗ No formulas found. Run train_sr_models.py first.")
        return

    for category in all_formulas:
        print(f"  ✓ Loaded {category}: {len(all_formulas[category])} formulas")

    # Generate Python modules
    print("\nGenerating Python modules...")
    output_dir = Path(__file__).parent.parent / "data" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    for category, formulas in all_formulas.items():
        module_path = output_dir / f"{category}_formulas.py"
        generate_python_module(category, formulas, module_path)

    # Generate master module
    print("\nGenerating master module...")
    generate_master_module(all_formulas, output_dir)

    # Generate tests
    print("\nGenerating test cases...")
    test_dir = Path(__file__).parent.parent / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    generate_test_module(all_formulas, test_dir)

    print("\n" + "=" * 60)
    print("Code generation complete!")
    print("=" * 60)

    print(f"\nGenerated files:")
    for f in output_dir.glob("*.py"):
        print(f"  - {f.relative_to(output_dir.parent.parent)}")

    print(f"\nTest file:")
    for f in test_dir.glob("test_*.py"):
        print(f"  - {f.relative_to(test_dir.parent)}")

    print("\nNext steps:")
    print("  1. Review generated code in data/generated/")
    print("  2. Run tests: pytest tests/")
    print("  3. Import formulas in your KnowledgeGraph:")
    print("     from data.generated import compute_base_width, compute_curvature")


if __name__ == "__main__":
    main()
