#!/bin/bash
# Quick start pipeline for Rose SR

set -e

echo "=================================="
echo "Rose Symbolic Regression Pipeline"
echo "=================================="

# Step 1: Generate dataset
echo ""
echo "[1/4] Generating training dataset..."
python scripts/generate_rose_dataset.py

# Step 2: Train SR models
echo ""
echo "[2/4] Training SR models (mock mode without PySR)..."
python scripts/train_sr_models.py

# Step 3: Generate Python code
echo ""
echo "[3/4] Generating Python code from formulas..."
python scripts/codegen_formulas.py

# Step 4: Generate sample CLI
echo ""
echo "[4/4] Generating sample rose CLI..."
python scripts/generate_rose_cli.py --size 2.5 --opening 0.7 --layers 2 --output examples/generated_rose.cli

echo ""
echo "=================================="
echo "Pipeline complete!"
echo "=================================="

echo ""
echo "Generated files:"
ls -lh data/processed/*.csv
echo ""
ls -lh data/models/*/all_formulas.json 2>/dev/null || echo "Models in data/models/"
echo ""
ls -lh data/generated/*.py 2>/dev/null || echo "Generated code in data/generated/"
echo ""
ls -lh examples/generated_rose.cli 2>/dev/null || true

echo ""
echo "Next steps:"
echo "  - Review data/generated/*.py for discovered formulas"
echo "  - Check examples/generated_rose.cli for sample output"
echo "  - Install PySR for real training: pip install pysr"
