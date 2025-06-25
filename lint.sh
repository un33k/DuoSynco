#!/bin/bash
# Lint script for DuoSynco project
# Runs all code quality checks

set -e  # Exit on any error

echo "🔍 Running DuoSynco Code Quality Checks..."

# Activate virtual environment (lint tools should be installed there)
source .venv/bin/activate

echo ""
echo "1️⃣ Running Black formatter..."
black src/ tests/

echo ""
echo "2️⃣ Running Flake8 linter..."
flake8 src/ tests/

echo ""
echo "3️⃣ Running MyPy type checker..."
mypy src/

echo ""
echo "4️⃣ Running Unit Tests..."
python test_runner.py --quick

echo ""
echo "✅ All code quality checks completed!"