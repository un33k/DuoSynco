#!/bin/bash
# Lint script for DuoSynco project
# Runs all code quality checks

set -e  # Exit on any error

echo "🔍 Running DuoSynco Linting Checks..."

# Activate virtual environment if available
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Activated virtual environment"
fi

echo ""
echo "1️⃣ Running flake8 (syntax errors)..."
flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics

echo ""
echo "2️⃣ Running flake8 (style warnings)..."
flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics --ignore=C901

echo ""
echo "3️⃣ Running black format check..."
black --check src/ tests/

echo ""
echo "4️⃣ Running mypy type checker..."
mypy src/ --ignore-missing-imports

echo ""
echo "✅ All linting checks completed successfully!"