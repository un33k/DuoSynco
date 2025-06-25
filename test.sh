#!/bin/bash
# Test script for DuoSynco project
# Runs pytest with coverage

set -e  # Exit on any error

echo "🧪 Running DuoSynco Tests..."

# Activate virtual environment if available
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Activated virtual environment"
fi

echo ""
echo "Running tests with coverage..."
pytest tests/ -v --cov=src --cov-report=xml --cov-report=term-missing

echo ""
echo "✅ All tests completed successfully!"