# DuoSynco Development Makefile

.PHONY: help install install-dev test lint format type-check clean run

# Default target
help:
	@echo "DuoSynco Development Commands:"
	@echo "  install      Install package and dependencies"
	@echo "  install-dev  Install with development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting (flake8)"
	@echo "  format       Format code (black)"
	@echo "  type-check   Run type checking (mypy)"
	@echo "  clean        Clean up temporary files"
	@echo "  run          Run DuoSynco with sample file"

# Virtual environment setup
venv:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip

# Install package
install: venv
	. .venv/bin/activate && pip install -e .

# Install with development dependencies
install-dev: venv
	. .venv/bin/activate && pip install -e ".[dev]"

# Run tests
test:
	./test.sh

# Run linting
lint:
	./lint.sh

# Format code
format:
	black src/ tests/

# Type checking
type-check:
	mypy src/

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf output/
	rm -rf temp/
	rm -rf tmp/

# Run DuoSynco
run:
	@if [ -f "sample_data/example.mp4" ]; then \
		./scripts/run.sh sample_data/example.mp4 --verbose; \
	else \
		echo "No sample file found. Please add a sample video to sample_data/example.mp4"; \
	fi

# Development setup
setup-dev: install-dev
	@echo "Development environment setup complete!"
	@echo "Activate virtual environment with: source .venv/bin/activate"