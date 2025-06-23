#!/bin/bash

# DuoSynco Bootstrap Script
# Sets up Python 3.11.9 via pyenv and creates virtual environment

set -e

PYTHON_VERSION="3.11.9"
VENV_DIR="./.venv"
AUTO_YES=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--yes)
            AUTO_YES=true
            shift
            ;;
        *)
            echo "Usage: $0 [-y|--yes]"
            echo "  -y, --yes    Auto-confirm all prompts"
            exit 1
            ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}✓${NC} $1"
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
    exit 1
}

ask() {
    if [[ "$AUTO_YES" == true ]]; then
        return 0
    fi
    
    local prompt="$1"
    local default="${2:-y}"
    
    echo -e "${BLUE}?${NC} $prompt (${default}/n): "
    read -r response
    response=${response:-$default}
    
    [[ "$response" =~ ^[Yy] ]]
}

# Check if pyenv is installed
check_pyenv() {
    if command -v pyenv >/dev/null 2>&1; then
        log "pyenv found: $(pyenv --version)"
        return 0
    else
        warn "pyenv not found"
        return 1
    fi
}

# Install pyenv
install_pyenv() {
    echo -e "${BLUE}Installing pyenv...${NC}"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew >/dev/null 2>&1; then
            brew install pyenv
        else
            error "Homebrew not found. Please install Homebrew first: https://brew.sh"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl https://pyenv.run | bash
        
        # Add to shell profile
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
        echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
        
        warn "Please restart your shell or run: source ~/.bashrc"
        warn "Then run this script again"
        exit 0
    else
        error "Unsupported OS. Please install pyenv manually: https://github.com/pyenv/pyenv"
    fi
    
    log "pyenv installed successfully"
}

# Install Python version
install_python() {
    log "Installing Python $PYTHON_VERSION..."
    
    if pyenv versions --bare | grep -q "^$PYTHON_VERSION$"; then
        log "Python $PYTHON_VERSION already installed"
    else
        pyenv install "$PYTHON_VERSION"
        log "Python $PYTHON_VERSION installed"
    fi
    
    # Set local version
    pyenv local "$PYTHON_VERSION"
    log "Set local Python version to $PYTHON_VERSION"
}

# Create virtual environment
create_venv() {
    if [[ -d "$VENV_DIR" ]]; then
        if ask "Virtual environment already exists. Recreate?"; then
            rm -rf "$VENV_DIR"
        else
            log "Using existing virtual environment"
            return 0
        fi
    fi
    
    log "Creating virtual environment in $VENV_DIR..."
    python -m venv "$VENV_DIR"
    log "Virtual environment created"
}

# Install dependencies
install_deps() {
    log "Activating virtual environment and installing dependencies..."
    
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -e ".[dev]"
    
    log "Dependencies installed successfully"
}

# Main execution
main() {
    echo -e "${BLUE}🚀 DuoSynco Bootstrap${NC}"
    echo "Setting up Python $PYTHON_VERSION environment..."
    echo
    
    # Check/install pyenv
    if ! check_pyenv; then
        if ask "Install pyenv?"; then
            install_pyenv
        else
            error "pyenv is required for this project"
        fi
    fi
    
    # Install Python
    install_python
    
    # Create virtual environment
    create_venv
    
    # Install dependencies
    install_deps
    
    echo
    log "Bootstrap complete!"
    echo -e "${GREEN}To activate the environment:${NC} source $VENV_DIR/bin/activate"
    echo -e "${GREEN}To run DuoSynco:${NC} ./scripts/run.sh input.mp4"
}

main "$@"