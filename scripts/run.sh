#!/bin/bash

# DuoSynco Execution Script
# Simple bash wrapper for running the Python CLI

set -e  # Exit on any error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        if ! command -v python &> /dev/null; then
            print_message $RED "❌ Python not found. Please install Python 3.8 or later."
            exit 1
        else
            PYTHON_CMD="python"
        fi
    else
        PYTHON_CMD="python3"
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        print_message $RED "❌ Python 3.8 or later required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    print_message $GREEN "✅ Python $PYTHON_VERSION found"
}

# Check if required packages are installed
check_dependencies() {
    print_message $BLUE "🔍 Checking dependencies..."
    
    cd "$PROJECT_ROOT"
    
    # Check if virtual environment exists
    if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
        print_message $YELLOW "⚠️  No virtual environment found. Creating one..."
        $PYTHON_CMD -m venv venv
        
        # Activate virtual environment
        if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
        elif [ -f "venv/Scripts/activate" ]; then
            source venv/Scripts/activate
        fi
        
        # Install dependencies
        print_message $BLUE "📦 Installing dependencies..."
        pip install --upgrade pip
        pip install -e .
    else
        # Activate existing virtual environment
        if [ -d "venv" ]; then
            if [ -f "venv/bin/activate" ]; then
                source venv/bin/activate
            elif [ -f "venv/Scripts/activate" ]; then
                source venv/Scripts/activate
            fi
        elif [ -d ".venv" ]; then
            if [ -f ".venv/bin/activate" ]; then
                source .venv/bin/activate
            elif [ -f ".venv/Scripts/activate" ]; then
                source .venv/Scripts/activate
            fi
        fi
    fi
    
    # Verify installation
    if ! $PYTHON_CMD -c "import src.main" &> /dev/null; then
        print_message $YELLOW "⚠️  DuoSynco not properly installed. Installing..."
        pip install -e .
    fi
}

# Check if FFmpeg is available (optional but recommended)
check_ffmpeg() {
    if command -v ffmpeg &> /dev/null; then
        FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -n1 | cut -d' ' -f3)
        print_message $GREEN "✅ FFmpeg $FFMPEG_VERSION found"
    else
        print_message $YELLOW "⚠️  FFmpeg not found - some features may be limited"
        echo "   Install FFmpeg for better performance:"
        echo "   - macOS: brew install ffmpeg"
        echo "   - Ubuntu: sudo apt install ffmpeg"
        echo "   - Windows: Download from https://ffmpeg.org/"
    fi
}

# Show usage information
show_usage() {
    echo "DuoSynco - Video/Audio Synchronization Tool"
    echo ""
    echo "Usage: $0 <input_file> [options]"
    echo ""
    echo "Arguments:"
    echo "  input_file           Path to input video/audio file"
    echo ""
    echo "Options:"
    echo "  -o, --output-dir     Output directory (default: ./output)"
    echo "  -s, --speakers       Number of speakers (default: 2)"
    echo "  -f, --format         Output format: mp4, avi, mov (default: mp4)"
    echo "  -q, --quality        Quality: low, medium, high (default: medium)"
    echo "  -v, --verbose        Enable verbose output"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 podcast.mp4"
    echo "  $0 interview.wav -o results -s 2 -q high -v"
    echo "  $0 meeting.mov --output-dir ./processed --speakers 3"
}

# Main execution
main() {
    print_message $BLUE "🎬 DuoSynco - Video/Audio Synchronization Tool"
    
    # Check if help requested or no arguments
    if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        show_usage
        exit 0
    fi
    
    # System checks
    check_python
    check_dependencies
    check_ffmpeg
    
    print_message $BLUE "🚀 Starting DuoSynco..."
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Run the Python CLI with all arguments
    $PYTHON_CMD -m src.main "$@"
    
    # Check exit code
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        print_message $GREEN "✅ DuoSynco completed successfully!"
    else
        print_message $RED "❌ DuoSynco failed with exit code $EXIT_CODE"
    fi
    
    exit $EXIT_CODE
}

# Run main function with all arguments
main "$@"