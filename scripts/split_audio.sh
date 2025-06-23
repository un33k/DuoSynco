#!/bin/bash

# DuoSynco Audio Splitting Script
# Quick script to split audio into separate speaker tracks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
SPEAKERS=2
OUTPUT_DIR="./output"
QUALITY="medium"
VERBOSE=false

# Print colored message
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Show usage
show_usage() {
    echo "DuoSynco Audio Splitter"
    echo ""
    echo "Usage: $0 <audio_file> [options]"
    echo ""
    echo "Arguments:"
    echo "  audio_file           Path to input audio file (mp3, wav, etc.)"
    echo ""
    echo "Options:"
    echo "  -s, --speakers N     Number of speakers to separate (default: 2)"
    echo "  -o, --output-dir DIR Output directory (default: ./output)"
    echo "  -q, --quality LEVEL  Quality: low, medium, high (default: medium)"
    echo "  -v, --verbose        Enable verbose output"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 sample_data/annunaki.mp3"
    echo "  $0 interview.wav -s 3 -o results -v"
    echo "  $0 podcast.mp3 --speakers 2 --quality high"
}

# Parse arguments
parse_args() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 1
    fi

    INPUT_FILE="$1"
    shift

    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--speakers)
                SPEAKERS="$2"
                shift 2
                ;;
            -o|--output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -q|--quality)
                QUALITY="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Validate input file
validate_input() {
    if [ ! -f "$INPUT_FILE" ]; then
        print_message $RED "❌ Error: Input file '$INPUT_FILE' not found"
        exit 1
    fi

    # Check if it's an audio file
    if ! file "$INPUT_FILE" | grep -i audio > /dev/null; then
        print_message $YELLOW "⚠️  Warning: File may not be a valid audio file"
    fi

    print_message $GREEN "✅ Input file validated: $INPUT_FILE"
}

# Main execution
main() {
    print_message $BLUE "🎵 DuoSynco Audio Splitter"
    
    # Parse command line arguments
    parse_args "$@"
    
    # Validate input
    validate_input
    
    # Get script directory and project root
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    
    print_message $BLUE "🚀 Starting audio splitting process..."
    
    if [ "$VERBOSE" = true ]; then
        print_message $BLUE "📋 Configuration:"
        echo "  Input file: $INPUT_FILE"
        echo "  Output directory: $OUTPUT_DIR"
        echo "  Expected speakers: $SPEAKERS"
        echo "  Quality: $QUALITY"
    fi
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Build arguments for the main script
    ARGS=("$INPUT_FILE")
    ARGS+=("--output-dir" "$OUTPUT_DIR")
    ARGS+=("--speakers" "$SPEAKERS")
    ARGS+=("--quality" "$QUALITY")
    
    if [ "$VERBOSE" = true ]; then
        ARGS+=("--verbose")
    fi
    
    # Run the main DuoSynco script
    print_message $BLUE "🔄 Processing with DuoSynco..."
    ./scripts/run.sh "${ARGS[@]}"
    
    # Success message
    print_message $GREEN "✅ Audio splitting complete!"
    print_message $GREEN "📁 Check the output directory: $OUTPUT_DIR"
}

# Run main function with all arguments
main "$@"