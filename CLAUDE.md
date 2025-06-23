# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DuoSynco is a Python-based video/audio synchronization tool that separates speakers in video files and creates isolated audio tracks. The tool uses speaker diarization to identify different voices and generates separate video files where each contains only one speaker's audio while maintaining perfect video synchronization.

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Running the Application
```bash
# Using Python module
python -m src.main input_video.mp4 --output-dir ./output

# Using bash script (recommended)
./scripts/run.sh input_video.mp4 --speakers 2 --quality high --verbose

# With various options
./scripts/run.sh interview.mp4 -o results -s 3 -f mp4 -q medium -v
```

### Testing
```bash
# Run tests (when implemented)
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Architecture

DuoSynco follows a modular architecture with clear separation of concerns:

### Core Components

1. **Audio Processing** (`src/audio/`)
   - `diarization.py`: Speaker identification using pyannote.audio
   - `processor.py`: Audio manipulation and format handling
   - `isolator.py`: Creates isolated audio tracks for each speaker

2. **Video Processing** (`src/video/`)
   - `processor.py`: Video file operations using MoviePy/FFmpeg
   - `synchronizer.py`: Syncs video with isolated audio tracks

3. **Utilities** (`src/utils/`)
   - `config.py`: Configuration management and settings
   - `file_handler.py`: File I/O operations and validation

4. **CLI Interface** (`src/main.py`)
   - Click-based command-line interface
   - Orchestrates the processing pipeline

### Processing Pipeline

1. **Input Validation**: Verify file format and accessibility
2. **Speaker Diarization**: Identify when each speaker talks
3. **Audio Isolation**: Create separate audio tracks per speaker
4. **Video Synchronization**: Replace original audio with isolated tracks
5. **Output Generation**: Save synchronized videos for each speaker

## Technology Stack

- **Core Language**: Python 3.8+
- **Audio Processing**: 
  - `pyannote.audio` for speaker diarization
  - `librosa` for audio analysis
  - `pydub` for audio manipulation
- **Video Processing**:
  - `moviepy` for video operations
  - `FFmpeg` for advanced video/audio processing
- **CLI Framework**: `click`
- **Dependencies**: See `requirements.txt` and `pyproject.toml`

## File Structure

```
DuoSynco/
├── src/
│   ├── __init__.py
│   ├── main.py                 # CLI entry point
│   ├── audio/
│   │   ├── diarization.py      # Speaker separation
│   │   ├── processor.py        # Audio processing
│   │   └── isolator.py         # Voice isolation
│   ├── video/
│   │   ├── processor.py        # Video processing
│   │   └── synchronizer.py     # Video-audio sync
│   └── utils/
│       ├── config.py           # Configuration
│       └── file_handler.py     # File operations
├── scripts/
│   └── run.sh                  # Bash execution script
├── tests/                      # Test files
├── sample_data/                # Test input files
├── output/                     # Generated output
├── requirements.txt            # Python dependencies
└── pyproject.toml             # Project configuration
```

## Configuration

The application supports multiple configuration methods:

1. **Command-line arguments** (highest priority)
2. **Environment variables** (prefix: `DUOSYNCO_`)
3. **Configuration files** (JSON format)
4. **Default values**

### Key Settings
- `quality`: Processing quality (low/medium/high)
- `output_format`: Video format (mp4/avi/mov)
- `verbose`: Enable detailed logging
- `speakers`: Expected number of speakers
- `cleanup_temp_files`: Auto-cleanup temporary files

## Dependencies

### Required
- `click`: CLI framework
- `librosa`: Audio analysis
- `pydub`: Audio manipulation
- `moviepy`: Video processing
- `pyannote.audio`: Speaker diarization
- `torch`: Deep learning backend

### Optional but Recommended
- `FFmpeg`: Advanced video/audio processing (install separately)
- `soundfile`: Audio I/O (comes with librosa)

## Development Notes

- **Performance**: Large video files may require significant memory and processing time
- **Quality**: Higher quality settings increase processing time but improve output
- **Format Support**: Supports common video formats (mp4, avi, mov, mkv) and audio formats (mp3, wav, aac)
- **Error Handling**: Comprehensive error handling with fallback methods
- **Temporary Files**: Uses system temp directory with automatic cleanup
- **Threading**: Currently single-threaded, but architecture supports future multi-threading

## Troubleshooting

- **FFmpeg not found**: Install FFmpeg separately for best performance
- **CUDA/GPU issues**: PyTorch may default to CPU; this is normal for most use cases
- **Memory errors**: Reduce quality setting or process shorter video segments
- **Poor speaker separation**: Try adjusting the number of expected speakers