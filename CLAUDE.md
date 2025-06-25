# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DuoSynco is a Python-based video/audio synchronization tool that separates speakers in video files and creates isolated audio tracks. The tool uses AssemblyAI's professional-grade speaker diarization API to identify different voices with high accuracy and generates separate video files where each contains only one speaker's audio while maintaining perfect video synchronization.

## Development Commands

### Environment Setup

```bash
# Create virtual environment (macOS/Linux only)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### Running the Application

```bash
# CRITICAL: Always source .zshrc first to load API keys, then activate virtual environment
source ~/.zshrc && source .venv/bin/activate

# API key should now be available from .zshrc
# No need to manually export ASSEMBLYAI_API_KEY

# Using Python module
python -m src.main input_video.mp4 --output-dir ./output

# Test with sample audio file
python -m src.main sample_data/annunaki.mp3 --verbose --speakers 2 --output-dir ./output

# With various options
python -m src.main interview.mp4 -o ./output -s 2 -f mp4 -l en --enhanced-processing -v
```

### Testing

```bash
# CRITICAL: Always source .zshrc first to load API keys, then activate virtual environment
source ~/.zshrc && source .venv/bin/activate

# Run tests (when implemented)
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Code Quality

```bash
# CRITICAL: Always source .zshrc first to load API keys, then activate virtual environment
source ~/.zshrc && source .venv/bin/activate

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
- **Speaker Diarization**:
  - Direct HTTP API calls to AssemblyAI for maximum control
  - `requests` library for cross-platform HTTP communication
  - Provider pattern supporting multiple services (AssemblyAI, future ElevenLabs)
- **Audio Processing**:
  - `librosa` for audio analysis
  - `pydub` for audio manipulation
  - `soundfile` for audio I/O
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
- `requests`: HTTP API client for cross-platform compatibility
- `numpy`: Numerical computing
- `soundfile`: Audio I/O operations
- `librosa`: Audio analysis
- `pydub`: Audio manipulation
- `moviepy`: Video processing

### Optional but Recommended

- `FFmpeg`: Advanced video/audio processing (install separately)
- `pytest`: Testing framework
- `black`: Code formatting
- `flake8`: Code linting

## Development Notes

- **Environment Setup**: **ALWAYS** run `source ~/.zshrc && source .venv/bin/activate` before any Python commands to load API keys and activate virtual environment
- **Performance**: Large video files may require significant memory and processing time
- **Quality**: Higher quality settings increase processing time but improve output
- **Format Support**: Supports common video formats (mp4, avi, mov, mkv) and audio formats (mp3, wav, aac)
- **Error Handling**: Comprehensive error handling with fallback methods
- **Temporary Files**: Uses system temp directory with automatic cleanup
- **Threading**: Currently single-threaded, but architecture supports future multi-threading

## Coding Standards

### Cross-Platform Path Operations
- **ALWAYS** use `pathlib.Path` for all file and directory operations
- **NEVER** use hardcoded path separators (`'/'` or `'\'` in strings)
- **NEVER** use `os.path.join()`, `os.path.exists()`, `os.path.dirname()`, etc.
- **ALWAYS** use Path object methods: `Path.exists()`, `Path.parent`, `Path.name`, `Path.suffix`
- **ALWAYS** use the `/` operator for path concatenation: `Path('dir') / 'file.txt'`
- **ALWAYS** define file extension and directory constants at module level
- **NEVER** hardcode relative paths like `'./output'` - use `Path('output')` instead

#### Examples:
```python
# ✅ GOOD - Cross-platform compatible
AUDIO_EXTENSIONS = ['.mp3', '.wav', '.m4a', '.aac']
DEFAULT_OUTPUT_DIR = Path('output')

audio_path = Path(audio_file)
if not audio_path.exists():
    raise FileNotFoundError(f"File not found: {audio_file}")

output_file = output_dir / f"{base_name}{AUDIO_EXTENSIONS[0]}"
file_size = audio_path.stat().st_size

# ❌ BAD - Platform-specific, brittle
if not os.path.exists(audio_file):
    raise FileNotFoundError(f"File not found: {audio_file}")

output_file = output_dir + "/" + base_name + ".mp3"
file_size = os.path.getsize(audio_file)
```

### Python Shebang Lines
- **NEVER** use `#!/usr/bin/env python3` in any Python files within this project
- For executable Python scripts that need a shebang, use `#!/usr/bin/env python` which will point to the virtual environment's Python interpreter
- Most Python module files should NOT have shebang lines at all since they are imported, not executed directly
- Only add shebangs to files that are intended to be executed directly (like main.py or standalone scripts)

### Python in Bash Anti-Pattern
- **NEVER** wrap or embed Python code inside bash scripts using `python -c` or heredocs
- **NEVER** use bash for complex logic - bash should only be used for simple script invocation
- **ALWAYS** create separate Python modules for any logic beyond basic file operations
- **ALWAYS** make code modular - separate functions into different files/modules for maintainability
- If you need a bash wrapper, keep it minimal (< 20 lines) and only for environment setup and Python invocation

### HTTP API Implementation
- **Direct API calls**: Use HTTP requests instead of SDKs for maximum control and cross-platform compatibility
- **Full request/response control**: Manage timeouts, retries, headers, and error handling explicitly
- **Minimal dependencies**: Only rely on `requests` library for HTTP communication
- **Debugging transparency**: Easy to inspect and log all API interactions
- **Provider independence**: Each provider implements its own HTTP client without shared SDK dependencies

### Claude Code Execution
- **For Claude Code**: ALWAYS run `source ~/.zshrc && source .venv/bin/activate &&` before any Python command
- This ensures API keys from .zshrc are loaded AND virtual environment is activated
- Example: `source ~/.zshrc && source .venv/bin/activate && python -m src.main --list-providers`

## CLI Standards and Conventions

### Option Naming Convention
- **ALWAYS provide both short and long options** for all CLI parameters (e.g., `-p` and `--provider`)
- **Exception**: Only skip the short option if it's already taken by another option
- **Examples**:
  - `--provider, -p` ✅ (both provided)
  - `--verbose, -v` ✅ (both provided) 
  - `--version` ✅ (only long, because `-v` is taken by `--verbose`)
  - `--help` ✅ (only long, because `-h` is taken by `--help`)

### Provider Architecture
- **Provider vs Feature vs Mode distinction**:
  - `--provider` (`-p`): The service provider (e.g., `assemblyai`, `elevenlabs`)
  - **Feature**: What the provider does (e.g., STT, TTS, diarization)
  - `--mode`: The workflow type (e.g., `diarization`, `edit`, `tts`)
- **Examples**:
  - `elevenlabs` is the provider
  - `stt` is a feature of ElevenLabs
  - `edit` is the mode that uses ElevenLabs STT feature
  - NOT `elevenlabs-stt` as a provider name

### Execution Path Examples
```bash
# ElevenLabs provider, STT feature, edit mode
python -m src.main input.mp4 -p elevenlabs --mode edit

# ElevenLabs provider, TTS feature, tts mode  
python -m src.main transcript.json -p elevenlabs --mode tts

# AssemblyAI provider, diarization feature, diarization mode (default)
python -m src.main input.mp4 -p assemblyai

# Note: TTS mode uses transcript file as input
python -m src.main transcript.json -p elevenlabs --mode tts --total-duration 120
```

### Short Option Assignments
- `-p`: `--provider`
- `-o`: `--output-dir`
- `-s`: `--speakers`
- `-f`: `--format`
- `-q`: `--quality`
- `-l`: `--language`
- `-v`: `--verbose`
- `-sp`: `--secondary-provider`
- `-sq`: `--stt-quality`
- `-ei`: `--edit-interactive`
- `-sm`: `--speaker-mapping`
- `-ot`: `--output-transcript`
- `-lep`: `--list-execution-paths`

## Troubleshooting

- **FFmpeg not found**: Install FFmpeg separately for best performance
- **CUDA/GPU issues**: PyTorch may default to CPU; this is normal for most use cases
- **Memory errors**: Reduce quality setting or process shorter video segments
- **Poor speaker separation**: Try adjusting the number of expected speakers
