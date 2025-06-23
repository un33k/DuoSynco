# DuoSynco

**Video/Audio Synchronization Tool with Speaker Separation**

DuoSynco is a Python-based CLI tool that separates speakers in video files and creates isolated audio tracks. Using advanced speaker diarization, it generates separate video files where each contains only one speaker's audio while maintaining perfect video synchronization.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (3.11.9 recommended)
- [pyenv](https://github.com/pyenv/pyenv) for Python version management
- [FFmpeg](https://ffmpeg.org/) (optional but recommended for best performance)

### Installation

#### Option 1: Bootstrap Script (Recommended)
```bash
# Clone the repository
git clone https://github.com/your-username/DuoSynco.git
cd DuoSynco

# Run bootstrap script (installs Python 3.11.9 + dependencies)
./bootstrap.sh

# Or non-interactive mode
./bootstrap.sh -y
```

#### Option 2: Manual Setup
```bash
# Install Python 3.11.9 via pyenv
pyenv install 3.11.9
pyenv local 3.11.9

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Using the convenient bash script
./scripts/run.sh input_video.mp4

# Using Python module directly
python -m src.main input_video.mp4 --output-dir ./output

# Quick test with provided sample (host + guest conversation)
./scripts/run.sh sample_data/TheAnnunaki.wav --speakers 2 --verbose

# With options
./scripts/run.sh interview.mp4 \
  --output-dir ./results \
  --speakers 3 \
  --quality high \
  --verbose
```

## 📋 Features

- **🎵 Speaker Diarization**: Automatically identifies different speakers using AI
- **🎬 Video Synchronization**: Maintains perfect sync between video and isolated audio
- **🔧 Multiple Formats**: Supports MP4, AVI, MOV, and common audio formats
- **⚙️ Quality Settings**: Configurable processing quality (low/medium/high)
- **🛠️ Robust Error Handling**: Comprehensive fallback methods
- **📊 Progress Tracking**: Verbose output with clear progress indicators

## 🏗️ Architecture

DuoSynco follows a modular architecture:

### Core Components
- **Audio Processing** (`src/audio/`): Speaker diarization, isolation, and processing
- **Video Processing** (`src/video/`): Video operations and synchronization
- **Utilities** (`src/utils/`): Configuration management and file handling
- **CLI Interface** (`src/main.py`): User-friendly command-line interface

### Processing Pipeline
1. **Input Validation** → Verify file format and accessibility
2. **Speaker Diarization** → Identify when each speaker talks
3. **Audio Isolation** → Create separate audio tracks per speaker  
4. **Video Synchronization** → Replace original audio with isolated tracks
5. **Output Generation** → Save synchronized videos for each speaker

## 🔧 Development

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"
```

### Development Commands
```bash
# Run tests
make test
# or: pytest tests/ -v

# Format code
make format
# or: black src/ tests/

# Lint code
make lint
# or: flake8 src/ tests/

# Type checking
make type-check
# or: mypy src/

# Clean up
make clean
```

### VS Code Integration
The project includes comprehensive VS Code configuration:
- Automatic `.venv` detection
- Debugging configurations
- Testing integration
- Code formatting and linting
- Recommended extensions

## 📁 Project Structure

```
DuoSynco/
├── src/
│   ├── audio/           # Audio processing modules
│   ├── video/           # Video processing modules
│   ├── utils/           # Utilities and configuration
│   └── main.py          # CLI entry point
├── tests/               # Test suite
├── scripts/
│   └── run.sh          # Convenient execution script
├── .vscode/            # VS Code configuration
├── bootstrap.sh        # Environment setup script
├── Makefile           # Development commands
└── pyproject.toml     # Project configuration
```

## ⚙️ Configuration

### Command Line Options
```bash
duosynco input.mp4 [OPTIONS]

Options:
  -o, --output-dir PATH    Output directory (default: ./output)
  -s, --speakers INTEGER   Number of speakers (default: 2)  
  -f, --format CHOICE      Output format: mp4, avi, mov (default: mp4)
  -q, --quality CHOICE     Quality: low, medium, high (default: medium)
  -v, --verbose           Enable verbose output
  -h, --help              Show help message
```

### Environment Variables
```bash
# Copy and modify .env.example
cp .env.example .env

# Available variables:
DUOSYNCO_QUALITY=medium
DUOSYNCO_FORMAT=mp4
DUOSYNCO_VERBOSE=false
DUOSYNCO_SPEAKERS=2
```

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_audio/ -v
```

### Quick Functional Test
```bash
# Test speaker separation with provided sample
./scripts/run.sh sample_data/TheAnnunaki.wav --speakers 2 --verbose

# Expected output: Two separate audio files in ./output/
# - TheAnnunaki_speaker_1.mp4 (host audio only)
# - TheAnnunaki_speaker_2.mp4 (guest audio only)
```

## 🔧 Dependencies

### Core Dependencies
- **click**: CLI framework
- **librosa**: Audio analysis and processing
- **moviepy**: Video processing
- **pyannote.audio**: Speaker diarization AI
- **torch**: Deep learning backend

### Optional Dependencies
- **FFmpeg**: Advanced video/audio processing (install separately)
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Code linting

## 🚧 Troubleshooting

### Common Issues

**FFmpeg not found**
```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/
```

**Memory errors with large files**
- Reduce quality setting: `--quality low`
- Process shorter segments
- Increase system memory

**Poor speaker separation**
- Adjust expected speakers: `--speakers 3`
- Ensure clear audio input
- Try different quality settings

### Getting Help
- Check the [troubleshooting guide](CLAUDE.md#troubleshooting)
- Review verbose output: `--verbose`
- Open an issue on GitHub

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/new-feature`
3. Run tests: `make test`
4. Submit a pull request

## 🔄 CI/CD

The project includes GitHub Actions CI that runs on:
- Multiple Python versions (3.8-3.11)
- Multiple platforms (Ubuntu, macOS, Windows)
- Automatic testing, linting, and type checking

---

**DuoSynco** - Making speaker separation simple and reliable.