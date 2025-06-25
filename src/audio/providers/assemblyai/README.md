# AssemblyAI Provider

Professional-grade speaker diarization with high accuracy and cost-effectiveness.

## Overview

AssemblyAI is the default provider for DuoSynco, offering reliable speaker separation using advanced AI models. It provides excellent accuracy while maintaining reasonable costs for processing audio/video files.

## Features

- ✅ **Professional Diarization**: Industry-leading speaker identification accuracy
- ✅ **Cost-Effective**: Competitive pricing for audio processing
- ✅ **Word-Level Timestamps**: Precise timing for speaker boundaries
- ✅ **Enhanced Processing**: Advanced voice separation techniques
- ✅ **Cross-Talk Cleanup**: Handles overlapping speech
- ✅ **Multiple Languages**: Support for various language inputs

## Usage Examples

### Basic Diarization

```bash
# Default AssemblyAI processing
./scripts/run.sh sample_data/annunaki-en.mp3 --speakers 2 --verbose

# With custom output directory
./scripts/run.sh interview.mp4 -o ./results --speakers 3
```

### Advanced Options

```bash
# High quality processing
./scripts/run.sh meeting.wav --quality high --enhanced-processing

# Specific language
./scripts/run.sh podcast.mp3 --language es --speakers 2
```

## Expected Output

### Command
```bash
./scripts/run.sh sample_data/annunaki-en.mp3 --speakers 2 --verbose
```

### Console Output
```
📁 Processing: sample_data/annunaki-en.mp3
📁 Output directory: output
👥 Expected speakers: 2
🌍 Language: en
⚡ Enhanced processing: True
🔍 Analyzing speakers with assemblyai...
✅ Speaker separation completed!
📊 Coverage: 97.3% (186.3s / 191.5s)
  speaker_0: 66.4s (34.7%)
  speaker_1: 119.9s (62.6%)
🎵 Audio-only processing completed!
```

### Generated Files
```
output/
├── annunaki-en_speaker_0_assemblyai.wav    # Speaker 0 isolated audio
├── annunaki-en_speaker_1_assemblyai.wav    # Speaker 1 isolated audio
└── annunaki-en_transcript_assemblyai.txt   # Full transcript with timestamps
```

## Performance Metrics

### Test Results (annunaki-en.mp3)
- **Total Duration**: 191.5 seconds
- **Coverage**: 97.3% (186.3s processed)
- **Speaker Distribution**:
  - Speaker 0: 66.4s (34.7%)
  - Speaker 1: 119.9s (62.6%)
- **Processing Time**: ~30-60 seconds (depending on file size)
- **Accuracy**: High-quality speaker boundaries with word-level precision

## API Integration

AssemblyAI uses direct HTTP API calls for maximum control:

- **Endpoint**: `https://api.assemblyai.com/v2/`
- **Features Used**:
  - File upload and transcription
  - Speaker diarization (`speaker_labels=True`)
  - Enhanced speech model (`speech_model=best`)
  - Word-level timestamps for precise boundaries

## Configuration

### Environment Variables
```bash
# Required API key
ASSEMBLYAI_API_KEY=your_api_key_here

# Optional settings
DUOSYNCO_QUALITY=medium
DUOSYNCO_ENHANCED_PROCESSING=true
```

### Quality Settings
- **Low**: Fast processing, basic accuracy
- **Medium**: Balanced speed and quality (default)
- **High**: Best accuracy, slower processing

## Cost Structure

AssemblyAI charges based on audio duration:
- **Transcription**: ~$0.37 per hour of audio
- **Speaker Diarization**: Included with transcription
- **Enhanced Features**: No additional cost

## Error Handling

The provider includes comprehensive error handling:
- API rate limiting with automatic retries
- Network timeout management
- File upload validation
- Graceful fallback for processing errors

## Troubleshooting

### Common Issues

**API Key Missing**
```bash
# Set in environment
export ASSEMBLYAI_API_KEY=your_key_here

# Or add to .env.local
echo "ASSEMBLYAI_API_KEY=your_key_here" >> .env.local
```

**Low Coverage Results**
- Try adjusting `--speakers` count
- Ensure clear audio input
- Use `--enhanced-processing` flag

**Processing Timeouts**
- Check internet connection
- Verify file isn't corrupted
- Try with smaller audio segments

### Debug Mode
```bash
# Enable verbose output for detailed logs
./scripts/run.sh input.mp3 --verbose

# Check API status
./scripts/run.sh --show-config
```

## Supported Formats

### Input Formats
- **Audio**: MP3, WAV, FLAC, AAC, M4A
- **Video**: MP4, AVI, MOV, MKV, WEBM

### Output Formats
- **Audio**: WAV (high quality, uncompressed)
- **Transcript**: TXT (human-readable with timestamps)

## Best Practices

1. **File Quality**: Use clear, high-quality audio for best results
2. **Speaker Count**: Accurately specify expected number of speakers
3. **File Size**: Optimal results with files under 2GB
4. **Language**: Specify language code for non-English content
5. **Environment**: Ensure stable internet connection for API calls

## Integration Notes

- Works seamlessly with ElevenLabs as secondary provider in edit mode
- Outputs compatible with video synchronization pipeline
- Transcript format suitable for TTS processing
- Word-level timestamps enable precise audio editing