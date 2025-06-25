# ElevenLabs Provider

Premium STT/TTS provider with advanced features and voice synthesis capabilities.

## Overview

ElevenLabs offers professional-grade Speech-to-Text (STT) and Text-to-Speech (TTS) services with exceptional quality. In DuoSynco, it provides multiple execution paths for different workflows including transcription, editing, and voice synthesis.

## Features

- ✅ **Premium STT**: High-accuracy speech recognition with 99 languages
- ✅ **Advanced TTS**: Natural voice synthesis with multiple voice options
- ✅ **Edit Workflow**: STT + Diarization combo for enhanced processing
- ✅ **Voice Cloning**: Preserve original speaker characteristics
- ✅ **Dialogue Generation**: AI-powered conversation synthesis
- ✅ **Multi-Modal**: Audio events detection and processing
- ✅ **Adaptive Timing**: Smart pacing for natural speech flow

## Execution Modes

### 1. Diarization Mode (STT Only)
Basic speech-to-text with speaker identification.

```bash
# STT transcription with speaker separation
./scripts/run.sh sample_data/annunaki-en.mp3 -p elevenlabs --speakers 2 --verbose
```

### 2. Edit Mode (STT + Diarization)
Premium workflow combining ElevenLabs STT with AssemblyAI diarization.

```bash
# Premium edit workflow
./scripts/run.sh sample_data/annunaki-en.mp3 -p elevenlabs --mode edit --speakers 2 --verbose
```

### 3. TTS Mode
Generate audio from transcript files.

```bash
# Convert transcript to audio
./scripts/run.sh transcript.json -p elevenlabs --mode tts --total-duration 120
```

### 4. Dialogue Mode
Generate AI conversations with character profiles.

```bash
# Create AI dialogue
./scripts/run.sh transcript.json -p elevenlabs --mode dialogue --dialogue-quality high
```

## Expected Outputs

### Edit Mode Example

#### Command
```bash
./scripts/run.sh sample_data/annunaki-en.mp3 -p elevenlabs --mode edit --speakers 2 --verbose
```

#### Console Output
```
🎤 Edit Mode Workflow Starting
📁 Input file: sample_data/annunaki-en.mp3
📁 Output directory: output
🛤️  Execution Path: ElevenLabs Premium Path
🗣️  STT Provider: elevenlabs-stt
🔧 Final Provider: assemblyai
⚡ STT Quality: high
💰 Cost Profile: Medium ($0.40/hour STT) + High (TTS per character)
🎯 Features: Premium STT quality, Speaker diarization, 99 languages, Audio events detection, Premium voice synthesis, Multiple voice options, Adaptive timing

🎤 Step 1: Transcribing audio with elevenlabs-stt...
✅ STT completed: 32 utterances, 2 speakers

✏️  Step 2: Text editing and speaker replacement...
💡 Speaker analysis suggestions:
  - Consider replacing 2 generic speaker names with meaningful names
  - Review 1 pairs of similar speaker names for potential merging
💾 Saved edited transcript: output/annunaki-en_edited_transcript.json

🎯 Step 3: Final audio separation with assemblyai...
✅ Edit workflow completed successfully!

📊 Results Summary:
  🎤 STT Transcription: 32 utterances
  ✏️  Edited Transcript: output/annunaki-en_edited_transcript.json
  🎯 Final Separation: 97.1% coverage
    speaker_0: 65.6s (34.3%)
    speaker_1: 120.3s (62.8%)
```

#### Generated Files
```
output/
├── annunaki-en_edited_transcript.json           # Editable transcript with speaker mapping
├── annunaki-en_final_speaker_0_assemblyai.wav   # Speaker 0 isolated audio
├── annunaki-en_final_speaker_1_assemblyai.wav   # Speaker 1 isolated audio
├── annunaki-en_final_transcript_assemblyai.txt  # Final human-readable transcript
├── annunaki-en_stt_elevenlabs_stt_transcript_debug.txt  # Debug STT output
└── annunaki-en_stt_stt_results_debug.json       # Raw STT results
```

### STT-Only Mode Example

#### Command
```bash
./scripts/run.sh sample_data/annunaki-en.mp3 -p elevenlabs --speakers 2 --verbose
```

#### Console Output
```
📁 Processing: sample_data/annunaki-en.mp3
📁 Output directory: output
👥 Expected speakers: 2
🌍 Language: en
⚡ Enhanced processing: True
🔍 Analyzing speakers with elevenlabs...
✅ Speaker separation completed!
📊 Coverage: 99.9% (191.2s / 191.5s)
  speaker_1: 121.3s (63.3%)
  speaker_0: 69.9s (36.5%)
🎵 Audio-only processing completed!
```

#### Generated Files
```
output/
├── annunaki-en_elevenlabs_stt_speakers.txt      # Speaker timeline
└── annunaki-en_elevenlabs_stt_transcript.txt    # Full transcript
```

## Performance Metrics

### Test Results (annunaki-en.mp3)

**Edit Mode:**
- **Total Duration**: 191.5 seconds
- **STT Coverage**: 99.9% (191.2s processed)
- **Final Coverage**: 97.1% (185.9s processed)
- **Speaker Distribution**:
  - Speaker 0: 65.6s (34.3%)
  - Speaker 1: 120.3s (62.8%)
- **Processing Time**: ~90-120 seconds (multi-step workflow)
- **Accuracy**: Premium STT + Enhanced diarization

**STT-Only Mode:**
- **Coverage**: 99.9% (191.2s/191.5s)
- **Speaker Distribution**:
  - Speaker 0: 69.9s (36.5%)
  - Speaker 1: 121.3s (63.3%)
- **Processing Time**: ~60-90 seconds
- **Accuracy**: High-quality transcription with speaker tags

## API Integration

ElevenLabs uses direct HTTP API calls with comprehensive features:

- **STT Endpoint**: Advanced speech recognition
- **TTS Endpoint**: Natural voice synthesis
- **Voice Cloning**: Preserve speaker characteristics
- **Language Support**: 99+ languages
- **Audio Events**: Background noise detection

## Configuration

### Environment Variables
```bash
# Required API key
ELEVENLABS_API_KEY=your_api_key_here

# Optional voice settings
VOICE_SPEAKER_0=N2lVS1w4EtoT3dr4eOWO  # Default voice for speaker 0
VOICE_SPEAKER_1=oWAxZDx7w5VPeCdBGZOi  # Default voice for speaker 1

# Quality settings
DUOSYNCO_STT_QUALITY=high
DUOSYNCO_TTS_QUALITY=high
```

### Quality Settings

**STT Quality:**
- **Low**: Fast processing, basic accuracy
- **Medium**: Balanced speed and quality
- **High**: Best accuracy (default)
- **Ultra**: Maximum quality, slower processing

**TTS Quality:**
- **Standard**: Basic voice synthesis
- **High**: Natural voice quality (default)
- **Ultra**: Premium voice synthesis

## Cost Structure

ElevenLabs pricing varies by feature:
- **STT**: ~$0.40 per hour of audio
- **TTS**: Per character (varies by voice/quality)
- **Voice Cloning**: Additional premium feature
- **API Calls**: Rate-limited based on subscription

## Workflow Modes

### Edit Mode (Recommended)
Combines ElevenLabs STT with AssemblyAI diarization:
1. **STT Phase**: ElevenLabs processes audio → transcript
2. **Edit Phase**: Generate editable JSON transcript
3. **Diarization Phase**: AssemblyAI separates speakers → audio files

**Benefits**: Premium STT quality + reliable diarization

### TTS Mode
Convert transcript files to synthesized audio:
```bash
# Basic TTS
./scripts/run.sh transcript.json -p elevenlabs --mode tts

# With voice mapping
./scripts/run.sh transcript.json -p elevenlabs --mode tts --voice-mapping '{"speaker_0": "voice-id-1", "speaker_1": "voice-id-2"}'

# Auto voice assignment from .env.local
./scripts/run.sh transcript.json -p elevenlabs --mode tts --voice-mapping auto
```

### Dialogue Mode
AI-powered conversation generation:
```bash
# Generate dialogue with character profiles
./scripts/run.sh transcript.json -p elevenlabs --mode dialogue --character-profiles profiles.json

# Interactive voice selection
./scripts/run.sh transcript.json -p elevenlabs --mode dialogue --interactive-voices
```

## Advanced Features

### Voice Cloning
```bash
# Enable voice cloning to preserve original speaker characteristics
./scripts/run.sh input.mp3 -p elevenlabs --mode tts --clone-voices
```

### Multi-Language Support
```bash
# Spanish processing
./scripts/run.sh spanish_audio.mp3 -p elevenlabs --language es

# Dialogue in multiple languages
./scripts/run.sh transcript.json -p elevenlabs --mode dialogue --dialogue-language fr
```

### Custom Models
```bash
# Use specific TTS model
./scripts/run.sh transcript.json -p elevenlabs --mode tts --model-id eleven_flash_v2_5
```

## Troubleshooting

### Common Issues

**API Key Issues**
```bash
# Verify API key
./scripts/run.sh --show-config

# Set API key
export ELEVENLABS_API_KEY=your_key_here
```

**STT-Only Mode No Audio**
- STT mode only generates transcripts, not audio files
- Use `--mode edit` for audio separation
- Use `--mode tts` for audio synthesis

**Rate Limiting**
- Check your ElevenLabs subscription limits
- Reduce concurrent processing
- Use quality settings to manage API usage

**Voice Mapping Errors**
```bash
# List available voices
./scripts/run.sh --list-voices

# Use auto mapping
./scripts/run.sh transcript.json -p elevenlabs --mode tts --voice-mapping auto
```

## Supported Formats

### Input Formats
- **Audio**: MP3, WAV, FLAC, AAC, M4A, OGG
- **Video**: MP4, AVI, MOV, MKV, WEBM
- **Transcripts**: JSON, TXT

### Output Formats
- **Audio**: WAV (diarization), MP3 (TTS)
- **Transcripts**: TXT, JSON
- **Debug**: Detailed logs and intermediate files

## Best Practices

1. **Mode Selection**: Use `edit` mode for best results
2. **Quality Settings**: Balance quality vs. processing time
3. **File Management**: Clean up intermediate files regularly
4. **API Limits**: Monitor usage to avoid rate limiting
5. **Voice Selection**: Test voices before bulk processing
6. **Language Codes**: Use correct ISO codes for non-English content

## Integration with AssemblyAI

ElevenLabs works seamlessly with AssemblyAI in edit mode:
- **STT Phase**: ElevenLabs processes speech → high-quality transcript
- **Diarization Phase**: AssemblyAI separates speakers → precise audio files
- **Combined Benefits**: Premium transcription + reliable audio separation

This hybrid approach provides the best of both providers while maintaining cost efficiency.