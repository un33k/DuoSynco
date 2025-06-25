# DuoSynco Provider-Based Execution Paths

## Overview

DuoSynco uses a unified `--provider` (`-p`) system to manage different execution paths, allowing you to choose the optimal provider and mode combination based on cost, features, and quality requirements.

## Provider Architecture

- **Provider**: The service provider (e.g., `assemblyai`, `elevenlabs`)
- **Feature**: What the provider does (STT, TTS, diarization)
- **Mode**: The workflow type (`diarization`, `edit`, `tts`)

## Available Providers

### 🎯 AssemblyAI Professional (`-p assemblyai`)
- **Cost**: Low ($0.37/hour)
- **Features**: Professional diarization, High accuracy, Fast processing, Custom vocabulary
- **Modes**: `diarization`, `edit`
- **Use Case**: Cost-effective solution for most use cases, excellent accuracy-to-cost ratio
- **Best For**: Production workflows, budget-conscious projects

```bash
# Standard diarization workflow
python -m src.main input.mp4 -p assemblyai

# Edit workflow with AssemblyAI
python -m src.main input.mp4 -p assemblyai --mode edit
```

### 🎯 ElevenLabs Premium (`-p elevenlabs`)
- **Cost**: Medium ($0.40/hour STT) + High (TTS per character)
- **Features**: Premium STT quality, Speaker diarization, 99 languages, Audio events detection, Premium voice synthesis, Multiple voice options, Adaptive timing
- **Modes**: `diarization`, `edit`, `tts`
- **Use Case**: High-quality transcription with detailed analysis OR voice synthesis
- **Best For**: Multi-language content, premium quality requirements, voice replacement

```bash
# High-quality STT with diarization
python -m src.main input.mp4 -p elevenlabs

# Premium edit workflow (uses STT feature)
python -m src.main input.mp4 -p elevenlabs --mode edit -sq ultra

# Voice synthesis (uses TTS feature)  
python -m src.main transcript.json -p elevenlabs --mode tts --total-duration 120
```

## Multi-Provider Workflows (Edit Mode)

The edit mode supports combining different providers for optimal results:

### ElevenLabs STT → AssemblyAI Final
High-quality initial transcription with cost-effective final separation:

```bash
python -m src.main input.mp4 -p elevenlabs -sp assemblyai --mode edit
```

### AssemblyAI STT → ElevenLabs Final
Cost-effective transcription with premium final processing:

```bash
python -m src.main input.mp4 -p assemblyai -sp elevenlabs --mode edit
```

## Command Examples

### Basic Operations
```bash
# List all execution paths
python -m src.main -lep

# List available providers
python -m src.main --list-providers

# Show current configuration
python -m src.main --show-config
```

### Cost-Optimized Workflow
```bash
# Most cost-effective approach
python -m src.main interview.mp4 -p assemblyai -q low -s 2
```

### Premium Quality Workflow
```bash
# Highest quality approach
python -m src.main podcast.mp4 -p elevenlabs --mode edit -sq ultra -ei
```

### Custom Multi-Step Workflow
```bash
# ElevenLabs STT for quality, AssemblyAI for cost-effective final separation
python -m src.main meeting.mp4 -p elevenlabs -sp assemblyai --mode edit -sm speakers.json
```

## Provider Selection Guidelines

### Choose AssemblyAI when:
- ✅ Cost is a primary concern
- ✅ English content with clear speech
- ✅ Standard diarization requirements
- ✅ Batch processing workflows

### Choose ElevenLabs when:
- ✅ Premium quality is required
- ✅ Multi-language content (99 languages)
- ✅ Complex audio environments
- ✅ Detailed speaker analysis needed
- ✅ Voice replacement/synthesis projects
- ✅ Multiple voice options needed
- ✅ High-quality synthetic speech required
- ✅ Content adaptation projects

## Cost Optimization Strategies

### 1. Hybrid Approach
Use ElevenLabs STT for initial high-quality transcription, then AssemblyAI for final separation:
```bash
python -m src.main input.mp4 -p elevenlabs -sp assemblyai --mode edit
```

### 2. Quality-First Approach
Use ElevenLabs throughout for maximum quality:
```bash
python -m src.main input.mp4 -p elevenlabs --mode edit -sq ultra
```

### 3. Cost-First Approach
Use AssemblyAI throughout for minimum cost:
```bash
python -m src.main input.mp4 -p assemblyai -q low
```

## Feature Comparison

| Feature | AssemblyAI | ElevenLabs |
|---------|------------|------------|
| Speaker Diarization | ✅ Professional | ✅ Premium |
| Multi-language | ✅ 50+ languages | ✅ 99 languages |
| Cost (STT) | $0.37/hour | $0.40/hour |
| Cost (TTS) | ❌ N/A | Per character |
| Quality | High | Premium |
| Speed | Fast | Medium |
| Audio Events | ❌ | ✅ |
| Voice Synthesis | ❌ | ✅ Premium |
| Modes Supported | diarization, edit | diarization, edit, tts |

## Integration with Existing Workflows

The new provider system is fully backward compatible:

```bash
# Old way (still works - defaults to assemblyai)
python -m src.main input.mp4 -s 2

# New way (explicit provider selection)
python -m src.main input.mp4 -p assemblyai -s 2

# Advanced edit workflow
python -m src.main input.mp4 -p elevenlabs --mode edit -sm custom_speakers.json

# TTS workflow (note: transcript.json as input file)
python -m src.main transcript.json -p elevenlabs --mode tts --total-duration 120
```

## Environment Variables

Set your API keys in `.env.local` or environment:

```bash
# AssemblyAI
export ASSEMBLYAI_API_KEY="your_assemblyai_key"

# ElevenLabs
export ELEVENLABS_API_KEY="your_elevenlabs_key"
```

## Troubleshooting

### Provider Not Available
```bash
python -m src.main --list-providers
```

### Execution Path Details
```bash
python -m src.main --list-execution-paths
```

### Configuration Issues
```bash
python -m src.main --show-config
```

This provider-based system gives you full control over cost, quality, and feature trade-offs while maintaining the simplicity of the original DuoSynco interface.