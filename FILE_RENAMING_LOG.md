# File Renaming Log

This document tracks the file renaming changes made to improve code organization and consistency.

## ElevenLabs Provider Files
**Location**: `src/audio/providers/elevenlabs/`

| Old Name | New Name | Description |
|----------|----------|-------------|
| `stt_provider.py` | `el_stt.py` | ElevenLabs Speech-to-Text provider |
| `tts_provider.py` | `el_tts.py` | ElevenLabs Text-to-Speech provider |
| `voice_manager.py` | `el_voice.py` | ElevenLabs voice management |

## AssemblyAI Provider Files
**Location**: `src/audio/providers/assemblyai/`

| Old Name | New Name | Description |
|----------|----------|-------------|
| `diarizer.py` | `ai_diarizer.py` | AssemblyAI speaker diarization provider |

## Main Audio Module Files
**Location**: `src/audio/`

| Old Name | New Name | Description |
|----------|----------|-------------|
| `diarization.py` | `audio_diarization.py` | Main speaker diarization orchestrator |
| `stt_generator.py` | `audio_stt.py` | Speech-to-text generator interface |
| `tts_generator.py` | `audio_tts.py` | Text-to-speech generator interface |

## Text Processing Module Files
**Location**: `src/text/`

| Old Name | New Name | Description |
|----------|----------|-------------|
| `transcript_editor.py` | `text_editor.py` | Transcript editing functionality |
| `speaker_replacer.py` | `text_replacer.py` | Speaker replacement functionality |

## Utils Module Files
**Location**: `src/utils/`

| Old Name | New Name | Description |
|----------|----------|-------------|
| `file_handler.py` | `util_files.py` | File handling utilities |
| `file_manager.py` | `util_manager.py` | Advanced file management |
| `env_loader.py` | `util_env.py` | Environment variable loading |

## Video Module Files
**Location**: `src/video/`

| Old Name | New Name | Description |
|----------|----------|-------------|
| `processor.py` | `video_processor.py` | Video processing functionality |
| `synchronizer.py` | `video_sync.py` | Video-audio synchronization |

## Updated Import Statements

### Main Module (`src/main.py`)
```python
# Updated imports
from .utils.util_files import FileHandler
from .utils.util_env import env, get_voice_mapping  
from .video.video_sync import VideoSynchronizer
from .audio.audio_diarization import SpeakerDiarizer
from .audio.audio_tts import TTSAudioGenerator
from .audio.audio_stt import STTAudioTranscriber
```

### Provider Files
```python
# ElevenLabs TTS provider
from .el_voice import VoiceManager

# All provider files
from ....utils.util_env import get_env
```

### Text Module (`src/text/__init__.py`)
```python
from .text_editor import TranscriptEditor
from .text_replacer import SpeakerReplacer
```

### Video Module
```python
# video_sync.py
from .video_processor import VideoProcessor
```

### ElevenLabs Provider (`src/audio/providers/elevenlabs/__init__.py`)
```python
from .el_tts import ElevenLabsTTSProvider
from .el_stt import ElevenLabsSTTProvider
```

### AssemblyAI Provider (`src/audio/providers/assemblyai/__init__.py`)
```python
from .ai_diarizer import AssemblyAIDiarizer
```

### Workflow Module (`src/workflows/edit_workflow.py`)
```python
from ..audio.audio_stt import STTAudioTranscriber
from ..audio.audio_diarization import SpeakerDiarizer
```

## Naming Convention Established

### Provider Files
- ElevenLabs: `el_` prefix (e.g., `el_stt.py`, `el_tts.py`, `el_voice.py`)
- AssemblyAI: `ai_` prefix (e.g., `ai_diarizer.py`)

### Main Module Files
- Audio: `audio_` prefix (e.g., `audio_stt.py`, `audio_tts.py`)
- Text: `text_` prefix (e.g., `text_editor.py`, `text_replacer.py`)
- Utils: `util_` prefix (e.g., `util_files.py`, `util_env.py`)
- Video: `video_` prefix (e.g., `video_processor.py`, `video_sync.py`)

## Benefits of This Naming Convention

1. **Clear Provider Identification**: Easy to identify which provider a file belongs to
2. **Module Consistency**: All files within a module follow the same naming pattern
3. **Namespace Clarity**: Avoids naming conflicts and makes imports more explicit
4. **Shortened Names**: More concise while maintaining clarity (e.g., `el_stt.py` vs `stt_provider.py`)
5. **Future Scalability**: Easy to add new providers following the same pattern

## Testing Verification

All functionality has been tested after renaming:
- ✅ Provider listing works correctly
- ✅ Execution paths display properly  
- ✅ CLI help functions normally
- ✅ All imports resolve correctly
- ✅ No broken dependencies

The file renaming maintains full backward compatibility while improving code organization and consistency across the entire DuoSynco codebase.