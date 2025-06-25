"""
ElevenLabs Providers for DuoSynco
TTS Provider: Generates separate audio tracks from transcript segments using ElevenLabs TTS API
STT Provider: Transcribes audio/video files with speaker diarization using ElevenLabs Scribe API
"""

from .tts import ElevenLabsTTSProvider
from .stt import ElevenLabsSTTProvider

__all__ = ['ElevenLabsTTSProvider', 'ElevenLabsSTTProvider']