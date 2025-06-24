"""
ElevenLabs Providers for DuoSynco
TTS Provider: Generates separate audio tracks from transcript segments using ElevenLabs TTS API
STT Provider: Transcribes audio/video files with speaker diarization using ElevenLabs Scribe API
"""

from .el_tts import ElevenLabsTTSProvider
from .el_stt import ElevenLabsSTTProvider

__all__ = ['ElevenLabsTTSProvider', 'ElevenLabsSTTProvider']