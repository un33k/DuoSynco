"""
Text processing module for DuoSynco
Handles transcript editing, speaker replacement, and text manipulation
"""

from .editor import TranscriptEditor
from .replacer import SpeakerReplacer

__all__ = ["TranscriptEditor", "SpeakerReplacer"]
