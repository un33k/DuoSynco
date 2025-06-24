"""
Text processing module for DuoSynco
Handles transcript editing, speaker replacement, and text manipulation
"""

from .text_editor import TranscriptEditor
from .text_replacer import SpeakerReplacer

__all__ = ['TranscriptEditor', 'SpeakerReplacer']