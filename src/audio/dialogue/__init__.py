"""
Audio Dialogue Module
Enhanced dialogue generation and management for multi-speaker conversations
"""

from .base import DialogueBase, DialogueSegment
from .converter import TranscriptToDialogueConverter
from .generator import DialogueGenerator
from .profile import CharacterProfile, CharacterManager
from .test import DialogueTester

__all__ = [
    'DialogueBase',
    'DialogueSegment', 
    'TranscriptToDialogueConverter',
    'DialogueGenerator',
    'CharacterProfile',
    'CharacterManager',
    'DialogueTester'
]