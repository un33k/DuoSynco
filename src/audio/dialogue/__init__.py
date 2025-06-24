"""
Audio Dialogue Module
Enhanced dialogue generation and management for multi-speaker conversations
"""

from .dialogue_base import DialogueBase, DialogueSegment
from .dialogue_converter import TranscriptToDialogueConverter
from .dialogue_generator import DialogueGenerator
from .character_profile import CharacterProfile, CharacterManager
from .dialogue_test import DialogueTester

__all__ = [
    'DialogueBase',
    'DialogueSegment', 
    'TranscriptToDialogueConverter',
    'DialogueGenerator',
    'CharacterProfile',
    'CharacterManager',
    'DialogueTester'
]