"""
Character Profile Management for Dialogue Generation
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
from ..providers.elevenlabs.el_voice import VoiceManager


@dataclass
class CharacterProfile:
    """
    Represents a character with voice and speaking characteristics
    """
    character_id: str
    name: str
    voice_id: str
    description: Optional[str] = None
    personality_traits: List[str] = field(default_factory=list)
    speaking_style: Optional[str] = None
    emotion_default: str = "neutral"
    voice_settings: Dict[str, Any] = field(default_factory=dict)
    language: str = "en"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary"""
        return {
            'character_id': self.character_id,
            'name': self.name,
            'voice_id': self.voice_id,
            'description': self.description,
            'personality_traits': self.personality_traits,
            'speaking_style': self.speaking_style,
            'emotion_default': self.emotion_default,
            'voice_settings': self.voice_settings,
            'language': self.language
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterProfile':
        """Create profile from dictionary"""
        return cls(**data)
    
    def get_voice_settings_with_style(self) -> Dict[str, Any]:
        """Get voice settings enhanced with character style"""
        settings = self.voice_settings.copy()
        
        # Add style-based adjustments
        if self.speaking_style:
            style_adjustments = {
                'calm': {'stability': 0.8, 'similarity_boost': 0.3},
                'energetic': {'stability': 0.4, 'similarity_boost': 0.7},
                'authoritative': {'stability': 0.9, 'similarity_boost': 0.5},
                'friendly': {'stability': 0.6, 'similarity_boost': 0.6},
                'dramatic': {'stability': 0.3, 'similarity_boost': 0.8}
            }
            
            if self.speaking_style in style_adjustments:
                settings.update(style_adjustments[self.speaking_style])
        
        return settings


class CharacterManager:
    """
    Manages character profiles and voice assignments
    """
    
    def __init__(self, voice_manager: Optional[VoiceManager] = None):
        self.voice_manager = voice_manager
        self.characters: Dict[str, CharacterProfile] = {}
        self.speaker_to_character_mapping: Dict[str, str] = {}
    
    def add_character(self, character: CharacterProfile) -> None:
        """Add a character profile"""
        self.characters[character.character_id] = character
    
    def get_character(self, character_id: str) -> Optional[CharacterProfile]:
        """Get character by ID"""
        return self.characters.get(character_id)
    
    def list_characters(self) -> List[CharacterProfile]:
        """List all characters"""
        return list(self.characters.values())
    
    def map_speaker_to_character(self, speaker_id: str, character_id: str) -> bool:
        """
        Map a speaker ID to a character
        
        Args:
            speaker_id: Original speaker identifier
            character_id: Character to map to
            
        Returns:
            True if mapping successful, False if character not found
        """
        if character_id in self.characters:
            self.speaker_to_character_mapping[speaker_id] = character_id
            return True
        return False
    
    def get_voice_id_for_speaker(self, speaker_id: str) -> Optional[str]:
        """Get voice ID for a speaker through character mapping"""
        character_id = self.speaker_to_character_mapping.get(speaker_id)
        if character_id and character_id in self.characters:
            return self.characters[character_id].voice_id
        return None
    
    def get_character_for_speaker(self, speaker_id: str) -> Optional[CharacterProfile]:
        """Get character profile for a speaker"""
        character_id = self.speaker_to_character_mapping.get(speaker_id)
        if character_id:
            return self.get_character(character_id)
        return None
    
    def create_speaker_voice_mapping(self) -> Dict[str, str]:
        """
        Create mapping from speaker IDs to voice IDs
        
        Returns:
            Dictionary mapping speaker_id -> voice_id
        """
        mapping = {}
        for speaker_id, character_id in self.speaker_to_character_mapping.items():
            character = self.get_character(character_id)
            if character:
                mapping[speaker_id] = character.voice_id
        return mapping
    
    def auto_assign_voices(self, speaker_ids: List[str], language: str = "en") -> Dict[str, str]:
        """
        Automatically assign voices to speakers
        
        Args:
            speaker_ids: List of speaker IDs to assign voices to
            language: Language for voice selection
            
        Returns:
            Dictionary mapping speaker_id -> voice_id
        """
        if not self.voice_manager:
            return {}
        
        # Get available voices for the language
        available_voices = self.voice_manager.get_voices_by_language(language)
        if not available_voices:
            available_voices = self.voice_manager.get_all_voices()
        
        # Create auto-assignment mapping
        voice_mapping = {}
        for i, speaker_id in enumerate(speaker_ids):
            if i < len(available_voices):
                voice_id = available_voices[i]['voice_id']
                voice_mapping[speaker_id] = voice_id
                
                # Create auto character profile
                character = CharacterProfile(
                    character_id=f"auto_{speaker_id}",
                    name=f"Speaker {i+1}",
                    voice_id=voice_id,
                    description=f"Auto-assigned character for {speaker_id}",
                    language=language
                )
                
                self.add_character(character)
                self.map_speaker_to_character(speaker_id, character.character_id)
        
        return voice_mapping
    
    def save_profiles(self, file_path: Path) -> None:
        """Save character profiles to file"""
        data = {
            'characters': {cid: char.to_dict() for cid, char in self.characters.items()},
            'speaker_mappings': self.speaker_to_character_mapping
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_profiles(self, file_path: Path) -> None:
        """Load character profiles from file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load characters
        self.characters.clear()
        for character_data in data.get('characters', {}).values():
            character = CharacterProfile.from_dict(character_data)
            self.add_character(character)
        
        # Load speaker mappings
        self.speaker_to_character_mapping = data.get('speaker_mappings', {})
    
    def create_character_from_voice(self, voice_id: str, character_id: str, name: str) -> CharacterProfile:
        """
        Create a character profile from an existing voice
        
        Args:
            voice_id: ElevenLabs voice ID
            character_id: Unique character identifier
            name: Character name
            
        Returns:
            Created character profile
        """
        # Get voice info if voice manager available
        voice_info = None
        if self.voice_manager:
            voice_info = self.voice_manager.get_voice_info(voice_id)
        
        character = CharacterProfile(
            character_id=character_id,
            name=name,
            voice_id=voice_id,
            description=f"Character using voice {voice_id}",
            language=voice_info.get('language', 'en') if voice_info else 'en'
        )
        
        self.add_character(character)
        return character
    
    def validate_character_voices(self) -> List[str]:
        """
        Validate that all character voice IDs exist
        
        Returns:
            List of invalid voice IDs
        """
        if not self.voice_manager:
            return []
        
        invalid_voices = []
        all_voices = {v['voice_id'] for v in self.voice_manager.get_all_voices()}
        
        for character in self.characters.values():
            if character.voice_id not in all_voices:
                invalid_voices.append(character.voice_id)
        
        return invalid_voices
    
    def get_character_statistics(self) -> Dict[str, Any]:
        """Get statistics about managed characters"""
        return {
            'total_characters': len(self.characters),
            'mapped_speakers': len(self.speaker_to_character_mapping),
            'languages': list(set(char.language for char in self.characters.values())),
            'speaking_styles': list(set(char.speaking_style for char in self.characters.values() if char.speaking_style)),
            'characters_by_language': {
                lang: len([char for char in self.characters.values() if char.language == lang])
                for lang in set(char.language for char in self.characters.values())
            }
        }