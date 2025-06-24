"""
Voice Management for ElevenLabs TTS
Handles voice selection, mapping, and voice discovery
"""

import logging
from typing import Dict, List, Optional, Any
import requests

logger = logging.getLogger(__name__)


class VoiceManager:
    """
    Manages ElevenLabs voice selection and mapping
    """
    
    # Default voice mappings for common speaker labels
    DEFAULT_VOICE_MAPPING = {
        'A': 'pNInz6obpgDQGcFmaJgB',  # Adam (male)
        'B': 'EXAVITQu4vr4xnSDxMaL',  # Bella (female)  
        'C': 'VR6AewLTigWG4xSOukaG',  # Arnold (male)
        'D': 'oWAxZDx7w5VEj9dCyTzz',  # Grace (female)
        'E': '2EiwWnXFnvU5JabPnv8n',  # Clyde (male)
        'F': 'IKne3meq5aSn9XLyUdCD',  # Charlie (female)
        'SPEAKER_00': 'pNInz6obpgDQGcFmaJgB',  # Adam
        'SPEAKER_01': 'EXAVITQu4vr4xnSDxMaL',  # Bella
        'SPEAKER_02': 'VR6AewLTigWG4xSOukaG',  # Arnold
        'SPEAKER_03': 'oWAxZDx7w5VEj9dCyTzz',  # Grace
    }
    
    def __init__(self, api_key: str, base_url: str = "https://api.elevenlabs.io/v1"):
        """
        Initialize voice manager
        
        Args:
            api_key: ElevenLabs API key
            base_url: Base URL for ElevenLabs API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "xi-api-key": api_key
        })
        
        self._available_voices: Optional[List[Dict]] = None
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Get list of available voices from ElevenLabs API
        
        Returns:
            List of voice dictionaries with id, name, category, etc.
        """
        if self._available_voices is None:
            try:
                response = self.session.get(f"{self.base_url}/voices")
                response.raise_for_status()
                data = response.json()
                self._available_voices = data.get('voices', [])
                logger.info("Retrieved %d available voices", len(self._available_voices))
            except Exception as e:
                logger.error("Failed to retrieve voices: %s", e)
                self._available_voices = []
                
        return self._available_voices
    
    def get_voice_info(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific voice
        
        Args:
            voice_id: ElevenLabs voice ID
            
        Returns:
            Voice information dictionary or None if not found
        """
        voices = self.get_available_voices()
        for voice in voices:
            if voice.get('voice_id') == voice_id:
                return voice
        return None
    
    def create_voice_mapping(
        self, 
        speakers: List[str], 
        custom_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Create speaker to voice ID mapping
        
        Args:
            speakers: List of speaker IDs (e.g., ['A', 'B', 'SPEAKER_00'])
            custom_mapping: Optional custom speaker to voice ID mapping
            
        Returns:
            Dictionary mapping speaker IDs to voice IDs
        """
        voice_mapping = {}
        
        for speaker in speakers:
            if custom_mapping and speaker in custom_mapping:
                # Use custom mapping first
                voice_mapping[speaker] = custom_mapping[speaker]
            elif speaker in self.DEFAULT_VOICE_MAPPING:
                # Use default mapping
                voice_mapping[speaker] = self.DEFAULT_VOICE_MAPPING[speaker]
            else:
                # Fallback to cycling through default voices
                speaker_index = len(voice_mapping) % len(self.DEFAULT_VOICE_MAPPING)
                default_voices = list(self.DEFAULT_VOICE_MAPPING.values())
                voice_mapping[speaker] = default_voices[speaker_index]
                
        logger.info("Created voice mapping: %s", voice_mapping)
        return voice_mapping
    
    def validate_voice_ids(self, voice_ids: List[str]) -> Dict[str, bool]:
        """
        Validate that voice IDs exist in ElevenLabs
        
        Args:
            voice_ids: List of voice IDs to validate
            
        Returns:
            Dictionary mapping voice IDs to their validity
        """
        available_voices = self.get_available_voices()
        available_ids = {v.get('voice_id') for v in available_voices}
        
        validation_results = {}
        for voice_id in voice_ids:
            validation_results[voice_id] = voice_id in available_ids
            
        logger.info("Voice validation results: %s", validation_results)
        return validation_results
    
    def get_voice_suggestions(self, gender: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get voice suggestions based on criteria
        
        Args:
            gender: Optional gender filter ('male', 'female')
            
        Returns:
            List of suggested voices
        """
        voices = self.get_available_voices()
        
        if gender:
            # Filter by gender if specified
            gender_keywords = {
                'male': ['male', 'man', 'masculine'],
                'female': ['female', 'woman', 'feminine']
            }
            
            filtered_voices = []
            for voice in voices:
                voice_name = voice.get('name', '').lower()
                labels = voice.get('labels', {})
                
                # Check if voice matches gender criteria
                if any(keyword in voice_name for keyword in gender_keywords.get(gender, [])):
                    filtered_voices.append(voice)
                elif labels.get('gender') == gender:
                    filtered_voices.append(voice)
                    
            return filtered_voices
        
        return voices
    
    def list_default_voices(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about the default voices used in mapping
        
        Returns:
            Dictionary mapping voice IDs to their information
        """
        default_info = {}
        unique_voice_ids = set(self.DEFAULT_VOICE_MAPPING.values())
        
        for voice_id in unique_voice_ids:
            voice_info = self.get_voice_info(voice_id)
            if voice_info:
                default_info[voice_id] = voice_info
            else:
                default_info[voice_id] = {
                    'voice_id': voice_id,
                    'name': 'Unknown',
                    'status': 'Not found'
                }
                
        return default_info