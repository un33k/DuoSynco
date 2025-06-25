"""
Dialogue Generator with ElevenLabs Text to Dialogue API Integration
"""

import requests
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import time

from .base import DialogueBase, DialogueSegment
from ...utils.env import get_env

logger = logging.getLogger(__name__)


class DialogueGenerator:
    """
    Generates audio from dialogue using ElevenLabs Text to Dialogue API
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.elevenlabs.io"):
        """
        Initialize dialogue generator
        
        Args:
            api_key: ElevenLabs API key (will auto-detect if not provided)
            base_url: Base URL for ElevenLabs API
        """
        self.api_key = api_key or get_env('ELEVENLABS_API_KEY')
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required")
        
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        })
    
    def generate_dialogue_audio(
        self,
        dialogue: DialogueBase,
        model_id: str = "eleven_v3",
        output_format: str = "mp3_44100_128",
        quality: str = "standard"
    ) -> bytes:
        """
        Generate audio from dialogue using ElevenLabs Text to Dialogue API
        
        Args:
            dialogue: DialogueBase object with voice-assigned segments
            model_id: ElevenLabs model to use (eleven_v3 for dialogue)
            output_format: Audio output format
            quality: Audio quality setting
            
        Returns:
            Audio data as bytes
        """
        # Validate dialogue has voice IDs
        missing_voices = dialogue.validate_voice_ids()
        if missing_voices:
            raise ValueError(f"Missing voice IDs for speakers: {missing_voices}")
        
        # Convert to ElevenLabs format
        dialogue_data = dialogue.to_elevenlabs_dialogue_format()
        
        if not dialogue_data:
            raise ValueError("No dialogue segments to generate")
        
        # Prepare API request
        request_data = {
            "text": dialogue_data,  # Array of {speaker_id, text} objects
            "model_id": model_id,
            "output_format": output_format
        }
        
        # Add quality settings
        if quality != "standard":
            request_data["voice_settings"] = {
                "stability": 0.7,
                "similarity_boost": 0.5,
                "style": 0.5,
                "use_speaker_boost": True
            }
        
        logger.info("Generating dialogue audio with %d segments", len(dialogue_data))
        
        try:
            # Make API request to Text to Dialogue endpoint
            response = self.session.post(
                f"{self.base_url}/v1/text-to-dialogue",
                json=request_data,
                timeout=300  # 5 minute timeout for long dialogues
            )
            
            response.raise_for_status()
            
            # Return audio data
            return response.content
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                error_detail = e.response.json().get('detail', 'Unknown error')
                raise ValueError(f"API request error: {error_detail}")
            elif e.response.status_code == 401:
                raise ValueError("Invalid API key or insufficient permissions")
            elif e.response.status_code == 402:
                raise ValueError("Insufficient credits for dialogue generation")
            elif e.response.status_code == 404:
                raise ValueError("Text to Dialogue API not available - may need alpha access")
            else:
                raise ValueError(f"API error {e.response.status_code}: {e.response.text}")
        
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Network error during dialogue generation: {e}")
    
    def generate_dialogue_with_fallback(
        self,
        dialogue: DialogueBase,
        output_file: Path,
        use_fallback: bool = True
    ) -> bool:
        """
        Generate dialogue audio with fallback to individual TTS calls
        
        Args:
            dialogue: DialogueBase object
            output_file: Output audio file path
            use_fallback: Whether to use fallback method if main API fails
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try Text to Dialogue API first
            audio_data = self.generate_dialogue_audio(dialogue)
            
            # Save audio file
            with open(output_file, 'wb') as f:
                f.write(audio_data)
            
            logger.info("Dialogue audio generated successfully: %s", output_file)
            return True
            
        except ValueError as e:
            logger.warning("Text to Dialogue API failed: %s", e)
            
            if use_fallback and "not available" in str(e).lower():
                logger.info("Falling back to individual TTS generation")
                return self._generate_dialogue_fallback(dialogue, output_file)
            else:
                raise
    
    def _generate_dialogue_fallback(
        self,
        dialogue: DialogueBase,
        output_file: Path
    ) -> bool:
        """
        Fallback method using individual TTS calls and audio concatenation
        
        Args:
            dialogue: DialogueBase object
            output_file: Output audio file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from ..providers.elevenlabs.tts import ElevenLabsTTSProvider
            from pydub import AudioSegment
            import tempfile
            
            # Initialize TTS provider
            tts_provider = ElevenLabsTTSProvider(api_key=self.api_key)
            
            # Generate individual audio segments
            audio_segments = []
            temp_files = []
            
            for i, segment in enumerate(dialogue.segments):
                if not segment.voice_id:
                    logger.warning("Skipping segment %d: no voice ID", i)
                    continue
                
                # Generate TTS for this segment
                temp_file = Path(tempfile.mktemp(suffix='.mp3'))
                temp_files.append(temp_file)
                
                success = tts_provider.generate_speech(
                    text=segment.text,
                    voice_id=segment.voice_id,
                    output_file=temp_file
                )
                
                if success:
                    # Load audio segment
                    audio_seg = AudioSegment.from_mp3(temp_file)
                    
                    # Add pause between speakers if needed
                    if audio_segments and i > 0:
                        prev_speaker = dialogue.segments[i-1].speaker_id
                        curr_speaker = segment.speaker_id
                        if prev_speaker != curr_speaker:
                            # Add 500ms pause between different speakers
                            pause = AudioSegment.silent(duration=500)
                            audio_segments.append(pause)
                    
                    audio_segments.append(audio_seg)
                else:
                    logger.warning("Failed to generate TTS for segment %d", i)
            
            if not audio_segments:
                logger.error("No audio segments generated")
                return False
            
            # Concatenate all segments
            final_audio = sum(audio_segments)
            
            # Export final audio
            final_audio.export(output_file, format="mp3")
            
            # Cleanup temp files
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            
            logger.info("Fallback dialogue generation completed: %s", output_file)
            return True
            
        except ImportError as e:
            logger.error("Missing dependencies for fallback generation: %s", e)
            return False
        except Exception as e:
            logger.error("Fallback generation failed: %s", e)
            return False
    
    def estimate_generation_cost(self, dialogue: DialogueBase) -> Dict[str, Any]:
        """
        Estimate cost and time for dialogue generation
        
        Args:
            dialogue: DialogueBase object
            
        Returns:
            Cost and time estimates
        """
        total_characters = sum(len(segment.text) for segment in dialogue.segments)
        unique_voices = len(set(segment.voice_id for segment in dialogue.segments if segment.voice_id))
        
        # Rough estimates (actual costs may vary)
        estimated_cost_per_1k_chars = 0.3  # USD
        estimated_time_per_segment = 2.0  # seconds
        
        estimated_cost = (total_characters / 1000) * estimated_cost_per_1k_chars
        estimated_time = len(dialogue.segments) * estimated_time_per_segment
        
        return {
            'total_characters': total_characters,
            'total_segments': len(dialogue.segments),
            'unique_voices': unique_voices,
            'estimated_cost_usd': round(estimated_cost, 2),
            'estimated_time_seconds': round(estimated_time, 1),
            'estimated_time_minutes': round(estimated_time / 60, 1)
        }
    
    def test_dialogue_api_availability(self) -> Dict[str, Any]:
        """
        Test if Text to Dialogue API is available
        
        Returns:
            API availability status
        """
        test_data = {
            "text": [
                {"speaker_id": "pNInz6obpgDQGcFmaJgB", "text": "Hello, this is a test."}
            ],
            "model_id": "eleven_v3",
            "output_format": "mp3_44100_128"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/v1/text-to-dialogue",
                json=test_data,
                timeout=30
            )
            
            if response.status_code == 200:
                return {
                    'available': True,
                    'status': 'API is available and working',
                    'response_size': len(response.content)
                }
            elif response.status_code == 404:
                return {
                    'available': False,
                    'status': 'Text to Dialogue API not found - may need alpha access',
                    'error_code': 404
                }
            elif response.status_code == 401:
                return {
                    'available': False,
                    'status': 'Unauthorized - check API key',
                    'error_code': 401
                }
            elif response.status_code == 402:
                return {
                    'available': False,
                    'status': 'Payment required - insufficient credits',
                    'error_code': 402
                }
            else:
                return {
                    'available': False,
                    'status': f'API error: {response.status_code}',
                    'error_code': response.status_code,
                    'error_detail': response.text[:200]
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'available': False,
                'status': f'Network error: {e}',
                'error_type': type(e).__name__
            }
    
    def preview_dialogue_generation(
        self,
        dialogue: DialogueBase,
        max_segments: int = 3
    ) -> Dict[str, Any]:
        """
        Preview what will be generated without making the full API call
        
        Args:
            dialogue: DialogueBase object
            max_segments: Maximum segments to include in preview
            
        Returns:
            Preview information
        """
        preview_segments = dialogue.segments[:max_segments]
        
        preview_data = {
            'preview_segments': [
                {
                    'speaker_id': seg.speaker_id,
                    'voice_id': seg.voice_id,
                    'text': seg.text[:100] + ('...' if len(seg.text) > 100 else ''),
                    'duration': seg.duration
                }
                for seg in preview_segments
            ],
            'total_segments': len(dialogue.segments),
            'preview_count': len(preview_segments),
            'cost_estimate': self.estimate_generation_cost(dialogue),
            'api_format': dialogue.to_elevenlabs_dialogue_format()[:max_segments]
        }
        
        return preview_data