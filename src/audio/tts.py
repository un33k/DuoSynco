"""
TTS Audio Generator Module
High-level interface for generating audio tracks from transcript segments
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from .providers.factory import ProviderFactory
from .providers.elevenlabs import ElevenLabsTTSProvider

logger = logging.getLogger(__name__)


class TTSAudioGenerator:
    """
    High-level TTS audio generation interface
    Generates separate audio tracks from transcript segments using TTS providers
    """
    
    def __init__(
        self,
        provider: str = "elevenlabs",
        api_key: Optional[str] = None
    ):
        """
        Initialize TTS audio generator
        
        Args:
            provider: Provider name ('elevenlabs', etc.)
            api_key: API key for the provider
        """
        if provider.lower() != 'elevenlabs':
            raise ValueError(
                f"Unsupported TTS provider: {provider}. "
                "Currently only 'elevenlabs' is supported for TTS generation."
            )
            
        self.provider_name = provider
        self.tts_provider = ElevenLabsTTSProvider(api_key=api_key)
        logger.info(
            "TTS generator initialized with %s backend",
            self.tts_provider.provider_name
        )
    
    def generate_audio_tracks(
        self,
        transcript_segments: List[Dict[str, Any]],
        total_duration: float,
        output_dir: str = "./output",
        base_filename: Optional[str] = None,
        voice_mapping: Optional[Dict[str, str]] = None,
        tts_settings: Optional[Dict[str, Any]] = None,
        max_workers: int = 3,
        quality: str = "high",
        timing_mode: str = "adaptive",
        gap_duration: float = 0.4
    ) -> Dict[str, Any]:
        """
        Generate separate audio tracks from transcript segments
        
        Args:
            transcript_segments: List of segments with format:
                [{"speaker": "A", "start": 0.0, "end": 5.0, "text": "Hello world"}, ...]
            total_duration: Total duration of conversation in seconds
            output_dir: Directory to save output files
            base_filename: Base name for output files (auto-generated if None)
            voice_mapping: Optional mapping of speaker IDs to ElevenLabs voice IDs
            tts_settings: Optional TTS generation settings
            max_workers: Maximum number of concurrent TTS requests
            quality: Quality level ('low', 'medium', 'high', 'ultra') - affects model and settings
            timing_mode: 'adaptive' (adjust timing based on voice speed) or 'strict' (preserve original timing)
            gap_duration: Gap between speakers in seconds
            
        Returns:
            Dictionary with results:
            {
                'audio_files': List[str],     # Paths to generated audio files
                'speakers': List[str],        # List of speaker IDs
                'voice_mapping': Dict,        # Speaker to voice ID mapping used
                'total_duration': float,      # Total duration in seconds
                'stats': Dict,               # Generation statistics
                'provider': str              # Provider used
            }
        """
        logger.info("Starting TTS generation for %d segments", len(transcript_segments))
        
        if not transcript_segments:
            raise ValueError("No transcript segments provided")
            
        if total_duration <= 0:
            raise ValueError("Total duration must be positive")
            
        # Validate transcript segments format
        self._validate_transcript_segments(transcript_segments)
        
        # Generate base filename if not provided
        if base_filename is None:
            base_filename = f"tts_generated_{int(total_duration)}s"
            
        # Apply quality-based settings
        quality_settings = self._get_quality_settings(quality)
        if tts_settings:
            quality_settings.update(tts_settings)
        
        # Generate audio tracks with timing options
        tracks = self.tts_provider.generate_tracks_from_transcript(
            transcript_segments=transcript_segments,
            total_duration=total_duration,
            voice_mapping=voice_mapping,
            tts_settings=quality_settings,
            max_workers=max_workers,
            timing_mode=timing_mode,
            gap_duration=gap_duration
        )
        
        # Save tracks as WAV files
        saved_files = self.tts_provider.save_tracks_as_wav(
            tracks=tracks,
            output_dir=output_dir,
            base_filename=base_filename
        )
        
        # Get final voice mapping used
        speakers = list(set(seg['speaker'] for seg in transcript_segments))
        if not voice_mapping:
            voice_mapping = self.tts_provider.voice_manager.create_voice_mapping(speakers)
            
        # Calculate statistics
        stats = self._calculate_stats(tracks, transcript_segments)
        
        # Prepare result
        result = {
            'audio_files': saved_files,
            'speakers': speakers,
            'voice_mapping': voice_mapping,
            'total_duration': total_duration,
            'stats': stats,
            'provider': self.tts_provider.provider_name
        }
        
        logger.info("TTS generation completed successfully")
        logger.info("Generated %d audio tracks", len(saved_files))
        logger.info("Total segments processed: %d", len(transcript_segments))
        
        return result
    
    def _validate_transcript_segments(self, segments: List[Dict[str, Any]]) -> None:
        """Validate transcript segments format"""
        required_keys = {'speaker', 'start', 'end', 'text'}
        
        for i, segment in enumerate(segments):
            missing_keys = required_keys - set(segment.keys())
            if missing_keys:
                raise ValueError(
                    f"Segment {i} missing required keys: {missing_keys}"
                )
                
            # Validate data types and values
            if not isinstance(segment['speaker'], str) or not segment['speaker']:
                raise ValueError(f"Segment {i}: speaker must be non-empty string")
                
            if not isinstance(segment['start'], (int, float)) or segment['start'] < 0:
                raise ValueError(f"Segment {i}: start must be non-negative number")
                
            if not isinstance(segment['end'], (int, float)) or segment['end'] < 0:
                raise ValueError(f"Segment {i}: end must be non-negative number")
                
            if segment['end'] <= segment['start']:
                raise ValueError(f"Segment {i}: end must be greater than start")
                
            if not isinstance(segment['text'], str):
                raise ValueError(f"Segment {i}: text must be string")
    
    def _calculate_stats(
        self,
        tracks: Dict[str, Any],
        transcript_segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate generation statistics"""
        stats = {
            'total_segments': len(transcript_segments),
            'speakers': {},
            'total_speech_duration': 0.0,
            'total_characters': 0,
        }
        
        # Calculate per-speaker statistics
        speaker_segments = {}
        for segment in transcript_segments:
            speaker = segment['speaker']
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(segment)
            
        for speaker, segments in speaker_segments.items():
            total_duration = sum(seg['end'] - seg['start'] for seg in segments)
            total_text = ' '.join(seg['text'] for seg in segments)
            
            stats['speakers'][speaker] = {
                'segments': len(segments),
                'duration': total_duration,
                'characters': len(total_text),
                'words': len(total_text.split())
            }
            
            stats['total_speech_duration'] += total_duration
            stats['total_characters'] += len(total_text)
            
        return stats
    
    def list_available_voices(self) -> Dict[str, Any]:
        """
        List available voices from the TTS provider
        
        Returns:
            Dictionary with available voices information
        """
        try:
            voices = self.tts_provider.voice_manager.get_available_voices()
            default_voices = self.tts_provider.voice_manager.list_default_voices()
            
            return {
                'available_voices': voices,
                'default_voices': default_voices,
                'total_voices': len(voices)
            }
        except Exception as e:
            logger.error("Failed to retrieve available voices: %s", e)
            return {
                'available_voices': [],
                'default_voices': {},
                'total_voices': 0,
                'error': str(e)
            }
    
    def create_voice_mapping(
        self,
        speakers: List[str],
        custom_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Create speaker to voice ID mapping
        
        Args:
            speakers: List of speaker IDs
            custom_mapping: Optional custom mapping
            
        Returns:
            Mapping dictionary
        """
        return self.tts_provider.voice_manager.create_voice_mapping(
            speakers, custom_mapping
        )
    
    @staticmethod
    def create_sample_transcript(
        speakers: List[str] = ["A", "B"],
        duration: float = 30.0
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Create a sample transcript for testing
        
        Args:
            speakers: List of speaker IDs
            duration: Total duration in seconds
            
        Returns:
            Tuple of (transcript_segments, total_duration)
        """
        sample_texts = {
            "A": [
                "Hello, how are you doing today?",
                "That sounds really interesting.",
                "I completely agree with that point.",
                "What do you think about this?"
            ],
            "B": [
                "I'm doing great, thanks for asking!",
                "Yes, it's been quite an adventure.",
                "Absolutely, that makes perfect sense.",
                "I think it's a fascinating topic to explore."
            ]
        }
        
        segments = []
        current_time = 0.0
        segment_duration = duration / (len(speakers) * 2)  # 2 segments per speaker
        
        for i in range(4):  # 4 total segments
            speaker = speakers[i % len(speakers)]
            text_index = i // len(speakers)
            
            if text_index < len(sample_texts[speaker]):
                text = sample_texts[speaker][text_index]
            else:
                text = f"This is additional text from speaker {speaker}."
                
            segments.append({
                'speaker': speaker,
                'start': current_time,
                'end': current_time + segment_duration,
                'text': text
            })
            
            current_time += segment_duration + 1.0  # 1 second gap
            
        return segments, duration
    
    def _get_quality_settings(self, quality: str) -> Dict[str, Any]:
        """
        Get TTS settings based on quality level
        
        Args:
            quality: Quality level ('low', 'medium', 'high', 'ultra')
            
        Returns:
            Dictionary with TTS settings optimized for the quality level
        """
        quality_profiles = {
            'low': {
                'model_id': 'eleven_turbo_v2_5',     # Faster, lower quality
                'stability': 0.4,
                'similarity_boost': 0.6,
                'style': 0.0,
                'use_speaker_boost': False,
                'output_format': 'mp3_22050_32'      # 32kbps - basic quality
            },
            'medium': {
                'model_id': 'eleven_multilingual_v2', # Balanced quality/speed
                'stability': 0.6,
                'similarity_boost': 0.75,
                'style': 0.1,
                'use_speaker_boost': True,
                'output_format': 'mp3_44100_32'      # 32kbps - good for voice
            },
            'high': {
                'model_id': 'eleven_multilingual_v2', # High quality, stable
                'stability': 0.75,
                'similarity_boost': 0.85,
                'style': 0.2,
                'use_speaker_boost': True,
                'output_format': 'mp3_44100_64'      # 64kbps - excellent voice quality
            },
            'ultra': {
                'model_id': 'eleven_multilingual_v2', # Best available model
                'stability': 0.8,                     # Maximum stability
                'similarity_boost': 0.9,              # Maximum similarity
                'style': 0.3,                         # Enhanced expressiveness
                'use_speaker_boost': True,
                'output_format': 'mp3_44100_128'     # 128kbps - premium voice quality
            }
        }
        
        return quality_profiles.get(quality.lower(), quality_profiles['high'])