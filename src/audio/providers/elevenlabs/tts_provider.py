"""
ElevenLabs TTS Provider
Generates separate audio tracks from transcript segments using ElevenLabs TTS API
"""

import os
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import tempfile
import concurrent.futures
import threading

import requests
import soundfile as sf
import numpy as np
from pydub import AudioSegment

from ..base import SpeakerDiarizationProvider
from .voice_manager import VoiceManager

logger = logging.getLogger(__name__)


class ElevenLabsTTSProvider(SpeakerDiarizationProvider):
    """
    ElevenLabs TTS provider for generating audio tracks from transcript segments
    Uses direct HTTP API for maximum control and cross-platform compatibility
    """
    
    BASE_URL = "https://api.elevenlabs.io/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize ElevenLabs TTS provider
        
        Args:
            api_key: ElevenLabs API key. If None, will try to get from env
        """
        super().__init__(api_key)
        self.api_key = api_key or self._get_api_key()
        if not self.api_key:
            raise ValueError(
                "ElevenLabs API key not found. "
                "Set ELEVEN_LABS_API_KEY environment variable"
            )
            
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        })
        
        self.voice_manager = VoiceManager(self.api_key, self.BASE_URL)
        self._validate_api_key()
        
        # Thread-safe storage for generated audio
        self._audio_cache = {}
        self._cache_lock = threading.Lock()
    
    @property
    def provider_name(self) -> str:
        """Return the name of this provider"""
        return "ElevenLabs"
    
    @property
    def requires_api_key(self) -> bool:
        """Return whether this provider requires an API key"""
        return True
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from multiple sources"""
        # Try environment variable (primary)
        api_key = os.getenv('ELEVEN_LABS_API_KEY')
        if api_key:
            return api_key
            
        # Try alternative environment variable name
        api_key = os.getenv('ELEVENLABS_API_KEY')
        if api_key:
            return api_key
            
        # Try .env file
        try:
            with open('.env', 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip().startswith('ELEVEN_LABS_API_KEY='):
                        return line.strip().split('=', 1)[1].strip('\'"')
                    elif line.strip().startswith('ELEVENLABS_API_KEY='):
                        return line.strip().split('=', 1)[1].strip('\'"')
        except FileNotFoundError:
            pass
            
        # Try local key file
        try:
            with open('elevenlabs_key.txt', 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            pass
            
        return None
    
    def _validate_api_key(self) -> None:
        """Validate API key by making a test request"""
        try:
            response = self.session.get(f"{self.BASE_URL}/voices")
            response.raise_for_status()
            logger.debug("API key validation successful")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise ValueError("Invalid ElevenLabs API key")
            raise
        except requests.exceptions.RequestException as e:
            logger.warning("Could not validate API key: %s", e)
    
    def generate_tts_audio(
        self,
        text: str,
        voice_id: str,
        settings: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        Generate TTS audio for given text and voice
        
        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID
            settings: Optional TTS settings (stability, similarity_boost, etc.)
            
        Returns:
            Audio data as bytes (MP3 format)
        """
        if not text.strip():
            # Return silence for empty text
            return b''
            
        url = f"{self.BASE_URL}/text-to-speech/{voice_id}"
        
        # High-quality TTS settings optimized for best audio quality
        default_settings = {
            "stability": 0.75,           # Higher stability for consistency
            "similarity_boost": 0.85,    # Higher similarity for better voice matching
            "style": 0.2,               # Slight style enhancement for expressiveness
            "use_speaker_boost": True    # Enable speaker boost for clarity
        }
        
        if settings:
            default_settings.update(settings)
            
        # Use highest quality model available
        model_id = settings.get('model_id', 'eleven_multilingual_v2') if settings else 'eleven_multilingual_v2'
        
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": default_settings,
            "output_format": "mp3_44100_128"  # High quality MP3 format
        }
        
        try:
            response = self.session.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            logger.debug("Generated TTS for text: '%.50s...' (voice: %s)", 
                        text[:50], voice_id)
            return response.content
            
        except requests.exceptions.RequestException as e:
            logger.error("TTS generation failed for voice %s: %s", voice_id, e)
            raise RuntimeError(f"TTS generation failed: {e}")
    
    def generate_tracks_from_transcript(
        self,
        transcript_segments: List[Dict[str, Any]],
        total_duration: float,
        voice_mapping: Optional[Dict[str, str]] = None,
        tts_settings: Optional[Dict[str, Any]] = None,
        max_workers: int = 3
    ) -> Dict[str, AudioSegment]:
        """
        Generate separate audio tracks from transcript segments
        
        Args:
            transcript_segments: List of segments with speaker, start, end, text
            total_duration: Total duration of conversation in seconds
            voice_mapping: Optional mapping of speaker IDs to voice IDs
            tts_settings: Optional TTS generation settings
            max_workers: Maximum number of concurrent TTS requests
            
        Returns:
            Dictionary mapping speaker IDs to their complete audio tracks
        """
        logger.info("Generating TTS tracks from %d segments", len(transcript_segments))
        
        if not transcript_segments:
            raise ValueError("No transcript segments provided")
            
        # Extract speakers and create voice mapping
        speakers = list(set(seg['speaker'] for seg in transcript_segments))
        if not voice_mapping:
            voice_mapping = self.voice_manager.create_voice_mapping(speakers)
            
        # Validate voice IDs
        voice_ids = list(voice_mapping.values())
        validation = self.voice_manager.validate_voice_ids(voice_ids)
        invalid_voices = [vid for vid, valid in validation.items() if not valid]
        if invalid_voices:
            logger.warning("Invalid voice IDs: %s", invalid_voices)
            
        # Generate TTS for each segment concurrently
        audio_segments = self._generate_segments_concurrent(
            transcript_segments, voice_mapping, tts_settings, max_workers
        )
        
        # Build complete tracks with timing
        tracks = self._build_timed_tracks(
            audio_segments, transcript_segments, total_duration, speakers
        )
        
        logger.info("Generated %d complete audio tracks", len(tracks))
        return tracks
    
    def _generate_segments_concurrent(
        self,
        transcript_segments: List[Dict[str, Any]],
        voice_mapping: Dict[str, str],
        tts_settings: Optional[Dict[str, Any]],
        max_workers: int
    ) -> Dict[int, AudioSegment]:
        """
        Generate TTS audio for all segments concurrently
        
        Returns:
            Dictionary mapping segment index to AudioSegment
        """
        logger.info("Generating TTS for %d segments with %d workers", 
                   len(transcript_segments), max_workers)
        
        audio_segments = {}
        
        def generate_segment_audio(seg_index: int, segment: Dict[str, Any]) -> Tuple[int, AudioSegment]:
            """Generate audio for a single segment"""
            speaker = segment['speaker']
            text = segment['text']
            voice_id = voice_mapping.get(speaker)
            
            if not voice_id:
                logger.warning("No voice mapping for speaker %s", speaker)
                return seg_index, AudioSegment.silent(duration=1000)  # 1 second silence
                
            try:
                # Generate TTS audio
                audio_data = self.generate_tts_audio(text, voice_id, tts_settings)
                
                if not audio_data:
                    # Return silence for empty audio
                    return seg_index, AudioSegment.silent(duration=1000)
                    
                # Convert MP3 bytes to AudioSegment
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                    tmp_file.write(audio_data)
                    tmp_file.flush()
                    
                    audio_segment = AudioSegment.from_mp3(tmp_file.name)
                    
                # Clean up temp file
                os.unlink(tmp_file.name)
                
                return seg_index, audio_segment
                
            except Exception as e:
                logger.error("Failed to generate audio for segment %d: %s", seg_index, e)
                # Return silence on error
                duration_ms = int((segment['end'] - segment['start']) * 1000)
                return seg_index, AudioSegment.silent(duration=max(duration_ms, 1000))
        
        # Use ThreadPoolExecutor for concurrent generation
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(generate_segment_audio, i, seg): i 
                for i, seg in enumerate(transcript_segments)
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_index):
                try:
                    seg_index, audio_segment = future.result()
                    audio_segments[seg_index] = audio_segment
                except Exception as e:
                    seg_index = future_to_index[future]
                    logger.error("Failed to process segment %d: %s", seg_index, e)
                    # Add silence for failed segments
                    segment = transcript_segments[seg_index]
                    duration_ms = int((segment['end'] - segment['start']) * 1000)
                    audio_segments[seg_index] = AudioSegment.silent(duration=max(duration_ms, 1000))
        
        return audio_segments
    
    def _build_timed_tracks(
        self,
        audio_segments: Dict[int, AudioSegment],
        transcript_segments: List[Dict[str, Any]],
        total_duration: float,
        speakers: List[str]
    ) -> Dict[str, AudioSegment]:
        """
        Build complete audio tracks with proper timing for each speaker
        
        Returns:
            Dictionary mapping speaker IDs to complete AudioSegments
        """
        logger.info("Building timed tracks for %d speakers", len(speakers))
        
        # Initialize empty tracks for each speaker
        total_duration_ms = int(total_duration * 1000)
        tracks = {
            speaker: AudioSegment.silent(duration=total_duration_ms)
            for speaker in speakers
        }
        
        # Process each segment
        for seg_index, segment in enumerate(transcript_segments):
            speaker = segment['speaker']
            start_time = segment['start']
            end_time = segment['end']
            
            if seg_index not in audio_segments:
                logger.warning("Missing audio for segment %d", seg_index)
                continue
                
            audio_segment = audio_segments[seg_index]
            
            # Calculate timing
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            target_duration_ms = end_ms - start_ms
            
            # Adjust audio segment to match target duration
            if len(audio_segment) > target_duration_ms:
                # Trim if too long
                audio_segment = audio_segment[:target_duration_ms]
            elif len(audio_segment) < target_duration_ms:
                # Pad with silence if too short
                silence_needed = target_duration_ms - len(audio_segment)
                # Add silence at the end
                audio_segment = audio_segment + AudioSegment.silent(duration=silence_needed)
            
            # Insert audio into the speaker's track at the correct time
            # Create a track with the audio at the right position
            pre_silence = AudioSegment.silent(duration=start_ms)
            post_silence = AudioSegment.silent(duration=total_duration_ms - end_ms)
            positioned_audio = pre_silence + audio_segment + post_silence
            
            # Overlay on the existing track (replace the silence with audio)
            tracks[speaker] = tracks[speaker].overlay(positioned_audio)
            
            logger.debug(
                "Added segment %d to %s track: %.2fs-%.2fs (%.2fs duration)",
                seg_index, speaker, start_time, end_time, target_duration_ms / 1000
            )
        
        return tracks
    
    def save_tracks_as_wav(
        self,
        tracks: Dict[str, AudioSegment],
        output_dir: str,
        base_filename: str
    ) -> List[str]:
        """
        Save audio tracks as WAV files
        
        Args:
            tracks: Dictionary of speaker tracks
            output_dir: Output directory
            base_filename: Base filename for output files
            
        Returns:
            List of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = []
        
        for speaker, track in tracks.items():
            audio_file = output_path / f"{base_filename}_{speaker.lower()}_elevenlabs_tts.wav"
            
            # Export as WAV
            track.export(str(audio_file), format="wav")
            saved_files.append(str(audio_file))
            
            duration = len(track) / 1000.0
            logger.info("Saved %s track: %s (%.1fs)", speaker, audio_file, duration)
        
        return saved_files
    
    # Required abstract methods from base class (adapted for TTS)
    def diarize(
        self,
        audio_file: str,
        speakers_expected: int = 2,
        language: str = "en",
        enhanced_processing: bool = True
    ) -> Tuple[Dict[str, np.ndarray], str, List[Dict]]:
        """
        This method is not used for TTS generation but required by base class
        Use generate_tracks_from_transcript instead for TTS functionality
        """
        raise NotImplementedError(
            "ElevenLabs provider is for TTS generation, not diarization. "
            "Use generate_tracks_from_transcript method instead."
        )
    
    def save_results(
        self,
        speaker_tracks: Dict[str, np.ndarray],
        transcript_text: str,
        output_dir: str,
        base_filename: str,
        sample_rate: int = 24000
    ) -> List[str]:
        """
        This method is not used for TTS generation but required by base class
        Use save_tracks_as_wav instead for TTS functionality
        """
        raise NotImplementedError(
            "ElevenLabs provider is for TTS generation, not diarization. "
            "Use save_tracks_as_wav method instead."
        )
    
    def __del__(self):
        """Clean up session on destruction"""
        if hasattr(self, 'session'):
            self.session.close()