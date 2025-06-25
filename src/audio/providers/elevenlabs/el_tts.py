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

import soundfile as sf
import numpy as np
from pydub import AudioSegment
from elevenlabs import ElevenLabs
from elevenlabs.types import VoiceSettings

from ..base import SpeakerDiarizationProvider
from .el_voice import VoiceManager

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
            
        # Initialize the ElevenLabs SDK client
        self.client = ElevenLabs(api_key=self.api_key)
        
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
        # Load from .env.local file first
        from ....utils.util_env import get_env
        
        # Try environment variables
        api_key = get_env('ELEVENLABS_API_KEY')
        if api_key:
            return api_key
            
        # Try alternative environment variable name
        api_key = get_env('ELEVEN_LABS_API_KEY')
        if api_key:
            return api_key
            
            
        # Try local key file (legacy fallback)
        try:
            key_file = Path('elevenlabs_key.txt')
            with open(key_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            pass
            
        return None
    
    def _validate_api_key(self) -> None:
        """Validate API key by making a test request"""
        try:
            # Use the SDK to validate the API key
            voices = self.client.voices.get_all()
            logger.debug("API key validation successful")
        except Exception as e:
            if "401" in str(e) or "Unauthorized" in str(e):
                raise ValueError("Invalid ElevenLabs API key")
            logger.warning("Could not validate API key: %s", e)
    
    def generate_tts_audio(
        self,
        text: str,
        voice_id: str,
        settings: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        Generate TTS audio for given text and voice using ElevenLabs SDK
        
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
        
        # High-quality TTS settings optimized for best audio quality
        default_voice_settings = {
            "stability": 0.75,           # Higher stability for consistency
            "similarity_boost": 0.85,    # Higher similarity for better voice matching
            "style": 0.2,               # Slight style enhancement for expressiveness
            "use_speaker_boost": True    # Enable speaker boost for clarity
        }
        
        # Extract model_id from settings if provided
        model_id = 'eleven_multilingual_v2'  # Default model
        output_format = "mp3_44100_128"       # High quality MP3 format
        
        if settings:
            # Update voice settings
            default_voice_settings.update({k: v for k, v in settings.items() 
                                         if k in ['stability', 'similarity_boost', 'style', 'use_speaker_boost']})
            # Extract other parameters
            model_id = settings.get('model_id', model_id)
            output_format = settings.get('output_format', output_format)
        
        # Create VoiceSettings object for the SDK
        voice_settings = VoiceSettings(
            stability=default_voice_settings['stability'],
            similarity_boost=default_voice_settings['similarity_boost'],
            style=default_voice_settings['style'],
            use_speaker_boost=default_voice_settings['use_speaker_boost']
        )
        
        try:
            # Log the API call details in verbose mode
            logger.info("🔗 ElevenLabs TTS SDK Call:")
            logger.info("   Voice: %s", voice_id)
            logger.info("   Model: %s", model_id)
            logger.info("   Output Format: %s", output_format)
            logger.info("   Text: '%.100s%s'", text[:100], "..." if len(text) > 100 else "")
            logger.debug("   Voice Settings: %s", voice_settings)
            
            # Use the SDK to generate TTS
            audio_generator = self.client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id=model_id,
                voice_settings=voice_settings,
                output_format=output_format
            )
            
            # Collect all audio chunks
            audio_data = b''.join(audio_generator)
            
            logger.info("✅ TTS generated successfully (%.1f KB)", len(audio_data) / 1024)
            logger.debug("Generated TTS for text: '%.50s...' (voice: %s)", 
                        text[:50], voice_id)
            return audio_data
            
        except Exception as e:
            logger.error("❌ TTS generation failed for voice %s: %s", voice_id, e)
            raise RuntimeError(f"TTS generation failed: {e}")
    
    def measure_tts_duration(
        self,
        text: str,
        voice_id: str,
        settings: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Measure the actual duration of TTS audio for given text
        
        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID
            settings: Optional TTS settings
            
        Returns:
            Duration in seconds
        """
        try:
            # Generate TTS audio
            audio_data = self.generate_tts_audio(text, voice_id, settings)
            
            if not audio_data:
                return 0.0
                
            # Convert to AudioSegment to measure duration
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file.flush()
                
                audio_segment = AudioSegment.from_mp3(tmp_file.name)
                duration = len(audio_segment) / 1000.0  # Convert ms to seconds
                
            # Clean up temp file
            os.unlink(tmp_file.name)
            
            logger.debug("Measured TTS duration: %.2fs for text: '%.30s...'", 
                        duration, text[:30])
            return duration
            
        except Exception as e:
            logger.error("Failed to measure TTS duration: %s", e)
            return 0.0
    
    def calculate_voice_speed_profile(
        self,
        voice_id: str,
        test_texts: Optional[List[str]] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate speaking speed profile for a voice
        
        Args:
            voice_id: ElevenLabs voice ID
            test_texts: Optional list of test texts (uses defaults if None)
            settings: Optional TTS settings
            
        Returns:
            Dictionary with speed metrics:
            {
                'chars_per_second': float,
                'words_per_second': float,
                'avg_duration': float,
                'total_chars': int,
                'total_words': int
            }
        """
        if test_texts is None:
            test_texts = [
                "Hello, this is a test of the voice speed.",
                "The quick brown fox jumps over the lazy dog.",
                "In this example, we are measuring how fast this voice speaks.",
                "Speaking speed can vary significantly between different voices and settings.",
                "This measurement helps us predict accurate timing for longer conversations."
            ]
        
        total_duration = 0.0
        total_chars = 0
        total_words = 0
        successful_tests = 0
        
        logger.info("Profiling voice speed for voice ID: %s", voice_id)
        
        for i, text in enumerate(test_texts):
            try:
                duration = self.measure_tts_duration(text, voice_id, settings)
                if duration > 0:
                    total_duration += duration
                    total_chars += len(text)
                    total_words += len(text.split())
                    successful_tests += 1
                    
                    logger.debug("Test %d: %.2fs for %d chars (%d words)", 
                               i+1, duration, len(text), len(text.split()))
                else:
                    logger.warning("Test %d failed for voice %s", i+1, voice_id)
                    
            except Exception as e:
                logger.error("Error in speed test %d for voice %s: %s", i+1, voice_id, e)
                continue
        
        if successful_tests == 0:
            logger.error("All speed tests failed for voice %s", voice_id)
            return {
                'chars_per_second': 0.0,
                'words_per_second': 0.0,
                'avg_duration': 0.0,
                'total_chars': 0,
                'total_words': 0
            }
        
        chars_per_second = total_chars / total_duration if total_duration > 0 else 0.0
        words_per_second = total_words / total_duration if total_duration > 0 else 0.0
        avg_duration = total_duration / successful_tests
        
        profile = {
            'chars_per_second': chars_per_second,
            'words_per_second': words_per_second,
            'avg_duration': avg_duration,
            'total_chars': total_chars,
            'total_words': total_words
        }
        
        logger.info("Voice speed profile for %s: %.1f chars/sec, %.1f words/sec", 
                   voice_id, chars_per_second, words_per_second)
        
        return profile
    
    def predict_tts_duration(
        self,
        text: str,
        voice_profile: Dict[str, float]
    ) -> float:
        """
        Predict TTS duration based on voice speed profile
        
        Args:
            text: Text to predict duration for
            voice_profile: Voice speed profile from calculate_voice_speed_profile
            
        Returns:
            Predicted duration in seconds
        """
        if not voice_profile or voice_profile['chars_per_second'] <= 0:
            # Fallback estimate: average human speech is ~150 words/minute
            words = len(text.split())
            return words / (150 / 60)  # Convert to words per second
            
        chars = len(text)
        predicted_duration = chars / voice_profile['chars_per_second']
        
        logger.debug("Predicted duration: %.2fs for %d chars", predicted_duration, chars)
        return predicted_duration
    
    def generate_adaptive_timeline(
        self,
        transcript_segments: List[Dict[str, Any]],
        voice_mapping: Dict[str, str],
        tts_settings: Optional[Dict[str, Any]] = None,
        gap_duration: float = 0.4,
        timing_mode: str = "adaptive"
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Generate adaptive timeline based on actual TTS durations
        
        Args:
            transcript_segments: Original transcript segments
            voice_mapping: Speaker to voice ID mapping
            tts_settings: TTS generation settings
            gap_duration: Gap between speakers in seconds
            timing_mode: 'adaptive' or 'strict'
            
        Returns:
            Tuple of (adjusted_segments, total_duration)
        """
        logger.info("Generating adaptive timeline for %d segments", len(transcript_segments))
        
        # First pass: Profile voice speeds for all speakers
        voice_profiles = {}
        speakers = list(set(seg['speaker'] for seg in transcript_segments))
        
        for speaker in speakers:
            voice_id = voice_mapping.get(speaker)
            if voice_id:
                logger.info("Profiling voice speed for speaker %s (voice: %s)", speaker, voice_id)
                voice_profiles[speaker] = self.calculate_voice_speed_profile(
                    voice_id, settings=tts_settings
                )
            else:
                logger.warning("No voice mapping for speaker %s", speaker)
                voice_profiles[speaker] = {
                    'chars_per_second': 0.0,
                    'words_per_second': 0.0,
                    'avg_duration': 0.0,
                    'total_chars': 0,
                    'total_words': 0
                }
        
        # Second pass: Adjust timeline based on predicted durations
        adjusted_segments = []
        current_time = 0.5  # Start at 0.5s
        
        for i, segment in enumerate(transcript_segments):
            speaker = segment['speaker']
            text = segment['text']
            
            # Predict actual TTS duration
            voice_profile = voice_profiles.get(speaker, {})
            predicted_duration = self.predict_tts_duration(text, voice_profile)
            
            # For adaptive mode, use predicted duration
            # For strict mode, try to fit within original window but allow overflow
            if timing_mode == "adaptive":
                actual_duration = predicted_duration
            else:  # strict mode
                original_duration = segment['end'] - segment['start']
                # Allow up to 20% overflow in strict mode
                max_allowed = original_duration * 1.2
                actual_duration = min(predicted_duration, max_allowed)
            
            # Create adjusted segment
            adjusted_segment = {
                'speaker': speaker,
                'start': current_time,
                'end': current_time + actual_duration,
                'text': text,
                'original_start': segment['start'],
                'original_end': segment['end'],
                'predicted_duration': predicted_duration,
                'timing_mode': timing_mode
            }
            
            adjusted_segments.append(adjusted_segment)
            
            # Move to next speaker's start time
            current_time = adjusted_segment['end'] + gap_duration
            
            logger.debug(
                "Segment %d (%s): %.2fs -> %.2fs (predicted: %.2fs)",
                i, speaker, segment['end'] - segment['start'], 
                actual_duration, predicted_duration
            )
        
        total_duration = current_time - gap_duration  # Remove final gap
        
        logger.info(
            "Adaptive timeline complete: %d segments, %.1fs total duration",
            len(adjusted_segments), total_duration
        )
        
        return adjusted_segments, total_duration
    
    def generate_tracks_from_transcript(
        self,
        transcript_segments: List[Dict[str, Any]],
        total_duration: float,
        voice_mapping: Optional[Dict[str, str]] = None,
        tts_settings: Optional[Dict[str, Any]] = None,
        max_workers: int = 3,
        timing_mode: str = "adaptive",
        gap_duration: float = 0.4
    ) -> Dict[str, AudioSegment]:
        """
        Generate separate audio tracks from transcript segments with adaptive timing
        
        Args:
            transcript_segments: List of segments with speaker, start, end, text
            total_duration: Total duration of conversation in seconds (may be adjusted)
            voice_mapping: Optional mapping of speaker IDs to voice IDs
            tts_settings: Optional TTS generation settings
            max_workers: Maximum number of concurrent TTS requests
            timing_mode: 'adaptive' (adjust timing) or 'strict' (try to preserve)
            gap_duration: Gap between speakers in seconds
            
        Returns:
            Dictionary mapping speaker IDs to their complete audio tracks
        """
        logger.info("Generating TTS tracks from %d segments (mode: %s)", 
                   len(transcript_segments), timing_mode)
        
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
        
        # Generate adaptive timeline if in adaptive mode
        if timing_mode == "adaptive":
            logger.info("Generating adaptive timeline based on voice speeds...")
            adjusted_segments, adjusted_total_duration = self.generate_adaptive_timeline(
                transcript_segments, voice_mapping, tts_settings, gap_duration, timing_mode
            )
            working_segments = adjusted_segments
            working_duration = adjusted_total_duration
            logger.info("Timeline adjusted: %.1fs -> %.1fs", total_duration, adjusted_total_duration)
        else:
            # Use original segments for strict mode
            working_segments = transcript_segments
            working_duration = total_duration
            
        # Generate TTS for each segment concurrently
        audio_segments = self._generate_segments_concurrent(
            working_segments, voice_mapping, tts_settings, max_workers
        )
        
        # Build complete tracks with timing
        tracks = self._build_timed_tracks(
            audio_segments, working_segments, working_duration, speakers
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
        """Clean up resources on destruction"""
        # SDK handles cleanup automatically
        pass