"""
ElevenLabs Speech-to-Text Provider
Transcribes audio/video files using ElevenLabs Scribe API with speaker diarization
"""

import os
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import tempfile

import requests
import numpy as np

from ..base import SpeakerDiarizationProvider

logger = logging.getLogger(__name__)


class ElevenLabsSTTProvider(SpeakerDiarizationProvider):
    """
    ElevenLabs Speech-to-Text provider for transcribing audio with speaker diarization
    Uses direct HTTP API for maximum control and cross-platform compatibility
    """
    
    BASE_URL = "https://api.elevenlabs.io/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize ElevenLabs STT provider
        
        Args:
            api_key: ElevenLabs API key. If None, will try to get from env
        """
        super().__init__(api_key)
        self.api_key = api_key or self._get_api_key()
        if not self.api_key:
            raise ValueError(
                "ElevenLabs API key not found. "
                "Set ELEVENLABS_API_KEY or ELEVEN_LABS_API_KEY environment variable"
            )
            
        # Create session with headers
        self.session = requests.Session()
        self.session.headers.update({
            "xi-api-key": self.api_key
        })
        
        self._validate_api_key()
    
    @property
    def provider_name(self) -> str:
        """Return the name of this provider"""
        return "ElevenLabs-STT"
    
    @property
    def requires_api_key(self) -> bool:
        """Return whether this provider requires an API key"""
        return True
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from multiple sources"""
        # Load from .env.local file first
        from ....utils.env import get_env
        
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
            response = self.session.get(f"{self.BASE_URL}/voices")
            response.raise_for_status()
            logger.debug("API key validation successful")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise ValueError("Invalid ElevenLabs API key")
            raise
        except requests.exceptions.RequestException as e:
            logger.warning("Could not validate API key: %s", e)
    
    def transcribe_audio(
        self,
        audio_file: str,
        model_id: str = "scribe_v1",
        language_code: Optional[str] = None,
        num_speakers: Optional[int] = None,
        diarize: bool = True,
        tag_audio_events: bool = False,
        timestamps_granularity: str = "word"
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using ElevenLabs STT API
        
        Args:
            audio_file: Path to audio/video file
            model_id: Transcription model ('scribe_v1' or 'scribe_v1_experimental')
            language_code: ISO language code (auto-detected if None)
            num_speakers: Expected number of speakers (1-32, auto if None)
            diarize: Enable speaker diarization
            tag_audio_events: Tag non-speech audio events
            timestamps_granularity: 'none', 'word', or 'character'
            
        Returns:
            Dictionary containing transcription results
        """
        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
        file_size = audio_path.stat().st_size
        if file_size > 1024 * 1024 * 1024:  # 1GB limit
            raise ValueError(f"File size {file_size/1024/1024:.1f}MB exceeds 1GB limit")
            
        logger.info("Transcribing audio file: %s (%.1fMB)", audio_file, file_size/1024/1024)
        
        url = f"{self.BASE_URL}/speech-to-text"
        
        # Prepare form data
        data = {
            "model_id": model_id,
            "diarize": str(diarize).lower(),
            "tag_audio_events": str(tag_audio_events).lower(),
            "timestamps_granularity": timestamps_granularity
        }
        
        # Add optional parameters
        if language_code:
            data["language_code"] = language_code
        if num_speakers:
            data["num_speakers"] = str(num_speakers)
            
        try:
            # Open file and send request
            with open(audio_file, 'rb') as f:
                files = {"file": (audio_path.name, f, "audio/mpeg")}
                
                logger.debug("Sending STT request with parameters: %s", data)
                response = self.session.post(
                    url, 
                    data=data, 
                    files=files, 
                    timeout=300  # 5 minute timeout for large files
                )
                response.raise_for_status()
                
            result = response.json()
            
            logger.info("Transcription completed: %d words, language: %s (%.1f%% confidence)", 
                       len(result.get('words', [])),
                       result.get('language_code', 'unknown'),
                       (result.get('language_probability', 0) * 100))
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error("STT transcription failed: %s", e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error("API error details: %s", error_detail)
                except:
                    logger.error("Response content: %s", e.response.text)
            raise RuntimeError(f"STT transcription failed: {e}")
    
    def convert_to_diarization_format(
        self, 
        stt_result: Dict[str, Any],
        audio_duration: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], str, float]:
        """
        Convert ElevenLabs STT result to standard diarization format
        
        Args:
            stt_result: Result from transcribe_audio
            audio_duration: Optional audio duration (calculated if None)
            
        Returns:
            Tuple of (utterances_list, transcript_text, total_duration)
        """
        words = stt_result.get('words', [])
        if not words:
            logger.warning("No words found in STT result")
            return [], "", 0.0
            
        # Calculate total duration
        if audio_duration is None:
            audio_duration = max(word.get('end', 0) for word in words)
            
        # Group words by speaker
        utterances = []
        current_speaker = None
        current_text = []
        current_start = None
        current_end = None
        
        for word in words:
            speaker_id = word.get('speaker_id', 'speaker_0')
            word_text = word.get('text', '')
            word_start = word.get('start', 0)
            word_end = word.get('end', 0)
            
            if speaker_id != current_speaker:
                # Save previous utterance
                if current_speaker is not None and current_text:
                    utterances.append({
                        'speaker': current_speaker,
                        'start': current_start,
                        'end': current_end,
                        'text': ' '.join(current_text).strip()
                    })
                
                # Start new utterance
                current_speaker = speaker_id
                current_text = [word_text]
                current_start = word_start
                current_end = word_end
            else:
                # Continue current utterance
                current_text.append(word_text)
                current_end = word_end
        
        # Add final utterance
        if current_speaker is not None and current_text:
            utterances.append({
                'speaker': current_speaker,
                'start': current_start,
                'end': current_end,
                'text': ' '.join(current_text).strip()
            })
        
        # Generate transcript text
        transcript_lines = []
        for utterance in utterances:
            timestamp = f"[{utterance['start']:.1f}s - {utterance['end']:.1f}s]"
            transcript_lines.append(f"{utterance['speaker']} {timestamp}: {utterance['text']}")
        
        transcript_text = '\n'.join(transcript_lines)
        
        logger.info("Converted to %d utterances from %d speakers", 
                   len(utterances), len(set(u['speaker'] for u in utterances)))
        
        return utterances, transcript_text, audio_duration
    
    def create_speaker_tracks_from_utterances(
        self,
        utterances: List[Dict[str, Any]],
        total_duration: float,
        sample_rate: int = 24000
    ) -> Dict[str, np.ndarray]:
        """
        Create placeholder speaker tracks from utterances
        (Since this is STT, we don't have actual separated audio)
        
        Args:
            utterances: List of utterance dictionaries
            total_duration: Total audio duration in seconds
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary of speaker IDs to placeholder audio arrays
        """
        speakers = list(set(u['speaker'] for u in utterances))
        total_samples = int(total_duration * sample_rate)
        
        # Create placeholder tracks (silence with markers for speech segments)
        speaker_tracks = {}
        
        for speaker in speakers:
            # Create silent track
            track = np.zeros(total_samples, dtype=np.float32)
            
            # Add low-level markers where this speaker talks
            # This is just for compatibility - actual audio separation would need different approach
            for utterance in utterances:
                if utterance['speaker'] == speaker:
                    start_sample = int(utterance['start'] * sample_rate)
                    end_sample = int(utterance['end'] * sample_rate)
                    # Add very quiet marker signal (0.01 amplitude sine wave)
                    if start_sample < total_samples and end_sample <= total_samples:
                        duration_samples = end_sample - start_sample
                        marker_signal = 0.01 * np.sin(2 * np.pi * 440 * np.linspace(0, duration_samples/sample_rate, duration_samples))
                        track[start_sample:end_sample] = marker_signal[:end_sample-start_sample]
            
            speaker_tracks[speaker] = track
            
        logger.info("Created %d placeholder speaker tracks", len(speaker_tracks))
        return speaker_tracks
    
    # Required abstract methods from base class
    def diarize(
        self,
        audio_file: str,
        speakers_expected: int = 2,
        language: str = "en",
        enhanced_processing: bool = True
    ) -> Tuple[Dict[str, np.ndarray], str, List[Dict]]:
        """
        Perform speaker diarization using ElevenLabs STT API
        
        Args:
            audio_file: Path to audio file
            speakers_expected: Number of speakers expected
            language: Language code (e.g., "en")
            enhanced_processing: Apply enhanced processing (uses best model)
            
        Returns:
            Tuple of (speaker_tracks_dict, transcript_text, utterances_list)
        """
        logger.info("Starting ElevenLabs STT diarization: %s", audio_file)
        
        # Choose model based on enhanced_processing flag
        model_id = "scribe_v1_experimental" if enhanced_processing else "scribe_v1"
        
        # Transcribe audio with speaker diarization
        stt_result = self.transcribe_audio(
            audio_file=audio_file,
            model_id=model_id,
            language_code=language if language != "en" else None,  # Auto-detect for English
            num_speakers=speakers_expected,
            diarize=True,
            tag_audio_events=False,  # Focus on speech
            timestamps_granularity="word"
        )
        
        # Convert to standard format
        utterances, transcript_text, total_duration = self.convert_to_diarization_format(stt_result)
        
        # Create placeholder speaker tracks
        speaker_tracks = self.create_speaker_tracks_from_utterances(
            utterances, total_duration
        )
        
        logger.info("ElevenLabs STT diarization completed: %d speakers, %.1fs duration", 
                   len(speaker_tracks), total_duration)
        
        return speaker_tracks, transcript_text, utterances
    
    def save_results(
        self,
        speaker_tracks: Dict[str, np.ndarray],
        transcript_text: str,
        output_dir: str,
        base_filename: str,
        sample_rate: int = 24000
    ) -> List[str]:
        """
        Save STT results (transcript and metadata)
        Note: Speaker tracks are placeholders since this is STT, not audio separation
        
        Args:
            speaker_tracks: Dictionary of speaker audio tracks (placeholders)
            transcript_text: Formatted transcript text
            output_dir: Output directory path
            base_filename: Base filename for output files
            sample_rate: Audio sample rate
            
        Returns:
            List of created file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = []
        
        # Determine if this is a transitory/debug file (contains "_stt" in base filename)
        debug_suffix = "_debug" if "_stt" in base_filename and not base_filename.endswith("_final") else ""
        logger.info(f"📝 File naming: base_filename='{base_filename}', debug_suffix='{debug_suffix}'")
        
        # Save transcript
        transcript_file = output_path / f"{base_filename}_elevenlabs_stt_transcript{debug_suffix}.txt"
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(transcript_text)
        saved_files.append(str(transcript_file))
        
        # Save speaker information
        speakers_file = output_path / f"{base_filename}_elevenlabs_stt_speakers{debug_suffix}.txt"
        with open(speakers_file, 'w', encoding='utf-8') as f:
            f.write("ElevenLabs STT Speaker Analysis\n")
            f.write("=" * 40 + "\n\n")
            for speaker_id in speaker_tracks.keys():
                f.write(f"Speaker: {speaker_id}\n")
                f.write(f"Track samples: {len(speaker_tracks[speaker_id])}\n")
                f.write(f"Duration: {len(speaker_tracks[speaker_id]) / sample_rate:.1f}s\n\n")
        saved_files.append(str(speakers_file))
        
        logger.info("Saved ElevenLabs STT results: %s", saved_files)
        return saved_files
    
    def __del__(self):
        """Clean up session on destruction"""
        if hasattr(self, 'session'):
            self.session.close()