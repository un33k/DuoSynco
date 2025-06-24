"""
AssemblyAI Speaker Diarization Provider
Direct HTTP API implementation for maximum control and cross-platform compatibility
"""

import os
import time
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import requests
import soundfile as sf
import numpy as np

from ..base import SpeakerDiarizationProvider

logger = logging.getLogger(__name__)


class AssemblyAIDiarizer(SpeakerDiarizationProvider):
    """
    Speaker diarization using AssemblyAI's HTTP API directly
    Provides full control over requests and cross-platform compatibility
    """

    BASE_URL = "https://api.assemblyai.com/v2"
    UPLOAD_URL = f"{BASE_URL}/upload"
    TRANSCRIPT_URL = f"{BASE_URL}/transcript"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize AssemblyAI diarizer with direct HTTP API

        Args:
            api_key: AssemblyAI API key. If None, will try to get from env
        """
        super().__init__(api_key)
        self.api_key = api_key or self._get_api_key()
        if not self.api_key:
            raise ValueError(
                "AssemblyAI API key not found. "
                "Set ASSEMBLYAI_API_KEY environment variable"
            )

        self.headers = {
            "authorization": self.api_key,
            "content-type": "application/json"
        }

        # Setup session for connection reuse
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Test API key validity
        self._validate_api_key()

    @property
    def provider_name(self) -> str:
        """Return the name of this provider"""
        return "AssemblyAI"

    @property
    def requires_api_key(self) -> bool:
        """Return whether this provider requires an API key"""
        return True

    def _get_api_key(self) -> Optional[str]:
        """Get API key from multiple sources"""
        # Load from .env.local file first
        from ....utils.env_loader import get_env
        
        # Try environment variable from .env.local
        api_key = get_env('ASSEMBLYAI_API_KEY')
        if api_key:
            return api_key

        # Try direct environment variable (fallback)
        api_key = os.getenv('ASSEMBLYAI_API_KEY')
        if api_key:
            return api_key

        # Try local key file (legacy fallback)
        try:
            with open('assemblyai_key.txt', 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            pass

        return None

    def _validate_api_key(self) -> None:
        """Validate API key by making a test request"""
        try:
            response = self.session.get(f"{self.BASE_URL}/transcript")
            response.raise_for_status()
            logger.debug("API key validation successful")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise ValueError("Invalid AssemblyAI API key")
            raise
        except requests.exceptions.RequestException as e:
            logger.warning("Could not validate API key: %s", e)

    def _upload_file(self, audio_file: str) -> str:
        """
        Upload audio file to AssemblyAI and return upload URL

        Args:
            audio_file: Path to audio file

        Returns:
            Upload URL for the file
        """
        logger.info("Uploading file: %s", audio_file)

        upload_headers = {"authorization": self.api_key}

        try:
            with open(audio_file, 'rb') as f:
                response = self.session.post(
                    self.UPLOAD_URL,
                    files={'file': f},
                    headers=upload_headers,
                    timeout=300  # 5 minute timeout for upload
                )
            response.raise_for_status()

            upload_url = response.json().get('upload_url')
            if not upload_url:
                raise ValueError("No upload URL returned from AssemblyAI")

            logger.info("File uploaded successfully")
            return upload_url

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to upload file: {e}")

    def _submit_transcription(
        self,
        audio_url: str,
        speakers_expected: int,
        language: str
    ) -> str:
        """
        Submit transcription request to AssemblyAI

        Args:
            audio_url: URL of uploaded audio file
            speakers_expected: Number of speakers expected
            language: Language code

        Returns:
            Transcript ID for polling
        """
        logger.info("Submitting transcription request")

        data = {
            "audio_url": audio_url,
            "speaker_labels": True,
            "speakers_expected": speakers_expected,
            "punctuate": True,
            "format_text": True,
            # Enhanced quality settings
            "speech_model": "best",
            "language_detection": True,
            "disfluencies": True,
            "sentiment_analysis": True,
            "auto_highlights": True
        }
        
        # Use language_code OR language_detection, not both
        if language and language != "auto":
            data["language_code"] = language
            del data["language_detection"]

        try:
            response = self.session.post(
                self.TRANSCRIPT_URL,
                json=data,
                timeout=30
            )
            response.raise_for_status()

            transcript_id = response.json().get('id')
            if not transcript_id:
                raise ValueError("No transcript ID returned from AssemblyAI")

            logger.info("Transcription submitted with ID: %s", transcript_id)
            return transcript_id

        except requests.exceptions.HTTPError as e:
            # Log the response body for debugging
            error_details = e.response.text if hasattr(e.response, 'text') else str(e)
            logger.error("HTTP Error %s: %s", e.response.status_code, error_details)
            raise RuntimeError(f"Failed to submit transcription: {e} - Details: {error_details}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to submit transcription: {e}")

    def _poll_transcription(self, transcript_id: str) -> Dict:
        """
        Poll for transcription completion

        Args:
            transcript_id: ID of the transcription job

        Returns:
            Complete transcription result
        """
        logger.info("Polling transcription status...")

        url = f"{self.TRANSCRIPT_URL}/{transcript_id}"
        max_wait_time = 1800  # 30 minutes max
        start_time = time.time()
        poll_interval = 5  # Start with 5 seconds

        while time.time() - start_time < max_wait_time:
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                result = response.json()

                status = result.get('status')
                logger.debug("Transcription status: %s", status)

                if status == 'completed':
                    logger.info("Transcription completed successfully")
                    return result
                elif status == 'error':
                    error_msg = result.get('error', 'Unknown error')
                    raise RuntimeError(f"Transcription failed: {error_msg}")
                elif status in ['queued', 'processing']:
                    time.sleep(poll_interval)
                    # Gradually increase poll interval up to 30 seconds
                    poll_interval = min(poll_interval + 2, 30)
                else:
                    raise RuntimeError(f"Unknown transcription status: {status}")

            except requests.exceptions.RequestException as e:
                logger.warning("Error polling transcription: %s", e)
                time.sleep(poll_interval)

        raise TimeoutError("Transcription timed out after 30 minutes")

    def diarize(
        self,
        audio_file: str,
        speakers_expected: int = 2,
        language: str = "en",
        enhanced_processing: bool = True
    ) -> Tuple[Dict[str, np.ndarray], str, List[Dict]]:
        """
        Perform speaker diarization on audio file using HTTP API

        Args:
            audio_file: Path to audio file
            speakers_expected: Number of speakers expected
            language: Language code (e.g., "en")
            enhanced_processing: Apply post-processing for better separation

        Returns:
            Tuple of (speaker_tracks_dict, transcript_text, utterances_list)
        """
        logger.info("Starting diarization for: %s", audio_file)

        try:
            # Step 1: Upload file
            audio_url = self._upload_file(audio_file)

            # Step 2: Submit transcription
            transcript_id = self._submit_transcription(
                audio_url, speakers_expected, language
            )

            # Step 3: Poll for completion
            result = self._poll_transcription(transcript_id)

            # Step 4: Process results with enhanced boundary detection
            utterances = result.get('utterances', [])
            if not utterances:
                raise RuntimeError("No speaker labels found in transcript")

            logger.info("Found %d speaker utterances", len(utterances))
            
            # Apply enhanced boundary detection using word-level timestamps
            words = result.get('words', [])
            if words:
                logger.info("Processing %d word-level timestamps for precise boundaries", len(words))
                utterances = self._enhance_speaker_boundaries(utterances, words)
            else:
                logger.warning("No word-level timestamps available, using basic utterance boundaries")

            # Load original audio
            audio_data, sample_rate = sf.read(audio_file)

            # Process utterances into speaker tracks
            if enhanced_processing:
                speaker_tracks = self._enhanced_separation(
                    utterances, audio_data, sample_rate
                )
            else:
                speaker_tracks = self._basic_separation(
                    utterances, audio_data, sample_rate
                )

            # Prepare transcript text
            transcript_text = self._format_transcript(utterances)

            # Prepare utterances list
            utterances_list = [
                {
                    'speaker': u.get('speaker'),
                    'start': u.get('start', 0) / 1000.0,
                    'end': u.get('end', 0) / 1000.0,
                    'text': u.get('text', '')
                }
                for u in utterances
            ]

            logger.info("Diarization completed successfully")
            return speaker_tracks, transcript_text, utterances_list

        except Exception as e:
            logger.error("Diarization failed: %s", e)
            raise

    def _basic_separation(
        self,
        utterances: List[Dict],
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Dict[str, np.ndarray]:
        """Basic speaker separation without post-processing"""
        speakers = set(u.get('speaker') for u in utterances if u.get('speaker'))
        speaker_tracks = {
            speaker: np.zeros_like(audio_data) for speaker in speakers
        }

        for utterance in utterances:
            speaker = utterance.get('speaker')
            start_ms = utterance.get('start', 0)
            end_ms = utterance.get('end', 0)

            if not speaker or start_ms >= end_ms:
                continue

            start_sample = int((start_ms / 1000.0) * sample_rate)
            end_sample = int((end_ms / 1000.0) * sample_rate)

            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)

            if start_sample < end_sample:
                speaker_tracks[speaker][start_sample:end_sample] = \
                    audio_data[start_sample:end_sample]

        return speaker_tracks

    def _enhanced_separation(
        self,
        utterances: List[Dict],
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Dict[str, np.ndarray]:
        """Enhanced speaker separation with voice bleed-through reduction"""
        logger.info("Applying enhanced voice separation techniques...")

        speakers = set(u.get('speaker') for u in utterances if u.get('speaker'))
        speaker_tracks = {
            speaker: np.zeros_like(audio_data) for speaker in speakers
        }

        # Sort utterances by start time
        sorted_utterances = sorted(
            utterances,
            key=lambda u: u.get('start', 0)
        )

        # Process each utterance with enhancements
        for utterance in sorted_utterances:
            speaker = utterance.get('speaker')
            start_ms = utterance.get('start', 0)
            end_ms = utterance.get('end', 0)
            text = utterance.get('text', '')

            if not speaker or start_ms >= end_ms:
                continue

            start_time = start_ms / 1000.0
            end_time = end_ms / 1000.0
            duration = end_time - start_time

            # Skip very short utterances (likely artifacts)
            if duration < 0.3 and len(text.split()) <= 2:
                logger.debug(
                    "Skipping short utterance: %s (%.2fs) - '%s'",
                    speaker, duration, text
                )
                continue

            # Convert to sample indices
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            # Add fade margins to reduce bleed-through
            fade_samples = int(0.1 * sample_rate)  # 100ms fade

            if start_sample > 0:
                start_sample += fade_samples
            if end_sample < len(audio_data):
                end_sample -= fade_samples

            # Ensure bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)

            # Apply audio with fade processing
            if start_sample < end_sample:
                audio_segment = audio_data[start_sample:end_sample].copy()

                # Apply fade-in/fade-out
                if len(audio_segment) > 2 * fade_samples:
                    fade_in = np.linspace(0, 1, fade_samples)
                    audio_segment[:fade_samples] *= fade_in

                    fade_out = np.linspace(1, 0, fade_samples)
                    audio_segment[-fade_samples:] *= fade_out

                speaker_tracks[speaker][start_sample:end_sample] = \
                    audio_segment

        # Cross-talk cleanup
        self._cleanup_cross_talk(speaker_tracks, sample_rate)

        return speaker_tracks

    def _enhance_speaker_boundaries(
        self, utterances: List[Dict], words: List[Dict]
    ) -> List[Dict]:
        """
        Enhance speaker boundaries using word-level timestamps for precise cuts
        
        Args:
            utterances: Original speaker utterances
            words: Word-level timestamps from AssemblyAI
            
        Returns:
            Enhanced utterances with precise word-boundary timing
        """
        logger.info("Enhancing speaker boundaries with word-level precision")
        
        enhanced_utterances = []
        
        for i, utterance in enumerate(utterances):
            speaker = utterance.get('speaker')
            start_ms = utterance.get('start', 0)
            end_ms = utterance.get('end', 0)
            text = utterance.get('text', '')
            
            # Find words that belong to this utterance
            utterance_words = [
                word for word in words
                if (word.get('start', 0) >= start_ms and 
                    word.get('end', 0) <= end_ms + 100)  # 100ms tolerance
            ]
            
            if not utterance_words:
                # Fallback to original timing if no words found
                enhanced_utterances.append(utterance)
                continue
            
            # Use first and last word for precise boundaries
            precise_start = utterance_words[0].get('start', start_ms)
            precise_end = utterance_words[-1].get('end', end_ms)
            
            # Check for natural speech gaps and eliminate artificial ones
            if i > 0:
                prev_utterance = enhanced_utterances[-1]
                prev_end = prev_utterance.get('end', 0)
                gap_ms = precise_start - prev_end
                
                # If gap is very small (< 200ms), eliminate it for seamless transition
                if gap_ms < 200:
                    logger.debug("Eliminating %dms gap between speakers", gap_ms)
                    precise_start = prev_end
                    
                # If gap is artificial (exactly 400ms), reduce to natural pause
                elif 390 <= gap_ms <= 410:  # 400ms ± 10ms tolerance
                    logger.debug("Converting artificial 400ms gap to natural 100ms pause")
                    precise_start = prev_end + 100
            
            enhanced_utterance = {
                'speaker': speaker,
                'start': precise_start,
                'end': precise_end,
                'text': text,
                'word_count': len(utterance_words),
                'enhanced': True
            }
            
            enhanced_utterances.append(enhanced_utterance)
            
            logger.debug(
                "Enhanced %s: %.2fs-%.2fs (%d words) - '%s'",
                speaker, precise_start/1000, precise_end/1000, 
                len(utterance_words), text[:50]
            )
        
        logger.info("Enhanced %d utterances with word-level boundaries", len(enhanced_utterances))
        return enhanced_utterances

    def _cleanup_cross_talk(
        self, speaker_tracks: Dict[str, np.ndarray], sample_rate: int
    ) -> None:
        """Remove overlapping segments using energy-based analysis"""
        logger.info("Performing cross-talk cleanup...")

        speakers = list(speaker_tracks.keys())

        for speaker in speakers:
            other_speakers = [s for s in speakers if s != speaker]
            speaker_audio = speaker_tracks[speaker]

            # Check every 100ms window
            window_size = sample_rate // 10
            for i in range(0, len(speaker_audio), window_size):
                window_end = min(i + window_size, len(speaker_audio))
                window = speaker_audio[i:window_end]

                # Check if this speaker has significant audio
                if np.max(np.abs(window)) > 0.01:
                    # Check overlap with other speakers
                    for other_speaker in other_speakers:
                        other_window = speaker_tracks[other_speaker][
                            i:window_end
                        ]
                        if np.max(np.abs(other_window)) > 0.01:
                            # Energy-based resolution
                            this_energy = np.mean(window ** 2)
                            other_energy = np.mean(other_window ** 2)

                            # Reduce quieter speaker
                            if other_energy > this_energy * 1.2:
                                speaker_tracks[speaker][i:window_end] *= 0.1

    def _format_transcript(self, utterances: List[Dict]) -> str:
        """Format transcript with speaker labels and timestamps"""
        lines = ["=== ASSEMBLYAI TRANSCRIPT WITH SPEAKER LABELS ===\n"]

        for utterance in utterances:
            speaker = utterance.get('speaker', 'Unknown')
            start_time = utterance.get('start', 0) / 1000.0
            end_time = utterance.get('end', 0) / 1000.0
            text = utterance.get('text', '')

            lines.append(
                f"{speaker} ({start_time:.2f}s-{end_time:.2f}s): {text}"
            )

        return "\n".join(lines)

    def save_results(
        self,
        speaker_tracks: Dict[str, np.ndarray],
        transcript_text: str,
        output_dir: str,
        base_filename: str,
        sample_rate: int = 24000
    ) -> List[str]:
        """
        Save separated audio files and transcript

        Args:
            speaker_tracks: Dictionary of speaker audio tracks
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

        # Save audio files
        for speaker in sorted(speaker_tracks.keys()):
            non_zero_samples = np.sum(speaker_tracks[speaker] != 0)
            speaker_duration = non_zero_samples / sample_rate

            if speaker_duration > 0:
                audio_file = (
                    output_path / f"{base_filename}_{speaker.lower()}_"
                    f"{self.provider_name.lower()}.wav"
                )
                sf.write(audio_file, speaker_tracks[speaker], sample_rate)
                saved_files.append(str(audio_file))
                logger.info(
                    "Saved %s: %s (%.1fs)",
                    speaker, audio_file, speaker_duration
                )

        # Save transcript
        transcript_file = (
            output_path / f"{base_filename}_transcript_"
            f"{self.provider_name.lower()}.txt"
        )
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(transcript_text)
        saved_files.append(str(transcript_file))

        logger.info("Saved transcript: %s", transcript_file)
        return saved_files

    def __del__(self):
        """Clean up session on destruction"""
        if hasattr(self, 'session'):
            self.session.close()
