"""
AssemblyAI Speaker Diarization Provider
Using the official AssemblyAI SDK for cleaner and more maintainable code
"""

import os
import time
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import assemblyai as aai
import soundfile as sf
import numpy as np

from ..base import SpeakerDiarizationProvider

logger = logging.getLogger(__name__)


class AssemblyAIDiarizer(SpeakerDiarizationProvider):
    """
    Speaker diarization using AssemblyAI's official SDK
    Provides cleaner, more maintainable code with automatic error handling
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize AssemblyAI diarizer with SDK

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

        # Configure the SDK
        aai.settings.api_key = self.api_key
        self.transcriber = aai.Transcriber()

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
        from ....utils.util_env import get_env
        
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
        """Validate API key by testing SDK connection"""
        try:
            # Try to list transcripts to validate the API key (no parameters needed)
            transcripts = self.transcriber.list_transcripts()
            logger.debug("API key validation successful")
        except Exception as e:
            if "401" in str(e) or "Unauthorized" in str(e):
                raise ValueError("Invalid AssemblyAI API key")
            logger.warning("Could not validate API key: %s", e)


    def diarize(
        self,
        audio_file: str,
        speakers_expected: int = 2,
        language: str = "en",
        enhanced_processing: bool = True
    ) -> Tuple[Dict[str, np.ndarray], str, List[Dict]]:
        """
        Perform speaker diarization on audio file using AssemblyAI SDK

        Args:
            audio_file: Path to audio file
            speakers_expected: Number of speakers expected
            language: Language code (e.g., "en")
            enhanced_processing: Apply post-processing for better separation

        Returns:
            Tuple of (speaker_tracks_dict, transcript_text, utterances_list)
        """
        logger.info("🎤 Starting AssemblyAI diarization for: %s", audio_file)

        try:
            # Prepare configuration parameters based on language and features
            config_params = {
                "speaker_labels": True,
                "punctuate": True,
                "format_text": True,
                "speech_model": aai.SpeechModel.best,
            }
            
            # Set language-specific parameters
            if language and language != "auto":
                config_params["language_code"] = language
                config_params["language_detection"] = False
                
                # Disable features not available for non-English languages
                if language != "en":
                    logger.info("Disabling advanced features for non-English language: %s", language)
                    config_params.update({
                        "sentiment_analysis": False,
                        "auto_highlights": False,
                        "disfluencies": False
                    })
                else:
                    config_params.update({
                        "sentiment_analysis": True,
                        "auto_highlights": True,
                        "disfluencies": True
                    })
            else:
                config_params["language_detection"] = True
                config_params.update({
                    "sentiment_analysis": True,
                    "auto_highlights": True,
                    "disfluencies": True
                })
            
            # Add domain-specific word boosting for conversation accuracy
            config_params["word_boost"] = [
                "yeah", "okay", "right", "exactly", "absolutely", "well",
                "um", "uh", "like", "you know", "I mean", "sort of",
                "speaker", "person", "voice", "conversation", "dialogue",
                "Anunnaki", "Mesopotamia", "Sumerian", "Babylonian", "Assyrian"
            ]
            config_params["boost_param"] = aai.WordBoost.high
            
            # Custom spelling for proper nouns and technical terms (dict format)
            config_params["custom_spelling"] = {
                "anunaki": "Anunnaki",
                "mesopotamia": "Mesopotamia", 
                "sumerian": "Sumerian",
                "babylonian": "Babylonian",
                "assyrian": "Assyrian",
                "enlil": "Enlil",
                "enki": "Enki",
                "inanna": "Inanna",
                "ishtar": "Ishtar"
            }
            
            # Create configuration with all parameters at once
            config = aai.TranscriptionConfig(**config_params)

            logger.info("🔗 AssemblyAI SDK Call:")
            logger.info("   Audio File: %s", audio_file)
            logger.info("   Language: %s", language)
            logger.info("   Speakers Expected: %d", speakers_expected)
            logger.info("   Enhanced Processing: %s", enhanced_processing)
            logger.info("   Config: Speaker Labels=%s, Speech Model=%s", 
                       config.speaker_labels, config.speech_model)

            # Transcribe with SDK (handles upload, submission, and polling automatically)
            transcript = self.transcriber.transcribe(audio_file, config=config)
            
            logger.info("✅ Transcription completed with status: %s", transcript.status)
            
            # Check for errors
            if transcript.status == aai.TranscriptStatus.error:
                raise RuntimeError(f"Transcription failed: {transcript.error}")

            # Extract utterances from SDK response
            utterances = []
            if transcript.utterances:
                logger.info("Found %d speaker utterances", len(transcript.utterances))
                utterances = [
                    {
                        'speaker': utt.speaker,
                        'start': utt.start,
                        'end': utt.end,
                        'text': utt.text,
                        'confidence': getattr(utt, 'confidence', 0.0)
                    }
                    for utt in transcript.utterances
                ]
            else:
                raise RuntimeError("No speaker labels found in transcript")
            
            # Apply enhanced boundary detection using word-level timestamps
            if transcript.words:
                logger.info("Processing %d word-level timestamps for precise boundaries", len(transcript.words))
                words = [
                    {
                        'text': word.text,
                        'start': word.start,
                        'end': word.end,
                        'confidence': word.confidence,
                        'speaker': getattr(word, 'speaker', None)
                    }
                    for word in transcript.words
                ]
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
            transcript_text = transcript.text if transcript.text else self._format_transcript(utterances)

            # Prepare utterances list for compatibility
            utterances_list = [
                {
                    'speaker': u.get('speaker'),
                    'start': u.get('start', 0) / 1000.0,  # Convert to seconds if needed
                    'end': u.get('end', 0) / 1000.0,      # Convert to seconds if needed
                    'text': u.get('text', '')
                }
                for u in utterances
            ]

            logger.info("✅ Diarization completed successfully with %d speakers", len(speaker_tracks))
            return speaker_tracks, transcript_text, utterances_list

        except Exception as e:
            logger.error("❌ Diarization failed: %s", e)
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

