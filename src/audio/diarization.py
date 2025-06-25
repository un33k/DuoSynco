"""
Speaker Diarization Module
High-level interface for speaker separation using multiple providers
"""

import logging
from typing import Dict, Optional
import numpy as np
from pathlib import Path

from .providers.factory import ProviderFactory

logger = logging.getLogger(__name__)


class SpeakerDiarizer:
    """
    High-level speaker diarization interface
    Supports multiple providers (AssemblyAI, ElevenLabs, etc.)
    """

    def __init__(self, provider: str = "assemblyai", api_key: Optional[str] = None):
        """
        Initialize speaker diarizer

        Args:
            provider: Provider name ('assemblyai', 'elevenlabs', etc.)
            api_key: API key for the provider
        """
        self.provider_name = provider
        self.diarizer = ProviderFactory.get_provider(provider, api_key)
        logger.info("Speaker diarizer initialized with %s backend", self.diarizer.provider_name)

    def separate_speakers(
        self,
        audio_file: str,
        output_dir: str = "output",
        speakers_expected: int = 2,
        language: str = "en",
        enhanced_processing: bool = True,
        base_filename: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Separate speakers in audio file and save results

        Args:
            audio_file: Path to input audio file
            output_dir: Directory to save output files
            speakers_expected: Number of speakers expected
            language: Language code (e.g., "en")
            enhanced_processing: Apply post-processing for better separation
            base_filename: Base name for output files (auto-generated if None)

        Returns:
            Dictionary with results:
            {
                'speaker_files': List[str],  # Paths to separated audio files
                'transcript_file': str,      # Path to transcript file
                'speakers': List[str],       # List of speaker IDs
                'utterances': List[Dict],    # Detailed utterance data
                'stats': Dict,               # Processing statistics
                'provider': str              # Provider used
            }
        """
        logger.info("Starting speaker separation for: %s", audio_file)

        # Generate base filename if not provided
        if base_filename is None:
            base_filename = Path(audio_file).stem

        # Perform diarization
        speaker_tracks, transcript_text, utterances = self.diarizer.diarize(
            audio_file=audio_file,
            speakers_expected=speakers_expected,
            language=language,
            enhanced_processing=enhanced_processing,
        )

        # Get audio properties for stats
        import soundfile as sf

        audio_data, sample_rate = sf.read(audio_file)
        original_duration = len(audio_data) / sample_rate

        # Save results
        saved_files = self.diarizer.save_results(
            speaker_tracks=speaker_tracks,
            transcript_text=transcript_text,
            output_dir=output_dir,
            base_filename=base_filename,
            sample_rate=sample_rate,
        )

        # Separate audio files from transcript file
        audio_files = [f for f in saved_files if f.endswith(".wav")]
        transcript_file = [f for f in saved_files if f.endswith(".txt")][0]

        # Calculate statistics
        stats = self._calculate_stats(speaker_tracks, sample_rate, original_duration)

        # Prepare result
        result = {
            "speaker_files": audio_files,
            "transcript_file": transcript_file,
            "speakers": list(speaker_tracks.keys()),
            "utterances": utterances,
            "stats": stats,
            "provider": self.diarizer.provider_name,
        }

        logger.info("Speaker separation completed successfully")
        logger.info("Generated %d speaker files", len(audio_files))
        logger.info(
            "Coverage: %.1f%% (%.1fs / %.1fs)",
            stats["total_coverage"],
            stats["total_speaker_duration"],
            stats["original_duration"],
        )

        return result

    def _calculate_stats(
        self,
        speaker_tracks: Dict[str, np.ndarray],
        sample_rate: int,
        original_duration: float,
    ) -> Dict[str, any]:
        """Calculate processing statistics"""
        stats = {
            "original_duration": original_duration,
            "speakers": {},
            "total_speaker_duration": 0.0,
            "total_coverage": 0.0,
        }

        for speaker, track in speaker_tracks.items():
            non_zero_samples = np.sum(track != 0)
            duration = non_zero_samples / sample_rate

            stats["speakers"][speaker] = {
                "duration": duration,
                "coverage": ((duration / original_duration) * 100 if original_duration > 0 else 0),
            }
            stats["total_speaker_duration"] += duration

        stats["total_coverage"] = (
            (stats["total_speaker_duration"] / original_duration) * 100
            if original_duration > 0
            else 0
        )

        return stats

    @staticmethod
    def list_providers() -> Dict[str, Dict]:
        """
        List all available providers

        Returns:
            Dictionary of available providers and their status
        """
        return ProviderFactory.list_providers()
