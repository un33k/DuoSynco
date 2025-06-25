"""
Base provider interface for speaker diarization
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np


class SpeakerDiarizationProvider(ABC):
    """
    Abstract base class for speaker diarization providers
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the provider with optional API key"""
        self.api_key = api_key

    @abstractmethod
    def diarize(
        self,
        audio_file: str,
        speakers_expected: int = 2,
        language: str = "en",
        enhanced_processing: bool = True,
    ) -> Tuple[Dict[str, np.ndarray], str, List[Dict]]:
        """
        Perform speaker diarization on audio file

        Args:
            audio_file: Path to audio file
            speakers_expected: Number of speakers expected
            language: Language code (e.g., "en")
            enhanced_processing: Apply post-processing for better separation

        Returns:
            Tuple of (speaker_tracks_dict, transcript_text, utterances_list)
        """
        pass

    @abstractmethod
    def save_results(
        self,
        speaker_tracks: Dict[str, np.ndarray],
        transcript_text: str,
        output_dir: str,
        base_filename: str,
        sample_rate: int = 24000,
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
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider"""
        pass

    @property
    @abstractmethod
    def requires_api_key(self) -> bool:
        """Return whether this provider requires an API key"""
        pass
