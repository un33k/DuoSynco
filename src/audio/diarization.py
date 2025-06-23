"""
Speaker Diarization Module
Identifies and separates different speakers in audio/video files
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

try:
    from pyannote.audio import Pipeline
    import torch
    import torchaudio
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

from ..utils.config import Config


@dataclass
class SpeakerSegment:
    """Represents a segment of audio with speaker information"""
    start_time: float
    end_time: float
    speaker_id: str
    confidence: float = 1.0


class SpeakerDiarizer:
    """
    Handles speaker diarization using pyannote.audio pipeline
    Identifies who speaks when in an audio/video file
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.pipeline: Optional[Pipeline] = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self) -> None:
        """Initialize the speaker diarization pipeline"""
        if not PYANNOTE_AVAILABLE:
            raise ImportError(
                "pyannote.audio is required for speaker diarization. "
                "Install with: pip install pyannote.audio"
            )
        
        try:
            # Use pre-trained speaker diarization pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.0",
                use_auth_token=None  # May require HuggingFace token for some models
            )
            
            if self.config.verbose:
                print("✅ Speaker diarization pipeline initialized")
                
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Warning: Could not initialize pyannote pipeline: {e}")
            self.pipeline = None
    
    def diarize(self, input_file: Path, num_speakers: int = 2) -> List[SpeakerSegment]:
        """
        Perform speaker diarization on the input file
        
        Args:
            input_file: Path to audio/video file
            num_speakers: Expected number of speakers
            
        Returns:
            List of SpeakerSegment objects with timing and speaker info
        """
        if self.pipeline is None:
            # Fallback to simple segmentation if pipeline not available
            return self._fallback_diarization(input_file, num_speakers)
        
        try:
            # Apply the pipeline to the audio file
            diarization = self.pipeline(str(input_file))
            
            # Convert to our SpeakerSegment format
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = SpeakerSegment(
                    start_time=turn.start,
                    end_time=turn.end,
                    speaker_id=speaker,
                    confidence=1.0  # pyannote doesn't provide confidence scores directly
                )
                segments.append(segment)
            
            if self.config.verbose:
                print(f"🔍 Identified {len(segments)} speaker segments")
                self._print_diarization_summary(segments)
            
            return segments
            
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Diarization failed: {e}, using fallback method")
            return self._fallback_diarization(input_file, num_speakers)
    
    def _fallback_diarization(self, input_file: Path, num_speakers: int) -> List[SpeakerSegment]:
        """
        Fallback method when advanced diarization is not available
        Creates simple alternating speaker segments
        """
        try:
            # Get audio duration using torchaudio or librosa
            if PYANNOTE_AVAILABLE:
                waveform, sample_rate = torchaudio.load(str(input_file))
                duration = waveform.shape[1] / sample_rate
            else:
                # Fallback to estimate (this would need librosa or other method)
                duration = 60.0  # Default assumption
            
            # Create alternating segments of roughly equal length
            segment_length = duration / (num_speakers * 2)  # Alternate between speakers
            segments = []
            
            current_time = 0.0
            speaker_id = 0
            
            while current_time < duration:
                end_time = min(current_time + segment_length, duration)
                
                segment = SpeakerSegment(
                    start_time=current_time,
                    end_time=end_time,
                    speaker_id=f"SPEAKER_{speaker_id}",
                    confidence=0.5  # Lower confidence for fallback
                )
                segments.append(segment)
                
                current_time = end_time
                speaker_id = (speaker_id + 1) % num_speakers
            
            if self.config.verbose:
                print(f"🔄 Using fallback diarization with {len(segments)} segments")
            
            return segments
            
        except Exception as e:
            if self.config.verbose:
                print(f"❌ Fallback diarization failed: {e}")
            return []
    
    def _print_diarization_summary(self, segments: List[SpeakerSegment]) -> None:
        """Print a summary of the diarization results"""
        if not segments:
            return
        
        # Group by speaker
        speaker_stats = {}
        for segment in segments:
            if segment.speaker_id not in speaker_stats:
                speaker_stats[segment.speaker_id] = {
                    'total_time': 0.0,
                    'segment_count': 0
                }
            
            duration = segment.end_time - segment.start_time
            speaker_stats[segment.speaker_id]['total_time'] += duration
            speaker_stats[segment.speaker_id]['segment_count'] += 1
        
        print("📊 Speaker Summary:")
        for speaker_id, stats in speaker_stats.items():
            print(f"  {speaker_id}: {stats['total_time']:.1f}s "
                  f"({stats['segment_count']} segments)")
    
    def get_speaker_timeline(self, segments: List[SpeakerSegment]) -> Dict[str, List[Tuple[float, float]]]:
        """
        Convert segments to speaker timeline format
        
        Returns:
            Dictionary mapping speaker_id to list of (start_time, end_time) tuples
        """
        timeline = {}
        
        for segment in segments:
            if segment.speaker_id not in timeline:
                timeline[segment.speaker_id] = []
            
            timeline[segment.speaker_id].append((segment.start_time, segment.end_time))
        
        # Sort timelines by start time
        for speaker_id in timeline:
            timeline[speaker_id].sort(key=lambda x: x[0])
        
        return timeline