"""
Voice Isolation Module
Creates isolated audio tracks for each speaker based on diarization results
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import tempfile
import os

from .diarization import SpeakerSegment, SpeakerDiarizer
from .processor import AudioProcessor
from ..utils.config import Config


class VoiceIsolator:
    """
    Creates isolated audio tracks for each speaker
    Takes diarization results and creates separate audio files
    where each file contains only one speaker's voice
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.audio_processor = AudioProcessor(config)
    
    def isolate_speakers(self, 
                        input_file: Path, 
                        speaker_segments: List[SpeakerSegment]) -> Dict[str, Path]:
        """
        Create isolated audio tracks for each speaker
        
        Args:
            input_file: Original audio/video file
            speaker_segments: List of speaker segments from diarization
            
        Returns:
            Dictionary mapping speaker_id to path of isolated audio file
        """
        if not speaker_segments:
            if self.config.verbose:
                print("⚠️  No speaker segments provided for isolation")
            return {}
        
        # Get original audio info
        audio_info = self.audio_processor.get_audio_info(input_file)
        total_duration = audio_info.duration
        
        if self.config.verbose:
            print(f"🎵 Creating isolated tracks for {total_duration:.1f}s audio")
        
        # Group segments by speaker
        speaker_timeline = self._group_segments_by_speaker(speaker_segments)
        
        # Create isolated tracks
        isolated_tracks = {}
        
        for speaker_id, segments in speaker_timeline.items():
            if self.config.verbose:
                print(f"  Processing {speaker_id}...")
            
            isolated_path = self._create_isolated_track(
                input_file, segments, speaker_id, total_duration
            )
            
            if isolated_path:
                isolated_tracks[speaker_id] = isolated_path
        
        if self.config.verbose:
            print(f"✅ Created {len(isolated_tracks)} isolated tracks")
        
        return isolated_tracks
    
    def _group_segments_by_speaker(self, 
                                  segments: List[SpeakerSegment]) -> Dict[str, List[Tuple[float, float]]]:
        """
        Group speaker segments by speaker ID
        
        Returns:
            Dictionary mapping speaker_id to list of (start_time, end_time) tuples
        """
        speaker_timeline = {}
        
        for segment in segments:
            if segment.speaker_id not in speaker_timeline:
                speaker_timeline[segment.speaker_id] = []
            
            speaker_timeline[segment.speaker_id].append(
                (segment.start_time, segment.end_time)
            )
        
        # Sort segments by start time for each speaker
        for speaker_id in speaker_timeline:
            speaker_timeline[speaker_id].sort(key=lambda x: x[0])
        
        return speaker_timeline
    
    def _create_isolated_track(self, 
                              input_file: Path, 
                              speaker_segments: List[Tuple[float, float]], 
                              speaker_id: str, 
                              total_duration: float) -> Optional[Path]:
        """
        Create an isolated audio track for a specific speaker
        
        Args:
            input_file: Original audio/video file
            speaker_segments: List of (start_time, end_time) for this speaker
            speaker_id: Speaker identifier
            total_duration: Total duration of the original audio
            
        Returns:
            Path to the created isolated audio file, or None if failed
        """
        try:
            # Create output filename
            output_dir = Path(tempfile.gettempdir()) / "duosynco_temp"
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / f"{input_file.stem}_{speaker_id}_isolated.wav"
            
            # Get audio info for sample rate
            audio_info = self.audio_processor.get_audio_info(input_file)
            sample_rate = audio_info.sample_rate
            
            # Create timeline: 1 = speaker active, 0 = silent
            timeline = self._create_speaker_timeline(speaker_segments, total_duration, sample_rate)
            
            # Method 1: Segment-based isolation (recommended)
            success = self._create_track_by_segments(
                input_file, speaker_segments, output_file, total_duration, sample_rate
            )
            
            if success:
                return output_file
            else:
                # Method 2: Timeline-based isolation (fallback)
                return self._create_track_by_timeline(
                    input_file, timeline, output_file, sample_rate
                )
        
        except Exception as e:
            if self.config.verbose:
                print(f"❌ Failed to create isolated track for {speaker_id}: {e}")
            return None
    
    def _create_speaker_timeline(self, 
                               segments: List[Tuple[float, float]], 
                               total_duration: float, 
                               sample_rate: int) -> np.ndarray:
        """
        Create a binary timeline indicating when speaker is active
        
        Returns:
            Binary array where 1 = speaker active, 0 = silent
        """
        # Create timeline array
        total_samples = int(total_duration * sample_rate)
        timeline = np.zeros(total_samples, dtype=np.float32)
        
        # Mark active segments
        for start_time, end_time in segments:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Ensure bounds
            start_sample = max(0, start_sample)
            end_sample = min(total_samples, end_sample)
            
            if start_sample < end_sample:
                timeline[start_sample:end_sample] = 1.0
        
        return timeline
    
    def _create_track_by_segments(self, 
                                 input_file: Path, 
                                 segments: List[Tuple[float, float]], 
                                 output_file: Path, 
                                 total_duration: float, 
                                 sample_rate: int) -> bool:
        """
        Create isolated track by extracting and combining segments
        """
        try:
            # Create full-length silent track
            total_samples = int(total_duration * sample_rate)
            isolated_audio = np.zeros(total_samples, dtype=np.float32)
            
            # Extract and place each segment
            for start_time, end_time in segments:
                # Extract segment from original audio
                segment_data = self.audio_processor.extract_audio_segment(
                    input_file, start_time, end_time
                )
                
                if segment_data is not None:
                    # Calculate position in output array
                    start_sample = int(start_time * sample_rate)
                    end_sample = min(
                        start_sample + len(segment_data),
                        total_samples
                    )
                    
                    # Ensure we don't exceed bounds
                    segment_length = end_sample - start_sample
                    if segment_length > 0:
                        isolated_audio[start_sample:end_sample] = segment_data[:segment_length]
            
            # Normalize the audio
            isolated_audio = self.audio_processor.normalize_audio(isolated_audio)
            
            # Save to file
            return self.audio_processor.save_audio(isolated_audio, output_file, sample_rate)
        
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Segment-based isolation failed: {e}")
            return False
    
    def _create_track_by_timeline(self, 
                                 input_file: Path, 
                                 timeline: np.ndarray, 
                                 output_file: Path, 
                                 sample_rate: int) -> Optional[Path]:
        """
        Create isolated track by applying timeline mask (fallback method)
        """
        try:
            # Load full audio file
            audio_info = self.audio_processor.get_audio_info(input_file)
            
            # This is a simplified approach - in practice, you'd need to
            # load the audio data and apply the timeline mask
            # For now, create a basic implementation
            
            duration = len(timeline) / sample_rate
            
            # Create segments based on timeline
            segments = []
            in_segment = False
            start_time = 0.0
            
            for i, active in enumerate(timeline):
                time = i / sample_rate
                
                if active and not in_segment:
                    # Start of new segment
                    start_time = time
                    in_segment = True
                elif not active and in_segment:
                    # End of segment
                    segments.append((start_time, time))
                    in_segment = False
            
            # Handle case where file ends while in segment
            if in_segment:
                segments.append((start_time, duration))
            
            # Use segment-based method with detected segments
            return self._create_track_by_segments(
                input_file, segments, output_file, duration, sample_rate
            ) and output_file or None
        
        except Exception as e:
            if self.config.verbose:
                print(f"❌ Timeline-based isolation failed: {e}")
            return None
    
    def cleanup_temp_files(self, isolated_tracks: Dict[str, Path]) -> None:
        """
        Clean up temporary isolated audio files
        
        Args:
            isolated_tracks: Dictionary of isolated track paths
        """
        for speaker_id, file_path in isolated_tracks.items():
            try:
                if file_path.exists():
                    file_path.unlink()
                    if self.config.verbose:
                        print(f"🗑️  Cleaned up temp file: {file_path}")
            except Exception as e:
                if self.config.verbose:
                    print(f"⚠️  Could not clean up {file_path}: {e}")
        
        # Try to remove temp directory if empty
        try:
            temp_dir = Path(tempfile.gettempdir()) / "duosynco_temp"
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()
        except Exception:
            pass  # Ignore cleanup errors