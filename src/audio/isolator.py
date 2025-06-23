"""
Voice Isolation Module
Creates isolated audio tracks for each speaker based on diarization results
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import tempfile
import os
import scipy.signal

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from .diarization import SpeakerSegment, SpeakerDiarizer
from .processor import AudioProcessor
from .ml_separator import MLVoiceSeparator
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
        self.ml_separator = MLVoiceSeparator(config)
    
    def isolate_speakers(self, 
                        input_file: Path, 
                        speaker_segments: List[SpeakerSegment]) -> Dict[str, Path]:
        """
        Create isolated audio tracks for each speaker using ML or spectral separation
        
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
        
        # Group segments by speaker
        speaker_timeline = self._group_segments_by_speaker(speaker_segments)
        
        # Try ML-based separation first
        if self.ml_separator.is_available():
            if self.config.verbose:
                print(f"🤖 Using ML-based voice separation for {total_duration:.1f}s audio")
            
            isolated_tracks = self.ml_separator.separate_speakers_from_segments(
                input_file, speaker_timeline, total_duration
            )
            
            # Verify tracks were created successfully
            if isolated_tracks:
                for speaker_id, track_path in isolated_tracks.items():
                    if self.config.verbose:
                        self._verify_track_length(track_path, total_duration)
                
                if self.config.verbose:
                    print(f"✅ Created {len(isolated_tracks)} ML-separated tracks")
                return isolated_tracks
        
        # Fallback to spectral separation
        if self.config.verbose:
            print(f"🎵 Using spectral voice separation for {total_duration:.1f}s audio")
        
        return self._fallback_spectral_separation(input_file, speaker_timeline, total_duration)
    
    def _verify_track_length(self, track_path: Path, expected_duration: float) -> None:
        """
        Verify that the created track has the correct duration
        """
        try:
            track_info = self.audio_processor.get_audio_info(track_path)
            actual_duration = track_info.duration
            
            if abs(actual_duration - expected_duration) < 0.1:  # Within 100ms tolerance
                print(f"    ✅ Track length verified: {actual_duration:.1f}s (expected {expected_duration:.1f}s)")
            else:
                print(f"    ⚠️  Track length mismatch: {actual_duration:.1f}s (expected {expected_duration:.1f}s)")
        
        except Exception as e:
            if self.config.verbose:
                print(f"    ⚠️  Could not verify track length: {e}")
    
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
        Create isolated track by extracting and combining segments with proper silence masking
        """
        try:
            # Create full-length silent track
            total_samples = int(total_duration * sample_rate)
            isolated_audio = np.zeros(total_samples, dtype=np.float32)
            
            if self.config.verbose:
                print(f"    Creating {total_duration:.1f}s track with {len(segments)} segments")
            
            # Extract and place each segment with fade transitions
            for i, (start_time, end_time) in enumerate(segments):
                # Extract segment from original audio
                segment_data = self.audio_processor.extract_audio_segment(
                    input_file, start_time, end_time
                )
                
                if segment_data is not None and len(segment_data) > 0:
                    # Calculate position in output array
                    start_sample = int(start_time * sample_rate)
                    segment_length = len(segment_data)
                    end_sample = min(start_sample + segment_length, total_samples)
                    
                    # Ensure we don't exceed bounds
                    actual_length = end_sample - start_sample
                    if actual_length > 0:
                        # Apply fade-in/fade-out to prevent clicks
                        processed_segment = self._apply_fade_transitions(
                            segment_data[:actual_length], sample_rate
                        )
                        
                        # Place in the correct position
                        isolated_audio[start_sample:end_sample] = processed_segment
                        
                        if self.config.verbose:
                            print(f"    Placed segment {i+1}: {start_time:.1f}s-{end_time:.1f}s")
            
            # Apply gentle normalization to maintain consistency
            isolated_audio = self._gentle_normalize(isolated_audio)
            
            # Apply final audio processing
            isolated_audio = self._post_process_audio(isolated_audio, sample_rate)
            
            # Save to file
            success = self.audio_processor.save_audio(isolated_audio, output_file, sample_rate)
            
            if success and self.config.verbose:
                print(f"    ✅ Saved isolated track: {output_file.name}")
            
            return success
        
        except Exception as e:
            if self.config.verbose:
                print(f"❌ Segment-based isolation failed: {e}")
            return False
    
    def _apply_fade_transitions(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply fade-in and fade-out to prevent audio clicks between segments
        """
        if len(audio_data) == 0:
            return audio_data
        
        # Fade duration: 10ms or 1/10 of segment length, whichever is smaller
        fade_samples = min(int(0.01 * sample_rate), len(audio_data) // 10)
        
        if fade_samples < 2:
            return audio_data
        
        result = audio_data.copy()
        
        # Fade-in
        fade_in = np.linspace(0, 1, fade_samples)
        result[:fade_samples] *= fade_in
        
        # Fade-out
        fade_out = np.linspace(1, 0, fade_samples)
        result[-fade_samples:] *= fade_out
        
        return result
    
    def _gentle_normalize(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply gentle normalization to maintain natural dynamics
        """
        if len(audio_data) == 0:
            return audio_data
        
        # Find the maximum absolute value
        max_val = np.max(np.abs(audio_data))
        
        if max_val == 0:
            return audio_data
        
        # Apply gentle compression (don't normalize to full scale)
        target_level = 0.8  # Leave some headroom
        normalized = audio_data * (target_level / max_val)
        
        return normalized
    
    def _post_process_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply final post-processing to the isolated track
        """
        if len(audio_data) == 0:
            return audio_data
        
        # Apply a gentle high-pass filter to remove DC offset and low-frequency noise
        # This is a simple first-order high-pass filter
        alpha = 0.99  # High-pass cutoff around 20 Hz at 44.1 kHz
        filtered = np.zeros_like(audio_data)
        
        if len(audio_data) > 1:
            filtered[0] = audio_data[0]
            for i in range(1, len(audio_data)):
                filtered[i] = alpha * (filtered[i-1] + audio_data[i] - audio_data[i-1])
        else:
            filtered = audio_data
        
        return filtered
    
    def _load_audio_fallback(self, input_file: Path) -> np.ndarray:
        """
        Fallback method to load audio when librosa is not available
        """
        try:
            # Use the audio processor to extract the full audio
            audio_info = self.audio_processor.get_audio_info(input_file)
            return self.audio_processor.extract_audio_segment(
                input_file, 0.0, audio_info.duration
            )
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Fallback audio loading failed: {e}")
            return np.array([])
    
    def _analyze_speaker_characteristics(self, 
                                       full_audio: np.ndarray, 
                                       sample_rate: int,
                                       speaker_timeline: Dict[str, List[Tuple[float, float]]]) -> Dict[str, Dict]:
        """
        Analyze spectral characteristics of each speaker for separation
        """
        speaker_profiles = {}
        
        for speaker_id, segments in speaker_timeline.items():
            if self.config.verbose:
                print(f"    Analyzing voice characteristics for {speaker_id}...")
            
            # Extract audio segments for this speaker
            speaker_audio_segments = []
            for start_time, end_time in segments:
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                end_sample = min(end_sample, len(full_audio))
                
                if start_sample < end_sample:
                    segment = full_audio[start_sample:end_sample]
                    speaker_audio_segments.append(segment)
            
            if speaker_audio_segments:
                # Analyze spectral characteristics
                profile = self._extract_speaker_profile(
                    np.concatenate(speaker_audio_segments), sample_rate
                )
                speaker_profiles[speaker_id] = profile
        
        return speaker_profiles
    
    def _extract_speaker_profile(self, speaker_audio: np.ndarray, sample_rate: int) -> Dict:
        """
        Extract spectral and tonal characteristics of a speaker
        """
        if len(speaker_audio) == 0:
            return {}
        
        try:
            if LIBROSA_AVAILABLE:
                # Advanced spectral analysis using librosa
                # Fundamental frequency (pitch) analysis
                pitches, magnitudes = librosa.piptrack(y=speaker_audio, sr=sample_rate, 
                                                     threshold=0.1, fmin=50, fmax=500)
                
                # Get average pitch
                pitch_values = pitches[magnitudes > np.percentile(magnitudes, 85)]
                avg_pitch = np.median(pitch_values) if len(pitch_values) > 0 else 150.0
                
                # Spectral centroid (brightness)
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(
                    y=speaker_audio, sr=sample_rate
                ))
                
                # MFCC features (voice timbre)
                mfccs = librosa.feature.mfcc(y=speaker_audio, sr=sample_rate, n_mfcc=13)
                mfcc_means = np.mean(mfccs, axis=1)
                
                # Spectral rolloff
                spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(
                    y=speaker_audio, sr=sample_rate
                ))
                
                return {
                    'avg_pitch': float(avg_pitch),
                    'spectral_centroid': float(spectral_centroid),
                    'mfcc_profile': mfcc_means.tolist(),
                    'spectral_rolloff': float(spectral_rolloff),
                    'pitch_range': (float(np.min(pitch_values)) if len(pitch_values) > 0 else 100.0,
                                  float(np.max(pitch_values)) if len(pitch_values) > 0 else 300.0)
                }
            else:
                # Basic analysis without librosa
                return self._basic_speaker_profile(speaker_audio, sample_rate)
                
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Speaker profile extraction failed: {e}")
            return self._basic_speaker_profile(speaker_audio, sample_rate)
    
    def _basic_speaker_profile(self, speaker_audio: np.ndarray, sample_rate: int) -> Dict:
        """
        Basic speaker analysis without advanced libraries
        """
        # Simple spectral analysis using FFT
        fft = np.fft.rfft(speaker_audio)
        freqs = np.fft.rfftfreq(len(speaker_audio), 1/sample_rate)
        
        # Find dominant frequencies
        magnitude = np.abs(fft)
        dominant_freq_idx = np.argmax(magnitude)
        dominant_freq = freqs[dominant_freq_idx]
        
        # Calculate spectral centroid (brightness measure)
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        
        return {
            'avg_pitch': float(dominant_freq),
            'spectral_centroid': float(spectral_centroid),
            'energy': float(np.mean(speaker_audio ** 2))
        }
    
    def _create_voice_separated_track(self, 
                                    full_audio: np.ndarray,
                                    sample_rate: int,
                                    speaker_segments: List[Tuple[float, float]], 
                                    speaker_id: str,
                                    speaker_profile: Optional[Dict],
                                    total_duration: float) -> Optional[Path]:
        """
        Create an isolated track using true voice separation techniques
        """
        try:
            # Create output filename
            output_dir = Path(tempfile.gettempdir()) / "duosynco_temp"
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"isolated_{speaker_id}_separated.wav"
            
            # Create full-length isolated track
            total_samples = int(total_duration * sample_rate)
            isolated_audio = np.zeros(total_samples, dtype=np.float32)
            
            if self.config.verbose:
                print(f"    Creating voice-separated track for {speaker_id}")
            
            # Process each segment with voice separation
            for i, (start_time, end_time) in enumerate(speaker_segments):
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                end_sample = min(end_sample, len(full_audio))
                
                if start_sample < end_sample:
                    # Extract original segment
                    original_segment = full_audio[start_sample:end_sample]
                    
                    # Apply voice separation
                    separated_segment = self._separate_voice_in_segment(
                        original_segment, sample_rate, speaker_profile
                    )
                    
                    # Place in output track
                    output_start = start_sample
                    output_end = min(output_start + len(separated_segment), total_samples)
                    
                    if output_start < output_end:
                        segment_length = output_end - output_start
                        isolated_audio[output_start:output_end] = separated_segment[:segment_length]
            
            # Post-process the separated audio
            isolated_audio = self._post_process_separated_audio(isolated_audio, sample_rate)
            
            # Save the isolated track
            success = self.audio_processor.save_audio(isolated_audio, output_file, sample_rate)
            
            if success:
                if self.config.verbose:
                    print(f"    ✅ Voice-separated track saved: {output_file.name}")
                return output_file
            else:
                return None
                
        except Exception as e:
            if self.config.verbose:
                print(f"❌ Voice separation failed for {speaker_id}: {e}")
            return None
    
    def _separate_voice_in_segment(self, 
                                 audio_segment: np.ndarray,
                                 sample_rate: int,
                                 speaker_profile: Optional[Dict]) -> np.ndarray:
        """
        Separate the target speaker's voice from a mixed audio segment
        """
        if len(audio_segment) == 0 or speaker_profile is None:
            return audio_segment
        
        try:
            if LIBROSA_AVAILABLE and 'avg_pitch' in speaker_profile:
                return self._advanced_voice_separation(audio_segment, sample_rate, speaker_profile)
            else:
                return self._basic_voice_separation(audio_segment, sample_rate, speaker_profile)
                
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Voice separation failed, using original: {e}")
            return audio_segment
    
    def _advanced_voice_separation(self, 
                                 audio_segment: np.ndarray,
                                 sample_rate: int,
                                 speaker_profile: Dict) -> np.ndarray:
        """
        Advanced voice separation using spectral analysis
        """
        # Get spectral representation
        stft = librosa.stft(audio_segment, hop_length=512)
        magnitude, phase = np.abs(stft), np.angle(stft)
        
        # Create frequency bins
        freqs = librosa.fft_frequencies(sr=sample_rate)
        
        # Create spectral mask based on speaker characteristics
        target_pitch = speaker_profile.get('avg_pitch', 150.0)
        spectral_centroid = speaker_profile.get('spectral_centroid', 2000.0)
        
        # Enhanced spectral mask
        mask = self._create_spectral_mask(freqs, magnitude, target_pitch, spectral_centroid)
        
        # Apply mask to isolate voice
        masked_stft = stft * mask
        
        # Convert back to time domain
        separated_audio = librosa.istft(masked_stft, hop_length=512)
        
        return separated_audio.astype(np.float32)
    
    def _basic_voice_separation(self, 
                              audio_segment: np.ndarray,
                              sample_rate: int,
                              speaker_profile: Dict) -> np.ndarray:
        """
        Basic voice separation using frequency filtering
        """
        # Get target frequency characteristics
        target_pitch = speaker_profile.get('avg_pitch', 150.0)
        
        # Create bandpass filter around the speaker's pitch range
        low_freq = max(target_pitch * 0.5, 80.0)   # Lower bound
        high_freq = min(target_pitch * 4.0, 4000.0)  # Upper bound
        
        # Apply bandpass filter
        nyquist = sample_rate * 0.5
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Ensure normalized frequencies are valid
        low_norm = max(0.01, min(low_norm, 0.99))
        high_norm = max(low_norm + 0.01, min(high_norm, 0.99))
        
        try:
            b, a = scipy.signal.butter(4, [low_norm, high_norm], btype='band')
            filtered_audio = scipy.signal.filtfilt(b, a, audio_segment)
            return filtered_audio.astype(np.float32)
        except Exception:
            # If filtering fails, return original
            return audio_segment
    
    def _create_spectral_mask(self, 
                            freqs: np.ndarray,
                            magnitude: np.ndarray, 
                            target_pitch: float,
                            spectral_centroid: float) -> np.ndarray:
        """
        Create a spectral mask to isolate the target speaker's voice
        """
        # Initialize mask
        mask = np.ones_like(magnitude)
        
        # Create frequency-based mask
        for i, freq in enumerate(freqs):
            # Boost frequencies around the target pitch and its harmonics
            distance_to_pitch = abs(freq - target_pitch)
            distance_to_harmonics = min(
                abs(freq - 2 * target_pitch),
                abs(freq - 3 * target_pitch),
                abs(freq - target_pitch / 2)
            )
            
            # Distance to spectral centroid
            distance_to_centroid = abs(freq - spectral_centroid)
            
            # Create mask value based on relevance to speaker
            if distance_to_pitch < target_pitch * 0.1:  # Very close to fundamental
                mask[i, :] *= 1.5
            elif distance_to_harmonics < target_pitch * 0.1:  # Close to harmonics
                mask[i, :] *= 1.2
            elif distance_to_centroid < spectral_centroid * 0.3:  # In brightness range
                mask[i, :] *= 1.1
            elif freq < 80 or freq > 8000:  # Outside speech range
                mask[i, :] *= 0.3
            
        # Apply energy-based masking
        energy_threshold = np.percentile(magnitude, 20)
        mask[magnitude < energy_threshold] *= 0.5
        
        # Smooth the mask to avoid artifacts
        from scipy.ndimage import gaussian_filter
        mask = gaussian_filter(mask, sigma=1.0)
        
        return mask
    
    def _post_process_separated_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Post-process the separated audio to improve quality
        """
        if len(audio) == 0:
            return audio
        
        # Apply gentle noise reduction
        audio = self._simple_noise_reduction(audio)
        
        # Apply gentle normalization
        audio = self._gentle_normalize(audio)
        
        # Apply high-pass filter to remove low-frequency artifacts
        audio = self._post_process_audio(audio, sample_rate)
        
        return audio
    
    def _simple_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """
        Simple noise reduction using spectral gating
        """
        if len(audio) == 0:
            return audio
        
        # Find noise floor (quiet sections)
        energy = audio ** 2
        noise_threshold = np.percentile(energy, 10)  # Bottom 10% as noise
        
        # Apply gentle gating
        gate_ratio = 0.3  # Don't completely silence, just reduce
        audio_gated = np.where(energy < noise_threshold, 
                              audio * gate_ratio, 
                              audio)
        
        return audio_gated
    
    def _fallback_spectral_separation(self, 
                                    input_file: Path,
                                    speaker_timeline: Dict[str, List[Tuple[float, float]]],
                                    total_duration: float) -> Dict[str, Path]:
        """
        Fallback to spectral-based voice separation when ML is not available
        """
        try:
            # Load full audio for analysis
            if LIBROSA_AVAILABLE:
                full_audio, sample_rate = librosa.load(str(input_file), sr=None)
            else:
                # Fallback to basic loading
                full_audio = self._load_audio_fallback(input_file)
                audio_info = self.audio_processor.get_audio_info(input_file)
                sample_rate = audio_info.sample_rate
            
            # Analyze speaker characteristics
            speaker_profiles = self._analyze_speaker_characteristics(
                full_audio, sample_rate, speaker_timeline
            )
            
            # Create isolated tracks using spectral separation
            isolated_tracks = {}
            
            for speaker_id, segments in speaker_timeline.items():
                if self.config.verbose:
                    total_speaker_time = sum(end - start for start, end in segments)
                    print(f"  Processing {speaker_id} ({len(segments)} segments, {total_speaker_time:.1f}s total)...")
                
                isolated_path = self._create_voice_separated_track(
                    full_audio, sample_rate, segments, speaker_id, 
                    speaker_profiles.get(speaker_id), total_duration
                )
                
                if isolated_path:
                    isolated_tracks[speaker_id] = isolated_path
                    
                    # Verify the output track length
                    if self.config.verbose:
                        self._verify_track_length(isolated_path, total_duration)
            
            if self.config.verbose:
                print(f"✅ Created {len(isolated_tracks)} spectrally-separated tracks")
            
            return isolated_tracks
            
        except Exception as e:
            if self.config.verbose:
                print(f"❌ Spectral separation failed: {e}")
            return {}
    
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