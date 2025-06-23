"""
FFmpeg-Based Voice Separation Module
Uses advanced FFmpeg audio filters for practical voice isolation
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import tempfile
import subprocess
import os

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from ..utils.config import Config


class FFmpegVoiceSeparator:
    """
    Advanced voice separation using FFmpeg audio filters
    Focuses on practical, reliable separation without complex ML dependencies
    """
    
    def __init__(self, config: Config):
        self.config = config
        self._check_ffmpeg()
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available with required filters"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False
    
    def is_available(self) -> bool:
        """Check if FFmpeg separation is available"""
        return self._check_ffmpeg()
    
    def separate_speakers_from_segments(self, 
                                      input_file: Path,
                                      speaker_segments: Dict[str, List[Tuple[float, float]]],
                                      total_duration: float) -> Dict[str, Path]:
        """
        Separate speakers using FFmpeg-based voice isolation
        
        Args:
            input_file: Original audio file
            speaker_segments: Dict mapping speaker_id to list of (start, end) times
            total_duration: Total duration of audio
            
        Returns:
            Dict mapping speaker_id to separated audio file path
        """
        if not self.is_available():
            if self.config.verbose:
                print("❌ FFmpeg separation not available")
            return {}
        
        try:
            if self.config.verbose:
                print("🎛️  Starting FFmpeg-based voice separation...")
            
            # Create temporary directory for processing
            temp_dir = Path(tempfile.gettempdir()) / "duosynco_ffmpeg"
            temp_dir.mkdir(exist_ok=True)
            
            # Analyze speaker characteristics for filtering
            speaker_profiles = self._analyze_speaker_frequencies(
                input_file, speaker_segments
            )
            
            # Process each speaker
            separated_tracks = {}
            
            for speaker_id, segments in speaker_segments.items():
                if self.config.verbose:
                    total_speaker_time = sum(end - start for start, end in segments)
                    print(f"  🎤 Processing {speaker_id} ({len(segments)} segments, {total_speaker_time:.1f}s)")
                
                separated_path = self._create_ffmpeg_separated_track(
                    input_file, segments, speaker_id, total_duration, 
                    speaker_profiles.get(speaker_id), temp_dir
                )
                
                if separated_path:
                    separated_tracks[speaker_id] = separated_path
            
            if self.config.verbose:
                print(f"✅ FFmpeg separation completed: {len(separated_tracks)} tracks")
            
            return separated_tracks
            
        except Exception as e:
            if self.config.verbose:
                print(f"❌ FFmpeg separation failed: {e}")
            return {}
    
    def _analyze_speaker_frequencies(self, 
                                   input_file: Path,
                                   speaker_segments: Dict[str, List[Tuple[float, float]]]) -> Dict[str, Dict]:
        """
        Analyze frequency characteristics of each speaker for targeted filtering
        """
        speaker_profiles = {}
        
        if not LIBROSA_AVAILABLE:
            # Return default profiles if librosa not available
            return {speaker_id: self._default_speaker_profile(i) 
                   for i, speaker_id in enumerate(speaker_segments.keys())}
        
        try:
            # Load audio for analysis
            audio, sample_rate = librosa.load(str(input_file), sr=None)
            
            for speaker_id, segments in speaker_segments.items():
                if self.config.verbose:
                    print(f"    🔍 Analyzing frequency profile for {speaker_id}...")
                
                # Extract speaker audio segments
                speaker_audio = []
                for start_time, end_time in segments[:5]:  # Analyze first 5 segments
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    end_sample = min(end_sample, len(audio))
                    
                    if start_sample < end_sample:
                        segment = audio[start_sample:end_sample]
                        speaker_audio.append(segment)
                
                if speaker_audio:
                    combined_audio = np.concatenate(speaker_audio)
                    profile = self._extract_frequency_profile(combined_audio, sample_rate)
                    speaker_profiles[speaker_id] = profile
        
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Speaker analysis failed: {e}, using defaults")
            return {speaker_id: self._default_speaker_profile(i) 
                   for i, speaker_id in enumerate(speaker_segments.keys())}
        
        return speaker_profiles
    
    def _extract_frequency_profile(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """
        Extract frequency characteristics for FFmpeg filtering
        """
        try:
            # Fundamental frequency analysis
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate, 
                                                 threshold=0.1, fmin=50, fmax=500)
            
            # Get dominant pitch
            pitch_values = pitches[magnitudes > np.percentile(magnitudes, 80)]
            fundamental_freq = np.median(pitch_values) if len(pitch_values) > 0 else 150.0
            
            # Spectral analysis
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            freqs = librosa.fft_frequencies(sr=sample_rate)
            
            # Find dominant frequency bands
            freq_energy = np.mean(magnitude, axis=1)
            dominant_freqs = freqs[freq_energy > np.percentile(freq_energy, 75)]
            
            # Calculate frequency ranges
            low_freq = max(fundamental_freq * 0.5, 80)
            high_freq = min(fundamental_freq * 8, 4000)
            
            # Voice characteristics
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
            
            return {
                'fundamental_freq': float(fundamental_freq),
                'low_freq': float(low_freq),
                'high_freq': float(high_freq),
                'spectral_centroid': float(spectral_centroid),
                'dominant_freqs': dominant_freqs.tolist()[:10]  # Top 10 frequencies
            }
            
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Frequency analysis failed: {e}")
            return self._default_speaker_profile(0)
    
    def _default_speaker_profile(self, speaker_index: int) -> Dict:
        """
        Generate default speaker profile when analysis fails
        """
        # Different default profiles for different speakers
        profiles = [
            {  # Speaker 0: Lower voice
                'fundamental_freq': 120.0,
                'low_freq': 80.0,
                'high_freq': 3000.0,
                'spectral_centroid': 1500.0,
                'dominant_freqs': [120, 240, 360, 480, 600]
            },
            {  # Speaker 1: Higher voice
                'fundamental_freq': 200.0,
                'low_freq': 100.0,
                'high_freq': 4000.0,
                'spectral_centroid': 2200.0,
                'dominant_freqs': [200, 400, 600, 800, 1000]
            }
        ]
        return profiles[speaker_index % len(profiles)]
    
    def _create_ffmpeg_separated_track(self,
                                     input_file: Path,
                                     speaker_segments: List[Tuple[float, float]],
                                     speaker_id: str,
                                     total_duration: float,
                                     speaker_profile: Optional[Dict],
                                     temp_dir: Path) -> Optional[Path]:
        """
        Create separated track using FFmpeg filters
        """
        try:
            output_file = temp_dir / f"ffmpeg_separated_{speaker_id}.wav"
            
            if not speaker_profile:
                speaker_profile = self._default_speaker_profile(0)
            
            # Build advanced FFmpeg filter chain
            filter_chain = self._build_ffmpeg_filter_chain(
                speaker_segments, total_duration, speaker_profile
            )
            
            if self.config.verbose:
                print(f"    🎛️  Applying FFmpeg filters for {speaker_id}")
            
            # Run FFmpeg with advanced filtering
            cmd = [
                'ffmpeg',
                '-i', str(input_file),
                '-filter_complex', filter_chain,
                '-map', '[output]',
                '-acodec', 'pcm_s16le',
                '-ar', '24000',
                '-ac', '1',
                '-y',
                str(output_file)
            ]
            
            if self.config.verbose:
                result = subprocess.run(cmd, capture_output=True, text=True)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if result.returncode == 0 and output_file.exists():
                if self.config.verbose:
                    print(f"    ✅ FFmpeg track saved: {output_file.name}")
                return output_file
            else:
                if self.config.verbose:
                    print(f"    ❌ FFmpeg failed: {result.stderr}")
                return None
        
        except Exception as e:
            if self.config.verbose:
                print(f"❌ FFmpeg track creation failed for {speaker_id}: {e}")
            return None
    
    def _build_ffmpeg_filter_chain(self,
                                  speaker_segments: List[Tuple[float, float]],
                                  total_duration: float,
                                  speaker_profile: Dict) -> str:
        """
        Build simplified FFmpeg filter chain for voice isolation
        """
        # Extract speaker characteristics
        low_freq = speaker_profile.get('low_freq', 100)
        high_freq = speaker_profile.get('high_freq', 3000)
        fundamental = speaker_profile.get('fundamental_freq', 150)
        
        # Create a simple time-based segmentation using select filter
        # This approach uses simpler time-based filtering
        time_ranges = []
        for start_time, end_time in speaker_segments:
            time_ranges.append(f"between(t,{start_time:.2f},{end_time:.2f})")
        
        # Create volume filter with simplified logic
        if time_ranges:
            # Only enable volume during speaker segments
            enable_condition = ' or '.join(time_ranges[:20])  # Limit to first 20 segments to avoid complexity
            volume_filter = f"volume=enable='{enable_condition}':volume=1.0"
        else:
            volume_filter = "volume=1.0"
        
        # Simplified frequency filtering based on speaker profile
        freq_filters = [
            # Basic band-pass filter for speaker's frequency range
            f"highpass=f={int(low_freq)}",
            f"lowpass=f={int(high_freq)}",
            
            # Single EQ boost at fundamental frequency
            f"equalizer=f={int(fundamental)}:width_type=h:width=2:g=3",
        ]
        
        # Basic noise reduction
        noise_filters = [
            "highpass=f=80",  # Remove low-frequency noise
            "lowpass=f=8000", # Remove high-frequency noise
        ]
        
        # Combine filters in simpler chain
        all_filters = freq_filters + [volume_filter] + noise_filters
        filter_chain = f"[0:a]{','.join(all_filters)}[output]"
        
        return filter_chain
    
    def cleanup_temp_files(self, separated_tracks: Dict[str, Path]) -> None:
        """
        Clean up temporary FFmpeg processing files
        """
        for speaker_id, file_path in separated_tracks.items():
            try:
                if file_path.exists():
                    file_path.unlink()
                    if self.config.verbose:
                        print(f"🗑️  Cleaned up FFmpeg temp file: {file_path.name}")
            except Exception as e:
                if self.config.verbose:
                    print(f"⚠️  Could not clean up {file_path}: {e}")
        
        # Try to remove temp directory if empty
        try:
            temp_dir = Path(tempfile.gettempdir()) / "duosynco_ffmpeg"
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()
        except Exception:
            pass  # Ignore cleanup errors