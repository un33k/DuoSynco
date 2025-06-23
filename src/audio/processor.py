"""
Audio Processing Utilities
Handles audio manipulation, format conversion, and basic processing
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

from ..utils.config import Config


@dataclass
class AudioInfo:
    """Audio file information"""
    sample_rate: int
    duration: float
    channels: int
    format: str


class AudioProcessor:
    """
    Handles basic audio processing operations
    """
    
    def __init__(self, config: Config):
        self.config = config
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check if required audio processing libraries are available"""
        if not LIBROSA_AVAILABLE and not PYDUB_AVAILABLE:
            raise ImportError(
                "Either librosa or pydub is required for audio processing. "
                "Install with: pip install librosa pydub"
            )
    
    def get_audio_info(self, file_path: Path) -> AudioInfo:
        """
        Get basic information about an audio file
        
        Args:
            file_path: Path to audio/video file
            
        Returns:
            AudioInfo object with file details
        """
        try:
            if LIBROSA_AVAILABLE:
                # Use librosa for detailed audio analysis
                y, sr = librosa.load(str(file_path), sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                
                return AudioInfo(
                    sample_rate=sr,
                    duration=duration,
                    channels=1 if len(y.shape) == 1 else y.shape[0],
                    format=file_path.suffix.lower()
                )
            
            elif PYDUB_AVAILABLE:
                # Use pydub as fallback
                audio = AudioSegment.from_file(str(file_path))
                
                return AudioInfo(
                    sample_rate=audio.frame_rate,
                    duration=len(audio) / 1000.0,  # Convert ms to seconds
                    channels=audio.channels,
                    format=file_path.suffix.lower()
                )
            
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Could not get audio info: {e}")
            
            # Return default info
            return AudioInfo(
                sample_rate=44100,
                duration=0.0,
                channels=2,
                format=file_path.suffix.lower()
            )
    
    def extract_audio_segment(self, 
                            file_path: Path, 
                            start_time: float, 
                            end_time: float) -> Optional[np.ndarray]:
        """
        Extract a specific time segment from an audio file
        
        Args:
            file_path: Path to source audio/video file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Audio data as numpy array, or None if extraction failed
        """
        try:
            if LIBROSA_AVAILABLE:
                # Load only the specified segment
                y, sr = librosa.load(
                    str(file_path), 
                    offset=start_time,
                    duration=end_time - start_time,
                    sr=None
                )
                return y
            
            elif PYDUB_AVAILABLE:
                # Use pydub for segment extraction
                audio = AudioSegment.from_file(str(file_path))
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)
                
                segment = audio[start_ms:end_ms]
                
                # Convert to numpy array
                samples = segment.get_array_of_samples()
                audio_data = np.array(samples, dtype=np.float32)
                
                # Normalize to [-1, 1] range
                if segment.sample_width == 1:
                    audio_data = audio_data / 128.0
                elif segment.sample_width == 2:
                    audio_data = audio_data / 32768.0
                elif segment.sample_width == 4:
                    audio_data = audio_data / 2147483648.0
                
                return audio_data
            
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Could not extract audio segment: {e}")
            return None
    
    def create_silence(self, duration: float, sample_rate: int = 44100) -> np.ndarray:
        """
        Create a silent audio segment
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate for the silence
            
        Returns:
            Silent audio data as numpy array
        """
        num_samples = int(duration * sample_rate)
        return np.zeros(num_samples, dtype=np.float32)
    
    def normalize_audio(self, audio_data: np.ndarray, target_level: float = -20.0) -> np.ndarray:
        """
        Normalize audio to a target level in dB
        
        Args:
            audio_data: Input audio data
            target_level: Target level in dB
            
        Returns:
            Normalized audio data
        """
        try:
            # Calculate current RMS level
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            if rms == 0:
                return audio_data
            
            # Convert to dB
            current_db = 20 * np.log10(rms)
            
            # Calculate gain needed
            gain_db = target_level - current_db
            gain_linear = 10 ** (gain_db / 20)
            
            # Apply gain
            normalized = audio_data * gain_linear
            
            # Prevent clipping
            max_val = np.max(np.abs(normalized))
            if max_val > 1.0:
                normalized = normalized / max_val
            
            return normalized
            
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Audio normalization failed: {e}")
            return audio_data
    
    def save_audio(self, 
                   audio_data: np.ndarray, 
                   output_path: Path, 
                   sample_rate: int = 44100) -> bool:
        """
        Save audio data to file
        
        Args:
            audio_data: Audio data to save
            output_path: Output file path
            sample_rate: Sample rate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if LIBROSA_AVAILABLE:
                # Use soundfile (comes with librosa) for saving
                sf.write(str(output_path), audio_data, sample_rate)
                return True
            
            elif PYDUB_AVAILABLE:
                # Convert numpy array to pydub AudioSegment
                # Scale to 16-bit range
                audio_scaled = (audio_data * 32767).astype(np.int16)
                
                # Create AudioSegment
                audio_segment = AudioSegment(
                    audio_scaled.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=2,
                    channels=1
                )
                
                # Export to file
                audio_segment.export(str(output_path), format="wav")
                return True
            
        except Exception as e:
            if self.config.verbose:
                print(f"❌ Could not save audio: {e}")
            return False
    
    def resample_audio(self, 
                      audio_data: np.ndarray, 
                      original_sr: int, 
                      target_sr: int) -> np.ndarray:
        """
        Resample audio to a different sample rate
        
        Args:
            audio_data: Input audio data
            original_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio data
        """
        if original_sr == target_sr:
            return audio_data
        
        try:
            if LIBROSA_AVAILABLE:
                return librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
            else:
                # Simple linear interpolation fallback
                ratio = target_sr / original_sr
                new_length = int(len(audio_data) * ratio)
                
                # Create new time indices
                old_indices = np.linspace(0, len(audio_data) - 1, len(audio_data))
                new_indices = np.linspace(0, len(audio_data) - 1, new_length)
                
                # Interpolate
                resampled = np.interp(new_indices, old_indices, audio_data)
                return resampled
                
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Audio resampling failed: {e}")
            return audio_data