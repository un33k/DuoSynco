"""
ML-Based Voice Separation Module
Uses Demucs and other ML models for high-quality voice separation
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import tempfile
import os

try:
    import demucs.separate
    import demucs.pretrained
    import torch
    import torchaudio
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

try:
    import speechbrain
    try:
        from speechbrain.inference import SepformerSeparation
    except ImportError:
        from speechbrain.pretrained import SepformerSeparation
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False

# Check for Spleeter CLI availability
def check_spleeter_cli():
    try:
        import subprocess
        result = subprocess.run(['spleeter', '--help'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

SPLEETER_CLI_AVAILABLE = check_spleeter_cli()

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False

from ..utils.config import Config
from .ffmpeg_separator import FFmpegVoiceSeparator


class MLVoiceSeparator:
    """
    ML-based voice separation using Demucs and other models
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.speechbrain_model = None
        self.ffmpeg_separator = FFmpegVoiceSeparator(config)
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the ML separation model - prioritize SpeechBrain for speaker separation"""
        
        # Try SpeechBrain first (better for speaker separation)
        if SPEECHBRAIN_AVAILABLE:
            try:
                if self.config.verbose:
                    print("🧠 Initializing SpeechBrain SepFormer model for speech separation...")
                
                # Use pre-trained speech separation model
                self.speechbrain_model = SepformerSeparation.from_hparams(
                    source="speechbrain/sepformer-wham",
                    savedir='pretrained_models/sepformer-wham'
                )
                
                if self.config.verbose:
                    print("✅ SpeechBrain speech separation model loaded successfully")
                return
                
            except Exception as e:
                if self.config.verbose:
                    print(f"⚠️  SpeechBrain failed: {e}, trying Demucs...")
        
        # Fallback to Demucs
        if DEMUCS_AVAILABLE:
            try:
                # Use a lightweight model for faster processing
                model_name = "htdemucs_ft" if self.config.quality == "high" else "htdemucs"
                
                if self.config.verbose:
                    print(f"🤖 Fallback to Demucs model: {model_name}")
                
                # Load Demucs model using new API
                self.model = demucs.pretrained.get_model(model_name)
                
                if self.config.verbose:
                    print("✅ Demucs model loaded as fallback")
                    
            except Exception as e:
                if self.config.verbose:
                    print(f"⚠️  Could not load any ML model: {e}")
                self.model = None
        else:
            if self.config.verbose:
                print("⚠️  No ML separation models available. Install speechbrain or demucs")
    
    def is_available(self) -> bool:
        """Check if ML separation is available"""
        return self.ffmpeg_separator.is_available() or \
               (SPEECHBRAIN_AVAILABLE and self.speechbrain_model is not None) or \
               (DEMUCS_AVAILABLE and self.model is not None) or \
               WHISPERX_AVAILABLE
    
    def separate_with_specific_backend(self,
                                     input_file: Path,
                                     speaker_segments: Dict[str, List[Tuple[float, float]]],
                                     total_duration: float,
                                     backend: str) -> Dict[str, Path]:
        """
        Separate speakers using a specific backend
        
        Args:
            input_file: Original audio file
            speaker_segments: Dict mapping speaker_id to list of (start, end) times
            total_duration: Total duration of audio
            backend: Specific backend to use ('ffmpeg', 'speechbrain', 'demucs')
            
        Returns:
            Dict mapping speaker_id to separated audio file path
        """
        if self.config.verbose:
            print(f"🎛️  Using {backend} backend for voice separation...")
        
        try:
            if backend == 'ffmpeg':
                return self._separate_with_ffmpeg(input_file, speaker_segments, total_duration)
            elif backend == 'speechbrain':
                return self._separate_with_speechbrain(input_file, speaker_segments, total_duration)
            elif backend == 'demucs':
                return self._separate_with_demucs(input_file, speaker_segments, total_duration)
            elif backend == 'whisperx':
                return self._separate_with_whisperx(input_file, speaker_segments, total_duration)
            else:
                if self.config.verbose:
                    print(f"❌ Unknown ML backend: {backend}")
                return {}
                
        except Exception as e:
            if self.config.verbose:
                print(f"❌ {backend} separation failed: {e}")
            return {}
    
    def _separate_with_ffmpeg(self,
                             input_file: Path,
                             speaker_segments: Dict[str, List[Tuple[float, float]]],
                             total_duration: float) -> Dict[str, Path]:
        """Separate using FFmpeg backend"""
        if not self.ffmpeg_separator.is_available():
            if self.config.verbose:
                print("❌ FFmpeg backend not available")
            return {}
        
        return self.ffmpeg_separator.separate_speakers_from_segments(
            input_file, speaker_segments, total_duration
        )
    
    def _separate_with_speechbrain(self,
                                  input_file: Path,
                                  speaker_segments: Dict[str, List[Tuple[float, float]]],
                                  total_duration: float) -> Dict[str, Path]:
        """Separate using SpeechBrain backend"""
        if not SPEECHBRAIN_AVAILABLE or self.speechbrain_model is None:
            if self.config.verbose:
                print("❌ SpeechBrain backend not available")
            return {}
        
        return self._ml_separate_with_model(input_file, speaker_segments, total_duration, 'speechbrain')
    
    def _separate_with_demucs(self,
                             input_file: Path,
                             speaker_segments: Dict[str, List[Tuple[float, float]]],
                             total_duration: float) -> Dict[str, Path]:
        """Separate using Demucs backend"""
        if not DEMUCS_AVAILABLE or self.model is None:
            if self.config.verbose:
                print("❌ Demucs backend not available")
            return {}
        
        return self._ml_separate_with_model(input_file, speaker_segments, total_duration, 'demucs')
    
    def _separate_with_whisperx(self,
                               input_file: Path,
                               speaker_segments: Dict[str, List[Tuple[float, float]]],
                               total_duration: float) -> Dict[str, Path]:
        """Separate using WhisperX backend - combines improved diarization with spectral separation"""
        if not WHISPERX_AVAILABLE:
            if self.config.verbose:
                print("❌ WhisperX backend not available")
            return {}
        
        try:
            # For WhisperX backend, we'll use it to get better diarization first
            # then apply spectral separation based on those segments
            from .whisperx_diarizer import WhisperXDiarizer
            
            if self.config.verbose:
                print("🎯 Using WhisperX for enhanced diarization + spectral separation")
            
            # Initialize WhisperX diarizer
            whisperx_diarizer = WhisperXDiarizer(self.config)
            
            if not whisperx_diarizer.is_available():
                if self.config.verbose:
                    print("❌ WhisperX diarizer not available")
                return {}
            
            # Get improved speaker segments from WhisperX
            num_speakers = len(speaker_segments)
            whisperx_segments = whisperx_diarizer.diarize(input_file, num_speakers)
            
            if not whisperx_segments:
                if self.config.verbose:
                    print("⚠️  WhisperX diarization returned no segments, using original")
                whisperx_segments_dict = speaker_segments
            else:
                # Convert WhisperX segments to our timeline format
                whisperx_segments_dict = {}
                for segment in whisperx_segments:
                    if segment.speaker_id not in whisperx_segments_dict:
                        whisperx_segments_dict[segment.speaker_id] = []
                    whisperx_segments_dict[segment.speaker_id].append(
                        (segment.start_time, segment.end_time)
                    )
                
                if self.config.verbose:
                    total_whisperx_time = sum(
                        sum(end - start for start, end in segments) 
                        for segments in whisperx_segments_dict.values()
                    )
                    print(f"🎯 WhisperX found {len(whisperx_segments_dict)} speakers, {total_whisperx_time:.1f}s total")
            
            # Now use spectral separation with the improved segments
            # Import the spectral separation method from the isolator
            from . import isolator
            voice_isolator = isolator.VoiceIsolator(self.config)
            
            # Use the spectral separation method directly
            separated_tracks = voice_isolator._fallback_spectral_separation(
                input_file, whisperx_segments_dict, total_duration
            )
            
            # Cleanup WhisperX models
            whisperx_diarizer.cleanup()
            
            if self.config.verbose:
                print(f"✅ WhisperX separation completed: {len(separated_tracks)} tracks")
            
            return separated_tracks
            
        except Exception as e:
            if self.config.verbose:
                print(f"❌ WhisperX separation failed: {e}")
            return {}
    
    def _ml_separate_with_model(self,
                               input_file: Path,
                               speaker_segments: Dict[str, List[Tuple[float, float]]],
                               total_duration: float,
                               model_type: str) -> Dict[str, Path]:
        """
        Generic ML separation using the specified model type
        """
        # Create temporary directory for processing
        temp_dir = Path(tempfile.gettempdir()) / "duosynco_ml"
        temp_dir.mkdir(exist_ok=True)
        
        # Load full audio
        if LIBROSA_AVAILABLE:
            audio, sample_rate = librosa.load(str(input_file), sr=None)
        else:
            # Fallback to basic loading
            audio, sample_rate = self._load_audio_fallback(input_file)
        
        # Process each speaker
        separated_tracks = {}
        
        for speaker_id, segments in speaker_segments.items():
            if self.config.verbose:
                total_speaker_time = sum(end - start for start, end in segments)
                print(f"  🎤 Processing {speaker_id} ({len(segments)} segments, {total_speaker_time:.1f}s)")
            
            separated_path = self._create_ml_separated_track(
                audio, sample_rate, segments, speaker_id, total_duration, temp_dir, model_type
            )
            
            if separated_path:
                separated_tracks[speaker_id] = separated_path
        
        if self.config.verbose:
            print(f"✅ {model_type.title()} separation completed: {len(separated_tracks)} tracks")
        
        return separated_tracks
    
    def _create_ml_separated_track(self,
                                 full_audio: np.ndarray,
                                 sample_rate: int,
                                 speaker_segments: List[Tuple[float, float]],
                                 speaker_id: str,
                                 total_duration: float,
                                 temp_dir: Path,
                                 model_type: str = 'auto') -> Optional[Path]:
        """
        Create separated track for one speaker using ML
        """
        try:
            # Create output file
            output_file = temp_dir / f"ml_separated_{speaker_id}.wav"
            
            # Create full-length silent track
            total_samples = int(total_duration * sample_rate)
            isolated_audio = np.zeros(total_samples, dtype=np.float32)
            
            if self.config.verbose:
                print(f"    🧠 ML processing {len(speaker_segments)} segments...")
            
            # Process each segment with ML separation
            for i, (start_time, end_time) in enumerate(speaker_segments):
                if self.config.verbose and i % 5 == 0:  # Progress every 5 segments
                    print(f"    Processing segment {i+1}/{len(speaker_segments)}")
                
                # Extract segment
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                end_sample = min(end_sample, len(full_audio))
                
                if start_sample < end_sample:
                    segment_audio = full_audio[start_sample:end_sample]
                    
                    # Apply ML separation to this segment
                    separated_segment = self._separate_segment_with_ml(
                        segment_audio, sample_rate, speaker_id, model_type
                    )
                    
                    # Place in full track
                    if separated_segment is not None and len(separated_segment) > 0:
                        segment_length = min(len(separated_segment), end_sample - start_sample)
                        isolated_audio[start_sample:start_sample + segment_length] = separated_segment[:segment_length]
            
            # Apply post-processing
            isolated_audio = self._post_process_ml_audio(isolated_audio, sample_rate)
            
            # Save the track
            if LIBROSA_AVAILABLE:
                sf.write(str(output_file), isolated_audio, sample_rate)
            else:
                # Fallback save method
                self._save_audio_fallback(isolated_audio, output_file, sample_rate)
            
            if self.config.verbose:
                print(f"    ✅ ML track saved: {output_file.name}")
            
            return output_file
            
        except Exception as e:
            if self.config.verbose:
                print(f"❌ ML track creation failed for {speaker_id}: {e}")
            return None
    
    def _separate_segment_with_ml(self,
                                segment_audio: np.ndarray,
                                sample_rate: int,
                                speaker_id: str,
                                model_type: str = 'auto') -> Optional[np.ndarray]:
        """
        Apply ML separation to a single audio segment
        """
        try:
            # Minimum segment length for ML processing
            min_duration = 1.0  # 1 second minimum for better results
            if len(segment_audio) < int(min_duration * sample_rate):
                # Too short for ML processing, return original
                return segment_audio
            
            # Use the specified model type
            if model_type == 'speechbrain' and self.speechbrain_model is not None:
                return self._separate_with_speechbrain_segment(segment_audio, sample_rate, speaker_id)
            elif model_type == 'demucs' and self.model is not None:
                return self._separate_with_demucs_segment(segment_audio, sample_rate, speaker_id)
            elif model_type == 'auto':
                # Auto selection for backward compatibility
                if self.speechbrain_model is not None:
                    return self._separate_with_speechbrain_segment(segment_audio, sample_rate, speaker_id)
                elif self.model is not None:
                    return self._separate_with_demucs_segment(segment_audio, sample_rate, speaker_id)
            
            return segment_audio
            
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  ML segment separation failed: {e}, using original")
            return segment_audio
    
    def _separate_with_speechbrain_segment(self, 
                                         segment_audio: np.ndarray,
                                         sample_rate: int,
                                         speaker_id: str) -> np.ndarray:
        """
        Separate using SpeechBrain SepFormer model
        """
        try:
            # Convert to torch tensor
            if len(segment_audio.shape) == 1:
                # SpeechBrain expects shape (batch, time)
                audio_tensor = torch.from_numpy(segment_audio).float().unsqueeze(0)
            else:
                audio_tensor = torch.from_numpy(segment_audio).float()
                if len(audio_tensor.shape) == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
            
            # Apply SpeechBrain separation
            with torch.no_grad():
                # SepFormer separates into multiple sources
                separated_sources = self.speechbrain_model.separate_batch(audio_tensor)
                
                # Get the number of sources
                num_sources = separated_sources.shape[1]
                
                # For 2-speaker separation, extract the appropriate source
                # Use speaker_id to determine which source to extract
                speaker_idx = 0 if "SPEAKER_0" in speaker_id else 1
                speaker_idx = min(speaker_idx, num_sources - 1)  # Ensure valid index
                
                # Extract the target speaker's audio
                separated_audio = separated_sources[0, speaker_idx, :].numpy()
                
                return separated_audio.astype(np.float32)
                
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  SpeechBrain separation failed: {e}")
            return segment_audio
    
    def _separate_with_demucs_segment(self, 
                                    segment_audio: np.ndarray,
                                    sample_rate: int,
                                    speaker_id: str) -> np.ndarray:
        """
        Separate using Demucs (fallback method)
        """
        try:
            # Convert to torch tensor
            if len(segment_audio.shape) == 1:
                # Convert mono to stereo for Demucs
                audio_tensor = torch.from_numpy(segment_audio).float()
                audio_tensor = torch.stack([audio_tensor, audio_tensor], dim=0)  # Make stereo
            else:
                audio_tensor = torch.from_numpy(segment_audio.T).float()
            
            # Ensure correct shape: (channels, time)
            if audio_tensor.shape[0] > audio_tensor.shape[1]:
                audio_tensor = audio_tensor.T
            
            # Apply Demucs separation
            with torch.no_grad():
                # Apply the model
                sources = demucs.separate.apply_model(
                    self.model, audio_tensor[None], device='cpu', progress=False
                )[0]
                
                # Extract vocals (typically index 3 in htdemucs: drums, bass, other, vocals)
                vocal_stem = sources[3]  # Vocals
                
                # Convert back to mono numpy array
                if vocal_stem.shape[0] == 2:  # Stereo
                    vocal_mono = vocal_stem.mean(0).numpy()  # Average to mono
                else:
                    vocal_mono = vocal_stem[0].numpy()  # First channel
                
                return vocal_mono.astype(np.float32)
                
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Demucs separation failed: {e}")
            return segment_audio
    
    def _post_process_ml_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Post-process ML-separated audio
        """
        if len(audio) == 0:
            return audio
        
        # Gentle normalization
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio * (0.8 / max_val)  # Leave headroom
        
        # Simple high-pass filter to remove low-frequency artifacts
        if len(audio) > 1:
            # Simple first-order high-pass
            alpha = 0.99
            filtered = np.zeros_like(audio)
            filtered[0] = audio[0]
            for i in range(1, len(audio)):
                filtered[i] = alpha * (filtered[i-1] + audio[i] - audio[i-1])
            audio = filtered
        
        return audio
    
    def _load_audio_fallback(self, input_file: Path) -> Tuple[np.ndarray, int]:
        """
        Fallback audio loading when librosa not available
        """
        try:
            # Try using torchaudio (comes with demucs)
            waveform, sample_rate = torchaudio.load(str(input_file))
            # Convert to mono numpy array
            if waveform.shape[0] == 2:  # Stereo
                audio_mono = waveform.mean(0).numpy()
            else:
                audio_mono = waveform[0].numpy()
            return audio_mono, sample_rate
        except Exception:
            # Last resort: return empty audio
            return np.array([]), 44100
    
    def _save_audio_fallback(self, audio: np.ndarray, output_file: Path, sample_rate: int) -> None:
        """
        Fallback audio saving when librosa not available
        """
        try:
            # Convert to 16-bit and save as WAV
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Simple WAV file creation (basic implementation)
            import wave
            with wave.open(str(output_file), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
                
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Fallback save failed: {e}")
    
    def cleanup_temp_files(self, separated_tracks: Dict[str, Path]) -> None:
        """
        Clean up temporary ML processing files
        """
        for speaker_id, file_path in separated_tracks.items():
            try:
                if file_path.exists():
                    file_path.unlink()
                    if self.config.verbose:
                        print(f"🗑️  Cleaned up ML temp file: {file_path.name}")
            except Exception as e:
                if self.config.verbose:
                    print(f"⚠️  Could not clean up {file_path}: {e}")
        
        # Try to remove temp directory if empty
        try:
            temp_dir = Path(tempfile.gettempdir()) / "duosynco_ml"
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()
        except Exception:
            pass  # Ignore cleanup errors


class MLModelManager:
    """
    Manages different ML models and their availability
    """
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available ML models"""
        models = []
        
        if DEMUCS_AVAILABLE:
            models.extend(['demucs', 'htdemucs', 'htdemucs_ft'])
        
        return models
    
    @staticmethod
    def recommend_model(quality: str = "medium") -> Optional[str]:
        """Recommend the best available model for given quality"""
        available = MLModelManager.get_available_models()
        
        if not available:
            return None
        
        if quality == "high" and "htdemucs_ft" in available:
            return "htdemucs_ft"
        elif "htdemucs" in available:
            return "htdemucs"
        elif "demucs" in available:
            return "demucs"
        
        return available[0] if available else None