"""
Configuration Management Module
Handles application configuration and settings
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path
import os


@dataclass
class Config:
    """
    Configuration class for DuoSynco application
    """
    
    # Processing quality settings
    quality: str = 'medium'  # 'low', 'medium', 'high'
    
    # Output format settings
    output_format: str = 'mp4'  # 'mp4', 'avi', 'mov'
    
    # Voice separation backend
    backend: str = 'speechbrain'  # 'ffmpeg', 'speechbrain', 'demucs', 'spectral', 'whisperx'
    
    # Logging and verbosity
    verbose: bool = False
    
    # Audio processing settings
    audio_sample_rate: int = 44100
    audio_channels: int = 2
    audio_bitrate: str = '128k'
    
    # Video processing settings
    video_fps: Optional[float] = None  # Use original FPS if None
    video_resolution: Optional[tuple] = None  # Use original resolution if None
    
    # Speaker diarization settings
    min_speakers: int = 1
    max_speakers: int = 10
    
    # TTS generation settings
    tts_provider: str = 'elevenlabs'  # 'elevenlabs'
    tts_voice_stability: float = 0.5  # 0.0-1.0
    tts_voice_similarity: float = 0.7  # 0.0-1.0  
    tts_max_workers: int = 3  # Concurrent TTS requests
    
    # File handling settings
    temp_dir: Optional[Path] = None
    cleanup_temp_files: bool = True
    
    # Performance settings
    num_threads: int = 1  # Number of processing threads
    memory_limit_mb: int = 2048  # Memory limit in MB
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        self._validate_settings()
        self._setup_temp_directory()
    
    def _validate_settings(self) -> None:
        """Validate configuration settings"""
        # Validate quality setting
        valid_qualities = ['low', 'medium', 'high']
        if self.quality not in valid_qualities:
            raise ValueError(f"Invalid quality setting: {self.quality}. "
                           f"Must be one of: {valid_qualities}")
        
        # Validate output format
        valid_formats = ['mp4', 'avi', 'mov']
        if self.output_format not in valid_formats:
            raise ValueError(f"Invalid output format: {self.output_format}. "
                           f"Must be one of: {valid_formats}")
        
        # Validate backend setting
        valid_backends = ['ffmpeg', 'speechbrain', 'demucs', 'spectral', 'whisperx']
        if self.backend not in valid_backends:
            raise ValueError(f"Invalid backend setting: {self.backend}. "
                           f"Must be one of: {valid_backends}")
        
        # Validate speaker range
        if self.min_speakers < 1:
            raise ValueError("min_speakers must be at least 1")
        if self.max_speakers < self.min_speakers:
            raise ValueError("max_speakers must be >= min_speakers")
        
        # Validate numeric settings
        if self.audio_sample_rate <= 0:
            raise ValueError("audio_sample_rate must be positive")
        if self.audio_channels not in [1, 2]:
            raise ValueError("audio_channels must be 1 (mono) or 2 (stereo)")
        if self.num_threads < 1:
            raise ValueError("num_threads must be at least 1")
        if self.memory_limit_mb < 256:
            raise ValueError("memory_limit_mb must be at least 256")
            
        # Validate TTS settings
        if not 0.0 <= self.tts_voice_stability <= 1.0:
            raise ValueError("tts_voice_stability must be between 0.0 and 1.0")
        if not 0.0 <= self.tts_voice_similarity <= 1.0:
            raise ValueError("tts_voice_similarity must be between 0.0 and 1.0")
        if self.tts_max_workers < 1:
            raise ValueError("tts_max_workers must be at least 1")
    
    def _setup_temp_directory(self) -> None:
        """Setup temporary directory for processing"""
        if self.temp_dir is None:
            import tempfile
            self.temp_dir = Path(tempfile.gettempdir()) / "duosynco"
        
        # Ensure temp directory exists
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """
        Create configuration from environment variables
        
        Returns:
            Config instance with settings from environment
        """
        from .util_env import get_env
        
        return cls(
            quality=get_env('DUOSYNCO_QUALITY', default='medium'),
            output_format=get_env('DUOSYNCO_FORMAT', default='mp4'),
            backend=get_env('DUOSYNCO_BACKEND', default='speechbrain'),
            verbose=get_env('DUOSYNCO_VERBOSE', default='false').lower() == 'true',
            audio_sample_rate=int(get_env('DUOSYNCO_SAMPLE_RATE', default='44100')),
            audio_channels=int(get_env('DUOSYNCO_CHANNELS', default='2')),
            num_threads=int(get_env('DUOSYNCO_THREADS', default='1')),
            memory_limit_mb=int(get_env('DUOSYNCO_MEMORY_MB', default='2048')),
            cleanup_temp_files=get_env('DUOSYNCO_CLEANUP', default='true').lower() == 'true'
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create configuration from dictionary
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            Config instance
        """
        # Filter valid keys
        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary
        
        Returns:
            Dictionary representation of configuration
        """
        result = {}
        for field_name, field_def in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            
            # Convert Path objects to strings
            if isinstance(value, Path):
                result[field_name] = str(value)
            else:
                result[field_name] = value
        
        return result
    
    def get_quality_settings(self) -> Dict[str, Any]:
        """
        Get processing settings based on quality level
        
        Returns:
            Dictionary with quality-specific settings
        """
        quality_profiles = {
            'low': {
                'video_crf': 28,
                'video_preset': 'fast',
                'audio_bitrate': '96k',
                'processing_priority': 'speed'
            },
            'medium': {
                'video_crf': 23,
                'video_preset': 'medium',
                'audio_bitrate': '128k',
                'processing_priority': 'balanced'
            },
            'high': {
                'video_crf': 18,
                'video_preset': 'slow',
                'audio_bitrate': '192k',
                'processing_priority': 'quality'
            }
        }
        
        return quality_profiles.get(self.quality, quality_profiles['medium'])
    
    def get_temp_file_path(self, filename: str) -> Path:
        """
        Get path for temporary file
        
        Args:
            filename: Name of temporary file
            
        Returns:
            Full path to temporary file
        """
        return self.temp_dir / filename
    
    def cleanup_temp_directory(self) -> None:
        """Clean up temporary files and directory"""
        if not self.cleanup_temp_files:
            return
        
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                if self.verbose:
                    print(f"🗑️  Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Could not clean up temp directory: {e}")
    
    def print_settings(self) -> None:
        """Print current configuration settings"""
        print("⚙️  DuoSynco Configuration:")
        print(f"  Quality: {self.quality}")
        print(f"  Output Format: {self.output_format}")
        print(f"  Backend: {self.backend}")
        print(f"  Audio: {self.audio_sample_rate}Hz, {self.audio_channels}ch, {self.audio_bitrate}")
        print(f"  Processing: {self.num_threads} thread(s), {self.memory_limit_mb}MB limit")
        print(f"  Temp Directory: {self.temp_dir}")
        print(f"  Verbose: {self.verbose}")
    
    def save_to_file(self, file_path: Path) -> None:
        """
        Save configuration to file
        
        Args:
            file_path: Path to save configuration file
        """
        import json
        
        config_dict = self.to_dict()
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        if self.verbose:
            print(f"💾 Configuration saved to: {file_path}")
    
    @staticmethod
    def get_available_backends() -> Dict[str, bool]:
        """
        Check availability of voice separation backends
        
        Returns:
            Dictionary mapping backend name to availability status
        """
        backends = {}
        
        # Check FFmpeg
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True)
            backends['ffmpeg'] = result.returncode == 0
        except Exception:
            backends['ffmpeg'] = False
        
        # Check SpeechBrain
        try:
            import speechbrain
            try:
                from speechbrain.inference import SepformerSeparation
            except ImportError:
                from speechbrain.pretrained import SepformerSeparation
            backends['speechbrain'] = True
        except ImportError:
            backends['speechbrain'] = False
        
        # Check Demucs
        try:
            import demucs.separate
            import demucs.pretrained
            backends['demucs'] = True
        except ImportError:
            backends['demucs'] = False
        
        # Check WhisperX
        try:
            import whisperx
            backends['whisperx'] = True
        except ImportError:
            backends['whisperx'] = False
        
        # Spectral is always available (uses basic libraries)
        backends['spectral'] = True
        
        return backends
    
    @staticmethod
    def get_valid_backends() -> List[str]:
        """
        Get list of valid backend names
        
        Returns:
            List of valid backend names
        """
        return ['ffmpeg', 'speechbrain', 'demucs', 'spectral', 'whisperx']
    
    def validate_backend_availability(self) -> bool:
        """
        Validate that the selected backend is available
        
        Returns:
            True if backend is available, False otherwise
        """
        available_backends = self.get_available_backends()
        return available_backends.get(self.backend, False)
    
    @classmethod
    def load_from_file(cls, file_path: Path) -> 'Config':
        """
        Load configuration from file
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Config instance loaded from file
        """
        import json
        
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)