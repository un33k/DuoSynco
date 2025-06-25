"""
Configuration management for Ali
"""

import os
from pathlib import Path
from typing import Dict, Optional


class AliConfig:
    """Configuration manager for Ali commands"""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize configuration
        
        Args:
            project_root: Path to project root, auto-detected if None
        """
        if project_root is None:
            # Auto-detect project root (find directory containing .env.local)
            current = Path.cwd()
            while current != current.parent:
                if (current / '.env.local').exists():
                    project_root = current
                    break
                current = current.parent
            
            if project_root is None:
                project_root = Path.cwd()
        
        self.project_root = project_root
        self.env_file = project_root / '.env.local'
        self._load_env()
    
    def _load_env(self):
        """Load environment variables from .env.local"""
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
    
    @property
    def defaults(self) -> Dict[str, str]:
        """Default configuration values"""
        return {
            'provider': 'elevenlabs',  # Default provider for STT  
            'language': 'en',          # Default to English
            'output_dir': str(self.project_root / 'output'),
            'verbose': True,
            'voice_mapping': 'auto',
            'mode': 'tts',
            'tts_quality': 'high',
            'timing_mode': 'adaptive',
            'gap_duration': '0.4',
            'model_id': 'eleven_multilingual_v2'
        }
    
    @property
    def voice_mapping(self) -> Dict[str, str]:
        """Get voice mapping from environment"""
        return {
            'speaker_0': os.getenv('VOICE_SPEAKER_0', 'N2lVS1w4EtoT3dr4eOWO'),
            'speaker_1': os.getenv('VOICE_SPEAKER_1', 'Xb7hH8MSUJpSbSDYk0k2'),
            'A': os.getenv('VOICE_SPEAKER_0', 'N2lVS1w4EtoT3dr4eOWO'),
            'B': os.getenv('VOICE_SPEAKER_1', 'Xb7hH8MSUJpSbSDYk0k2')
        }
    
    @property
    def api_key(self) -> Optional[str]:
        """Get ElevenLabs API key"""
        return os.getenv('ELEVENLABS_API_KEY')
    
    def get_sample_data_dir(self) -> Path:
        """Get sample data directory"""
        return self.project_root / 'sample_data'
    
    def get_output_dir(self) -> Path:
        """Get output directory"""
        return self.project_root / 'output'