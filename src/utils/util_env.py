"""
Environment Variable Loader
Uses python-dotenv for .env.local file loading
"""

import os
from pathlib import Path
from typing import Optional, Dict

from dotenv import load_dotenv


def find_project_root() -> Path:
    """Find the project root directory"""
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / ".git").exists():
            return current
        current = current.parent
    return Path.cwd()


def load_env_file(env_file: str = ".env.local") -> None:
    """Load environment variables from file"""
    project_root = find_project_root()
    env_path = project_root / env_file
    
    if env_path.exists():
        load_dotenv(env_path, override=True)


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable value"""
    return os.getenv(key, default)


def get_voice_mapping() -> Dict[str, str]:
    """Get voice mapping from environment variables"""
    voice_mapping = {}
    
    # Support both old and new naming conventions for backwards compatibility
    voice_0 = get_env('VOICE_SPEAKER_0') or get_env('VOICE_SPEAKER_A')
    voice_1 = get_env('VOICE_SPEAKER_1') or get_env('VOICE_SPEAKER_B')
    
    if voice_0:
        voice_mapping['speaker_0'] = voice_0
        voice_mapping['A'] = voice_0  # Backwards compatibility
    if voice_1:
        voice_mapping['speaker_1'] = voice_1
        voice_mapping['B'] = voice_1  # Backwards compatibility
    
    return voice_mapping


# Load environment on import
load_env_file()