"""
Environment Variable Loader
Uses python-dotenv for .env.local file loading
"""

import os
from pathlib import Path
from typing import Optional, Dict

from dotenv import load_dotenv, dotenv_values

# Environment file constants for cross-platform compatibility
ENV_LOCAL_FILE = Path(".env.local")
ENV_FILE = Path(".env")
DEFAULT_ENV_FILES = [ENV_LOCAL_FILE, ENV_FILE]


def find_project_root() -> Path:
    """Find the project root directory"""
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / ".git").exists():
            return current
        current = current.parent
    return Path.cwd()


def load_env_file(env_file: Optional[str] = None) -> None:
    """Load environment variables from file"""
    project_root = find_project_root()
    if env_file is None:
        env_path = project_root / ENV_LOCAL_FILE
    else:
        env_path = project_root / Path(env_file)

    if env_path.exists():
        load_dotenv(env_path, override=True)


def get_env(
    key: str, file_path: Optional[str] = None, default: Optional[str] = None
) -> Optional[str]:
    """
    Retrieve an environment variable in this order:
    1. From a specific file if file_path is provided and contains the key
    2. From .env.local if exists
    3. From .env if exists
    4. From system environment
    5. Return default if not found

    Args:
        key: Environment variable key to retrieve
        file_path: Optional specific file path to check first
        default: Default value if key not found anywhere

    Returns:
        Environment variable value or default
    """
    project_root = find_project_root()
    checked_files = []

    # 1. Custom file
    if file_path:
        custom_path = Path(file_path)
        if not custom_path.is_absolute():
            custom_path = project_root / custom_path

        if custom_path.exists():
            val = dotenv_values(custom_path).get(key)
            if val is not None:
                return val
            checked_files.append(str(custom_path))

    # 2. .env.local
    env_local_path = project_root / ENV_LOCAL_FILE
    if env_local_path.exists():
        val = dotenv_values(env_local_path).get(key)
        if val is not None:
            return val
        checked_files.append(str(env_local_path))

    # 3. .env
    env_path = project_root / ENV_FILE
    if env_path.exists():
        val = dotenv_values(env_path).get(key)
        if val is not None:
            return val
        checked_files.append(str(env_path))

    # 4. System environment
    val = os.environ.get(key)
    if val is not None:
        return val

    # 5. Default fallback
    return default


class EnvConfig:
    """
    Environment configuration object for backwards compatibility
    Delegates voice-related calls to the audio.voice module
    """

    @staticmethod
    def get_voice_mapping() -> Dict[str, str]:
        """Get voice mapping - delegates to audio.voice module"""
        from ..audio.voice import get_voice_mapping

        return get_voice_mapping()

    @staticmethod
    def print_config() -> None:
        """Print current environment configuration"""
        project_root = find_project_root()
        print(f"Project Root: {project_root}")

        # Check for environment files
        env_files = DEFAULT_ENV_FILES
        for env_file in env_files:
            env_path = project_root / env_file
            if env_path.exists():
                print(f"✅ Found: {env_file}")
            else:
                print(f"❌ Missing: {env_file}")

        # Show voice mapping
        from ..audio.voice import get_voice_mapping

        voice_mapping = get_voice_mapping()
        if voice_mapping:
            print("Voice Mapping:")
            for speaker, voice_id in voice_mapping.items():
                print(f"  {speaker}: {voice_id}")
        else:
            print("No voice mapping configured")


# Create a default instance for backwards compatibility
env = EnvConfig()

# Load environment on import
load_env_file()
