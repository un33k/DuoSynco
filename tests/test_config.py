"""
Unit tests for utils.config module
Tests configuration validation, environment loading, and settings management
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.config import Config


class TestConfig:
    """Test cases for Config class"""

    def test_default_config_creation(self):
        """Test creating config with default values"""
        config = Config()

        assert config.quality == "medium"
        assert config.output_format == "mp4"
        assert config.backend == "speechbrain"
        assert config.verbose is False
        assert config.audio_sample_rate == 44100
        assert config.audio_channels == 2
        assert config.num_threads == 1
        assert config.memory_limit_mb == 2048

    def test_custom_config_creation(self):
        """Test creating config with custom values"""
        config = Config(quality="high", output_format="avi", verbose=True, num_threads=4)

        assert config.quality == "high"
        assert config.output_format == "avi"
        assert config.verbose is True
        assert config.num_threads == 4

    def test_invalid_quality_validation(self):
        """Test validation rejects invalid quality setting"""
        with pytest.raises(ValueError, match="Invalid quality setting"):
            Config(quality="invalid")

    def test_invalid_output_format_validation(self):
        """Test validation rejects invalid output format"""
        with pytest.raises(ValueError, match="Invalid output format"):
            Config(output_format="invalid")

    def test_invalid_backend_validation(self):
        """Test validation rejects invalid backend"""
        with pytest.raises(ValueError, match="Invalid backend setting"):
            Config(backend="invalid")

    def test_invalid_speaker_range_validation(self):
        """Test validation rejects invalid speaker ranges"""
        with pytest.raises(ValueError, match="min_speakers must be at least 1"):
            Config(min_speakers=0)

        with pytest.raises(ValueError, match="max_speakers must be >= min_speakers"):
            Config(min_speakers=5, max_speakers=3)

    def test_invalid_numeric_settings(self):
        """Test validation rejects invalid numeric settings"""
        with pytest.raises(ValueError, match="audio_sample_rate must be positive"):
            Config(audio_sample_rate=-1)

        with pytest.raises(ValueError, match="audio_channels must be 1"):
            Config(audio_channels=3)

        with pytest.raises(ValueError, match="num_threads must be at least 1"):
            Config(num_threads=0)

        with pytest.raises(ValueError, match="memory_limit_mb must be at least 256"):
            Config(memory_limit_mb=100)

    def test_invalid_tts_settings(self):
        """Test validation rejects invalid TTS settings"""
        with pytest.raises(ValueError, match="tts_voice_stability must be between 0.0 and 1.0"):
            Config(tts_voice_stability=1.5)

        with pytest.raises(ValueError, match="tts_voice_similarity must be between 0.0 and 1.0"):
            Config(tts_voice_similarity=-0.5)

        with pytest.raises(ValueError, match="tts_max_workers must be at least 1"):
            Config(tts_max_workers=0)

    def test_temp_directory_setup(self):
        """Test temporary directory is created"""
        config = Config()
        assert config.temp_dir is not None
        assert config.temp_dir.exists()

    @patch("src.utils.config.get_env")
    def test_from_env(self, mock_get_env):
        """Test creating config from environment variables"""
        # Mock environment variable responses
        mock_get_env.side_effect = lambda key, default=None: {
            "DUOSYNCO_QUALITY": "high",
            "DUOSYNCO_FORMAT": "avi",
            "DUOSYNCO_BACKEND": "ffmpeg",
            "DUOSYNCO_VERBOSE": "true",
            "DUOSYNCO_SAMPLE_RATE": "48000",
            "DUOSYNCO_CHANNELS": "1",
            "DUOSYNCO_THREADS": "8",
            "DUOSYNCO_MEMORY_MB": "4096",
            "DUOSYNCO_CLEANUP": "false",
        }.get(key, default)

        config = Config.from_env()

        assert config.quality == "high"
        assert config.output_format == "avi"
        assert config.backend == "ffmpeg"
        assert config.verbose is True
        assert config.audio_sample_rate == 48000
        assert config.audio_channels == 1
        assert config.num_threads == 8
        assert config.memory_limit_mb == 4096
        assert config.cleanup_temp_files is False

    def test_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            "quality": "high",
            "output_format": "avi",
            "verbose": True,
            "num_threads": 4,
            "invalid_key": "should_be_ignored",  # Should be filtered out
        }

        config = Config.from_dict(config_dict)

        assert config.quality == "high"
        assert config.output_format == "avi"
        assert config.verbose is True
        assert config.num_threads == 4
        # Invalid key should be ignored, not cause an error

    def test_to_dict(self):
        """Test converting config to dictionary"""
        config = Config(quality="high", verbose=True)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["quality"] == "high"
        assert config_dict["verbose"] is True
        assert "temp_dir" in config_dict
        # Path should be converted to string
        assert isinstance(config_dict["temp_dir"], str)

    def test_get_quality_settings(self):
        """Test quality-specific settings retrieval"""
        config = Config(quality="high")
        quality_settings = config.get_quality_settings()

        assert isinstance(quality_settings, dict)
        assert "video_crf" in quality_settings
        assert "video_preset" in quality_settings
        assert "audio_bitrate" in quality_settings
        assert "processing_priority" in quality_settings

        # High quality should have better settings
        assert quality_settings["video_crf"] == 18
        assert quality_settings["video_preset"] == "slow"
        assert quality_settings["audio_bitrate"] == "192k"

    def test_get_temp_file_path(self):
        """Test temporary file path generation"""
        config = Config()
        temp_path = config.get_temp_file_path("test_file.txt")

        assert isinstance(temp_path, Path)
        assert temp_path.name == "test_file.txt"
        assert temp_path.parent == config.temp_dir

    def test_save_and_load_from_file(self):
        """Test saving and loading config from file"""
        original_config = Config(quality="high", verbose=True, num_threads=4)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_file = Path(f.name)

        try:
            # Save config
            original_config.save_to_file(config_file)
            assert config_file.exists()

            # Load config
            loaded_config = Config.load_from_file(config_file)

            assert loaded_config.quality == "high"
            assert loaded_config.verbose is True
            assert loaded_config.num_threads == 4

        finally:
            # Clean up
            if config_file.exists():
                config_file.unlink()

    @patch("subprocess.run")
    def test_get_available_backends(self, mock_subprocess):
        """Test backend availability checking"""
        # Mock subprocess for ffmpeg check
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        with patch("importlib.import_module") as mock_import:
            # Mock successful imports for testing
            mock_import.return_value = MagicMock()

            backends = Config.get_available_backends()

            assert isinstance(backends, dict)
            assert "ffmpeg" in backends
            assert "speechbrain" in backends
            assert "spectral" in backends
            # spectral should always be True (uses basic libraries)
            assert backends["spectral"] is True

    def test_get_valid_backends(self):
        """Test getting list of valid backend names"""
        valid_backends = Config.get_valid_backends()

        assert isinstance(valid_backends, list)
        assert "ffmpeg" in valid_backends
        assert "speechbrain" in valid_backends
        assert "demucs" in valid_backends
        assert "spectral" in valid_backends
        assert "whisperx" in valid_backends

    def test_validate_backend_availability(self):
        """Test backend availability validation"""
        with patch.object(Config, "get_available_backends") as mock_get_backends:
            mock_get_backends.return_value = {"speechbrain": True, "ffmpeg": False, "demucs": False}

            config = Config(backend="speechbrain")
            assert config.validate_backend_availability() is True

            config = Config(backend="ffmpeg")
            assert config.validate_backend_availability() is False
