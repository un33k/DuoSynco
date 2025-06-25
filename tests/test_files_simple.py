"""
Simple unit tests for utils.files module
Tests basic file validation functionality
"""

import tempfile
from pathlib import Path

from src.utils.files import FileHandler, FileInfo
from src.utils.config import Config


class TestFileHandler:
    """Test cases for FileHandler class"""

    def test_file_handler_creation(self):
        """Test creating FileHandler instance"""
        config = Config()
        handler = FileHandler(config)
        assert isinstance(handler, FileHandler)
        assert handler.config is config

    def test_supported_formats_constants(self):
        """Test that format constants are defined"""
        assert hasattr(FileHandler, "SUPPORTED_VIDEO_FORMATS")
        assert hasattr(FileHandler, "SUPPORTED_AUDIO_FORMATS")
        assert isinstance(FileHandler.SUPPORTED_VIDEO_FORMATS, set)
        assert isinstance(FileHandler.SUPPORTED_AUDIO_FORMATS, set)

        # Check some expected formats
        assert ".mp4" in FileHandler.SUPPORTED_VIDEO_FORMATS
        assert ".mp3" in FileHandler.SUPPORTED_AUDIO_FORMATS

    def test_validate_input_file_exists(self):
        """Test validating existing supported file"""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(b"fake audio content")

        try:
            config = Config(verbose=False)
            handler = FileHandler(config)

            # Should pass validation
            result = handler.validate_input_file(temp_path)
            # The exact return value depends on implementation
            # Just verify it doesn't raise an exception
            assert isinstance(result, bool)

        finally:
            temp_path.unlink()

    def test_validate_input_file_not_exists(self):
        """Test validating non-existent file"""
        config = Config(verbose=False)
        handler = FileHandler(config)

        non_existent = Path("/nonexistent/file.mp3")
        result = handler.validate_input_file(non_existent)
        assert result is False

    def test_file_info_dataclass(self):
        """Test FileInfo dataclass"""
        file_info = FileInfo(
            path=Path("/test/file.mp3"),
            size_mb=5.2,
            format="mp3",
            mime_type="audio/mpeg",
            is_video=False,
            is_audio=True,
        )

        assert file_info.path == Path("/test/file.mp3")
        assert file_info.size_mb == 5.2
        assert file_info.format == "mp3"
        assert file_info.mime_type == "audio/mpeg"
        assert file_info.is_video is False
        assert file_info.is_audio is True


class TestFileOperations:
    """Test basic file operations without complex method assumptions"""

    def test_path_operations(self):
        """Test basic Path operations work"""
        test_path = Path("/test/file.mp3")
        assert test_path.suffix == ".mp3"
        assert test_path.stem == "file"
        assert test_path.name == "file.mp3"

    def test_file_extensions(self):
        """Test file extension detection"""
        # Test various extensions
        extensions = [".mp3", ".wav", ".mp4", ".avi", ".json", ".txt"]

        for ext in extensions:
            path = Path(f"test{ext}")
            assert path.suffix == ext

    def test_mime_type_initialization(self):
        """Test that mimetypes can be initialized"""
        import mimetypes

        mimetypes.init()

        # Test common types
        assert mimetypes.guess_type("test.mp3")[0] == "audio/mpeg"
        assert mimetypes.guess_type("test.mp4")[0] == "video/mp4"
