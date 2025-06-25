"""
Unit tests for utils.files module
Tests file handling utilities and validation
"""

import tempfile
from pathlib import Path

from src.utils.files import FileHandler


class TestFileHandler:
    """Test cases for FileHandler class"""

    def test_file_handler_creation(self):
        """Test creating FileHandler instance"""
        handler = FileHandler()
        assert isinstance(handler, FileHandler)

    def test_validate_audio_file_exists(self):
        """Test validating existing audio file"""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(b"fake audio content")

        try:
            handler = FileHandler()
            result = handler.validate_audio_file(str(temp_path))

            assert result["exists"] is True
            assert result["extension"] == ".mp3"
            assert result["is_audio"] is True
            assert result["size_bytes"] > 0

        finally:
            temp_path.unlink()

    def test_validate_audio_file_not_exists(self):
        """Test validating non-existent audio file"""
        handler = FileHandler()
        result = handler.validate_audio_file("/nonexistent/file.mp3")

        assert result["exists"] is False
        assert result["is_audio"] is False
        assert result["error"] is not None

    def test_validate_video_file_exists(self):
        """Test validating existing video file"""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(b"fake video content")

        try:
            handler = FileHandler()
            result = handler.validate_video_file(str(temp_path))

            assert result["exists"] is True
            assert result["extension"] == ".mp4"
            assert result["is_video"] is True
            assert result["size_bytes"] > 0

        finally:
            temp_path.unlink()

    def test_validate_video_file_not_exists(self):
        """Test validating non-existent video file"""
        handler = FileHandler()
        result = handler.validate_video_file("/nonexistent/file.mp4")

        assert result["exists"] is False
        assert result["is_video"] is False
        assert result["error"] is not None

    def test_get_file_info(self):
        """Test getting file information"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            test_content = "Hello, World!"
            temp_file.write(test_content.encode())

        try:
            handler = FileHandler()
            info = handler.get_file_info(str(temp_path))

            assert info["exists"] is True
            assert info["size_bytes"] == len(test_content)
            assert info["extension"] == ".txt"
            assert "modified_time" in info
            assert "is_readable" in info
            assert "is_writable" in info

        finally:
            temp_path.unlink()

    def test_get_file_info_nonexistent(self):
        """Test getting info for non-existent file"""
        handler = FileHandler()
        info = handler.get_file_info("/nonexistent/file.txt")

        assert info["exists"] is False
        assert info["error"] is not None

    def test_create_output_directory(self):
        """Test creating output directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "new_output_dir"

            handler = FileHandler()
            result = handler.create_output_directory(str(output_dir))

            assert result["created"] is True
            assert result["path"] == str(output_dir)
            assert output_dir.exists()
            assert output_dir.is_dir()

    def test_create_output_directory_existing(self):
        """Test creating output directory that already exists"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)  # Already exists

            handler = FileHandler()
            result = handler.create_output_directory(str(output_dir))

            assert result["created"] is False  # Already existed
            assert result["path"] == str(output_dir)
            assert output_dir.exists()

    def test_clean_filename(self):
        """Test cleaning filename for filesystem safety"""
        handler = FileHandler()

        # Test various problematic characters
        test_cases = [
            ("normal_file.txt", "normal_file.txt"),
            ("file with spaces.txt", "file_with_spaces.txt"),
            ("file/with\\slashes.txt", "file_with_slashes.txt"),
            ("file:with*special?.txt", "file_with_special.txt"),
            ("file<with>pipes|.txt", "file_with_pipes.txt"),
            ("file\"with'quotes.txt", "file_with_quotes.txt"),
        ]

        for input_name, expected in test_cases:
            result = handler.clean_filename(input_name)
            assert result == expected

    def test_get_available_filename(self):
        """Test getting available filename when conflicts exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            # Create existing files
            (base_dir / "test.txt").touch()
            (base_dir / "test_1.txt").touch()
            (base_dir / "test_2.txt").touch()

            handler = FileHandler()
            available = handler.get_available_filename(str(base_dir / "test.txt"))

            # Should suggest test_3.txt
            assert available == str(base_dir / "test_3.txt")

    def test_get_available_filename_no_conflict(self):
        """Test getting available filename with no conflicts"""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            test_file = base_dir / "new_file.txt"

            handler = FileHandler()
            available = handler.get_available_filename(str(test_file))

            # Should return original filename
            assert available == str(test_file)

    def test_copy_file(self):
        """Test copying files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            # Create source file
            source = base_dir / "source.txt"
            content = "Test content for copying"
            source.write_text(content)

            # Copy to destination
            dest = base_dir / "destination.txt"

            handler = FileHandler()
            result = handler.copy_file(str(source), str(dest))

            assert result["success"] is True
            assert result["source"] == str(source)
            assert result["destination"] == str(dest)
            assert dest.exists()
            assert dest.read_text() == content

    def test_copy_file_nonexistent_source(self):
        """Test copying non-existent source file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            source = base_dir / "nonexistent.txt"
            dest = base_dir / "destination.txt"

            handler = FileHandler()
            result = handler.copy_file(str(source), str(dest))

            assert result["success"] is False
            assert result["error"] is not None

    def test_move_file(self):
        """Test moving files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            # Create source file
            source = base_dir / "source.txt"
            content = "Test content for moving"
            source.write_text(content)

            # Move to destination
            dest = base_dir / "moved.txt"

            handler = FileHandler()
            result = handler.move_file(str(source), str(dest))

            assert result["success"] is True
            assert result["source"] == str(source)
            assert result["destination"] == str(dest)
            assert not source.exists()  # Source should be gone
            assert dest.exists()
            assert dest.read_text() == content

    def test_delete_file(self):
        """Test deleting files"""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(b"content to be deleted")

        # Verify file exists
        assert temp_path.exists()

        handler = FileHandler()
        result = handler.delete_file(str(temp_path))

        assert result["success"] is True
        assert result["path"] == str(temp_path)
        assert not temp_path.exists()

    def test_delete_nonexistent_file(self):
        """Test deleting non-existent file"""
        handler = FileHandler()
        result = handler.delete_file("/nonexistent/file.txt")

        assert result["success"] is False
        assert result["error"] is not None

    def test_get_directory_size(self):
        """Test getting directory size"""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            # Create some files with known sizes
            file1 = base_dir / "file1.txt"
            file2 = base_dir / "file2.txt"
            file1.write_text("Hello")  # 5 bytes
            file2.write_text("World!")  # 6 bytes

            # Create subdirectory with file
            sub_dir = base_dir / "subdir"
            sub_dir.mkdir()
            file3 = sub_dir / "file3.txt"
            file3.write_text("Test")  # 4 bytes

            handler = FileHandler()
            result = handler.get_directory_size(str(base_dir))

            assert result["total_size"] == 15  # 5 + 6 + 4
            assert result["file_count"] == 3
            assert result["directory_count"] == 1  # subdir

    def test_find_files_by_extension(self):
        """Test finding files by extension"""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            # Create files with different extensions
            (base_dir / "audio.mp3").touch()
            (base_dir / "video.mp4").touch()
            (base_dir / "text.txt").touch()
            (base_dir / "another_audio.wav").touch()

            # Create subdirectory with more files
            sub_dir = base_dir / "subdir"
            sub_dir.mkdir()
            (sub_dir / "sub_audio.mp3").touch()

            handler = FileHandler()

            # Find mp3 files
            mp3_files = handler.find_files_by_extension(str(base_dir), ".mp3")
            assert len(mp3_files) == 2
            assert any("audio.mp3" in f for f in mp3_files)
            assert any("sub_audio.mp3" in f for f in mp3_files)

            # Find audio files (multiple extensions)
            audio_files = handler.find_files_by_extension(str(base_dir), [".mp3", ".wav"])
            assert len(audio_files) == 3

    def test_backup_file(self):
        """Test creating file backup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            # Create original file
            original = base_dir / "original.txt"
            content = "Original content"
            original.write_text(content)

            handler = FileHandler()
            result = handler.backup_file(str(original))

            assert result["success"] is True
            assert result["original"] == str(original)
            assert "backup_path" in result

            # Verify backup exists and has same content
            backup_path = Path(result["backup_path"])
            assert backup_path.exists()
            assert backup_path.read_text() == content
            assert backup_path.name.startswith("original")
            assert ".backup." in backup_path.name

    def test_is_audio_file(self):
        """Test audio file detection"""
        handler = FileHandler()

        # Test valid audio extensions
        audio_extensions = [".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"]
        for ext in audio_extensions:
            assert handler.is_audio_file(f"test{ext}") is True
            assert handler.is_audio_file(f"test{ext.upper()}") is True  # Case insensitive

        # Test invalid extensions
        assert handler.is_audio_file("test.txt") is False
        assert handler.is_audio_file("test.mp4") is False
        assert handler.is_audio_file("test") is False

    def test_is_video_file(self):
        """Test video file detection"""
        handler = FileHandler()

        # Test valid video extensions
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"]
        for ext in video_extensions:
            assert handler.is_video_file(f"test{ext}") is True
            assert handler.is_video_file(f"test{ext.upper()}") is True  # Case insensitive

        # Test invalid extensions
        assert handler.is_video_file("test.txt") is False
        assert handler.is_video_file("test.mp3") is False
        assert handler.is_video_file("test") is False
