"""
Pytest configuration and shared fixtures for DuoSynco tests
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import os


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_audio_file(temp_dir):
    """Create a sample audio file for testing"""
    audio_file = temp_dir / "test_audio.mp3"
    audio_file.write_text("fake audio content")
    return audio_file


@pytest.fixture
def sample_video_file(temp_dir):
    """Create a sample video file for testing"""
    video_file = temp_dir / "test_video.mp4"
    video_file.write_text("fake video content")
    return video_file


@pytest.fixture
def sample_transcript_data():
    """Sample transcript data for testing"""
    return {
        "transcript": "Hello world. This is a test. How are you?",
        "utterances": [
            {"start": 0.0, "end": 2.0, "text": "Hello world.", "speaker": "speaker_0"},
            {"start": 2.5, "end": 4.5, "text": "This is a test.", "speaker": "speaker_1"},
            {"start": 5.0, "end": 7.0, "text": "How are you?", "speaker": "speaker_0"},
        ],
        "speakers": ["speaker_0", "speaker_1"],
        "duration": 7.0,
        "language": "en",
    }


@pytest.fixture
def sample_transcript_file(temp_dir, sample_transcript_data):
    """Create a sample transcript JSON file"""
    transcript_file = temp_dir / "test_transcript.json"
    with open(transcript_file, "w") as f:
        json.dump(sample_transcript_data, f)
    return transcript_file


@pytest.fixture
def sample_text_transcript_file(temp_dir):
    """Create a sample text transcript file"""
    content = """speaker_0: Hello world.
speaker_1: This is a test.
speaker_0: How are you?"""

    transcript_file = temp_dir / "test_transcript.txt"
    transcript_file.write_text(content)
    return transcript_file


@pytest.fixture
def mock_api_responses():
    """Mock API responses for testing"""
    return {
        "elevenlabs_voices": [
            {
                "voice_id": "test_voice_1",
                "name": "Test Voice 1",
                "labels": {"gender": "male", "age": "adult"},
                "preview_url": "https://example.com/preview1.mp3",
            },
            {
                "voice_id": "test_voice_2",
                "name": "Test Voice 2",
                "labels": {"gender": "female", "age": "adult"},
                "preview_url": "https://example.com/preview2.mp3",
            },
        ],
        "assemblyai_transcript": {
            "id": "test_transcript_id",
            "status": "completed",
            "text": "Hello world. This is a test.",
            "utterances": [
                {"start": 0, "end": 2000, "text": "Hello world.", "speaker": "A"},
                {"start": 2500, "end": 4500, "text": "This is a test.", "speaker": "B"},
            ],
        },
        "elevenlabs_tts": {"audio_base64": "fake_audio_data_base64_encoded"},
    }


@pytest.fixture(autouse=True)
def mock_external_apis():
    """Automatically mock external API calls to prevent real network requests"""
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post, patch(
        "requests.put"
    ) as mock_put, patch("requests.delete") as mock_delete:

        # Configure default mock responses
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"mocked": True}
        mock_get.return_value.content = b"mocked content"

        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"mocked": True}
        mock_post.return_value.content = b"mocked content"

        mock_put.return_value.status_code = 200
        mock_put.return_value.json.return_value = {"mocked": True}

        mock_delete.return_value.status_code = 200
        mock_delete.return_value.json.return_value = {"mocked": True}

        yield {"get": mock_get, "post": mock_post, "put": mock_put, "delete": mock_delete}


@pytest.fixture
def clean_environment():
    """Provide a clean environment for testing"""
    # Store original environment
    original_env = dict(os.environ)

    # Clear test-related environment variables
    test_vars = [
        "DUOSYNCO_QUALITY",
        "DUOSYNCO_FORMAT",
        "DUOSYNCO_BACKEND",
        "DUOSYNCO_VERBOSE",
        "ELEVENLABS_API_KEY",
        "ASSEMBLYAI_API_KEY",
        "VOICE_SPEAKER_0",
        "VOICE_SPEAKER_1",
    ]

    for var in test_vars:
        os.environ.pop(var, None)

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_file_operations():
    """Mock file system operations for testing"""
    with patch("pathlib.Path.exists") as mock_exists, patch(
        "pathlib.Path.mkdir"
    ) as mock_mkdir, patch("pathlib.Path.unlink") as mock_unlink, patch(
        "pathlib.Path.stat"
    ) as mock_stat, patch(
        "shutil.copy2"
    ) as mock_copy, patch(
        "shutil.move"
    ) as mock_move:

        # Configure default behaviors
        mock_exists.return_value = True
        mock_mkdir.return_value = None
        mock_unlink.return_value = None

        # Mock file stats
        mock_stat_result = type("MockStat", (), {"st_size": 1024, "st_mtime": 1234567890.0})()
        mock_stat.return_value = mock_stat_result

        yield {
            "exists": mock_exists,
            "mkdir": mock_mkdir,
            "unlink": mock_unlink,
            "stat": mock_stat,
            "copy": mock_copy,
            "move": mock_move,
        }


@pytest.fixture
def mock_audio_processing():
    """Mock audio processing libraries to avoid dependencies"""
    with patch("librosa.load") as mock_librosa_load, patch(
        "soundfile.write"
    ) as mock_soundfile_write, patch("pydub.AudioSegment") as mock_pydub:

        # Mock librosa.load to return fake audio data
        mock_librosa_load.return_value = ([0.1, 0.2, 0.3], 22050)  # (audio_data, sample_rate)

        # Mock soundfile.write to do nothing
        mock_soundfile_write.return_value = None

        # Mock pydub AudioSegment
        mock_audio_segment = mock_pydub.return_value
        mock_audio_segment.export.return_value = None
        mock_audio_segment.duration_seconds = 10.0

        yield {
            "librosa_load": mock_librosa_load,
            "soundfile_write": mock_soundfile_write,
            "pydub": mock_pydub,
        }


@pytest.fixture
def mock_video_processing():
    """Mock video processing libraries"""
    with patch("moviepy.editor.VideoFileClip") as mock_video_clip:

        # Mock video clip
        mock_clip = mock_video_clip.return_value.__enter__.return_value
        mock_clip.duration = 10.0
        mock_clip.fps = 30.0
        mock_clip.size = (1920, 1080)
        mock_clip.write_videofile.return_value = None

        yield {"video_clip": mock_video_clip}


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "quality": "medium",
        "output_format": "mp4",
        "backend": "speechbrain",
        "verbose": False,
        "audio_sample_rate": 44100,
        "audio_channels": 2,
        "num_threads": 1,
        "memory_limit_mb": 2048,
        "cleanup_temp_files": True,
    }


# Custom markers for test categorization
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line(
        "markers", "external_api: mark test as requiring external API (should be skipped in CI)"
    )
