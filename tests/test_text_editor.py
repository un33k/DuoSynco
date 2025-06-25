"""
Unit tests for text.editor module
Tests transcript editing and manipulation functionality
"""

import pytest
import tempfile
import json
from pathlib import Path

from src.text.editor import TranscriptEditor


class TestTranscriptEditor:
    """Test cases for TranscriptEditor class"""

    def test_transcript_editor_creation(self):
        """Test creating TranscriptEditor instance"""
        editor = TranscriptEditor()
        assert isinstance(editor, TranscriptEditor)
        assert editor.transcript_data is None
        assert editor.original_data is None
        assert editor.edit_history == []
        assert editor.current_file is None

    def test_load_json_transcript(self):
        """Test loading JSON transcript file"""
        # Create sample transcript data
        transcript_data = {
            "transcript": "Hello world. This is a test.",
            "utterances": [
                {"start": 0.0, "end": 2.0, "text": "Hello world.", "speaker": "speaker_0"},
                {"start": 2.5, "end": 5.0, "text": "This is a test.", "speaker": "speaker_1"},
            ],
            "speakers": ["speaker_0", "speaker_1"],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
            json.dump(transcript_data, temp_file)
            temp_path = Path(temp_file.name)

        try:
            editor = TranscriptEditor()
            editor.load_transcript(str(temp_path), format="json")

            assert editor.transcript_data is not None
            assert editor.current_file == str(temp_path)
            assert "utterances" in editor.transcript_data
            assert len(editor.transcript_data["utterances"]) == 2
            assert editor.original_data == editor.transcript_data

        finally:
            temp_path.unlink()

    def test_load_text_transcript(self):
        """Test loading text transcript file"""
        text_content = """speaker_0: Hello world.
speaker_1: This is a test.
speaker_0: How are you?"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
            temp_file.write(text_content)
            temp_path = Path(temp_file.name)

        try:
            editor = TranscriptEditor()
            editor.load_transcript(str(temp_path), format="txt")

            assert editor.transcript_data is not None
            assert editor.current_file == str(temp_path)
            assert "utterances" in editor.transcript_data
            assert len(editor.transcript_data["utterances"]) == 3

        finally:
            temp_path.unlink()

    def test_load_transcript_auto_detect_json(self):
        """Test auto-detecting JSON format"""
        transcript_data = {"test": "data"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
            json.dump(transcript_data, temp_file)
            temp_path = Path(temp_file.name)

        try:
            editor = TranscriptEditor()
            editor.load_transcript(str(temp_path), format="auto")

            assert editor.transcript_data is not None

        finally:
            temp_path.unlink()

    def test_load_transcript_auto_detect_txt(self):
        """Test auto-detecting text format"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
            temp_file.write("speaker_0: Hello")
            temp_path = Path(temp_file.name)

        try:
            editor = TranscriptEditor()
            editor.load_transcript(str(temp_path), format="auto")

            assert editor.transcript_data is not None

        finally:
            temp_path.unlink()

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error"""
        editor = TranscriptEditor()

        with pytest.raises(FileNotFoundError):
            editor.load_transcript("/nonexistent/file.json")

    def test_save_json_transcript(self):
        """Test saving transcript as JSON"""
        # First load some data
        transcript_data = {
            "transcript": "Hello world.",
            "utterances": [
                {"start": 0.0, "end": 2.0, "text": "Hello world.", "speaker": "speaker_0"}
            ],
        }

        editor = TranscriptEditor()
        editor.transcript_data = transcript_data

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            editor.save_transcript(str(temp_path), format="json")

            # Verify file was created and contains correct data
            assert temp_path.exists()
            with open(temp_path, "r") as f:
                saved_data = json.load(f)
            assert saved_data == transcript_data

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_save_text_transcript(self):
        """Test saving transcript as text"""
        transcript_data = {
            "utterances": [
                {"start": 0.0, "end": 2.0, "text": "Hello world.", "speaker": "speaker_0"},
                {"start": 2.5, "end": 5.0, "text": "This is a test.", "speaker": "speaker_1"},
            ]
        }

        editor = TranscriptEditor()
        editor.transcript_data = transcript_data

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            editor.save_transcript(str(temp_path), format="txt")

            # Verify file was created and contains expected content
            assert temp_path.exists()
            content = temp_path.read_text()
            assert "speaker_0: Hello world." in content
            assert "speaker_1: This is a test." in content

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_save_with_backup(self):
        """Test saving with backup creation"""
        # Create original file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
            json.dump({"original": "data"}, temp_file)
            temp_path = Path(temp_file.name)

        try:
            # Load and modify
            editor = TranscriptEditor()
            editor.transcript_data = {"modified": "data"}

            # Save with backup
            editor.save_transcript(str(temp_path), format="json", backup_original=True)

            # Check backup was created
            backup_files = list(temp_path.parent.glob(f"{temp_path.stem}.backup.*"))
            assert len(backup_files) > 0

            # Check original file was updated
            with open(temp_path, "r") as f:
                saved_data = json.load(f)
            assert saved_data == {"modified": "data"}

        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
            for backup in temp_path.parent.glob(f"{temp_path.stem}.backup.*"):
                backup.unlink()

    def test_get_speakers(self):
        """Test getting list of speakers"""
        transcript_data = {
            "utterances": [
                {"speaker": "speaker_0", "text": "Hello"},
                {"speaker": "speaker_1", "text": "Hi"},
                {"speaker": "speaker_0", "text": "How are you?"},
                {"speaker": "speaker_2", "text": "Fine"},
            ]
        }

        editor = TranscriptEditor()
        editor.transcript_data = transcript_data

        speakers = editor.get_speakers()
        assert isinstance(speakers, list)
        assert len(speakers) == 3
        assert "speaker_0" in speakers
        assert "speaker_1" in speakers
        assert "speaker_2" in speakers

    def test_get_speakers_no_data(self):
        """Test getting speakers when no data loaded"""
        editor = TranscriptEditor()
        speakers = editor.get_speakers()
        assert speakers == []

    def test_get_utterances_by_speaker(self):
        """Test getting utterances for specific speaker"""
        transcript_data = {
            "utterances": [
                {"speaker": "speaker_0", "text": "Hello", "start": 0.0},
                {"speaker": "speaker_1", "text": "Hi", "start": 1.0},
                {"speaker": "speaker_0", "text": "How are you?", "start": 2.0},
            ]
        }

        editor = TranscriptEditor()
        editor.transcript_data = transcript_data

        speaker_0_utterances = editor.get_utterances_by_speaker("speaker_0")
        assert len(speaker_0_utterances) == 2
        assert speaker_0_utterances[0]["text"] == "Hello"
        assert speaker_0_utterances[1]["text"] == "How are you?"

        speaker_1_utterances = editor.get_utterances_by_speaker("speaker_1")
        assert len(speaker_1_utterances) == 1
        assert speaker_1_utterances[0]["text"] == "Hi"

    def test_replace_speaker_name(self):
        """Test replacing speaker names"""
        transcript_data = {
            "utterances": [
                {"speaker": "speaker_0", "text": "Hello"},
                {"speaker": "speaker_1", "text": "Hi"},
                {"speaker": "speaker_0", "text": "Bye"},
            ],
            "speakers": ["speaker_0", "speaker_1"],
        }

        editor = TranscriptEditor()
        editor.transcript_data = transcript_data

        # Replace speaker_0 with "Alice"
        editor.replace_speaker_name("speaker_0", "Alice")

        # Check utterances were updated
        for utterance in editor.transcript_data["utterances"]:
            if utterance["text"] in ["Hello", "Bye"]:
                assert utterance["speaker"] == "Alice"

        # Check speakers list was updated
        assert "Alice" in editor.transcript_data["speakers"]
        assert "speaker_0" not in editor.transcript_data["speakers"]

    def test_replace_text_content(self):
        """Test replacing text content in utterances"""
        transcript_data = {
            "utterances": [
                {"speaker": "speaker_0", "text": "Hello world"},
                {"speaker": "speaker_1", "text": "Hello there"},
            ]
        }

        editor = TranscriptEditor()
        editor.transcript_data = transcript_data

        # Replace "Hello" with "Hi"
        replacements = editor.replace_text_content("Hello", "Hi")

        assert replacements == 2  # Should replace in both utterances
        assert editor.transcript_data["utterances"][0]["text"] == "Hi world"
        assert editor.transcript_data["utterances"][1]["text"] == "Hi there"

    def test_filter_by_time_range(self):
        """Test filtering utterances by time range"""
        transcript_data = {
            "utterances": [
                {"speaker": "speaker_0", "text": "First", "start": 0.0, "end": 1.0},
                {"speaker": "speaker_1", "text": "Second", "start": 1.5, "end": 2.5},
                {"speaker": "speaker_0", "text": "Third", "start": 3.0, "end": 4.0},
                {"speaker": "speaker_1", "text": "Fourth", "start": 4.5, "end": 5.5},
            ]
        }

        editor = TranscriptEditor()
        editor.transcript_data = transcript_data

        # Filter to utterances between 1.0 and 4.0 seconds
        filtered = editor.filter_by_time_range(1.0, 4.0)

        assert len(filtered) == 2
        assert filtered[0]["text"] == "Second"
        assert filtered[1]["text"] == "Third"

    def test_get_statistics(self):
        """Test getting transcript statistics"""
        transcript_data = {
            "utterances": [
                {"speaker": "speaker_0", "text": "Hello world", "start": 0.0, "end": 2.0},
                {"speaker": "speaker_1", "text": "Hi there friend", "start": 2.5, "end": 5.0},
                {"speaker": "speaker_0", "text": "Goodbye", "start": 5.5, "end": 7.0},
            ]
        }

        editor = TranscriptEditor()
        editor.transcript_data = transcript_data

        stats = editor.get_statistics()

        assert stats["total_utterances"] == 3
        assert stats["total_speakers"] == 2
        assert stats["total_duration"] == 7.0
        assert stats["total_words"] == 6  # "Hello", "world", "Hi", "there", "friend", "Goodbye"
        assert "speaker_0" in stats["speaker_stats"]
        assert "speaker_1" in stats["speaker_stats"]
        assert stats["speaker_stats"]["speaker_0"]["utterance_count"] == 2
        assert stats["speaker_stats"]["speaker_1"]["utterance_count"] == 1

    def test_undo_last_edit(self):
        """Test undoing last edit"""
        transcript_data = {"utterances": [{"speaker": "speaker_0", "text": "Hello"}]}

        editor = TranscriptEditor()
        editor.transcript_data = transcript_data.copy()
        editor.original_data = transcript_data.copy()

        # Make an edit
        editor.replace_speaker_name("speaker_0", "Alice")
        assert editor.transcript_data["utterances"][0]["speaker"] == "Alice"
        assert len(editor.edit_history) > 0

        # Undo the edit
        editor.undo_last_edit()
        assert editor.transcript_data["utterances"][0]["speaker"] == "speaker_0"

    def test_merge_speakers(self):
        """Test merging two speakers"""
        transcript_data = {
            "utterances": [
                {"speaker": "speaker_0", "text": "Hello"},
                {"speaker": "speaker_1", "text": "Hi"},
                {"speaker": "speaker_0", "text": "Bye"},
            ],
            "speakers": ["speaker_0", "speaker_1"],
        }

        editor = TranscriptEditor()
        editor.transcript_data = transcript_data

        # Merge speaker_1 into speaker_0
        editor.merge_speakers("speaker_1", "speaker_0")

        # All utterances should now be speaker_0
        for utterance in editor.transcript_data["utterances"]:
            assert utterance["speaker"] == "speaker_0"

        # speaker_1 should be removed from speakers list
        assert "speaker_1" not in editor.transcript_data["speakers"]
        assert "speaker_0" in editor.transcript_data["speakers"]

    def test_split_utterance(self):
        """Test splitting a long utterance"""
        transcript_data = {
            "utterances": [
                {
                    "speaker": "speaker_0",
                    "text": "Hello world. How are you today?",
                    "start": 0.0,
                    "end": 4.0,
                }
            ]
        }

        editor = TranscriptEditor()
        editor.transcript_data = transcript_data

        # Split at index 0, at position of the period
        editor.split_utterance(0, 12)  # After "Hello world."

        # Should now have 2 utterances
        utterances = editor.transcript_data["utterances"]
        assert len(utterances) == 2
        assert utterances[0]["text"] == "Hello world."
        assert utterances[1]["text"] == " How are you today?"
        assert utterances[0]["end"] < utterances[1]["start"]

    def test_validate_transcript_structure(self):
        """Test validating transcript data structure"""
        editor = TranscriptEditor()

        # Valid structure
        valid_data = {
            "utterances": [{"speaker": "speaker_0", "text": "Hello", "start": 0.0, "end": 1.0}]
        }

        assert editor.validate_transcript_structure(valid_data) is True

        # Invalid structure - missing utterances
        invalid_data = {"transcript": "Hello"}
        assert editor.validate_transcript_structure(invalid_data) is False

        # Invalid structure - utterance missing required fields
        invalid_utterances = {"utterances": [{"speaker": "speaker_0"}]}  # Missing text, start, end
        assert editor.validate_transcript_structure(invalid_utterances) is False
