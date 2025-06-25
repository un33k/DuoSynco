"""
Unit tests for main CLI module
Tests command-line interface and workflow coordination
"""

import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from src.main import cli


class TestMainCLI:
    """Test cases for main CLI functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()
        self.temp_dir = None

    def teardown_method(self):
        """Clean up test fixtures"""
        if self.temp_dir:
            import shutil

            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_temp_audio_file(self, content="fake audio"):
        """Create a temporary audio file for testing"""
        if not self.temp_dir:
            self.temp_dir = tempfile.mkdtemp()

        temp_file = Path(self.temp_dir) / "test_audio.mp3"
        temp_file.write_text(content)
        return temp_file

    def create_temp_transcript_file(self, data=None):
        """Create a temporary transcript file for testing"""
        if not self.temp_dir:
            self.temp_dir = tempfile.mkdtemp()

        if data is None:
            data = {
                "utterances": [
                    {"speaker": "speaker_0", "text": "Hello world", "start": 0.0, "end": 2.0}
                ]
            }

        temp_file = Path(self.temp_dir) / "test_transcript.json"
        with open(temp_file, "w") as f:
            json.dump(data, f)
        return temp_file

    def test_cli_help(self):
        """Test CLI help message"""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "DuoSynco" in result.output
        assert "Sync videos with isolated speaker audio tracks" in result.output

    def test_list_providers(self):
        """Test listing available providers"""
        result = self.runner.invoke(cli, ["--list-providers"])
        assert result.exit_code == 0
        assert "assemblyai" in result.output.lower()
        assert "elevenlabs" in result.output.lower()

    def test_list_execution_paths(self):
        """Test listing execution paths"""
        result = self.runner.invoke(cli, ["--list-execution-paths"])
        assert result.exit_code == 0
        assert "execution path" in result.output.lower()

    @patch("src.main.Config")
    def test_show_config(self, mock_config):
        """Test showing configuration"""
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance

        result = self.runner.invoke(cli, ["--show-config"])
        assert result.exit_code == 0
        mock_config_instance.print_settings.assert_called_once()

    @patch("src.main.VoiceManager")
    def test_list_voices_with_api_key(self, mock_voice_manager):
        """Test listing voices with API key"""
        mock_manager = MagicMock()
        mock_manager.get_all_voices.return_value = [
            {"voice_id": "test1", "name": "Test Voice 1"},
            {"voice_id": "test2", "name": "Test Voice 2"},
        ]
        mock_voice_manager.return_value = mock_manager

        result = self.runner.invoke(cli, ["--list-voices", "--api-key", "test_key"])
        assert result.exit_code == 0
        assert "Test Voice 1" in result.output
        assert "Test Voice 2" in result.output

    def test_list_voices_without_api_key(self):
        """Test listing voices without API key shows error"""
        result = self.runner.invoke(cli, ["--list-voices"])
        assert result.exit_code == 1
        assert "API key is required" in result.output

    def test_missing_input_file_error(self):
        """Test error when no input file provided"""
        result = self.runner.invoke(cli, [])
        assert result.exit_code == 1
        assert "input file is required" in result.output.lower()

    def test_nonexistent_input_file_error(self):
        """Test error when input file doesn't exist"""
        result = self.runner.invoke(cli, ["/nonexistent/file.mp3"])
        assert result.exit_code == 1
        assert "does not exist" in result.output

    @patch("src.main.AudioDiarizer")
    @patch("src.main.FileHandler")
    def test_diarization_mode_success(self, mock_file_handler, mock_diarizer):
        """Test successful diarization mode execution"""
        # Setup mocks
        audio_file = self.create_temp_audio_file()

        mock_handler = MagicMock()
        mock_handler.validate_audio_file.return_value = {
            "exists": True,
            "is_audio": True,
            "size_bytes": 1000,
        }
        mock_file_handler.return_value = mock_handler

        mock_diarizer_instance = MagicMock()
        mock_diarizer_instance.separate_speakers.return_value = {
            "speaker_files": ["speaker_0.wav", "speaker_1.wav"],
            "transcript_file": "transcript.txt",
            "stats": {
                "total_coverage": 95.0,
                "total_speaker_duration": 60.0,
                "original_duration": 63.0,
                "speakers": {
                    "speaker_0": {"duration": 30.0, "coverage": 47.5},
                    "speaker_1": {"duration": 30.0, "coverage": 47.5},
                },
            },
        }
        mock_diarizer.return_value = mock_diarizer_instance

        result = self.runner.invoke(
            cli,
            [
                str(audio_file),
                "--mode",
                "diarization",
                "--provider",
                "assemblyai",
                "--speakers",
                "2",
            ],
        )

        assert result.exit_code == 0
        assert "Speaker separation completed" in result.output
        mock_diarizer_instance.separate_speakers.assert_called_once()

    @patch("src.main.TTSAudioGenerator")
    @patch("src.main.FileHandler")
    def test_tts_mode_success(self, mock_file_handler, mock_tts_generator):
        """Test successful TTS mode execution"""
        # Create test transcript
        transcript_file = self.create_temp_transcript_file()

        mock_handler = MagicMock()
        mock_handler.validate_audio_file.return_value = {
            "exists": True,
            "is_audio": False,
            "extension": ".json",
        }
        mock_file_handler.return_value = mock_handler

        mock_generator = MagicMock()
        mock_generator.generate_tts_from_transcript.return_value = {
            "audio_files": ["speaker_0_tts.wav", "speaker_1_tts.wav"],
            "success": True,
        }
        mock_tts_generator.return_value = mock_generator

        result = self.runner.invoke(
            cli,
            [
                str(transcript_file),
                "--mode",
                "tts",
                "--provider",
                "elevenlabs",
                "--api-key",
                "test_key",
                "--total-duration",
                "60",
            ],
        )

        assert result.exit_code == 0
        mock_generator.generate_tts_from_transcript.assert_called_once()

    @patch("src.main.EditWorkflow")
    @patch("src.main.FileHandler")
    def test_edit_mode_success(self, mock_file_handler, mock_edit_workflow):
        """Test successful edit mode execution"""
        audio_file = self.create_temp_audio_file()

        mock_handler = MagicMock()
        mock_handler.validate_audio_file.return_value = {
            "exists": True,
            "is_audio": True,
            "size_bytes": 1000,
        }
        mock_file_handler.return_value = mock_handler

        mock_workflow = MagicMock()
        mock_workflow.run_edit_workflow.return_value = {
            "speaker_files": ["edited_speaker_0.wav"],
            "transcript_file": "edited_transcript.txt",
            "stats": {
                "total_coverage": 90.0,
                "total_speaker_duration": 55.0,
                "original_duration": 60.0,
                "speakers": {"speaker_0": {"duration": 55.0, "coverage": 90.0}},
            },
        }
        mock_edit_workflow.return_value = mock_workflow

        result = self.runner.invoke(
            cli,
            [
                str(audio_file),
                "--mode",
                "edit",
                "--provider",
                "elevenlabs",
                "--secondary-provider",
                "assemblyai",
                "--api-key",
                "test_key",
            ],
        )

        assert result.exit_code == 0
        mock_workflow.run_edit_workflow.assert_called_once()

    @patch("src.main.DialogueWorkflow")
    @patch("src.main.Config")
    def test_dialogue_mode_success(self, mock_config, mock_dialogue_workflow):
        """Test successful dialogue mode execution"""
        transcript_file = self.create_temp_transcript_file()

        mock_config_instance = MagicMock()
        mock_config_instance.verbose = False
        mock_config.return_value = mock_config_instance

        mock_workflow = MagicMock()
        mock_workflow._check_components.return_value = True
        mock_workflow.run_stt_to_dialogue_workflow.return_value = {
            "success": True,
            "files_generated": {"dialogue_json": "dialogue.json", "audio": "dialogue.mp3"},
        }
        mock_dialogue_workflow.return_value = mock_workflow

        result = self.runner.invoke(
            cli,
            [
                str(transcript_file),
                "--mode",
                "dialogue",
                "--provider",
                "elevenlabs",
                "--api-key",
                "test_key",
            ],
        )

        assert result.exit_code == 0
        mock_workflow.run_stt_to_dialogue_workflow.assert_called_once()

    def test_invalid_mode_error(self):
        """Test error with invalid mode"""
        audio_file = self.create_temp_audio_file()

        result = self.runner.invoke(cli, [str(audio_file), "--mode", "invalid_mode"])

        assert result.exit_code == 2  # Click parameter error
        assert "Invalid value" in result.output

    def test_invalid_provider_error(self):
        """Test error with invalid provider"""
        audio_file = self.create_temp_audio_file()

        result = self.runner.invoke(cli, [str(audio_file), "--provider", "invalid_provider"])

        assert result.exit_code == 2  # Click parameter error
        assert "Invalid value" in result.output

    @patch("src.main.AudioDiarizer")
    def test_diarization_with_custom_options(self, mock_diarizer):
        """Test diarization with various custom options"""
        audio_file = self.create_temp_audio_file()

        mock_diarizer_instance = MagicMock()
        mock_diarizer_instance.separate_speakers.return_value = {
            "speaker_files": ["speaker_0.wav"],
            "transcript_file": "transcript.txt",
            "stats": {"total_coverage": 85.0, "speakers": {}},
        }
        mock_diarizer.return_value = mock_diarizer_instance

        with patch("src.main.FileHandler") as mock_file_handler:
            mock_handler = MagicMock()
            mock_handler.validate_audio_file.return_value = {
                "exists": True,
                "is_audio": True,
                "size_bytes": 1000,
            }
            mock_file_handler.return_value = mock_handler

            result = self.runner.invoke(
                cli,
                [
                    str(audio_file),
                    "--speakers",
                    "3",
                    "--quality",
                    "high",
                    "--language",
                    "es",
                    "--format",
                    "avi",
                    "--verbose",
                ],
            )

            assert result.exit_code == 0
            # Verify that custom options were passed
            call_args = mock_diarizer_instance.separate_speakers.call_args
            assert call_args is not None

    def test_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist"""
        audio_file = self.create_temp_audio_file()
        output_dir = Path(self.temp_dir) / "new_output"

        with patch("src.main.AudioDiarizer") as mock_diarizer:
            mock_diarizer_instance = MagicMock()
            mock_diarizer_instance.separate_speakers.return_value = {
                "speaker_files": ["speaker_0.wav"],
                "transcript_file": "transcript.txt",
                "stats": {"total_coverage": 85.0, "speakers": {}},
            }
            mock_diarizer.return_value = mock_diarizer_instance

            with patch("src.main.FileHandler") as mock_file_handler:
                mock_handler = MagicMock()
                mock_handler.validate_audio_file.return_value = {
                    "exists": True,
                    "is_audio": True,
                    "size_bytes": 1000,
                }
                mock_file_handler.return_value = mock_handler

                result = self.runner.invoke(cli, [str(audio_file), "--output-dir", str(output_dir)])

                assert result.exit_code == 0
                # Directory should be created by the application
                assert output_dir.exists()

    def test_error_handling(self):
        """Test error handling in CLI"""
        audio_file = self.create_temp_audio_file()

        with patch("src.main.AudioDiarizer") as mock_diarizer:
            mock_diarizer.side_effect = Exception("Simulated processing error")

            with patch("src.main.FileHandler") as mock_file_handler:
                mock_handler = MagicMock()
                mock_handler.validate_audio_file.return_value = {
                    "exists": True,
                    "is_audio": True,
                    "size_bytes": 1000,
                }
                mock_file_handler.return_value = mock_handler

                result = self.runner.invoke(cli, [str(audio_file)])

                assert result.exit_code == 1
                assert "Error" in result.output

    @patch("src.main.json.loads")
    def test_voice_mapping_json_parsing(self, mock_json_loads):
        """Test voice mapping JSON string parsing"""
        audio_file = self.create_temp_audio_file()
        mock_json_loads.return_value = {"speaker_0": "voice1", "speaker_1": "voice2"}

        with patch("src.main.AudioDiarizer") as mock_diarizer:
            mock_diarizer_instance = MagicMock()
            mock_diarizer_instance.separate_speakers.return_value = {
                "speaker_files": ["speaker_0.wav"],
                "transcript_file": "transcript.txt",
                "stats": {"total_coverage": 85.0, "speakers": {}},
            }
            mock_diarizer.return_value = mock_diarizer_instance

            with patch("src.main.FileHandler") as mock_file_handler:
                mock_handler = MagicMock()
                mock_handler.validate_audio_file.return_value = {
                    "exists": True,
                    "is_audio": True,
                    "size_bytes": 1000,
                }
                mock_file_handler.return_value = mock_handler

                result = self.runner.invoke(
                    cli, [str(audio_file), "--voice-mapping", '{"speaker_0": "voice1"}']
                )

                assert result.exit_code == 0
                mock_json_loads.assert_called_once_with('{"speaker_0": "voice1"}')

    def test_verbose_output(self):
        """Test verbose output flag"""
        audio_file = self.create_temp_audio_file()

        with patch("src.main.AudioDiarizer") as mock_diarizer:
            mock_diarizer_instance = MagicMock()
            mock_diarizer_instance.separate_speakers.return_value = {
                "speaker_files": ["speaker_0.wav"],
                "transcript_file": "transcript.txt",
                "stats": {"total_coverage": 85.0, "speakers": {}},
            }
            mock_diarizer.return_value = mock_diarizer_instance

            with patch("src.main.FileHandler") as mock_file_handler:
                mock_handler = MagicMock()
                mock_handler.validate_audio_file.return_value = {
                    "exists": True,
                    "is_audio": True,
                    "size_bytes": 1000,
                }
                mock_file_handler.return_value = mock_handler

                result = self.runner.invoke(cli, [str(audio_file), "--verbose"])

                assert result.exit_code == 0
                # Verbose mode should provide more detailed output
                # (Exact assertions depend on implementation)
