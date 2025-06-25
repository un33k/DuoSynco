"""
Edit Workflow Orchestration
Manages the complete STT -> Edit -> Final Diarization workflow
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class EditWorkflow:
    """
    Orchestrates the complete edit workflow:
    1. Speech-to-Text with Speaker Diarization (ElevenLabs)
    2. Text Editing and Speaker Replacement
    3. Final Audio Separation (AssemblyAI)
    """

    def __init__(
        self,
        stt_provider: str = "elevenlabs-stt",
        final_provider: str = "assemblyai",
        api_key: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        """
        Initialize edit workflow

        Args:
            stt_provider: STT provider name
            final_provider: Final diarization provider name
            api_key: API key for providers
            progress_callback: Optional callback for progress updates (message, progress_0_to_1)
        """
        self.stt_provider = stt_provider
        self.final_provider = final_provider
        self.api_key = api_key
        self.progress_callback = progress_callback

        # Workflow state
        self.current_step = 0
        self.total_steps = 3
        self.results = {}
        self.intermediate_files = []

        logger.info(
            "EditWorkflow initialized: STT=%s, Final=%s", stt_provider, final_provider
        )

    def execute(
        self,
        input_file: str,
        output_dir: str,
        speakers_expected: int = 2,
        language: str = "en",
        stt_quality: str = "high",
        edit_config: Optional[Dict[str, Any]] = None,
        save_intermediate: bool = True,
        cleanup_intermediate: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the complete edit workflow

        Args:
            input_file: Path to input audio/video file
            output_dir: Output directory for results
            speakers_expected: Expected number of speakers
            language: Language code
            stt_quality: STT quality level
            edit_config: Configuration for editing step
            save_intermediate: Whether to save intermediate results
            cleanup_intermediate: Whether to cleanup intermediate files

        Returns:
            Dictionary with complete workflow results
        """
        start_time = time.time()
        self._report_progress("Starting edit workflow", 0.0)

        try:
            # Step 1: Speech-to-Text
            self.current_step = 1
            self._report_progress("Step 1: Speech-to-Text transcription", 0.1)

            stt_result = self._execute_stt_step(
                input_file,
                output_dir,
                speakers_expected,
                language,
                stt_quality,
                save_intermediate,
            )

            self.results["stt"] = stt_result
            self._report_progress("STT completed", 0.4)

            # Step 2: Text Editing
            self.current_step = 2
            self._report_progress("Step 2: Text editing and speaker replacement", 0.5)

            edit_result = self._execute_edit_step(
                stt_result, output_dir, edit_config or {}, save_intermediate
            )

            self.results["edit"] = edit_result
            self._report_progress("Editing completed", 0.7)

            # Step 3: Final Diarization
            self.current_step = 3
            self._report_progress("Step 3: Final audio separation", 0.8)

            final_result = self._execute_final_step(
                input_file, output_dir, edit_result, language, save_intermediate
            )

            self.results["final"] = final_result
            self._report_progress("Final separation completed", 1.0)

            # Compile complete results
            total_time = time.time() - start_time

            complete_result = {
                "workflow_success": True,
                "total_time": total_time,
                "steps_completed": self.total_steps,
                "stt_result": stt_result,
                "edit_result": edit_result,
                "final_result": final_result,
                "intermediate_files": self.intermediate_files,
                "summary": self._generate_summary(),
            }

            # Save workflow metadata
            if save_intermediate:
                metadata_file = self._save_workflow_metadata(
                    output_dir, complete_result
                )
                complete_result["metadata_file"] = metadata_file

            # Cleanup if requested
            if cleanup_intermediate:
                self._cleanup_intermediate_files()

            logger.info("Edit workflow completed successfully in %.1fs", total_time)
            return complete_result

        except Exception as e:
            error_result = {
                "workflow_success": False,
                "error": str(e),
                "step_failed": self.current_step,
                "partial_results": self.results,
                "intermediate_files": self.intermediate_files,
            }

            logger.error("Edit workflow failed at step %d: %s", self.current_step, e)
            self._report_progress(f"Workflow failed: {str(e)}", 1.0)

            return error_result

    def _execute_stt_step(
        self,
        input_file: str,
        output_dir: str,
        speakers_expected: int,
        language: str,
        quality: str,
        save_intermediate: bool,
    ) -> Dict[str, Any]:
        """Execute STT step"""
        from ..audio.stt import STTAudioTranscriber

        logger.info("Executing STT step: %s", input_file)

        # Initialize STT transcriber
        stt_transcriber = STTAudioTranscriber(
            provider=self.stt_provider, api_key=self.api_key
        )

        # Perform transcription
        base_filename = f"{Path(input_file).stem}_stt"

        result = stt_transcriber.transcribe_audio_file(
            audio_file=input_file,
            output_dir=output_dir,
            base_filename=base_filename,
            speakers_expected=speakers_expected,
            language=language,
            quality=quality,
            enhanced_processing=True,
            save_results=save_intermediate,
        )

        # Track intermediate files
        if save_intermediate and "files" in result:
            self.intermediate_files.extend(result["files"])

        logger.info(
            "STT step completed: %d utterances, %d speakers",
            len(result["utterances"]),
            len(result["speakers"]),
        )

        return result

    def _execute_edit_step(
        self,
        stt_result: Dict[str, Any],
        output_dir: str,
        edit_config: Dict[str, Any],
        save_intermediate: bool,
    ) -> Dict[str, Any]:
        """Execute editing step"""
        from ..text import TranscriptEditor, SpeakerReplacer

        logger.info("Executing edit step")

        # Initialize editor and replacer
        editor = TranscriptEditor()
        replacer = SpeakerReplacer(editor)

        # Load STT results as transcript data
        transcript_data = {
            "utterances": stt_result["utterances"],
            "speakers": stt_result["speakers"],
            "duration": stt_result["duration"],
            "language": stt_result.get("language", "en"),
            "provider": stt_result.get("provider", "unknown"),
        }

        editor.transcript_data = transcript_data
        editor.original_data = transcript_data.copy()

        edit_result = {
            "original_utterances": len(transcript_data["utterances"]),
            "original_speakers": len(transcript_data["speakers"]),
            "modifications": [],
            "speaker_replacements": {},
            "text_edits": 0,
        }

        # Apply speaker mapping rules if provided
        speaker_mapping_file = edit_config.get("speaker_mapping_file")
        if speaker_mapping_file and Path(speaker_mapping_file).exists():
            logger.info("Loading speaker mapping rules: %s", speaker_mapping_file)
            replacer.load_replacement_rules(speaker_mapping_file)

            replacement_results = replacer.apply_replacement_rules()
            edit_result["speaker_replacements"] = replacement_results
            edit_result["modifications"].append(
                {
                    "type": "speaker_replacement",
                    "rules_applied": len(replacement_results),
                    "utterances_modified": sum(replacement_results.values()),
                }
            )

        # Apply automatic normalization if requested
        if edit_config.get("auto_normalize", False):
            logger.info("Applying automatic speaker name normalization")
            normalization_map = replacer.normalize_speaker_names(apply_changes=True)
            if normalization_map:
                edit_result["modifications"].append(
                    {
                        "type": "auto_normalization",
                        "normalizations": len(normalization_map),
                        "mapping": normalization_map,
                    }
                )

        # Apply bulk text replacements if provided
        bulk_replacements = edit_config.get("bulk_text_replacements", [])
        for replacement in bulk_replacements:
            search_pattern = replacement.get("search")
            replacement_text = replacement.get("replace")
            use_regex = replacement.get("regex", False)
            speaker_filter = replacement.get("speaker_filter")

            if search_pattern and replacement_text:
                modified_count = replacer.editor.bulk_text_replace(
                    search_pattern, replacement_text, use_regex, speaker_filter
                )

                edit_result["text_edits"] += modified_count
                edit_result["modifications"].append(
                    {
                        "type": "bulk_text_replacement",
                        "pattern": search_pattern,
                        "replacement": replacement_text,
                        "utterances_modified": modified_count,
                    }
                )

        # Get final statistics
        final_stats = editor.get_statistics()
        edit_result.update(
            {
                "final_utterances": final_stats.get("total_utterances", 0),
                "final_speakers": final_stats.get("speaker_count", 0),
                "final_duration": final_stats.get("total_duration", 0),
                "edit_history": editor.get_edit_history(),
            }
        )

        # Save edited transcript
        if save_intermediate:
            output_path = Path(output_dir)
            edited_transcript_file = (
                output_path / f"{Path(output_dir).name}_edited_transcript.json"
            )

            saved_file = editor.save_transcript(
                str(edited_transcript_file), format="json", backup_original=False
            )

            edit_result["edited_transcript_file"] = saved_file
            self.intermediate_files.append(saved_file)

        # Update transcript data for next step
        edit_result["edited_transcript_data"] = editor.transcript_data

        logger.info(
            "Edit step completed: %d modifications, %d final speakers",
            len(edit_result["modifications"]),
            edit_result["final_speakers"],
        )

        return edit_result

    def _execute_final_step(
        self,
        input_file: str,
        output_dir: str,
        edit_result: Dict[str, Any],
        language: str,
        save_intermediate: bool,
    ) -> Dict[str, Any]:
        """Execute final diarization step"""
        from ..audio.diarization import SpeakerDiarizer

        logger.info("Executing final diarization step")

        # Initialize final diarizer
        final_diarizer = SpeakerDiarizer(
            provider=self.final_provider, api_key=self.api_key
        )

        # Get number of speakers from edited transcript
        final_speakers_count = edit_result.get("final_speakers", 2)

        # Perform final speaker separation
        base_filename = f"{Path(input_file).stem}_final"

        result = final_diarizer.separate_speakers(
            audio_file=input_file,
            output_dir=output_dir,
            speakers_expected=final_speakers_count,
            language=language,
            enhanced_processing=True,
            base_filename=base_filename,
        )

        # Track output files
        if "speaker_files" in result:
            self.intermediate_files.extend(result["speaker_files"])
        if "transcript_file" in result:
            self.intermediate_files.append(result["transcript_file"])

        logger.info(
            "Final step completed: %d speaker files, %.1f%% coverage",
            len(result.get("speaker_files", [])),
            result.get("stats", {}).get("total_coverage", 0),
        )

        return result

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate workflow summary"""
        summary = {
            "workflow_type": "edit",
            "providers_used": {"stt": self.stt_provider, "final": self.final_provider},
            "steps_completed": [],
        }

        # STT step summary
        if "stt" in self.results:
            stt_result = self.results["stt"]
            summary["steps_completed"].append(
                {
                    "step": 1,
                    "name": "Speech-to-Text",
                    "utterances_detected": len(stt_result.get("utterances", [])),
                    "speakers_detected": len(stt_result.get("speakers", [])),
                    "duration": stt_result.get("duration", 0),
                    "language": stt_result.get("language", "unknown"),
                }
            )

        # Edit step summary
        if "edit" in self.results:
            edit_result = self.results["edit"]
            summary["steps_completed"].append(
                {
                    "step": 2,
                    "name": "Text Editing",
                    "modifications_applied": len(edit_result.get("modifications", [])),
                    "speaker_replacements": len(
                        edit_result.get("speaker_replacements", {})
                    ),
                    "text_edits": edit_result.get("text_edits", 0),
                    "final_speakers": edit_result.get("final_speakers", 0),
                }
            )

        # Final step summary
        if "final" in self.results:
            final_result = self.results["final"]
            summary["steps_completed"].append(
                {
                    "step": 3,
                    "name": "Final Separation",
                    "speaker_files_created": len(final_result.get("speaker_files", [])),
                    "coverage_percentage": final_result.get("stats", {}).get(
                        "total_coverage", 0
                    ),
                    "total_speaker_duration": final_result.get("stats", {}).get(
                        "total_speaker_duration", 0
                    ),
                }
            )

        return summary

    def _save_workflow_metadata(self, output_dir: str, result: Dict[str, Any]) -> str:
        """Save workflow metadata to JSON file"""
        output_path = Path(output_dir)
        metadata_file = (
            output_path
            / f"edit_workflow_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        # Prepare metadata (remove large objects)
        metadata = {
            "workflow_type": "edit",
            "timestamp": datetime.now().isoformat(),
            "providers": {"stt": self.stt_provider, "final": self.final_provider},
            "summary": result.get("summary", {}),
            "total_time": result.get("total_time", 0),
            "steps_completed": result.get("steps_completed", 0),
            "intermediate_files": result.get("intermediate_files", []),
            "success": result.get("workflow_success", False),
        }

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info("Saved workflow metadata: %s", metadata_file)
        return str(metadata_file)

    def _cleanup_intermediate_files(self) -> None:
        """Clean up intermediate files"""
        cleaned_count = 0

        for file_path in self.intermediate_files:
            try:
                if Path(file_path).exists():
                    Path(file_path).unlink()
                    cleaned_count += 1
            except Exception as e:
                logger.warning("Failed to cleanup file %s: %s", file_path, e)

        logger.info("Cleaned up %d intermediate files", cleaned_count)
        self.intermediate_files.clear()

    def _report_progress(self, message: str, progress: float) -> None:
        """Report progress to callback if available"""
        logger.info("Progress (%.1f%%): %s", progress * 100, message)

        if self.progress_callback:
            try:
                self.progress_callback(message, progress)
            except Exception as e:
                logger.warning("Progress callback failed: %s", e)

    @staticmethod
    def create_edit_config_template(output_file: str) -> str:
        """
        Create a template configuration file for edit workflow

        Args:
            output_file: Path to output template file

        Returns:
            Path to created template file
        """
        template = {
            "_instructions": [
                "Edit workflow configuration template",
                "Customize the settings below for your workflow needs",
                "Remove sections you don't need",
            ],
            "speaker_mapping_file": "path/to/speaker_mapping.json",
            "auto_normalize": True,
            "bulk_text_replacements": [
                {"search": "um", "replace": "", "regex": False, "speaker_filter": None},
                {
                    "search": "\\b(uh|uhm|er)\\b",
                    "replace": "",
                    "regex": True,
                    "speaker_filter": None,
                },
            ],
            "quality_settings": {
                "stt_quality": "high",
                "final_enhanced_processing": True,
            },
            "output_options": {
                "save_intermediate": True,
                "cleanup_intermediate": False,
                "include_metadata": True,
            },
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2, ensure_ascii=False)

        logger.info("Created edit config template: %s", output_file)
        return output_file
