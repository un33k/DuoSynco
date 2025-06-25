"""
STT Audio Transcriber Module
High-level interface for transcribing audio files with speaker diarization
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from .providers.elevenlabs import ElevenLabsSTTProvider

logger = logging.getLogger(__name__)


class STTAudioTranscriber:
    """
    High-level STT audio transcription interface
    Transcribes audio/video files with speaker diarization using STT providers
    """

    def __init__(self, provider: str = "elevenlabs-stt", api_key: Optional[str] = None):
        """
        Initialize STT audio transcriber

        Args:
            provider: Provider name ('elevenlabs-stt', etc.)
            api_key: API key for the provider
        """
        if provider.lower() != "elevenlabs-stt":
            raise ValueError(
                f"Unsupported STT provider: {provider}. "
                "Currently only 'elevenlabs-stt' is supported for STT transcription."
            )

        self.provider_name = provider
        self.stt_provider = ElevenLabsSTTProvider(api_key=api_key)
        logger.info(
            "STT transcriber initialized with %s backend",
            self.stt_provider.provider_name,
        )

    def transcribe_audio_file(
        self,
        audio_file: str,
        output_dir: str = "output",
        base_filename: Optional[str] = None,
        speakers_expected: int = 2,
        language: str = "en",
        quality: str = "high",
        enhanced_processing: bool = True,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Transcribe audio file with speaker diarization

        Args:
            audio_file: Path to audio/video file
            output_dir: Directory to save output files
            base_filename: Base name for output files (auto-generated if None)
            speakers_expected: Number of speakers expected
            language: Language code (e.g., "en")
            quality: Quality level ('low', 'medium', 'high', 'ultra')
            enhanced_processing: Apply enhanced processing (uses experimental model)
            save_results: Whether to save transcript files

        Returns:
            Dictionary with results:
            {
                'transcript_text': str,       # Formatted transcript
                'utterances': List[Dict],     # List of utterance segments
                'speakers': List[str],        # List of detected speaker IDs
                'duration': float,            # Total duration in seconds
                'language': str,              # Detected/specified language
                'stats': Dict,                # Transcription statistics
                'provider': str,              # Provider used
                'files': List[str]            # Saved file paths (if save_results=True)
            }
        """
        logger.info("Starting STT transcription: %s", audio_file)

        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # Generate base filename if not provided
        if base_filename is None:
            base_filename = f"stt_{audio_path.stem}"

        # Get quality-based settings
        self._get_quality_settings(quality)

        # Perform STT transcription with diarization
        speaker_tracks, transcript_text, utterances = self.stt_provider.diarize(
            audio_file=audio_file,
            speakers_expected=speakers_expected,
            language=language,
            enhanced_processing=enhanced_processing,
        )

        # Calculate statistics
        stats = self._calculate_stats(utterances, transcript_text)
        total_duration = stats["total_duration"]

        # Prepare result
        result = {
            "transcript_text": transcript_text,
            "utterances": utterances,
            "speakers": list(speaker_tracks.keys()),
            "duration": total_duration,
            "language": language,
            "stats": stats,
            "provider": self.stt_provider.provider_name,
            "files": [],
        }

        # Save results if requested
        if save_results:
            saved_files = self.stt_provider.save_results(
                speaker_tracks=speaker_tracks,
                transcript_text=transcript_text,
                output_dir=output_dir,
                base_filename=base_filename,
            )
            result["files"] = saved_files

            # Also save JSON format for easy processing
            json_file = self._save_json_results(result, output_dir, base_filename)
            result["files"].append(json_file)

        logger.info("STT transcription completed successfully")
        logger.info(
            "Detected %d speakers, %.1fs duration",
            len(result["speakers"]),
            total_duration,
        )
        logger.info("Generated %d utterances", len(utterances))

        return result

    def transcribe_with_custom_settings(
        self,
        audio_file: str,
        model_id: str = "scribe_v1",
        language_code: Optional[str] = None,
        num_speakers: Optional[int] = None,
        diarize: bool = True,
        tag_audio_events: bool = False,
        timestamps_granularity: str = "word",
        output_dir: str = "output",
        base_filename: Optional[str] = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Transcribe audio with custom ElevenLabs STT settings

        Args:
            audio_file: Path to audio/video file
            model_id: Transcription model ('scribe_v1' or 'scribe_v1_experimental')
            language_code: ISO language code (auto-detected if None)
            num_speakers: Expected number of speakers (1-32, auto if None)
            diarize: Enable speaker diarization
            tag_audio_events: Tag non-speech audio events
            timestamps_granularity: 'none', 'word', or 'character'
            output_dir: Directory to save output files
            base_filename: Base name for output files
            save_results: Whether to save transcript files

        Returns:
            Dictionary with detailed transcription results
        """
        logger.info("Starting custom STT transcription: %s", audio_file)

        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # Generate base filename if not provided
        if base_filename is None:
            base_filename = f"stt_custom_{audio_path.stem}"

        # Direct STT transcription
        stt_result = self.stt_provider.transcribe_audio(
            audio_file=audio_file,
            model_id=model_id,
            language_code=language_code,
            num_speakers=num_speakers,
            diarize=diarize,
            tag_audio_events=tag_audio_events,
            timestamps_granularity=timestamps_granularity,
        )

        # Convert to standard format
        utterances, transcript_text, total_duration = (
            self.stt_provider.convert_to_diarization_format(stt_result)
        )

        # Create speaker tracks
        speaker_tracks = self.stt_provider.create_speaker_tracks_from_utterances(
            utterances, total_duration
        )

        # Calculate statistics
        stats = self._calculate_stats(utterances, transcript_text)

        # Prepare result
        result = {
            "raw_stt_result": stt_result,
            "transcript_text": transcript_text,
            "utterances": utterances,
            "speakers": list(speaker_tracks.keys()),
            "duration": total_duration,
            "language": stt_result.get("language_code", "unknown"),
            "language_confidence": stt_result.get("language_probability", 0.0),
            "stats": stats,
            "provider": self.stt_provider.provider_name,
            "model_used": model_id,
            "settings": {
                "diarize": diarize,
                "tag_audio_events": tag_audio_events,
                "timestamps_granularity": timestamps_granularity,
            },
            "files": [],
        }

        # Save results if requested
        if save_results:
            saved_files = self.stt_provider.save_results(
                speaker_tracks=speaker_tracks,
                transcript_text=transcript_text,
                output_dir=output_dir,
                base_filename=base_filename,
            )
            result["files"] = saved_files

            # Save detailed JSON results
            json_file = self._save_json_results(result, output_dir, base_filename)
            result["files"].append(json_file)

        logger.info(
            "Custom STT transcription completed: %d speakers, %.1fs",
            len(result["speakers"]),
            total_duration,
        )

        return result

    def _calculate_stats(
        self, utterances: List[Dict[str, Any]], transcript_text: str
    ) -> Dict[str, Any]:
        """Calculate transcription statistics"""
        if not utterances:
            return {
                "total_utterances": 0,
                "total_duration": 0.0,
                "speakers": {},
                "total_characters": len(transcript_text),
                "total_words": len(transcript_text.split()) if transcript_text else 0,
                "average_utterance_duration": 0.0,
            }

        stats = {
            "total_utterances": len(utterances),
            "speakers": {},
            "total_duration": 0.0,
            "total_characters": len(transcript_text),
            "total_words": len(transcript_text.split()) if transcript_text else 0,
        }

        # Calculate overall duration
        if utterances:
            stats["total_duration"] = max(u.get("end", 0) for u in utterances)

        # Calculate per-speaker statistics
        speaker_stats = {}
        total_speech_time = 0.0

        for utterance in utterances:
            speaker = utterance.get("speaker", "unknown")
            duration = utterance.get("end", 0) - utterance.get("start", 0)
            text = utterance.get("text", "")

            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "utterances": 0,
                    "total_duration": 0.0,
                    "characters": 0,
                    "words": 0,
                }

            speaker_stats[speaker]["utterances"] += 1
            speaker_stats[speaker]["total_duration"] += duration
            speaker_stats[speaker]["characters"] += len(text)
            speaker_stats[speaker]["words"] += len(text.split())
            total_speech_time += duration

        stats["speakers"] = speaker_stats
        stats["total_speech_time"] = total_speech_time
        stats["silence_time"] = stats["total_duration"] - total_speech_time
        stats["average_utterance_duration"] = (
            total_speech_time / len(utterances) if utterances else 0.0
        )

        return stats

    def _save_json_results(
        self, results: Dict[str, Any], output_dir: str, base_filename: str
    ) -> str:
        """Save results as JSON file"""
        import json

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Add debug suffix for transitory STT results files
        debug_suffix = (
            "_debug" if "_stt" in base_filename and not base_filename.endswith("_final") else ""
        )
        logger.info(
            f"📝 JSON file naming: base_filename='{base_filename}', debug_suffix='{debug_suffix}'"
        )
        json_file = output_path / f"{base_filename}_stt_results{debug_suffix}.json"

        # Prepare JSON-serializable data
        json_data = {
            "transcript_text": results["transcript_text"],
            "utterances": results["utterances"],
            "speakers": results["speakers"],
            "duration": results["duration"],
            "language": results["language"],
            "stats": results["stats"],
            "provider": results["provider"],
        }

        # Add optional fields if present
        if "language_confidence" in results:
            json_data["language_confidence"] = results["language_confidence"]
        if "model_used" in results:
            json_data["model_used"] = results["model_used"]
        if "settings" in results:
            json_data["settings"] = results["settings"]

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        logger.info("Saved JSON results: %s", json_file)
        return str(json_file)

    def _get_quality_settings(self, quality: str) -> Dict[str, Any]:
        """
        Get STT settings based on quality level

        Args:
            quality: Quality level ('low', 'medium', 'high', 'ultra')

        Returns:
            Dictionary with STT settings optimized for the quality level
        """
        quality_profiles = {
            "low": {
                "model_id": "scribe_v1",
                "enhanced_processing": False,
                "diarize": True,
                "tag_audio_events": False,
                "timestamps_granularity": "none",
            },
            "medium": {
                "model_id": "scribe_v1",
                "enhanced_processing": True,
                "diarize": True,
                "tag_audio_events": False,
                "timestamps_granularity": "word",
            },
            "high": {
                "model_id": "scribe_v1_experimental",
                "enhanced_processing": True,
                "diarize": True,
                "tag_audio_events": False,
                "timestamps_granularity": "word",
            },
            "ultra": {
                "model_id": "scribe_v1_experimental",
                "enhanced_processing": True,
                "diarize": True,
                "tag_audio_events": True,
                "timestamps_granularity": "character",
            },
        }

        return quality_profiles.get(quality.lower(), quality_profiles["high"])

    @staticmethod
    def supported_formats() -> List[str]:
        """
        Get list of supported audio/video formats

        Returns:
            List of supported file extensions
        """
        return [
            ".mp3",
            ".wav",
            ".m4a",
            ".aac",
            ".ogg",
            ".flac",  # Audio formats
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm",
            ".wmv",  # Video formats
        ]

    def validate_audio_file(self, audio_file: str) -> Dict[str, Any]:
        """
        Validate audio file for STT processing

        Args:
            audio_file: Path to audio/video file

        Returns:
            Dictionary with validation results
        """
        result = {
            "valid": False,
            "exists": False,
            "size_ok": False,
            "format_supported": False,
            "file_size_mb": 0.0,
            "file_extension": "",
            "errors": [],
        }

        try:
            # Check if file exists
            audio_path = Path(audio_file)
            if not audio_path.exists():
                result["errors"].append("File does not exist")
                return result
            result["exists"] = True

            # Check file size (1GB limit for ElevenLabs)
            file_size = audio_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            result["file_size_mb"] = file_size_mb

            if file_size > 1024 * 1024 * 1024:  # 1GB
                result["errors"].append(f"File size {file_size_mb:.1f}MB exceeds 1GB limit")
            else:
                result["size_ok"] = True

            # Check file format
            file_path = Path(audio_file)
            extension = file_path.suffix.lower()
            result["file_extension"] = extension

            if extension in self.supported_formats():
                result["format_supported"] = True
            else:
                result["errors"].append(f"Unsupported format: {extension}")

            # Overall validation
            result["valid"] = result["exists"] and result["size_ok"] and result["format_supported"]

        except Exception as e:
            result["errors"].append(f"Validation error: {str(e)}")

        return result
