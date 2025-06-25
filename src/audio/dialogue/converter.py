"""
Transcript to Dialogue Converter
Converts STT transcripts with timelines to dialogue format with voice IDs
"""

import re
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json

from .base import DialogueBase, DialogueSegment
from .profile import CharacterManager
from ..providers.elevenlabs.voice import VoiceManager


class TranscriptToDialogueConverter:
    """
    Converts STT transcripts with speaker timelines to dialogue format for TTS generation
    """

    def __init__(self, voice_manager: Optional[VoiceManager] = None):
        self.voice_manager = voice_manager
        self.character_manager = CharacterManager(voice_manager)

    def parse_transcript_file(self, transcript_file: Path) -> List[DialogueSegment]:
        """
        Parse transcript file with speaker timelines

        Expected format:
        speaker_0 [0.1s - 8.6s]: text content here
        speaker_1 [8.6s - 9.4s]: more text content

        Args:
            transcript_file: Path to transcript file

        Returns:
            List of dialogue segments
        """
        segments = []

        with open(transcript_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse each line with the pattern: speaker_id [start_time - end_time]: text
        pattern = r"(\w+)\s*\[([0-9.]+)s\s*-\s*([0-9.]+)s\]:\s*(.+)"

        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            match = re.match(pattern, line)
            if match:
                speaker_id, start_time_str, end_time_str, text = match.groups()

                start_time = float(start_time_str)
                end_time = float(end_time_str)
                duration = end_time - start_time

                segment = DialogueSegment(
                    speaker_id=speaker_id,
                    text=text.strip(),
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                )

                segments.append(segment)

        return segments

    def parse_stt_json(self, stt_json: Dict[str, Any]) -> List[DialogueSegment]:
        """
        Parse STT JSON output to dialogue segments

        Args:
            stt_json: STT provider JSON output

        Returns:
            List of dialogue segments
        """
        segments = []

        # Handle different STT output formats
        if "utterances" in stt_json:
            # ElevenLabs STT format
            for utterance in stt_json["utterances"]:
                segment = DialogueSegment(
                    speaker_id=utterance.get("speaker", "unknown"),
                    text=utterance.get("text", ""),
                    start_time=utterance.get("start", 0.0),
                    end_time=utterance.get("end", 0.0),
                    duration=utterance.get("end", 0.0) - utterance.get("start", 0.0),
                    confidence=utterance.get("confidence", 1.0),
                )
                segments.append(segment)

        elif "segments" in stt_json:
            # Generic segments format
            for segment_data in stt_json["segments"]:
                segment = DialogueSegment(
                    speaker_id=segment_data.get("speaker_id", "unknown"),
                    text=segment_data.get("text", ""),
                    start_time=segment_data.get("start_time", 0.0),
                    end_time=segment_data.get("end_time", 0.0),
                    duration=segment_data.get("duration", 0.0),
                    confidence=segment_data.get("confidence", 1.0),
                )
                segments.append(segment)

        return segments

    def apply_voice_mapping(
        self, segments: List[DialogueSegment], voice_mapping: Dict[str, str]
    ) -> List[DialogueSegment]:
        """
        Apply voice ID mapping to dialogue segments

        Args:
            segments: List of dialogue segments
            voice_mapping: Dictionary mapping speaker_id -> voice_id

        Returns:
            Updated segments with voice IDs
        """
        for segment in segments:
            if segment.speaker_id in voice_mapping:
                segment.voice_id = voice_mapping[segment.speaker_id]

        return segments

    def auto_assign_voices(
        self,
        segments: List[DialogueSegment],
        language: str = "en",
        gender_preferences: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[DialogueSegment], Dict[str, str]]:
        """
        Automatically assign voice IDs to speakers

        Args:
            segments: List of dialogue segments
            language: Target language for voices
            gender_preferences: Optional gender preferences for speakers

        Returns:
            Tuple of (updated segments, voice mapping used)
        """
        if not self.voice_manager:
            return segments, {}

        # Get unique speakers
        speakers = list(set(segment.speaker_id for segment in segments))

        # Create voice mapping using enhanced voice manager
        voice_mapping = self.voice_manager.create_dialogue_voice_mapping(
            speakers=speakers, language=language, gender_preferences=gender_preferences
        )

        # Apply mapping to segments
        updated_segments = self.apply_voice_mapping(segments, voice_mapping)

        return updated_segments, voice_mapping

    def convert_transcript_to_dialogue(
        self,
        transcript_source: str,  # File path or JSON string
        language: str = "en",
        custom_voice_mapping: Optional[Dict[str, str]] = None,
        gender_preferences: Optional[Dict[str, str]] = None,
        auto_assign: bool = True,
    ) -> DialogueBase:
        """
        Convert transcript to dialogue format with voice assignments

        Args:
            transcript_source: Path to transcript file or JSON string
            language: Target language for voice selection
            custom_voice_mapping: Optional custom speaker -> voice mapping
            gender_preferences: Optional gender preferences for auto-assignment
            auto_assign: Whether to auto-assign voices if not in custom mapping

        Returns:
            DialogueBase object with voice-assigned segments
        """
        segments = []

        # Determine source type and parse accordingly
        if transcript_source.startswith("{") or transcript_source.startswith("["):
            # JSON string
            try:
                stt_json = json.loads(transcript_source)
                segments = self.parse_stt_json(stt_json)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format in transcript source")
        else:
            # File path
            transcript_path = Path(transcript_source)
            if not transcript_path.exists():
                raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

            if transcript_path.suffix.lower() == ".json":
                with open(transcript_path, "r", encoding="utf-8") as f:
                    stt_json = json.load(f)
                segments = self.parse_stt_json(stt_json)
            else:
                # Text file with timeline format
                segments = self.parse_transcript_file(transcript_path)

        if not segments:
            raise ValueError("No dialogue segments found in transcript source")

        # Apply voice mapping
        voice_mapping_used = {}

        if custom_voice_mapping:
            segments = self.apply_voice_mapping(segments, custom_voice_mapping)
            voice_mapping_used.update(custom_voice_mapping)

        if auto_assign:
            # Auto-assign voices for unmapped speakers
            unmapped_speakers = [seg.speaker_id for seg in segments if not seg.voice_id]
            if unmapped_speakers:
                auto_segments, auto_mapping = self.auto_assign_voices(
                    [seg for seg in segments if not seg.voice_id],
                    language=language,
                    gender_preferences=gender_preferences,
                )
                voice_mapping_used.update(auto_mapping)

                # Update original segments with auto-assigned voices
                for segment in segments:
                    if not segment.voice_id and segment.speaker_id in auto_mapping:
                        segment.voice_id = auto_mapping[segment.speaker_id]

        # Create dialogue object
        dialogue = DialogueBase(segments)
        dialogue.metadata = {
            "source": transcript_source,
            "language": language,
            "voice_mapping": voice_mapping_used,
            "conversion_timestamp": str(Path().absolute()),
            "total_segments": len(segments),
            "speakers": list(set(seg.speaker_id for seg in segments)),
        }

        return dialogue

    def optimize_dialogue_for_tts(self, dialogue: DialogueBase) -> DialogueBase:
        """
        Optimize dialogue for TTS generation

        Args:
            dialogue: Original dialogue object

        Returns:
            Optimized dialogue object
        """
        optimized_segments = []

        for segment in dialogue.segments:
            # Clean up text for better TTS
            cleaned_text = self._clean_text_for_tts(segment.text)

            # Create optimized segment
            optimized_segment = DialogueSegment(
                speaker_id=segment.speaker_id,
                text=cleaned_text,
                voice_id=segment.voice_id,
                start_time=segment.start_time,
                end_time=segment.end_time,
                duration=segment.duration,
                confidence=segment.confidence,
                emotion=segment.emotion,
                style=segment.style,
            )

            optimized_segments.append(optimized_segment)

        # Create optimized dialogue
        optimized_dialogue = DialogueBase(optimized_segments)
        optimized_dialogue.metadata = dialogue.metadata.copy()
        optimized_dialogue.metadata["optimized"] = True

        return optimized_dialogue

    def _clean_text_for_tts(self, text: str) -> str:
        """
        Clean text for optimal TTS generation

        Args:
            text: Original text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Handle common TTS issues
        # Add pause for long sentences
        text = re.sub(r"([.!?])\s*([A-Z])", r"\\1 \\2", text)

        # Normalize punctuation
        text = re.sub(r"[.]{2,}", "...", text)
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)

        return text

    def export_for_elevenlabs_dialogue_api(
        self, dialogue: DialogueBase, output_file: Optional[Path] = None
    ) -> str:
        """
        Export dialogue in ElevenLabs Text to Dialogue API format

        Args:
            dialogue: Dialogue object to export
            output_file: Optional file to save the export

        Returns:
            JSON string in ElevenLabs format
        """
        # Validate that all segments have voice IDs
        missing_voices = dialogue.validate_voice_ids()
        if missing_voices:
            raise ValueError(f"Missing voice IDs for speakers: {missing_voices}")

        # Convert to ElevenLabs format
        elevenlabs_format = dialogue.to_elevenlabs_dialogue_format()

        # Create final export structure
        export_data = {
            "dialogue": elevenlabs_format,
            "metadata": dialogue.metadata,
            "total_segments": len(elevenlabs_format),
            "estimated_duration": dialogue.get_total_duration(),
        }

        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(json_str)

        return json_str

    def analyze_dialogue_complexity(self, dialogue: DialogueBase) -> Dict[str, Any]:
        """
        Analyze dialogue complexity for TTS optimization

        Args:
            dialogue: Dialogue to analyze

        Returns:
            Complexity analysis results
        """
        segments = dialogue.segments

        # Basic statistics
        total_segments = len(segments)
        unique_speakers = len(dialogue.get_speakers())
        total_words = sum(len(seg.text.split()) for seg in segments)
        avg_segment_length = total_words / total_segments if total_segments > 0 else 0

        # Timing analysis
        if all(seg.duration for seg in segments):
            total_duration = sum(seg.duration for seg in segments)
            avg_segment_duration = total_duration / total_segments
            words_per_minute = (total_words / total_duration) * 60 if total_duration > 0 else 0
        else:
            total_duration = None
            avg_segment_duration = None
            words_per_minute = None

        # Complexity indicators
        long_segments = sum(1 for seg in segments if len(seg.text.split()) > 20)
        short_segments = sum(1 for seg in segments if len(seg.text.split()) < 3)

        # Language complexity (basic analysis)
        complex_punctuation = sum(
            1 for seg in segments if any(p in seg.text for p in ["...", "--", "()", "[]"])
        )

        analysis = {
            "basic_stats": {
                "total_segments": total_segments,
                "unique_speakers": unique_speakers,
                "total_words": total_words,
                "avg_segment_length": round(avg_segment_length, 2),
            },
            "timing_stats": {
                "total_duration": total_duration,
                "avg_segment_duration": (
                    round(avg_segment_duration, 2) if avg_segment_duration else None
                ),
                "estimated_wpm": (round(words_per_minute, 2) if words_per_minute else None),
            },
            "complexity_indicators": {
                "long_segments": long_segments,
                "short_segments": short_segments,
                "complex_punctuation": complex_punctuation,
                "complexity_score": (
                    (long_segments + complex_punctuation) / total_segments
                    if total_segments > 0
                    else 0
                ),
            },
            "recommendations": [],
        }

        # Add recommendations
        if long_segments > total_segments * 0.3:
            analysis["recommendations"].append(
                "Consider breaking down long segments for better TTS flow"
            )

        if short_segments > total_segments * 0.2:
            analysis["recommendations"].append(
                "Consider combining very short segments to improve naturalness"
            )

        if unique_speakers > 4:
            analysis["recommendations"].append(
                "Large number of speakers may require careful voice selection"
            )

        return analysis
