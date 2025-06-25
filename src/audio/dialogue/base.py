"""
Base classes for dialogue functionality
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class DialogueSegment:
    """
    Represents a single dialogue segment with speaker and content
    """

    speaker_id: str
    text: str
    voice_id: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    confidence: Optional[float] = None
    emotion: Optional[str] = None
    style: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary format"""
        return {
            "speaker_id": self.speaker_id,
            "text": self.text,
            "voice_id": self.voice_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "confidence": self.confidence,
            "emotion": self.emotion,
            "style": self.style,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueSegment":
        """Create segment from dictionary"""
        return cls(**data)

    def get_elevenlabs_format(self) -> Dict[str, str]:
        """Convert to ElevenLabs Text to Dialogue API format"""
        result = {"speaker_id": self.voice_id or self.speaker_id, "text": self.text}

        # Add optional parameters if available
        if self.emotion:
            result["emotion"] = self.emotion
        if self.style:
            result["style"] = self.style

        return result


class DialogueBase:
    """
    Base class for dialogue operations
    """

    def __init__(self, segments: Optional[List[DialogueSegment]] = None):
        self.segments = segments or []
        self.metadata: Dict[str, Any] = {}

    def add_segment(self, segment: DialogueSegment) -> None:
        """Add a dialogue segment"""
        self.segments.append(segment)

    def get_speakers(self) -> List[str]:
        """Get unique speaker IDs"""
        return list(set(segment.speaker_id for segment in self.segments))

    def get_segments_by_speaker(self, speaker_id: str) -> List[DialogueSegment]:
        """Get all segments for a specific speaker"""
        return [segment for segment in self.segments if segment.speaker_id == speaker_id]

    def get_total_duration(self) -> Optional[float]:
        """Calculate total dialogue duration"""
        if not self.segments or not all(seg.duration for seg in self.segments):
            return None
        return sum(seg.duration for seg in self.segments if seg.duration)

    def to_elevenlabs_dialogue_format(self) -> List[Dict[str, str]]:
        """Convert to ElevenLabs Text to Dialogue API format"""
        return [segment.get_elevenlabs_format() for segment in self.segments]

    def to_json(self, file_path: Optional[Path] = None) -> str:
        """Export dialogue to JSON format"""
        data = {
            "metadata": self.metadata,
            "segments": [segment.to_dict() for segment in self.segments],
        }

        json_str = json.dumps(data, indent=2, ensure_ascii=False)

        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json_str)

        return json_str

    @classmethod
    def from_json(cls, json_data: str) -> "DialogueBase":
        """Create dialogue from JSON data"""
        data = json.loads(json_data)

        dialogue = cls()
        dialogue.metadata = data.get("metadata", {})

        for segment_data in data.get("segments", []):
            segment = DialogueSegment.from_dict(segment_data)
            dialogue.add_segment(segment)

        return dialogue

    @classmethod
    def from_json_file(cls, file_path: Path) -> "DialogueBase":
        """Load dialogue from JSON file"""
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = f.read()
        return cls.from_json(json_data)

    def replace_speaker_voice_ids(self, voice_mapping: Dict[str, str]) -> None:
        """
        Replace speaker IDs with voice IDs using provided mapping

        Args:
            voice_mapping: Dictionary mapping speaker_id -> voice_id
        """
        for segment in self.segments:
            if segment.speaker_id in voice_mapping:
                segment.voice_id = voice_mapping[segment.speaker_id]

    def validate_voice_ids(self) -> List[str]:
        """
        Validate that all segments have voice IDs assigned

        Returns:
            List of speaker IDs missing voice assignments
        """
        missing_voices = []
        for segment in self.segments:
            if not segment.voice_id:
                if segment.speaker_id not in missing_voices:
                    missing_voices.append(segment.speaker_id)
        return missing_voices

    def get_conversation_statistics(self) -> Dict[str, Any]:
        """Get statistics about the conversation"""
        speakers = self.get_speakers()
        stats = {
            "total_segments": len(self.segments),
            "unique_speakers": len(speakers),
            "speakers": {},
            "total_duration": self.get_total_duration(),
            "total_words": sum(len(seg.text.split()) for seg in self.segments),
        }

        for speaker in speakers:
            speaker_segments = self.get_segments_by_speaker(speaker)
            speaker_duration = sum(seg.duration for seg in speaker_segments if seg.duration)
            speaker_words = sum(len(seg.text.split()) for seg in speaker_segments)

            stats["speakers"][speaker] = {
                "segments": len(speaker_segments),
                "duration": speaker_duration,
                "words": speaker_words,
                "voice_id": speaker_segments[0].voice_id if speaker_segments else None,
            }

        return stats
