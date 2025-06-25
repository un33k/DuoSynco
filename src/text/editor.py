"""
Transcript Editor Module
Handles loading, editing, and saving transcript files with speaker manipulation capabilities
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import copy

logger = logging.getLogger(__name__)


class TranscriptEditor:
    """
    Transcript editor for manipulating speaker diarization results
    Supports loading, editing, speaker replacement, and saving transcripts
    """

    def __init__(self) -> None:
        """Initialize transcript editor"""
        self.transcript_data: Optional[Dict[str, Any]] = None
        self.original_data: Optional[Dict[str, Any]] = None
        self.edit_history: List[Dict[str, Any]] = []
        self.current_file: Optional[str] = None

    def load_transcript(self, file_path: str, format: str = "auto") -> Dict[str, Any]:
        """
        Load transcript from file

        Args:
            file_path: Path to transcript file
            format: File format ('json', 'txt', 'auto')

        Returns:
            Dictionary with loaded transcript data
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Transcript file not found: {file_path}")

        self.current_file = str(file_path)
        file_path_obj = Path(file_path)

        # Auto-detect format
        if format == "auto":
            format = self._detect_format(file_path_obj)

        logger.info("Loading transcript: %s (format: %s)", file_path, format)

        if format == "json":
            self.transcript_data = self._load_json_transcript(file_path_obj)
        elif format == "txt":
            self.transcript_data = self._load_text_transcript(file_path_obj)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Keep original copy for comparison
        self.original_data = copy.deepcopy(self.transcript_data)
        self.edit_history = []

        logger.info(
            "Loaded transcript with %d utterances",
            len(self.transcript_data.get("utterances", [])),
        )

        return self.transcript_data

    def save_transcript(
        self,
        file_path: Optional[str] = None,
        format: str = "json",
        backup_original: bool = True,
    ) -> str:
        """
        Save transcript to file

        Args:
            file_path: Output file path (uses current file if None)
            format: Output format ('json', 'txt')
            backup_original: Whether to backup original file

        Returns:
            Path to saved file
        """
        if self.transcript_data is None:
            raise ValueError("No transcript data loaded")

        if file_path is None:
            if self.current_file is None:
                raise ValueError("No file path specified and no current file")
            file_path = self.current_file

        file_path_obj = Path(file_path)

        # Backup original if requested
        if backup_original and file_path_obj.exists():
            backup_path = file_path_obj.with_suffix(
                f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_path_obj.suffix}"
            )
            logger.info("Creating backup: %s", backup_path)
            import shutil

            shutil.copy2(file_path_obj, backup_path)

        # Save in requested format
        if format == "json":
            self._save_json_transcript(file_path_obj)
        elif format == "txt":
            self._save_text_transcript(file_path_obj)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info("Saved transcript: %s", file_path)
        return str(file_path)

    def replace_speaker_id(
        self,
        old_speaker_id: str,
        new_speaker_id: str,
        utterance_range: Optional[Tuple[int, int]] = None,
    ) -> int:
        """
        Replace speaker ID in utterances

        Args:
            old_speaker_id: Current speaker ID to replace
            new_speaker_id: New speaker ID
            utterance_range: Optional range (start_idx, end_idx) to limit replacement

        Returns:
            Number of utterances modified
        """
        if self.transcript_data is None:
            raise ValueError("No transcript data loaded")

        utterances = self.transcript_data.get("utterances", [])
        if not utterances:
            return 0

        # Determine range
        start_idx = 0
        end_idx = len(utterances)
        if utterance_range:
            start_idx = max(0, utterance_range[0])
            end_idx = min(len(utterances), utterance_range[1])

        modified_count = 0

        for i in range(start_idx, end_idx):
            if utterances[i].get("speaker") == old_speaker_id:
                # Record edit
                self._record_edit(
                    "replace_speaker",
                    {
                        "utterance_index": i,
                        "old_speaker": old_speaker_id,
                        "new_speaker": new_speaker_id,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

                utterances[i]["speaker"] = new_speaker_id
                modified_count += 1

        # Update speakers list if present
        if "speakers" in self.transcript_data:
            speakers = self.transcript_data["speakers"]
            if old_speaker_id in speakers and new_speaker_id not in speakers:
                speakers.remove(old_speaker_id)
                speakers.append(new_speaker_id)

        logger.info(
            "Replaced speaker ID '%s' -> '%s' in %d utterances",
            old_speaker_id,
            new_speaker_id,
            modified_count,
        )

        return modified_count

    def edit_utterance_text(
        self, utterance_index: int, new_text: str, adjust_timing: bool = False
    ) -> bool:
        """
        Edit text content of specific utterance

        Args:
            utterance_index: Index of utterance to edit
            new_text: New text content
            adjust_timing: Whether to adjust timing based on text length

        Returns:
            True if successful, False otherwise
        """
        if self.transcript_data is None:
            raise ValueError("No transcript data loaded")

        utterances = self.transcript_data.get("utterances", [])
        if not utterances or utterance_index >= len(utterances):
            return False

        utterance = utterances[utterance_index]
        old_text = utterance.get("text", "")

        # Record edit
        self._record_edit(
            "edit_text",
            {
                "utterance_index": utterance_index,
                "old_text": old_text,
                "new_text": new_text,
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Update text
        utterance["text"] = new_text

        # Optionally adjust timing
        if adjust_timing:
            self._adjust_utterance_timing(utterance_index, old_text, new_text)

        logger.info(
            "Edited utterance %d text: '%s...' -> '%s...'",
            utterance_index,
            old_text[:30],
            new_text[:30],
        )

        return True

    def bulk_text_replace(
        self,
        search_pattern: str,
        replacement: str,
        use_regex: bool = False,
        speaker_filter: Optional[str] = None,
    ) -> int:
        """
        Perform bulk text replacement across all utterances

        Args:
            search_pattern: Text/pattern to search for
            replacement: Replacement text
            use_regex: Whether to use regex pattern matching
            speaker_filter: Only replace in utterances from this speaker

        Returns:
            Number of utterances modified
        """
        if self.transcript_data is None:
            raise ValueError("No transcript data loaded")

        utterances = self.transcript_data.get("utterances", [])
        if not utterances:
            return 0

        modified_count = 0

        for i, utterance in enumerate(utterances):
            # Skip if speaker filter doesn't match
            if speaker_filter and utterance.get("speaker") != speaker_filter:
                continue

            old_text = utterance.get("text", "")

            # Perform replacement
            if use_regex:
                try:
                    new_text = re.sub(search_pattern, replacement, old_text)
                except re.error as e:
                    logger.error("Regex error in utterance %d: %s", i, e)
                    continue
            else:
                new_text = old_text.replace(search_pattern, replacement)

            # Check if text changed
            if new_text != old_text:
                # Record edit
                self._record_edit(
                    "bulk_replace",
                    {
                        "utterance_index": i,
                        "search_pattern": search_pattern,
                        "replacement": replacement,
                        "old_text": old_text,
                        "new_text": new_text,
                        "use_regex": use_regex,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

                utterance["text"] = new_text
                modified_count += 1

        logger.info(
            "Bulk replace '%s' -> '%s': %d utterances modified",
            search_pattern,
            replacement,
            modified_count,
        )

        return modified_count

    def merge_utterances(self, start_index: int, end_index: int, separator: str = " ") -> bool:
        """
        Merge consecutive utterances from the same speaker

        Args:
            start_index: First utterance index
            end_index: Last utterance index (inclusive)
            separator: Text separator for merged content

        Returns:
            True if successful, False otherwise
        """
        if self.transcript_data is None:
            raise ValueError("No transcript data loaded")

        utterances = self.transcript_data.get("utterances", [])
        if not utterances or start_index >= len(utterances) or end_index >= len(utterances):
            return False

        if start_index >= end_index:
            return False

        # Check if all utterances are from same speaker
        speaker = utterances[start_index].get("speaker")
        for i in range(start_index, end_index + 1):
            if utterances[i].get("speaker") != speaker:
                logger.warning("Cannot merge utterances from different speakers")
                return False

        # Merge utterances
        merged_texts = []
        for i in range(start_index, end_index + 1):
            merged_texts.append(utterances[i].get("text", ""))

        merged_text = separator.join(merged_texts)
        start_time = utterances[start_index].get("start", 0)
        end_time = utterances[end_index].get("end", 0)

        # Record edit
        self._record_edit(
            "merge_utterances",
            {
                "start_index": start_index,
                "end_index": end_index,
                "merged_text": merged_text,
                "original_count": end_index - start_index + 1,
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Create merged utterance
        merged_utterance = {
            "speaker": speaker,
            "start": start_time,
            "end": end_time,
            "text": merged_text,
        }

        # Replace original utterances with merged one
        utterances[start_index : end_index + 1] = [merged_utterance]

        logger.info(
            "Merged %d utterances (indices %d-%d) for speaker %s",
            end_index - start_index + 1,
            start_index,
            end_index,
            speaker,
        )

        return True

    def split_utterance(
        self,
        utterance_index: int,
        split_points: List[Tuple[str, float]],
        new_speaker_id: Optional[str] = None,
    ) -> bool:
        """
        Split utterance into multiple parts

        Args:
            utterance_index: Index of utterance to split
            split_points: List of (text_part, relative_time) tuples
            new_speaker_id: Optional new speaker ID for split parts

        Returns:
            True if successful, False otherwise
        """
        if self.transcript_data is None:
            raise ValueError("No transcript data loaded")

        utterances = self.transcript_data.get("utterances", [])
        if not utterances or utterance_index >= len(utterances):
            return False

        original_utterance = utterances[utterance_index]
        original_speaker = original_utterance.get("speaker")
        original_start = original_utterance.get("start", 0)
        original_end = original_utterance.get("end", 0)
        original_duration = original_end - original_start

        # Create new utterances
        new_utterances = []
        current_time = original_start

        for i, (text_part, relative_time) in enumerate(split_points):
            part_duration = relative_time * original_duration
            speaker = new_speaker_id if new_speaker_id and i > 0 else original_speaker

            new_utterance = {
                "speaker": speaker,
                "start": current_time,
                "end": current_time + part_duration,
                "text": text_part.strip(),
            }
            new_utterances.append(new_utterance)
            current_time += part_duration

        # Record edit
        self._record_edit(
            "split_utterance",
            {
                "utterance_index": utterance_index,
                "original_text": original_utterance.get("text", ""),
                "split_count": len(split_points),
                "new_speaker_id": new_speaker_id,
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Replace original with split utterances
        utterances[utterance_index : utterance_index + 1] = new_utterances

        # Update speakers list if new speaker added
        if new_speaker_id and "speakers" in self.transcript_data:
            speakers = self.transcript_data["speakers"]
            if new_speaker_id not in speakers:
                speakers.append(new_speaker_id)

        logger.info("Split utterance %d into %d parts", utterance_index, len(split_points))

        return True

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about current transcript

        Returns:
            Dictionary with transcript statistics
        """
        if self.transcript_data is None:
            return {}

        utterances = self.transcript_data.get("utterances", [])
        if not utterances:
            return {"total_utterances": 0}

        # Calculate statistics
        stats = {
            "total_utterances": len(utterances),
            "speakers": {},
            "total_duration": 0.0,
            "total_text_length": 0,
            "edit_count": len(self.edit_history),
        }

        # Calculate overall duration
        if utterances:
            stats["total_duration"] = max(u.get("end", 0) for u in utterances)

        # Per-speaker statistics
        speaker_stats = {}
        for utterance in utterances:
            speaker = utterance.get("speaker", "unknown")
            text = utterance.get("text", "")
            duration = utterance.get("end", 0) - utterance.get("start", 0)

            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "utterances": 0,
                    "total_duration": 0.0,
                    "total_characters": 0,
                    "total_words": 0,
                }

            speaker_stats[speaker]["utterances"] += 1
            speaker_stats[speaker]["total_duration"] += duration
            speaker_stats[speaker]["total_characters"] += len(text)
            speaker_stats[speaker]["total_words"] += len(text.split())
            stats["total_text_length"] += len(text)

        stats["speakers"] = speaker_stats
        stats["speaker_count"] = len(speaker_stats)

        return stats

    def get_edit_history(self) -> List[Dict[str, Any]]:
        """
        Get history of edits made to transcript

        Returns:
            List of edit records
        """
        return copy.deepcopy(self.edit_history)

    def undo_last_edit(self) -> bool:
        """
        Undo the last edit (basic implementation)

        Returns:
            True if successful, False otherwise
        """
        if not self.edit_history or self.original_data is None:
            return False

        # For now, just restore from original
        # A more sophisticated implementation would maintain edit states
        logger.warning("Undo restores to original state - all edits will be lost")
        self.transcript_data = copy.deepcopy(self.original_data)
        self.edit_history = []

        return True

    def _detect_format(self, file_path: Path) -> str:
        """Detect file format based on extension and content"""
        extension = file_path.suffix.lower()

        if extension == ".json":
            return "json"
        elif extension in [".txt", ".text"]:
            return "txt"
        else:
            # Try to detect by content
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line.startswith("{") or first_line.startswith("["):
                        return "json"
                    else:
                        return "txt"
            except Exception:
                return "txt"  # Default to text

    def _load_json_transcript(self, file_path: Path) -> Dict[str, Any]:
        """Load transcript from JSON file"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate required fields
        if "utterances" not in data:
            raise ValueError("JSON transcript must contain 'utterances' field")

        return data

    def _load_text_transcript(self, file_path: Path) -> Dict[str, Any]:
        """Load transcript from text file and parse it"""
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        utterances = []

        # Parse text format: "Speaker [start-end]: text"
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Try to parse format: "Speaker [X.Xs - Y.Ys]: text"
            match = re.match(r"^(.+?)\s*\[(\d+\.?\d*)s?\s*-\s*(\d+\.?\d*)s?\]:\s*(.+)$", line)
            if match:
                speaker = match.group(1).strip()
                start_time = float(match.group(2))
                end_time = float(match.group(3))
                text = match.group(4).strip()

                utterances.append(
                    {
                        "speaker": speaker,
                        "start": start_time,
                        "end": end_time,
                        "text": text,
                    }
                )
            else:
                logger.warning("Could not parse line %d: %s", line_num + 1, line)

        # Extract unique speakers
        speakers = list(set(u["speaker"] for u in utterances))

        return {
            "utterances": utterances,
            "speakers": speakers,
            "total_duration": max(u["end"] for u in utterances) if utterances else 0.0,
            "source_file": str(file_path),
            "format": "text",
        }

    def _save_json_transcript(self, file_path: Path) -> None:
        """Save transcript to JSON file"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.transcript_data, f, indent=2, ensure_ascii=False)

    def _save_text_transcript(self, file_path: Path) -> None:
        """Save transcript to text file"""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# Edited Transcript\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Total utterances: {len(self.transcript_data.get('utterances', []))}\n\n")

            for utterance in self.transcript_data.get("utterances", []):
                speaker = utterance.get("speaker", "Unknown")
                start = utterance.get("start", 0)
                end = utterance.get("end", 0)
                text = utterance.get("text", "")

                f.write(f"{speaker} [{start:.1f}s - {end:.1f}s]: {text}\n")

    def _record_edit(self, edit_type: str, details: Dict[str, Any]) -> None:
        """Record an edit operation"""
        edit_record = {
            "type": edit_type,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        }
        self.edit_history.append(edit_record)

    def _adjust_utterance_timing(self, utterance_index: int, old_text: str, new_text: str) -> None:
        """Adjust utterance timing based on text length change"""
        if not old_text:
            return

        utterances = self.transcript_data.get("utterances", [])
        if utterance_index >= len(utterances):
            return

        utterance = utterances[utterance_index]

        # Simple timing adjustment based on character count ratio
        old_length = len(old_text)
        new_length = len(new_text)

        if old_length > 0:
            length_ratio = new_length / old_length
            original_duration = utterance.get("end", 0) - utterance.get("start", 0)
            new_duration = original_duration * length_ratio

            # Don't make durations too short or too long
            new_duration = max(0.5, min(new_duration, original_duration * 2))

            utterance["end"] = utterance.get("start", 0) + new_duration

            logger.debug(
                "Adjusted timing for utterance %d: %.1fs -> %.1fs",
                utterance_index,
                original_duration,
                new_duration,
            )
