"""
Speaker Replacement Module
Advanced functionality for replacing speaker identities in transcripts
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Set
import re
from pathlib import Path
import json

from .editor import TranscriptEditor

logger = logging.getLogger(__name__)


class SpeakerReplacer:
    """
    Advanced speaker replacement system for transcript manipulation
    Handles speaker ID mapping, name normalization, and intelligent replacement
    """
    
    def __init__(self, transcript_editor: Optional[TranscriptEditor] = None):
        """
        Initialize speaker replacer
        
        Args:
            transcript_editor: Optional existing transcript editor instance
        """
        self.editor = transcript_editor or TranscriptEditor()
        self.replacement_rules = {}
        self.speaker_aliases = {}
        self.forbidden_replacements = set()
        
    def load_replacement_rules(self, rules_file: str) -> None:
        """
        Load speaker replacement rules from JSON file
        
        Args:
            rules_file: Path to JSON file with replacement rules
            
        Expected format:
        {
            "replacements": {
                "speaker_0": "Alice",
                "speaker_1": "Bob"
            },
            "aliases": {
                "A": ["Alice", "ALICE", "alice"],
                "B": ["Bob", "BOB", "bob"]
            },
            "forbidden": ["admin", "system", "unknown"]
        }
        """
        if not Path(rules_file).exists():
            raise FileNotFoundError(f"Rules file not found: {rules_file}")
            
        with open(rules_file, 'r', encoding='utf-8') as f:
            rules = json.load(f)
            
        self.replacement_rules = rules.get('replacements', {})
        self.speaker_aliases = rules.get('aliases', {})
        self.forbidden_replacements = set(rules.get('forbidden', []))
        
        logger.info("Loaded replacement rules: %d replacements, %d aliases, %d forbidden", 
                   len(self.replacement_rules), len(self.speaker_aliases), len(self.forbidden_replacements))
    
    def save_replacement_rules(self, rules_file: str) -> None:
        """
        Save current replacement rules to JSON file
        
        Args:
            rules_file: Path to output JSON file
        """
        rules = {
            'replacements': self.replacement_rules,
            'aliases': self.speaker_aliases,
            'forbidden': list(self.forbidden_replacements)
        }
        
        with open(rules_file, 'w', encoding='utf-8') as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)
            
        logger.info("Saved replacement rules to: %s", rules_file)
    
    def add_replacement_rule(
        self, 
        old_speaker: str, 
        new_speaker: str,
        validate: bool = True
    ) -> bool:
        """
        Add a speaker replacement rule
        
        Args:
            old_speaker: Original speaker ID
            new_speaker: New speaker ID
            validate: Whether to validate the replacement
            
        Returns:
            True if rule added successfully
        """
        if validate:
            if not self._validate_replacement(old_speaker, new_speaker):
                return False
                
        self.replacement_rules[old_speaker] = new_speaker
        logger.info("Added replacement rule: '%s' -> '%s'", old_speaker, new_speaker)
        return True
    
    def add_speaker_aliases(self, canonical_name: str, aliases: List[str]) -> None:
        """
        Add aliases for a canonical speaker name
        
        Args:
            canonical_name: The canonical/preferred speaker name
            aliases: List of alternative names/spellings
        """
        if canonical_name not in self.speaker_aliases:
            self.speaker_aliases[canonical_name] = []
            
        for alias in aliases:
            if alias not in self.speaker_aliases[canonical_name]:
                self.speaker_aliases[canonical_name].append(alias)
                
        logger.info("Added %d aliases for speaker '%s'", len(aliases), canonical_name)
    
    def normalize_speaker_names(self, apply_changes: bool = True) -> Dict[str, str]:
        """
        Normalize speaker names using aliases and common patterns
        
        Args:
            apply_changes: Whether to apply changes to transcript
            
        Returns:
            Dictionary of normalization mappings applied
        """
        if self.editor.transcript_data is None:
            raise ValueError("No transcript loaded in editor")
            
        utterances = self.editor.transcript_data.get('utterances', [])
        if not utterances:
            return {}
            
        # Find all current speakers
        current_speakers = set(u.get('speaker', '') for u in utterances)
        normalization_map = {}
        
        # Apply alias-based normalization
        for canonical, aliases in self.speaker_aliases.items():
            for speaker in current_speakers:
                if speaker in aliases and speaker != canonical:
                    if self._validate_replacement(speaker, canonical):
                        normalization_map[speaker] = canonical
        
        # Apply pattern-based normalization
        pattern_map = self._find_pattern_normalizations(current_speakers)
        normalization_map.update(pattern_map)
        
        # Apply changes if requested
        if apply_changes:
            for old_speaker, new_speaker in normalization_map.items():
                self.editor.replace_speaker_id(old_speaker, new_speaker)
                
        logger.info("Normalized %d speaker names", len(normalization_map))
        return normalization_map
    
    def apply_replacement_rules(self, selective_rules: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Apply all or selected replacement rules to transcript
        
        Args:
            selective_rules: Optional list of old speaker IDs to process
            
        Returns:
            Dictionary mapping old speaker IDs to number of utterances changed
        """
        if self.editor.transcript_data is None:
            raise ValueError("No transcript loaded in editor")
            
        results = {}
        rules_to_apply = self.replacement_rules
        
        if selective_rules:
            rules_to_apply = {k: v for k, v in self.replacement_rules.items() 
                            if k in selective_rules}
        
        for old_speaker, new_speaker in rules_to_apply.items():
            count = self.editor.replace_speaker_id(old_speaker, new_speaker)
            results[old_speaker] = count
            
        logger.info("Applied %d replacement rules", len(rules_to_apply))
        return results
    
    def smart_speaker_replacement(
        self,
        speaker_mapping: Dict[str, str],
        confidence_threshold: float = 0.8,
        preview_only: bool = False
    ) -> Dict[str, Any]:
        """
        Intelligent speaker replacement with confidence scoring
        
        Args:
            speaker_mapping: Dictionary of old -> new speaker mappings
            confidence_threshold: Minimum confidence to apply replacement
            preview_only: If True, only return preview without applying changes
            
        Returns:
            Dictionary with replacement results and confidence scores
        """
        if self.editor.transcript_data is None:
            raise ValueError("No transcript loaded in editor")
            
        utterances = self.editor.transcript_data.get('utterances', [])
        if not utterances:
            return {'replacements': {}, 'confidence_scores': {}, 'applied': 0}
            
        # Analyze replacement confidence
        confidence_scores = {}
        for old_speaker, new_speaker in speaker_mapping.items():
            confidence = self._calculate_replacement_confidence(old_speaker, new_speaker, utterances)
            confidence_scores[old_speaker] = confidence
            
        # Apply replacements above threshold
        applied_replacements = {}
        total_applied = 0
        
        for old_speaker, new_speaker in speaker_mapping.items():
            confidence = confidence_scores.get(old_speaker, 0.0)
            
            if confidence >= confidence_threshold:
                if not preview_only:
                    count = self.editor.replace_speaker_id(old_speaker, new_speaker)
                    applied_replacements[old_speaker] = count
                    total_applied += count
                else:
                    # Count what would be replaced
                    count = sum(1 for u in utterances if u.get('speaker') == old_speaker)
                    applied_replacements[old_speaker] = count
            else:
                logger.warning("Skipping replacement '%s' -> '%s' (confidence: %.2f < %.2f)", 
                             old_speaker, new_speaker, confidence, confidence_threshold)
        
        result = {
            'replacements': applied_replacements,
            'confidence_scores': confidence_scores,
            'applied': total_applied,
            'preview_only': preview_only
        }
        
        return result
    
    def detect_speaker_patterns(self) -> Dict[str, Any]:
        """
        Detect patterns in speaker naming and suggest improvements
        
        Returns:
            Dictionary with detected patterns and suggestions
        """
        if self.editor.transcript_data is None:
            raise ValueError("No transcript loaded in editor")
            
        utterances = self.editor.transcript_data.get('utterances', [])
        if not utterances:
            return {}
            
        speakers = [u.get('speaker', '') for u in utterances]
        unique_speakers = set(speakers)
        
        patterns = {
            'generic_names': [],
            'numbered_speakers': [],
            'case_inconsistencies': {},
            'similar_names': [],
            'suggestions': []
        }
        
        # Detect generic/system names
        generic_patterns = [r'^speaker[_\s]*\d+$', r'^person[_\s]*\d+$', r'^voice[_\s]*\d+$', 
                          r'^speaker[_\s]*[a-z]$', r'^unknown$', r'^unnamed$']
        
        for speaker in unique_speakers:
            for pattern in generic_patterns:
                if re.match(pattern, speaker.lower()):
                    patterns['generic_names'].append(speaker)
                    break
        
        # Detect numbered speakers
        for speaker in unique_speakers:
            if re.match(r'^.*\d+$', speaker):
                patterns['numbered_speakers'].append(speaker)
        
        # Detect case inconsistencies
        speaker_lower_map = {}
        for speaker in unique_speakers:
            lower = speaker.lower()
            if lower in speaker_lower_map:
                if lower not in patterns['case_inconsistencies']:
                    patterns['case_inconsistencies'][lower] = []
                patterns['case_inconsistencies'][lower].append(speaker)
            else:
                speaker_lower_map[lower] = speaker
        
        # Only keep case inconsistencies with multiple variants
        patterns['case_inconsistencies'] = {k: v for k, v in patterns['case_inconsistencies'].items() if len(v) > 1}
        
        # Detect similar names (Levenshtein distance)
        similar_pairs = self._find_similar_speaker_names(unique_speakers)
        patterns['similar_names'] = similar_pairs
        
        # Generate suggestions
        suggestions = []
        
        if patterns['generic_names']:
            suggestions.append(f"Consider replacing {len(patterns['generic_names'])} generic speaker names with meaningful names")
            
        if patterns['case_inconsistencies']:
            suggestions.append(f"Normalize case inconsistencies for {len(patterns['case_inconsistencies'])} speaker name groups")
            
        if patterns['similar_names']:
            suggestions.append(f"Review {len(patterns['similar_names'])} pairs of similar speaker names for potential merging")
            
        patterns['suggestions'] = suggestions
        
        return patterns
    
    def generate_speaker_mapping_template(self, output_file: str) -> str:
        """
        Generate a template file for speaker mapping
        
        Args:
            output_file: Path to output template file
            
        Returns:
            Path to generated template file
        """
        if self.editor.transcript_data is None:
            raise ValueError("No transcript loaded in editor")
            
        utterances = self.editor.transcript_data.get('utterances', [])
        speakers = list(set(u.get('speaker', '') for u in utterances))
        
        # Generate statistics for each speaker
        speaker_stats = {}
        for speaker in speakers:
            speaker_utterances = [u for u in utterances if u.get('speaker') == speaker]
            total_duration = sum(u.get('end', 0) - u.get('start', 0) for u in speaker_utterances)
            sample_texts = [u.get('text', '')[:50] + '...' for u in speaker_utterances[:3]]
            
            speaker_stats[speaker] = {
                'utterance_count': len(speaker_utterances),
                'total_duration': total_duration,
                'sample_texts': sample_texts
            }
        
        # Generate template
        template = {
            '_instructions': [
                "This is a speaker mapping template.",
                "Replace the 'new_name' values with desired speaker names.",
                "Remove or comment out speakers you don't want to change.",
                "Use the statistics to help identify speakers."
            ],
            'replacements': {},
            'aliases': {},
            'forbidden': ['system', 'admin', 'unknown'],
            '_statistics': speaker_stats
        }
        
        # Add replacement placeholders
        for speaker in sorted(speakers):
            template['replacements'][speaker] = f"NEW_NAME_FOR_{speaker.upper()}"
            
        # Save template
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
            
        logger.info("Generated speaker mapping template: %s", output_file)
        return output_file
    
    def _validate_replacement(self, old_speaker: str, new_speaker: str) -> bool:
        """Validate if a speaker replacement is allowed"""
        if new_speaker in self.forbidden_replacements:
            logger.warning("Replacement blocked - '%s' is in forbidden list", new_speaker)
            return False
            
        if not new_speaker or not new_speaker.strip():
            logger.warning("Replacement blocked - new speaker name is empty")
            return False
            
        if old_speaker == new_speaker:
            logger.warning("Replacement blocked - old and new speaker names are identical")
            return False
            
        return True
    
    def _find_pattern_normalizations(self, speakers: Set[str]) -> Dict[str, str]:
        """Find pattern-based normalizations"""
        normalizations = {}
        
        # Group speakers by lowercase version
        lowercase_groups = {}
        for speaker in speakers:
            lower = speaker.lower().strip()
            if lower not in lowercase_groups:
                lowercase_groups[lower] = []
            lowercase_groups[lower].append(speaker)
        
        # For each group with multiple variants, choose the "best" one
        for lower, variants in lowercase_groups.items():
            if len(variants) > 1:
                # Choose the variant with proper capitalization or most common format
                best_variant = self._choose_best_speaker_variant(variants)
                for variant in variants:
                    if variant != best_variant:
                        normalizations[variant] = best_variant
        
        return normalizations
    
    def _choose_best_speaker_variant(self, variants: List[str]) -> str:
        """Choose the best variant from a list of similar speaker names"""
        # Prefer properly capitalized names
        for variant in variants:
            if variant.istitle() and len(variant) > 1:
                return variant
                
        # Prefer names without numbers or underscores
        for variant in variants:
            if not re.search(r'[\d_]', variant):
                return variant
                
        # Prefer shorter names
        return min(variants, key=len)
    
    def _calculate_replacement_confidence(
        self, 
        old_speaker: str, 
        new_speaker: str, 
        utterances: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for a speaker replacement"""
        confidence = 0.5  # Base confidence
        
        # Check if speakers exist in transcript
        old_count = sum(1 for u in utterances if u.get('speaker') == old_speaker)
        if old_count == 0:
            return 0.0  # No utterances to replace
            
        # Higher confidence for more utterances
        if old_count > 10:
            confidence += 0.2
        elif old_count > 5:
            confidence += 0.1
            
        # Check if new speaker already exists (lower confidence for conflicts)
        new_count = sum(1 for u in utterances if u.get('speaker') == new_speaker)
        if new_count > 0:
            confidence -= 0.3  # Penalty for potential conflicts
            
        # Check name quality
        if self._is_generic_name(old_speaker):
            confidence += 0.2  # Good to replace generic names
            
        if self._is_generic_name(new_speaker):
            confidence -= 0.2  # Don't replace with generic names
            
        # Check aliases
        for canonical, aliases in self.speaker_aliases.items():
            if new_speaker == canonical and old_speaker in aliases:
                confidence += 0.3  # Strong confidence for alias matches
                
        return max(0.0, min(1.0, confidence))
    
    def _is_generic_name(self, speaker: str) -> bool:
        """Check if a speaker name is generic/system-generated"""
        generic_patterns = [
            r'^speaker[_\s]*\d+$', r'^person[_\s]*\d+$', r'^voice[_\s]*\d+$',
            r'^speaker[_\s]*[a-z]$', r'^unknown$', r'^unnamed$', r'^user\d*$'
        ]
        
        for pattern in generic_patterns:
            if re.match(pattern, speaker.lower()):
                return True
                
        return False
    
    def _find_similar_speaker_names(self, speakers: Set[str]) -> List[Tuple[str, str, float]]:
        """Find pairs of similar speaker names using edit distance"""
        similar_pairs = []
        speakers_list = list(speakers)
        
        for i in range(len(speakers_list)):
            for j in range(i + 1, len(speakers_list)):
                speaker1 = speakers_list[i]
                speaker2 = speakers_list[j]
                
                # Calculate similarity (simple approach)
                similarity = self._calculate_name_similarity(speaker1, speaker2)
                if similarity > 0.7:  # Threshold for similarity
                    similar_pairs.append((speaker1, speaker2, similarity))
        
        return similar_pairs
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names (0.0 to 1.0)"""
        if name1 == name2:
            return 1.0
            
        # Simple approach: check common prefixes, suffixes, and character overlap
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        # Check if one is contained in the other
        if name1_lower in name2_lower or name2_lower in name1_lower:
            return 0.8
            
        # Check character overlap
        set1 = set(name1_lower)
        set2 = set(name2_lower)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
            
        return intersection / union