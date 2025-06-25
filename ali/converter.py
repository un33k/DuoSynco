"""
Transcript format conversion utilities
"""

import json
from pathlib import Path
from typing import Dict, List, Any

from .config import AliConfig


class TranscriptConverter:
    """Handles conversion between different transcript formats"""
    
    def __init__(self, config: AliConfig):
        self.config = config
    
    def to_tts_format(self, input_file: Path) -> Path:
        """
        Convert transcript to TTS format
        
        Args:
            input_file: Path to input transcript file
            
        Returns:
            Path to converted TTS format file
        """
        output_file = input_file.parent / f"{input_file.stem}_tts_format.json"
        
        print(f"🔄 Converting transcript to TTS format...")
        
        # Read the transcript
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract utterances
        utterances = data.get('utterances', [])
        
        # Convert to TTS format
        tts_segments = []
        current_time = 0.0
        gap_duration = float(self.config.defaults['gap_duration'])
        
        for utterance in utterances:
            # Calculate duration based on text length
            text_length = len(utterance['text'])
            estimated_duration = max(1.0, text_length / 3.0)
            
            speaker = utterance['speaker']
            tts_segment = {
                'speaker': speaker,
                'start': current_time,
                'end': current_time + estimated_duration,
                'text': utterance['text'],
                'voice_id': self.config.voice_mapping.get(speaker, self.config.voice_mapping['speaker_0'])
            }
            
            tts_segments.append(tts_segment)
            current_time += estimated_duration + gap_duration
        
        # Create TTS-compatible file
        tts_data = {
            'metadata': {
                'source': input_file.name,
                'language': self.config.defaults['language'],
                'total_segments': len(tts_segments),
                'voice_mapping': self.config.voice_mapping
            },
            'segments': tts_segments
        }
        
        # Save converted file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tts_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Created TTS format: {output_file}")
        print(f"Total segments: {len(tts_segments)}")
        print(f"Estimated duration: {current_time:.1f}s")
        
        return output_file
    
    def calculate_duration(self, tts_file: Path) -> int:
        """
        Calculate total duration from TTS format file
        
        Args:
            tts_file: Path to TTS format file
            
        Returns:
            Total duration in seconds (rounded up)
        """
        with open(tts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        segments = data.get('segments', [])
        if segments:
            return int(segments[-1]['end']) + 1
        else:
            return 300  # Default fallback