"""
Translation content generator
"""

from typing import Dict, Any, List, Optional
import logging
import random

from ..base import ContentGenerator

logger = logging.getLogger(__name__)


class TranslateGenerator(ContentGenerator):
    """Generate translated content while preserving timing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize translation generator"""
        super().__init__(config)
        self.generator_type = "Translate"
        
        # Voice pools for random selection
        self.voice_pools = {
            "male": ["professional", "casual", "energetic"],
            "female": ["professional", "warm", "dynamic"]
        }
        
    def generate_script(
        self,
        input_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate translated script from subtitles
        
        Args:
            input_data: Original subtitles with timing
            **kwargs: target_language, preserve_timing, etc.
            
        Returns:
            Translated script with timing preserved
        """
        self.validate_input(input_data)
        
        subtitles = input_data.get("subtitles", [])
        source_language = input_data.get("source_language", "en")
        target_language = kwargs.get("target_language", "es")
        preserve_timing = kwargs.get("preserve_timing", True)
        
        script = {
            "type": "translation",
            "source_language": source_language,
            "target_language": target_language,
            "preserve_timing": preserve_timing,
            "segments": []
        }
        
        # Generate translated segments
        segments = self._generate_translated_segments(
            subtitles, 
            target_language,
            preserve_timing
        )
        script["segments"] = segments
        
        return script
        
    def assign_speakers(
        self,
        script: Dict[str, Any],
        speaker_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Assign voice to translated content"""
        
        # Random gender selection if not specified
        if speaker_preferences and "gender" in speaker_preferences:
            gender = speaker_preferences["gender"]
        else:
            gender = random.choice(["male", "female"])
            
        # Random voice style
        voice_style = random.choice(self.voice_pools[gender])
        
        voice_info = {
            "gender": gender,
            "style": voice_style,
            "language": script["target_language"]
        }
        
        # Assign same voice to all segments
        for segment in script["segments"]:
            segment["speaker"] = "translator"
            segment["speaker_info"] = voice_info
            
        script["speaker_assignments"] = {"translator": voice_info}
        
        return script
        
    def _generate_translated_segments(
        self,
        subtitles: List[Dict[str, Any]],
        target_language: str,
        preserve_timing: bool
    ) -> List[Dict[str, Any]]:
        """Generate translated segments"""
        
        segments = []
        
        for subtitle in subtitles:
            # In production, this would use translation API
            original_text = subtitle.get("text", "")
            start_time = subtitle.get("start", 0)
            end_time = subtitle.get("end", 0)
            
            # Simulated translation (in production, use actual translation service)
            translated_text = f"[Translated to {target_language}] {original_text}"
            
            segment = {
                "original_text": original_text,
                "text": translated_text,
                "start": start_time,
                "end": end_time,
                "type": "subtitle"
            }
            
            if preserve_timing:
                # Adjust translation to fit timing
                segment["timing_adjusted"] = True
                
            segments.append(segment)
            
        return segments
        
    def extract_youtube_subtitles(
        self,
        video_url: str,
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """Extract subtitles from YouTube video"""
        
        # In production, this would use youtube-dl or YouTube API
        # Return format compatible with our system
        
        logger.info(f"Extracting subtitles from: {video_url}")
        
        # Simulated subtitle extraction
        subtitles = [
            {
                "text": "Welcome to this video",
                "start": 0,
                "end": 3
            },
            {
                "text": "Today we'll discuss important topics",
                "start": 3,
                "end": 7
            }
        ]
        
        return subtitles
        
    def generate_ai_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str = ""
    ) -> str:
        """Generate prompt for AI translation"""
        
        prompt = f"""Translate the following text from {source_lang} to {target_lang}:

Text: "{text}"

Requirements:
- Maintain the original meaning and tone
- Keep similar length for timing synchronization
- Use natural, conversational language
- Preserve any emphasis or emotion

Context: {context}

Provide only the translated text without any explanation.
"""
        
        return prompt
        
    def adjust_timing_for_translation(
        self,
        segment: Dict[str, Any],
        target_language: str
    ) -> Dict[str, Any]:
        """Adjust timing based on language characteristics"""
        
        # Different languages have different speaking rates
        language_multipliers = {
            "es": 1.1,  # Spanish typically takes 10% longer
            "fr": 1.15, # French takes 15% longer
            "de": 1.2,  # German takes 20% longer
            "ja": 0.9,  # Japanese can be more concise
            "zh": 0.85  # Chinese is often more concise
        }
        
        multiplier = language_multipliers.get(target_language, 1.0)
        
        original_duration = segment["end"] - segment["start"]
        new_duration = original_duration * multiplier
        
        segment["adjusted_end"] = segment["start"] + new_duration
        segment["speed_factor"] = 1.0 / multiplier  # For TTS speed adjustment
        
        return segment