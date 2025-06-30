"""
Debate content generator
"""

from typing import Dict, Any, List, Optional
import logging

from ..base import ContentGenerator

logger = logging.getLogger(__name__)


class DebateGenerator(ContentGenerator):
    """Generate debate-style content with multiple perspectives"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize debate generator"""
        super().__init__(config)
        self.generator_type = "Debate"
        
        # Default speaker roles
        self.default_speakers = {
            "moderator": {"gender": "neutral", "role": "moderator"},
            "pro": {"gender": "male", "role": "proponent"},
            "con": {"gender": "female", "role": "opponent"}
        }
        
    def generate_script(
        self,
        input_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate debate script from input
        
        Args:
            input_data: Topic, article, or prompt
            **kwargs: Additional parameters (style, duration, etc.)
            
        Returns:
            Debate script with segments
        """
        self.validate_input(input_data)
        
        topic = input_data.get("text", "")
        duration = kwargs.get("duration", 300)  # 5 minutes default
        style = kwargs.get("style", "balanced")  # balanced, heated, academic
        
        # Check if LLM provider is available
        llm_provider = kwargs.get("llm_provider")
        
        if llm_provider:
            # Use LLM to generate debate
            from ...ai.prompts.debate import DebatePrompts
            prompts = DebatePrompts()
            
            system_prompt, user_prompt = prompts.get_debate_prompt(
                topic=topic,
                style=style,
                duration=duration
            )
            
            # Generate with LLM
            result = llm_provider.generate(
                prompt=user_prompt,
                system_prompt=system_prompt
            )
            
            # Parse response
            script = prompts.parse_debate_response(result)
            script["type"] = "debate"
            
            return script
        else:
            # Fallback to template generation
            script = {
                "type": "debate",
                "topic": topic,
                "duration": duration,
                "style": style,
                "speakers": ["moderator", "pro", "con"],
                "segments": []
            }
            
            # Generate debate segments
            segments = self._generate_debate_segments(topic, duration, style)
            script["segments"] = segments
            
            return script
        
    def assign_speakers(
        self,
        script: Dict[str, Any],
        speaker_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Assign speakers to debate roles"""
        
        # Merge preferences with defaults
        speakers = self.default_speakers.copy()
        if speaker_preferences:
            for role, prefs in speaker_preferences.items():
                if role in speakers:
                    speakers[role].update(prefs)
                    
        # Assign speakers to segments
        for segment in script["segments"]:
            role = segment.get("role", "moderator")
            if role in speakers:
                segment["speaker"] = role
                segment["speaker_info"] = speakers[role]
                
        script["speaker_assignments"] = speakers
        
        return script
        
    def _generate_debate_segments(
        self,
        topic: str,
        duration: float,
        style: str
    ) -> List[Dict[str, Any]]:
        """Generate debate segments"""
        
        # This is a template structure
        # In production, AI would generate actual content
        
        segments = [
            {
                "role": "moderator",
                "text": f"Welcome to today's debate on {topic}. We have two perspectives to explore.",
                "start": 0,
                "end": 10,
                "type": "introduction"
            },
            {
                "role": "pro",
                "text": f"I believe that {topic} represents a positive development because...",
                "start": 10,
                "end": 40,
                "type": "argument"
            },
            {
                "role": "con",
                "text": f"While I understand that perspective, I must disagree. {topic} poses significant challenges...",
                "start": 40,
                "end": 70,
                "type": "counter-argument"
            },
            {
                "role": "moderator",
                "text": "Interesting points from both sides. Let's explore this further.",
                "start": 70,
                "end": 80,
                "type": "transition"
            }
        ]
        
        # Continue generating segments based on duration
        current_time = 80
        segment_duration = 30
        
        while current_time < duration - 20:
            # Alternate between pro and con
            if len(segments) % 2 == 0:
                role = "pro"
            else:
                role = "con"
                
            segments.append({
                "role": role,
                "text": f"Another important aspect to consider about {topic}...",
                "start": current_time,
                "end": current_time + segment_duration,
                "type": "argument"
            })
            
            current_time += segment_duration
            
        # Add conclusion
        segments.append({
            "role": "moderator",
            "text": f"Thank you both for this insightful debate on {topic}. We've heard compelling arguments from both sides.",
            "start": current_time,
            "end": duration,
            "type": "conclusion"
        })
        
        return segments
        
    def generate_ai_prompt(
        self,
        topic: str,
        style: str,
        additional_context: str = ""
    ) -> str:
        """Generate prompt for AI to create debate content"""
        
        prompt = f"""Create a balanced debate on the topic: "{topic}"

Style: {style}
Duration: Approximately 5 minutes when spoken

Format the debate with three speakers:
1. Moderator - Neutral, guides the discussion
2. Proponent - Argues in favor 
3. Opponent - Argues against

Structure:
- Introduction by moderator
- Initial arguments from both sides
- Rebuttals and counter-arguments
- Additional points
- Conclusion by moderator

{additional_context}

Make the arguments thoughtful, evidence-based, and respectful. 
Each speaker should have distinct perspectives and speaking styles.
"""
        
        return prompt