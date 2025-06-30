"""
News content generator
"""

from typing import Dict, Any, List, Optional
import logging

from ..base import ContentGenerator

logger = logging.getLogger(__name__)


class NewsGenerator(ContentGenerator):
    """Generate news-style content from various sources"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize news generator"""
        super().__init__(config)
        self.generator_type = "News"
        
        # News anchor presets
        self.anchor_presets = {
            "male": {"gender": "male", "style": "professional", "tone": "authoritative"},
            "female": {"gender": "female", "style": "professional", "tone": "engaging"}
        }
        
    def generate_script(
        self,
        input_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate news script from input
        
        Args:
            input_data: Article, tweet, or news topic
            **kwargs: Additional parameters (anchor, style, duration)
            
        Returns:
            News script with segments
        """
        self.validate_input(input_data)
        
        content = input_data.get("text", "")
        source_url = input_data.get("url", "")
        anchor_gender = kwargs.get("anchor", "female")
        style = kwargs.get("style", "standard")  # standard, breaking, feature
        duration = kwargs.get("duration", 120)  # 2 minutes default
        
        # Check if LLM provider is available
        llm_provider = kwargs.get("llm_provider")
        
        if llm_provider:
            # Use LLM to generate news script
            from ...ai.prompts.news import NewsPrompts
            prompts = NewsPrompts()
            
            system_prompt, user_prompt = prompts.get_news_prompt(
                content=content,
                style=style,
                anchor_gender=anchor_gender
            )
            
            # Generate with LLM
            result = llm_provider.generate(
                prompt=user_prompt,
                system_prompt=system_prompt
            )
            
            # Parse response
            script = prompts.parse_news_response(result)
            script["type"] = "news"
            script["source"] = source_url
            script["anchor"] = anchor_gender
            
            return script
        else:
            # Fallback to template generation
            script = {
                "type": "news",
                "content": content,
                "source": source_url,
                "duration": duration,
                "style": style,
                "anchor": anchor_gender,
                "segments": []
            }
            
            # Generate news segments
            segments = self._generate_news_segments(content, duration, style)
            script["segments"] = segments
            
            return script
        
    def assign_speakers(
        self,
        script: Dict[str, Any],
        speaker_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Assign anchor to news segments"""
        
        anchor_gender = script.get("anchor", "female")
        anchor_info = self.anchor_presets.get(anchor_gender, self.anchor_presets["female"])
        
        # Override with preferences if provided
        if speaker_preferences and "anchor" in speaker_preferences:
            anchor_info.update(speaker_preferences["anchor"])
            
        # Assign anchor to all segments
        for segment in script["segments"]:
            segment["speaker"] = "anchor"
            segment["speaker_info"] = anchor_info
            
        script["speaker_assignments"] = {"anchor": anchor_info}
        
        return script
        
    def _generate_news_segments(
        self,
        content: str,
        duration: float,
        style: str
    ) -> List[Dict[str, Any]]:
        """Generate news presentation segments"""
        
        segments = []
        
        if style == "breaking":
            # Breaking news format
            segments.append({
                "text": "Breaking news just in.",
                "start": 0,
                "end": 3,
                "type": "breaking_intro"
            })
            start_time = 3
        else:
            # Standard intro
            segments.append({
                "text": "Good evening, I'm your news anchor with today's top story.",
                "start": 0,
                "end": 5,
                "type": "intro"
            })
            start_time = 5
            
        # Main content segments
        # In production, AI would intelligently segment the content
        content_duration = duration - 15  # Reserve time for intro/outro
        
        segments.append({
            "text": f"Here's what we know: {content[:200]}...",
            "start": start_time,
            "end": start_time + content_duration,
            "type": "main_story"
        })
        
        # Conclusion
        segments.append({
            "text": "We'll continue to follow this story and bring you updates as they develop. This has been your news update.",
            "start": duration - 10,
            "end": duration,
            "type": "conclusion"
        })
        
        return segments
        
    def generate_ai_prompt(
        self,
        content: str,
        style: str,
        source: str = ""
    ) -> str:
        """Generate prompt for AI to create news content"""
        
        prompt = f"""Transform the following content into a professional news broadcast script:

Content: {content}
Source: {source}
Style: {style} news format
Duration: 2 minutes when spoken

Requirements:
- Professional news anchor tone
- Clear and concise language
- Highlight key facts and developments
- Include appropriate transitions
- End with a professional sign-off

Structure:
1. Opening/Hook
2. Main story details
3. Context and implications
4. Conclusion/Sign-off

Make it engaging yet informative, suitable for broadcast news.
"""
        
        return prompt
        
    def format_for_teleprompter(
        self,
        script: Dict[str, Any]
    ) -> str:
        """Format script for teleprompter display"""
        
        teleprompter_text = ""
        
        for segment in script["segments"]:
            text = segment["text"]
            # Add pauses and emphasis markers
            text = text.replace(".", ".<pause>")
            text = text.replace(",", ",<brief>")
            
            teleprompter_text += f"{text}\n\n"
            
        return teleprompter_text