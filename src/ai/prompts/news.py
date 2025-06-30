"""
News generation prompts
"""

from typing import Dict, Any, Tuple
import json
import re


class NewsPrompts:
    """Prompts for generating news content"""
    
    def get_news_prompt(
        self,
        content: str,
        style: str = "standard",
        anchor_gender: str = "female"
    ) -> Tuple[str, str]:
        """
        Get system and user prompts for news generation
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = f"""You are a professional news script writer who creates 
engaging, informative news broadcasts. You write for a {anchor_gender} news anchor 
with a clear, authoritative, yet approachable delivery style.

Your news scripts always include:
1. A compelling hook or lead
2. Clear presentation of facts
3. Context and background when needed
4. Smooth transitions between points
5. Professional sign-off

Write in a conversational yet professional tone suitable for broadcast."""

        style_guidelines = {
            "standard": "Traditional news format with measured delivery",
            "breaking": "Urgent tone with immediate impact",
            "feature": "More storytelling approach with human interest angle"
        }
        
        user_prompt = f"""Transform the following content into a {style} news broadcast script:

Content: {content}

Style Guidelines: {style_guidelines.get(style, '')}

Create a 2-minute news segment that includes:
- Attention-grabbing opening
- Key facts and developments
- Context and implications
- Professional conclusion

Return a JSON object with this structure:
{{
  "headline": "news headline",
  "style": "{style}",
  "duration": 120,
  "segments": [
    {{
      "text": "what the anchor says",
      "start": start_time_seconds,
      "end": end_time_seconds,
      "type": "intro|main|context|conclusion",
      "emphasis": ["words", "to", "emphasize"]
    }}
  ],
  "teleprompter": "full script formatted for teleprompter"
}}

Make the script engaging and suitable for a professional news broadcast."""
        
        return system_prompt, user_prompt
        
    def parse_news_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured news data"""
        try:
            # Try to parse as JSON first
            if response.strip().startswith('{'):
                return json.loads(response)
            
            # Extract JSON from response if wrapped
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
                
            # Fallback: create structure from text
            return self._create_news_structure(response)
            
        except Exception as e:
            return {
                "headline": "News Update",
                "style": "standard",
                "duration": 120,
                "segments": [
                    {
                        "text": response,
                        "start": 0,
                        "end": 120,
                        "type": "main",
                        "emphasis": []
                    }
                ],
                "teleprompter": response,
                "error": str(e)
            }
            
    def _create_news_structure(self, text: str) -> Dict[str, Any]:
        """Create news structure from plain text"""
        lines = text.strip().split('\n')
        segments = []
        current_time = 0
        
        # Extract headline from first line
        headline = lines[0] if lines else "News Update"
        
        # Process remaining lines into segments
        for line in lines[1:]:
            if line.strip():
                # Estimate duration
                words = len(line.split())
                duration = words * 0.35  # News pace ~0.35 sec/word
                
                segments.append({
                    "text": line.strip(),
                    "start": current_time,
                    "end": current_time + duration,
                    "type": "main",
                    "emphasis": []
                })
                
                current_time += duration
                
        return {
            "headline": headline,
            "style": "standard",
            "duration": current_time,
            "segments": segments,
            "teleprompter": text
        }