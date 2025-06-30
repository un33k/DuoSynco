"""
Debate generation prompts
"""

from typing import Dict, Any, Tuple, List
import json
import re


class DebatePrompts:
    """Prompts for generating debate content"""
    
    def get_debate_prompt(
        self,
        topic: str,
        style: str = "balanced",
        duration: int = 300
    ) -> Tuple[str, str]:
        """
        Get system and user prompts for debate generation
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = """You are an expert debate script writer who creates engaging, 
balanced debates with multiple perspectives. You create natural-sounding dialogue 
that explores topics thoroughly while maintaining respect between participants.

Your debates always include:
1. A neutral moderator who guides the discussion
2. A proponent (Person A) who argues in favor
3. An opponent (Person B) who argues against
4. Natural speech patterns and conversational flow
5. Evidence-based arguments
6. Respectful disagreement

Format your output as a JSON structure with speakers and timestamped segments."""

        style_guidelines = {
            "balanced": "Keep arguments equally weighted and respectful",
            "heated": "Include more passionate exchanges while remaining civil",
            "academic": "Use scholarly language and cite evidence frequently"
        }
        
        user_prompt = f"""Create a {duration}-second debate on the topic: "{topic}"

Style: {style} - {style_guidelines.get(style, '')}

Structure the debate with:
- Opening statements (20% of time)
- Main arguments and rebuttals (60% of time)
- Closing statements (20% of time)

Return a JSON object with this structure:
{{
  "topic": "debate topic",
  "duration": total_seconds,
  "speakers": ["moderator", "proponent", "opponent"],
  "segments": [
    {{
      "speaker": "speaker_name",
      "text": "what they say",
      "start": start_time_seconds,
      "end": end_time_seconds,
      "type": "introduction|argument|rebuttal|conclusion"
    }}
  ]
}}

Make the dialogue natural and engaging. Each speaker should have a distinct voice and perspective."""
        
        return system_prompt, user_prompt
        
    def parse_debate_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured debate data"""
        try:
            # Try to parse as JSON first
            if response.strip().startswith('{'):
                return json.loads(response)
            
            # Extract JSON from response if wrapped in other text
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
                
            # Fallback: parse as text
            return self._parse_text_response(response)
            
        except Exception as e:
            # Return a basic structure on error
            return {
                "topic": "Parsed debate",
                "duration": 300,
                "speakers": ["moderator", "proponent", "opponent"],
                "segments": [
                    {
                        "speaker": "moderator",
                        "text": response,
                        "start": 0,
                        "end": 300,
                        "type": "argument"
                    }
                ],
                "error": str(e)
            }
            
    def _parse_text_response(self, text: str) -> Dict[str, Any]:
        """Parse text-based response into debate structure"""
        lines = text.strip().split('\n')
        segments = []
        current_time = 0
        
        for line in lines:
            if ':' in line:
                speaker, content = line.split(':', 1)
                speaker = speaker.strip().lower()
                
                # Map common speaker names
                if 'mod' in speaker:
                    speaker = 'moderator'
                elif 'pro' in speaker or 'a:' in speaker:
                    speaker = 'proponent'
                elif 'con' in speaker or 'opp' in speaker or 'b:' in speaker:
                    speaker = 'opponent'
                    
                # Estimate duration based on text length
                duration = max(len(content.split()) * 0.4, 5)  # ~0.4 sec per word
                
                segments.append({
                    "speaker": speaker,
                    "text": content.strip(),
                    "start": current_time,
                    "end": current_time + duration,
                    "type": "argument"
                })
                
                current_time += duration
                
        return {
            "topic": "Debate",
            "duration": current_time,
            "speakers": ["moderator", "proponent", "opponent"],
            "segments": segments
        }
        
    def format_as_text(self, debate_data: Dict[str, Any]) -> str:
        """Format debate data as readable text"""
        output = []
        output.append(f"DEBATE: {debate_data.get('topic', 'Unknown Topic')}")
        output.append(f"Duration: {debate_data.get('duration', 0)} seconds")
        output.append("=" * 50)
        output.append("")
        
        for segment in debate_data.get('segments', []):
            speaker = segment['speaker'].upper()
            time_range = f"[{segment['start']:.1f}s - {segment['end']:.1f}s]"
            output.append(f"{speaker} {time_range}:")
            output.append(segment['text'])
            output.append("")
            
        return '\n'.join(output)