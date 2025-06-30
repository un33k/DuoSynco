"""
Translation prompts
"""

from typing import Tuple


class TranslatePrompts:
    """Prompts for translation tasks"""
    
    def get_translation_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        preserve_length: bool = True,
        context: str = ""
    ) -> Tuple[str, str]:
        """
        Get system and user prompts for translation
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = f"""You are an expert translator specializing in video subtitle 
translation. You maintain the original meaning while adapting to the target language's 
natural flow and cultural context.

Your translations are:
1. Accurate to the original meaning
2. Natural in the target language
3. Appropriate for the context
4. Concise when needed for timing"""

        length_instruction = ""
        if preserve_length:
            length_instruction = """
IMPORTANT: Keep the translation approximately the same length as the original 
for subtitle timing synchronization. If needed, use shorter synonyms or phrases."""
        
        user_prompt = f"""Translate the following from {source_lang} to {target_lang}:

Original: "{text}"

{length_instruction}

Context: {context if context else 'General video content'}

Provide only the translated text without any explanation or notes."""
        
        return system_prompt, user_prompt