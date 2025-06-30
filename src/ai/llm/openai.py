"""
OpenAI/GPT LLM provider implementation
"""

import openai
from typing import Dict, Any, Optional, List, AsyncGenerator
import logging
from .base import LLMProvider
from ...utils.util_env import get_env

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI API integration for GPT models"""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize OpenAI provider"""
        super().__init__(
            api_key or get_env('OPENAI_API_KEY'),
            config
        )
        self.client = openai.OpenAI(api_key=self.api_key)
        self.provider_name = "OpenAI"
        
    def get_default_model(self) -> str:
        """Get default OpenAI model"""
        return "gpt-4-turbo-preview"
        
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text using OpenAI API"""
        self.validate_config()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=kwargs.get('model', self.model),
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                top_p=kwargs.get('top_p', 1.0),
                frequency_penalty=kwargs.get('frequency_penalty', 0),
                presence_penalty=kwargs.get('presence_penalty', 0)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
            
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text using OpenAI API"""
        self.validate_config()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            stream = await self.client.chat.completions.create(
                model=kwargs.get('model', self.model),
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise
            
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Chat completion with message history"""
        self.validate_config()
        
        try:
            response = self.client.chat.completions.create(
                model=kwargs.get('model', self.model),
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            raise
            
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for OpenAI usage"""
        # GPT-4 Turbo pricing (as of 2024)
        pricing = {
            "gpt-4-turbo-preview": {
                "input": 0.01 / 1000,   # $0.01 per 1K tokens
                "output": 0.03 / 1000   # $0.03 per 1K tokens
            },
            "gpt-4": {
                "input": 0.03 / 1000,
                "output": 0.06 / 1000
            },
            "gpt-3.5-turbo": {
                "input": 0.0005 / 1000,
                "output": 0.0015 / 1000
            }
        }
        
        model_pricing = pricing.get(self.model, pricing["gpt-4-turbo-preview"])
        
        input_cost = input_tokens * model_pricing["input"]
        output_cost = output_tokens * model_pricing["output"]
        
        return input_cost + output_cost