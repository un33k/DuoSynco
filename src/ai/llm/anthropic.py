"""
Anthropic/Claude LLM provider implementation
"""

import anthropic
from typing import Dict, Any, Optional, List, AsyncGenerator
import logging
from .base import LLMProvider
from ...utils.util_env import get_env

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic API integration for Claude models"""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize Anthropic provider"""
        super().__init__(
            api_key or get_env('ANTHROPIC_API_KEY'),
            config
        )
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.provider_name = "Anthropic"
        
    def get_default_model(self) -> str:
        """Get default Claude model"""
        return "claude-3-opus-20240229"
        
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text using Anthropic API"""
        self.validate_config()
        
        try:
            # Anthropic uses a different message format
            messages = [{"role": "user", "content": prompt}]
            
            response = self.client.messages.create(
                model=kwargs.get('model', self.model),
                messages=messages,
                system=system_prompt if system_prompt else None,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature)
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise
            
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text using Anthropic API"""
        self.validate_config()
        
        try:
            messages = [{"role": "user", "content": prompt}]
            
            async with self.client.messages.stream(
                model=kwargs.get('model', self.model),
                messages=messages,
                system=system_prompt if system_prompt else None,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature)
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise
            
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Chat completion with message history"""
        self.validate_config()
        
        try:
            # Extract system message if present
            system_prompt = None
            filtered_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    filtered_messages.append(msg)
                    
            response = self.client.messages.create(
                model=kwargs.get('model', self.model),
                messages=filtered_messages,
                system=system_prompt,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature)
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic chat error: {e}")
            raise
            
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for Anthropic usage"""
        # Claude 3 pricing (as of 2024)
        pricing = {
            "claude-3-opus-20240229": {
                "input": 0.015 / 1000,   # $15 per million tokens
                "output": 0.075 / 1000   # $75 per million tokens
            },
            "claude-3-sonnet-20240229": {
                "input": 0.003 / 1000,   # $3 per million tokens
                "output": 0.015 / 1000   # $15 per million tokens
            },
            "claude-3-haiku-20240307": {
                "input": 0.00025 / 1000,  # $0.25 per million tokens
                "output": 0.00125 / 1000  # $1.25 per million tokens
            }
        }
        
        model_pricing = pricing.get(self.model, pricing["claude-3-opus-20240229"])
        
        input_cost = input_tokens * model_pricing["input"]
        output_cost = output_tokens * model_pricing["output"]
        
        return input_cost + output_cost