"""
Base class for LLM providers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, AsyncGenerator
import logging

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLM provider
        
        Args:
            api_key: API key for the provider
            config: Provider-specific configuration
        """
        self.api_key = api_key
        self.config = config or {}
        self.provider_name = self.__class__.__name__
        self.model = self.config.get('model', self.get_default_model())
        self.temperature = self.config.get('temperature', 0.7)
        self.max_tokens = self.config.get('max_tokens', 4000)
        
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            **kwargs: Provider-specific parameters
            
        Returns:
            Generated text
        """
        pass
        
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate text stream from prompt
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            **kwargs: Provider-specific parameters
            
        Yields:
            Text chunks
        """
        pass
        
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Chat completion with message history
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Provider-specific parameters
            
        Returns:
            Generated response
        """
        pass
        
    @abstractmethod
    def get_default_model(self) -> str:
        """Get default model for this provider"""
        pass
        
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token
        return len(text) // 4
        
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for token usage
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        # Override in subclasses with actual pricing
        return 0.0
        
    def validate_config(self) -> bool:
        """
        Validate provider configuration
        
        Returns:
            True if configuration is valid
        """
        if not self.api_key:
            raise ValueError(f"{self.provider_name} requires an API key")
        return True
        
    def format_for_provider(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Any:
        """
        Format prompt for provider's expected input
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            
        Returns:
            Provider-specific format
        """
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}"
        return prompt