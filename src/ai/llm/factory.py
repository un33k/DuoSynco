"""
Factory for creating LLM provider instances
"""

from typing import Optional, Dict, Any, Type
from .base import LLMProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider


class LLMFactory:
    """Factory for creating LLM provider instances"""
    
    PROVIDERS: Dict[str, Type[LLMProvider]] = {
        "openai": OpenAIProvider,
        "gpt": OpenAIProvider,  # Alias
        "anthropic": AnthropicProvider,
        "claude": AnthropicProvider,  # Alias
    }
    
    @classmethod
    def create_provider(
        cls,
        provider_name: str,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> LLMProvider:
        """
        Create an LLM provider instance
        
        Args:
            provider_name: Name of the provider
            api_key: API key for the provider
            config: Provider-specific configuration
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If provider not found
        """
        provider_name_lower = provider_name.lower()
        
        if provider_name_lower not in cls.PROVIDERS:
            available = ", ".join(cls.PROVIDERS.keys())
            raise ValueError(
                f"LLM provider '{provider_name}' not found. "
                f"Available providers: {available}"
            )
            
        provider_class = cls.PROVIDERS[provider_name_lower]
        return provider_class(api_key=api_key, config=config)
        
    @classmethod
    def list_providers(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all available LLM providers
        
        Returns:
            Dictionary of provider information
        """
        providers = {}
        
        for name, provider_class in cls.PROVIDERS.items():
            # Skip aliases
            if name in ["gpt", "claude"]:
                continue
                
            try:
                # Try to instantiate with dummy key
                instance = provider_class(api_key="test")
                providers[name] = {
                    "available": True,
                    "name": instance.provider_name,
                    "default_model": instance.get_default_model()
                }
            except Exception as e:
                providers[name] = {
                    "available": False,
                    "error": str(e)
                }
                
        return providers
        
    @classmethod
    def get_provider_from_config(
        cls,
        config: Dict[str, Any]
    ) -> LLMProvider:
        """
        Create provider from configuration dict
        
        Args:
            config: Configuration with 'provider', 'api_key', etc.
            
        Returns:
            Configured provider instance
        """
        provider_name = config.get('provider', 'openai')
        api_key = config.get('api_key')
        provider_config = config.get('config', {})
        
        return cls.create_provider(
            provider_name=provider_name,
            api_key=api_key,
            config=provider_config
        )