"""
Factory for creating lipsync provider instances
"""

from typing import Optional, Dict, Type
from .base import LipsyncProvider
from .collossyan import CollossyanProvider
from .heygen import HeyGenProvider


class LipsyncProviderFactory:
    """Factory for creating lipsync provider instances"""
    
    PROVIDERS: Dict[str, Type[LipsyncProvider]] = {
        "collossyan": CollossyanProvider,
        "heygen": HeyGenProvider,
    }
    
    @classmethod
    def create_provider(
        cls,
        provider_name: str,
        api_key: Optional[str] = None
    ) -> LipsyncProvider:
        """
        Create a lipsync provider instance
        
        Args:
            provider_name: Name of the provider
            api_key: API key for the provider
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If provider not found
        """
        provider_name_lower = provider_name.lower()
        
        if provider_name_lower not in cls.PROVIDERS:
            available = ", ".join(cls.PROVIDERS.keys())
            raise ValueError(
                f"Lipsync provider '{provider_name}' not found. "
                f"Available providers: {available}"
            )
            
        provider_class = cls.PROVIDERS[provider_name_lower]
        return provider_class(api_key=api_key)
        
    @classmethod
    def list_providers(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all available lipsync providers
        
        Returns:
            Dictionary of provider information
        """
        providers = {}
        
        for name, provider_class in cls.PROVIDERS.items():
            try:
                # Try to instantiate with dummy key
                instance = provider_class(api_key="test")
                providers[name] = {
                    "available": True,
                    "name": instance.provider_name,
                    "requires_api_key": instance.requires_api_key
                }
            except Exception as e:
                providers[name] = {
                    "available": False,
                    "error": str(e)
                }
                
        return providers