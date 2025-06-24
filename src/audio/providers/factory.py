"""
Provider factory for speaker diarization services
"""

from typing import Optional
from .base import SpeakerDiarizationProvider
from .assemblyai import AssemblyAIDiarizer
from .elevenlabs import ElevenLabsTTSProvider, ElevenLabsSTTProvider


class ProviderFactory:
    """Factory for creating speaker diarization providers"""

    AVAILABLE_PROVIDERS = {
        'assemblyai': AssemblyAIDiarizer,
        'elevenlabs': ElevenLabsTTSProvider,
        'elevenlabs-stt': ElevenLabsSTTProvider,
    }

    @classmethod
    def get_provider(
        cls,
        provider_name: str,
        api_key: Optional[str] = None
    ) -> SpeakerDiarizationProvider:
        """
        Get a speaker diarization provider instance

        Args:
            provider_name: Name of the provider ('assemblyai', 'elevenlabs', 'elevenlabs-stt')
            api_key: API key for the provider

        Returns:
            Provider instance

        Raises:
            ValueError: If provider not found or not available
        """
        provider_name_lower = provider_name.lower()

        if provider_name_lower not in cls.AVAILABLE_PROVIDERS:
            available = ', '.join(cls.AVAILABLE_PROVIDERS.keys())
            raise ValueError(
                f"Provider '{provider_name}' not found. "
                f"Available providers: {available}"
            )

        provider_class = cls.AVAILABLE_PROVIDERS[provider_name_lower]
        return provider_class(api_key=api_key)

    @classmethod
    def list_providers(cls) -> dict:
        """
        List all available providers

        Returns:
            Dictionary mapping provider names to their availability
        """
        providers = {}
        for name, provider_class in cls.AVAILABLE_PROVIDERS.items():
            # Try to instantiate to check availability
            try:
                # Create with dummy key to test imports
                provider_instance = provider_class(api_key="test")
                providers[name] = {
                    'available': True,
                    'requires_api_key': provider_instance.requires_api_key,
                    'name': provider_instance.provider_name
                }
            except ImportError as e:
                providers[name] = {
                    'available': False,
                    'requires_api_key': True,
                    'error': str(e)
                }
            except Exception:
                # Provider might be available but need real API key
                try:
                    temp_instance = provider_class()
                    providers[name] = {
                        'available': True,
                        'requires_api_key': temp_instance.requires_api_key,
                        'name': temp_instance.provider_name
                    }
                except Exception:
                    providers[name] = {
                        'available': False,
                        'requires_api_key': True,
                        'error': 'Cannot instantiate provider'
                    }

        return providers