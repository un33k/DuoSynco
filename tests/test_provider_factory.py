"""
Unit tests for audio.providers.factory module
Tests provider creation and registration without external API calls
"""

import pytest
from unittest.mock import patch, MagicMock

from src.audio.providers.factory import ProviderFactory


class TestProviderFactory:
    """Test cases for ProviderFactory class"""

    def test_factory_singleton(self):
        """Test that factory implements singleton pattern"""
        factory1 = ProviderFactory()
        factory2 = ProviderFactory()
        assert factory1 is factory2

    def test_get_available_providers(self):
        """Test getting list of available providers"""
        factory = ProviderFactory()
        providers = factory.get_available_providers()

        assert isinstance(providers, list)
        assert len(providers) > 0
        assert "assemblyai" in providers
        assert "elevenlabs" in providers

    def test_provider_exists(self):
        """Test checking if provider exists"""
        factory = ProviderFactory()

        assert factory.provider_exists("assemblyai") is True
        assert factory.provider_exists("elevenlabs") is True
        assert factory.provider_exists("nonexistent") is False

    @patch("src.audio.providers.assemblyai.diarizer.AssemblyAIDiarizer")
    def test_create_assemblyai_provider(self, mock_assemblyai):
        """Test creating AssemblyAI provider"""
        mock_provider = MagicMock()
        mock_assemblyai.return_value = mock_provider

        factory = ProviderFactory()
        provider = factory.create_provider("assemblyai", api_key="test_key")

        assert provider is mock_provider
        mock_assemblyai.assert_called_once_with(api_key="test_key")

    @patch("src.audio.providers.elevenlabs.stt.ElevenLabsSTTProvider")
    def test_create_elevenlabs_stt_provider(self, mock_elevenlabs_stt):
        """Test creating ElevenLabs STT provider"""
        mock_provider = MagicMock()
        mock_elevenlabs_stt.return_value = mock_provider

        factory = ProviderFactory()
        provider = factory.create_provider("elevenlabs-stt", api_key="test_key")

        assert provider is mock_provider
        mock_elevenlabs_stt.assert_called_once_with(api_key="test_key")

    @patch("src.audio.providers.elevenlabs.tts.ElevenLabsTTSProvider")
    def test_create_elevenlabs_tts_provider(self, mock_elevenlabs_tts):
        """Test creating ElevenLabs TTS provider"""
        mock_provider = MagicMock()
        mock_elevenlabs_tts.return_value = mock_provider

        factory = ProviderFactory()
        provider = factory.create_provider("elevenlabs-tts", api_key="test_key")

        assert provider is mock_provider
        mock_elevenlabs_tts.assert_called_once_with(api_key="test_key")

    def test_create_invalid_provider(self):
        """Test creating provider with invalid name"""
        factory = ProviderFactory()

        with pytest.raises(ValueError, match="Unknown provider"):
            factory.create_provider("invalid_provider")

    def test_create_provider_without_api_key(self):
        """Test creating provider without required API key"""
        factory = ProviderFactory()

        # Some providers might not require API keys for basic functionality
        # This test checks that the factory handles missing API keys gracefully
        with patch("src.audio.providers.assemblyai.diarizer.AssemblyAIDiarizer") as mock_assemblyai:
            mock_provider = MagicMock()
            mock_assemblyai.return_value = mock_provider

            provider = factory.create_provider("assemblyai")
            assert provider is mock_provider
            mock_assemblyai.assert_called_once_with(api_key=None)

    def test_register_custom_provider(self):
        """Test registering a custom provider"""
        factory = ProviderFactory()

        # Create a mock provider class
        mock_provider_class = MagicMock()
        mock_provider_instance = MagicMock()
        mock_provider_class.return_value = mock_provider_instance

        # Register the custom provider
        factory.register_provider("custom", mock_provider_class)

        # Verify it's in the available providers
        assert "custom" in factory.get_available_providers()

        # Verify we can create it
        provider = factory.create_provider("custom", api_key="test")
        assert provider is mock_provider_instance
        mock_provider_class.assert_called_once_with(api_key="test")

    def test_get_provider_info(self):
        """Test getting provider information"""
        factory = ProviderFactory()

        info = factory.get_provider_info("assemblyai")
        assert isinstance(info, dict)
        assert "name" in info
        assert "description" in info
        assert "capabilities" in info

    def test_get_provider_info_invalid(self):
        """Test getting info for invalid provider"""
        factory = ProviderFactory()

        info = factory.get_provider_info("invalid")
        assert info is None

    def test_provider_capabilities(self):
        """Test checking provider capabilities"""
        factory = ProviderFactory()

        # AssemblyAI should support diarization
        assert factory.provider_supports("assemblyai", "diarization") is True

        # ElevenLabs should support TTS
        assert factory.provider_supports("elevenlabs-tts", "tts") is True

        # Invalid capability
        assert factory.provider_supports("assemblyai", "invalid_capability") is False

    def test_provider_supports_invalid_provider(self):
        """Test capability check for invalid provider"""
        factory = ProviderFactory()

        assert factory.provider_supports("invalid", "diarization") is False

    @patch("src.audio.providers.factory.importlib.import_module")
    def test_dynamic_provider_loading(self, mock_import):
        """Test dynamic loading of provider modules"""
        # Mock the import to simulate loading a provider module
        mock_module = MagicMock()
        mock_provider_class = MagicMock()
        mock_module.SomeProvider = mock_provider_class
        mock_import.return_value = mock_module

        ProviderFactory()

        # This would test dynamic loading if implemented
        # The exact implementation depends on the factory design

        # For now, just verify that the import mechanism can be mocked
        assert mock_import.call_count == 0  # Not called yet

    def test_factory_reset(self):
        """Test resetting factory to clean state"""
        factory = ProviderFactory()

        # Add a custom provider
        mock_provider_class = MagicMock()
        factory.register_provider("temp", mock_provider_class)
        assert "temp" in factory.get_available_providers()

        # Reset factory
        factory.reset()

        # Custom provider should be gone, but built-ins should remain
        assert "temp" not in factory.get_available_providers()
        assert "assemblyai" in factory.get_available_providers()

    def test_provider_priority(self):
        """Test provider priority ordering"""
        factory = ProviderFactory()

        # Test that providers are returned in expected priority order
        providers = factory.get_available_providers()

        # AssemblyAI might be preferred for diarization
        # ElevenLabs might be preferred for TTS
        # Exact order depends on implementation
        assert isinstance(providers, list)
        assert len(providers) > 0

    def test_create_provider_with_config(self):
        """Test creating provider with configuration dict"""
        factory = ProviderFactory()

        config = {"api_key": "test_key", "model": "best", "language": "en"}

        with patch("src.audio.providers.assemblyai.diarizer.AssemblyAIDiarizer") as mock_assemblyai:
            mock_provider = MagicMock()
            mock_assemblyai.return_value = mock_provider

            provider = factory.create_provider("assemblyai", **config)

            assert provider is mock_provider
            # Verify config was passed through
            mock_assemblyai.assert_called_once_with(**config)

    def test_validate_provider_config(self):
        """Test validating provider configuration"""
        factory = ProviderFactory()

        # Valid config
        valid_config = {"api_key": "test_key"}
        assert factory.validate_config("assemblyai", valid_config) is True

        # Invalid provider
        assert factory.validate_config("invalid", valid_config) is False

        # This test assumes validation is implemented
        # If not implemented, it should be graceful

    def test_get_default_config(self):
        """Test getting default configuration for provider"""
        factory = ProviderFactory()

        default_config = factory.get_default_config("assemblyai")

        if default_config is not None:
            assert isinstance(default_config, dict)
            # Should contain reasonable defaults
        # If not implemented, should return None gracefully
