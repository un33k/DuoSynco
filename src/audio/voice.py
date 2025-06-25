"""
Voice Configuration Module
Handles voice-related environment variables and speaker mappings
"""

from typing import Dict
from ..utils.util_env import get_env


def get_voice_mapping() -> Dict[str, str]:
    """
    Get voice mapping from environment variables

    Supports both old and new naming conventions for backwards compatibility:
    - New: VOICE_SPEAKER_0, VOICE_SPEAKER_1
    - Old: VOICE_SPEAKER_A, VOICE_SPEAKER_B

    Returns:
        Dictionary mapping speaker IDs to voice IDs
    """
    voice_mapping = {}

    # Support both old and new naming conventions for backwards compatibility
    voice_0 = get_env("VOICE_SPEAKER_0") or get_env("VOICE_SPEAKER_A")
    voice_1 = get_env("VOICE_SPEAKER_1") or get_env("VOICE_SPEAKER_B")

    if voice_0:
        voice_mapping["speaker_0"] = voice_0
        voice_mapping["A"] = voice_0  # Backwards compatibility
    if voice_1:
        voice_mapping["speaker_1"] = voice_1
        voice_mapping["B"] = voice_1  # Backwards compatibility

    return voice_mapping


def get_default_voice_mapping() -> Dict[str, str]:
    """
    Get default voice mapping with fallback values

    Returns:
        Dictionary with default ElevenLabs voice IDs
    """
    return {
        "speaker_0": get_env("VOICE_SPEAKER_0", default="N2lVS1w4EtoT3dr4eOWO"),
        "speaker_1": get_env("VOICE_SPEAKER_1", default="Xb7hH8MSUJpSbSDYk0k2"),
        "A": get_env("VOICE_SPEAKER_0", default="N2lVS1w4EtoT3dr4eOWO"),  # Backwards compatibility
        "B": get_env("VOICE_SPEAKER_1", default="Xb7hH8MSUJpSbSDYk0k2"),  # Backwards compatibility
    }


def get_voice_for_speaker(speaker_id: str) -> str:
    """
    Get the voice ID for a specific speaker

    Args:
        speaker_id: Speaker identifier (e.g., 'speaker_0', 'A', etc.)

    Returns:
        Voice ID for the speaker, or default if not configured
    """
    voice_mapping = get_default_voice_mapping()
    default_voice = "N2lVS1w4EtoT3dr4eOWO"
    return voice_mapping.get(speaker_id, voice_mapping.get("speaker_0", default_voice))


def is_voice_configured(speaker_id: str) -> bool:
    """
    Check if a voice is configured for a specific speaker

    Args:
        speaker_id: Speaker identifier

    Returns:
        True if voice is configured in environment variables
    """
    voice_mapping = get_voice_mapping()
    return speaker_id in voice_mapping
