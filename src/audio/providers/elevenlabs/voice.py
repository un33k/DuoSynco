"""
Voice Management for ElevenLabs TTS
Handles voice selection, mapping, and voice discovery
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests

logger = logging.getLogger(__name__)


class VoiceManager:
    """
    Manages ElevenLabs voice selection and mapping
    """

    # Default voice mappings for common speaker labels
    DEFAULT_VOICE_MAPPING = {
        "A": "pNInz6obpgDQGcFmaJgB",  # Adam (male)
        "B": "EXAVITQu4vr4xnSDxMaL",  # Bella (female)
        "C": "VR6AewLTigWG4xSOukaG",  # Arnold (male)
        "D": "oWAxZDx7w5VEj9dCyTzz",  # Grace (female)
        "E": "2EiwWnXFnvU5JabPnv8n",  # Clyde (male)
        "F": "IKne3meq5aSn9XLyUdCD",  # Charlie (female)
        "SPEAKER_00": "pNInz6obpgDQGcFmaJgB",  # Adam
        "SPEAKER_01": "EXAVITQu4vr4xnSDxMaL",  # Bella
        "SPEAKER_02": "VR6AewLTigWG4xSOukaG",  # Arnold
        "SPEAKER_03": "oWAxZDx7w5VEj9dCyTzz",  # Grace
    }

    def __init__(self, api_key: str, base_url: str = "https://api.elevenlabs.io/v1"):
        """
        Initialize voice manager

        Args:
            api_key: ElevenLabs API key
            base_url: Base URL for ElevenLabs API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(
            {"Accept": "application/json", "xi-api-key": api_key}
        )

        self._available_voices: Optional[List[Dict]] = None

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Get list of available voices from ElevenLabs API

        Returns:
            List of voice dictionaries with id, name, category, etc.
        """
        if self._available_voices is None:
            try:
                response = self.session.get(f"{self.base_url}/voices")
                response.raise_for_status()
                data = response.json()
                self._available_voices = data.get("voices", [])
                logger.info(
                    "Retrieved %d available voices", len(self._available_voices)
                )
            except Exception as e:
                logger.error("Failed to retrieve voices: %s", e)
                self._available_voices = []

        return self._available_voices

    def get_voice_info(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific voice

        Args:
            voice_id: ElevenLabs voice ID

        Returns:
            Voice information dictionary or None if not found
        """
        voices = self.get_available_voices()
        for voice in voices:
            if voice.get("voice_id") == voice_id:
                return voice
        return None

    def create_voice_mapping(
        self, speakers: List[str], custom_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Create speaker to voice ID mapping

        Args:
            speakers: List of speaker IDs (e.g., ['A', 'B', 'SPEAKER_00'])
            custom_mapping: Optional custom speaker to voice ID mapping

        Returns:
            Dictionary mapping speaker IDs to voice IDs
        """
        voice_mapping = {}

        for speaker in speakers:
            if custom_mapping and speaker in custom_mapping:
                # Use custom mapping first
                voice_mapping[speaker] = custom_mapping[speaker]
            elif speaker in self.DEFAULT_VOICE_MAPPING:
                # Use default mapping
                voice_mapping[speaker] = self.DEFAULT_VOICE_MAPPING[speaker]
            else:
                # Fallback to cycling through default voices
                speaker_index = len(voice_mapping) % len(self.DEFAULT_VOICE_MAPPING)
                default_voices = list(self.DEFAULT_VOICE_MAPPING.values())
                voice_mapping[speaker] = default_voices[speaker_index]

        logger.info("Created voice mapping: %s", voice_mapping)
        return voice_mapping

    def validate_voice_ids(self, voice_ids: List[str]) -> Dict[str, bool]:
        """
        Validate that voice IDs exist in ElevenLabs

        Args:
            voice_ids: List of voice IDs to validate

        Returns:
            Dictionary mapping voice IDs to their validity
        """
        available_voices = self.get_available_voices()
        available_ids = {v.get("voice_id") for v in available_voices}

        validation_results = {}
        for voice_id in voice_ids:
            validation_results[voice_id] = voice_id in available_ids

        logger.info("Voice validation results: %s", validation_results)
        return validation_results

    def get_voice_suggestions(
        self, gender: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get voice suggestions based on criteria

        Args:
            gender: Optional gender filter ('male', 'female')

        Returns:
            List of suggested voices
        """
        voices = self.get_available_voices()

        if gender:
            # Filter by gender if specified
            gender_keywords = {
                "male": ["male", "man", "masculine"],
                "female": ["female", "woman", "feminine"],
            }

            filtered_voices = []
            for voice in voices:
                voice_name = voice.get("name", "").lower()
                labels = voice.get("labels", {})

                # Check if voice matches gender criteria
                if any(
                    keyword in voice_name for keyword in gender_keywords.get(gender, [])
                ):
                    filtered_voices.append(voice)
                elif labels.get("gender") == gender:
                    filtered_voices.append(voice)

            return filtered_voices

        return voices

    def list_default_voices(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about the default voices used in mapping

        Returns:
            Dictionary mapping voice IDs to their information
        """
        default_info = {}
        unique_voice_ids = set(self.DEFAULT_VOICE_MAPPING.values())

        for voice_id in unique_voice_ids:
            voice_info = self.get_voice_info(voice_id)
            if voice_info:
                default_info[voice_id] = voice_info
            else:
                default_info[voice_id] = {
                    "voice_id": voice_id,
                    "name": "Unknown",
                    "status": "Not found",
                }

        return default_info

    def get_all_voices(self) -> List[Dict[str, Any]]:
        """
        Get all available voices (alias for get_available_voices)

        Returns:
            List of all voice dictionaries
        """
        return self.get_available_voices()

    def get_voices_by_language(self, language: str) -> List[Dict[str, Any]]:
        """
        Get voices filtered by language

        Args:
            language: Language code (e.g., 'en', 'es', 'fr', 'fa')

        Returns:
            List of voices supporting the language
        """
        voices = self.get_available_voices()
        language_voices = []

        for voice in voices:
            # Check if voice supports the language
            labels = voice.get("labels", {})
            supported_languages = labels.get("languages", [])

            if language in supported_languages:
                language_voices.append(voice)
            elif language == "en" and not supported_languages:
                # Default to English if no language specified
                language_voices.append(voice)

        return language_voices

    def get_voices_by_gender(self, gender: str) -> List[Dict[str, Any]]:
        """
        Get voices filtered by gender

        Args:
            gender: 'male' or 'female'

        Returns:
            List of voices matching gender
        """
        return self.get_voice_suggestions(gender)

    def create_dialogue_voice_mapping(
        self,
        speakers: List[str],
        language: str = "en",
        gender_preferences: Optional[Dict[str, str]] = None,
        custom_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Create optimized voice mapping for dialogue generation

        Args:
            speakers: List of speaker IDs
            language: Target language for voices
            gender_preferences: Optional mapping of speaker_id -> preferred gender
            custom_mapping: Optional custom speaker to voice ID mapping

        Returns:
            Dictionary mapping speaker IDs to optimal voice IDs for dialogue
        """
        voice_mapping = {}

        # Get available voices for the language
        available_voices = self.get_voices_by_language(language)
        if not available_voices:
            # Fallback to all voices if language-specific not found
            available_voices = self.get_available_voices()

        # Separate voices by gender if preferences specified
        male_voices = []
        female_voices = []
        neutral_voices = []

        for voice in available_voices:
            labels = voice.get("labels", {})
            gender = labels.get("gender", "").lower()

            if gender == "male":
                male_voices.append(voice)
            elif gender == "female":
                female_voices.append(voice)
            else:
                neutral_voices.append(voice)

        # Assign voices to speakers
        male_index = 0
        female_index = 0
        neutral_index = 0

        for speaker in speakers:
            # Check for custom mapping first
            if custom_mapping and speaker in custom_mapping:
                voice_mapping[speaker] = custom_mapping[speaker]
                continue

            # Check gender preference
            preferred_gender = (
                gender_preferences.get(speaker) if gender_preferences else None
            )

            if preferred_gender == "male" and male_voices:
                voice_mapping[speaker] = male_voices[male_index % len(male_voices)][
                    "voice_id"
                ]
                male_index += 1
            elif preferred_gender == "female" and female_voices:
                voice_mapping[speaker] = female_voices[
                    female_index % len(female_voices)
                ]["voice_id"]
                female_index += 1
            elif neutral_voices:
                voice_mapping[speaker] = neutral_voices[
                    neutral_index % len(neutral_voices)
                ]["voice_id"]
                neutral_index += 1
            elif available_voices:
                # Fallback to any available voice
                fallback_index = len(voice_mapping) % len(available_voices)
                voice_mapping[speaker] = available_voices[fallback_index]["voice_id"]
            else:
                # Last resort: use default mapping
                if speaker in self.DEFAULT_VOICE_MAPPING:
                    voice_mapping[speaker] = self.DEFAULT_VOICE_MAPPING[speaker]
                else:
                    default_voices = list(self.DEFAULT_VOICE_MAPPING.values())
                    voice_mapping[speaker] = default_voices[
                        len(voice_mapping) % len(default_voices)
                    ]

        logger.info(
            "Created dialogue voice mapping for language '%s': %s",
            language,
            voice_mapping,
        )
        return voice_mapping

    def analyze_voice_compatibility(self, voice_ids: List[str]) -> Dict[str, Any]:
        """
        Analyze compatibility of voices for dialogue generation

        Args:
            voice_ids: List of voice IDs to analyze

        Returns:
            Compatibility analysis results
        """
        voices_info = []
        for voice_id in voice_ids:
            info = self.get_voice_info(voice_id)
            if info:
                voices_info.append(info)

        if not voices_info:
            return {"compatible": False, "reason": "No valid voices found"}

        # Check language compatibility
        languages = set()
        for voice in voices_info:
            labels = voice.get("labels", {})
            voice_languages = labels.get("languages", ["en"])
            languages.update(voice_languages)

        # Check gender diversity
        genders = []
        for voice in voices_info:
            labels = voice.get("labels", {})
            gender = labels.get("gender", "unknown")
            genders.append(gender)

        gender_diversity = len(set(genders))

        analysis = {
            "compatible": True,
            "voice_count": len(voices_info),
            "languages": list(languages),
            "genders": genders,
            "gender_diversity": gender_diversity,
            "voices": [
                {
                    "voice_id": v["voice_id"],
                    "name": v.get("name", "Unknown"),
                    "gender": v.get("labels", {}).get("gender", "unknown"),
                    "languages": v.get("labels", {}).get("languages", ["en"]),
                }
                for v in voices_info
            ],
        }

        # Add recommendations
        recommendations = []
        if gender_diversity == 1 and len(voices_info) > 1:
            recommendations.append(
                "Consider using voices of different genders for better dialogue distinction"
            )

        if len(languages) > 1:
            recommendations.append(
                "Multiple languages detected - ensure all voices support the target language"
            )

        analysis["recommendations"] = recommendations

        return analysis

    def clone_voice(
        self,
        name: str,
        audio_files: List[str],
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        remove_background_noise: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a voice clone using ElevenLabs Instant Voice Cloning API

        Args:
            name: Name for the cloned voice
            audio_files: List of audio file paths for voice cloning
            description: Optional description of the voice
            labels: Optional labels dictionary for the voice
            remove_background_noise: Whether to remove background noise

        Returns:
            Dictionary containing voice creation results
        """
        try:
            # Prepare multipart form data
            files = []
            for audio_file in audio_files:
                if not Path(audio_file).exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_file}")

                with open(audio_file, "rb") as f:
                    files.append(
                        ("files", (Path(audio_file).name, f.read(), "audio/wav"))
                    )

            # Prepare form data
            data = {
                "name": name,
                "remove_background_noise": str(remove_background_noise).lower(),
            }

            if description:
                data["description"] = description

            if labels:
                data["labels"] = json.dumps(labels)

            # Make API request
            response = self.session.post(
                f"{self.base_url}/voices/add",
                files=files,
                data=data,
                timeout=120,  # Voice creation can take time
            )

            response.raise_for_status()
            result = response.json()

            voice_id = result.get("voice_id")
            logger.info("Successfully created voice clone: %s (ID: %s)", name, voice_id)

            return {
                "success": True,
                "voice_id": voice_id,
                "name": name,
                "description": description,
                "labels": labels,
                "audio_files": audio_files,
            }

        except Exception as e:
            logger.error("Failed to clone voice '%s': %s", name, e)
            return {
                "success": False,
                "error": str(e),
                "name": name,
                "audio_files": audio_files,
            }

    def clone_voices_from_samples(
        self,
        speaker_samples: Dict[str, str],
        language: str = "fa",
        name_prefix: str = "cloned",
    ) -> Dict[str, str]:
        """
        Clone voices for multiple speakers from audio samples

        Args:
            speaker_samples: Dict mapping speaker_id to audio file path
            language: Language code for the voices
            name_prefix: Prefix for generated voice names

        Returns:
            Dict mapping speaker_id to cloned voice_id
        """
        voice_mapping = {}

        for speaker_id, audio_file in speaker_samples.items():
            voice_name = f"{name_prefix}_{speaker_id}_{language}"

            # Add language-specific labels
            labels = {
                "language": language,
                "source": "voice_cloning",
                "speaker_id": speaker_id,
            }

            description = f"Voice cloned from {speaker_id} speaking {language}"

            result = self.clone_voice(
                name=voice_name,
                audio_files=[audio_file],
                description=description,
                labels=labels,
                remove_background_noise=True,  # Clean up audio
            )

            if result["success"]:
                voice_mapping[speaker_id] = result["voice_id"]
                logger.info("Mapped %s -> %s", speaker_id, result["voice_id"])
            else:
                logger.error(
                    "Failed to clone voice for %s: %s", speaker_id, result["error"]
                )
                # Fallback to default voice
                if speaker_id in self.DEFAULT_VOICE_MAPPING:
                    voice_mapping[speaker_id] = self.DEFAULT_VOICE_MAPPING[speaker_id]
                    logger.warning(
                        "Using fallback voice for %s: %s",
                        speaker_id,
                        voice_mapping[speaker_id],
                    )

        return voice_mapping

    def delete_voice(self, voice_id: str) -> bool:
        """
        Delete a cloned voice

        Args:
            voice_id: ID of the voice to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.session.delete(f"{self.base_url}/voices/{voice_id}")
            response.raise_for_status()
            logger.info("Successfully deleted voice: %s", voice_id)
            return True
        except Exception as e:
            logger.error("Failed to delete voice %s: %s", voice_id, e)
            return False

    def list_cloned_voices(self) -> List[Dict[str, Any]]:
        """
        Get list of user's cloned voices

        Returns:
            List of cloned voice dictionaries
        """
        voices = self.get_available_voices()
        cloned_voices = []

        for voice in voices:
            # Check if this is a cloned voice (has certain characteristics)
            labels = voice.get("labels", {})
            if (
                labels.get("source") == "voice_cloning"
                or "cloned" in voice.get("name", "").lower()
            ):
                cloned_voices.append(voice)

        return cloned_voices
