"""
Dialogue Generation Workflow
Orchestrates the complete STT -> Text Editing -> Voice Assignment -> TTS pipeline
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

from ..audio.dialogue import (
    DialogueBase,
    TranscriptToDialogueConverter,
    DialogueGenerator,
    CharacterManager,
)
from ..audio.providers.elevenlabs.voice import VoiceManager
from ..utils.config import Config

logger = logging.getLogger(__name__)


class DialogueWorkflow:
    """
    Complete workflow for dialogue generation from STT transcripts
    """

    def __init__(self, config: Config):
        """
        Initialize dialogue workflow

        Args:
            config: Configuration object
        """
        self.config = config
        self.voice_manager = None
        self.character_manager = None
        self.converter = None
        self.generator = None

        # Initialize components if API key available
        if hasattr(config, "elevenlabs_api_key") and config.elevenlabs_api_key:
            self._initialize_components()

    def _initialize_components(self):
        """Initialize workflow components"""
        try:
            self.voice_manager = VoiceManager(self.config.elevenlabs_api_key)
            self.character_manager = CharacterManager(self.voice_manager)
            self.converter = TranscriptToDialogueConverter(self.voice_manager)
            self.generator = DialogueGenerator(self.config.elevenlabs_api_key)

            if self.config.verbose:
                print("✅ Dialogue workflow components initialized")

        except Exception as e:
            logger.error("Failed to initialize dialogue components: %s", e)
            if self.config.verbose:
                print(f"⚠️  Warning: Dialogue components not available: {e}")

    def run_stt_to_dialogue_workflow(
        self,
        transcript_file: Path,
        output_dir: Path,
        language: str = "en",
        custom_voice_mapping: Optional[Dict[str, str]] = None,
        gender_preferences: Optional[Dict[str, str]] = None,
        use_character_profiles: bool = False,
        character_profiles_file: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run complete STT to dialogue workflow

        Args:
            transcript_file: Path to STT transcript file
            output_dir: Output directory for generated files
            language: Target language for voice selection
            custom_voice_mapping: Optional custom speaker -> voice mapping
            gender_preferences: Optional gender preferences for speakers
            use_character_profiles: Whether to use character profile system
            character_profiles_file: Optional character profiles file

        Returns:
            Workflow results with file paths and statistics
        """
        if not self._check_components():
            raise ValueError("Dialogue workflow components not initialized")

        results = {
            "input_file": str(transcript_file),
            "output_dir": str(output_dir),
            "success": False,
            "files_generated": {},
            "statistics": {},
            "errors": [],
        }

        try:
            # Step 1: Load character profiles if specified
            if (
                use_character_profiles
                and character_profiles_file
                and character_profiles_file.exists()
            ):
                if self.config.verbose:
                    print("📋 Loading character profiles...")
                self.character_manager.load_profiles(character_profiles_file)
                results["character_profiles_loaded"] = True

            # Step 2: Convert transcript to dialogue
            if self.config.verbose:
                print("🔄 Converting transcript to dialogue format...")

            # Merge custom voice mapping with environment voice mapping
            from ..audio.voice import get_voice_mapping

            env_voice_mapping = get_voice_mapping()

            final_voice_mapping = {}
            if env_voice_mapping:
                final_voice_mapping.update(env_voice_mapping)
            if custom_voice_mapping:
                final_voice_mapping.update(custom_voice_mapping)

            if self.config.verbose and final_voice_mapping:
                print(f"🗣️  Using voice mapping: {final_voice_mapping}")

            dialogue = self.converter.convert_transcript_to_dialogue(
                transcript_source=str(transcript_file),
                language=language,
                custom_voice_mapping=final_voice_mapping,
                gender_preferences=gender_preferences,
                auto_assign=True,
            )

            # Step 3: Analyze dialogue
            dialogue_stats = self.converter.analyze_dialogue_complexity(dialogue)
            results["statistics"] = dialogue_stats

            if self.config.verbose:
                print(
                    f"📊 Analyzed dialogue: {dialogue_stats['basic_stats']['total_segments']} segments, "
                    f"{dialogue_stats['basic_stats']['unique_speakers']} speakers"
                )

            # Step 4: Save dialogue files
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save dialogue JSON
            dialogue_file = output_dir / f"{transcript_file.stem}_dialogue.json"
            dialogue.to_json(dialogue_file)
            results["files_generated"]["dialogue_json"] = str(dialogue_file)

            # Save ElevenLabs format
            elevenlabs_file = (
                output_dir / f"{transcript_file.stem}_elevenlabs_dialogue.json"
            )
            self.converter.export_for_elevenlabs_dialogue_api(dialogue, elevenlabs_file)
            results["files_generated"]["elevenlabs_format"] = str(elevenlabs_file)

            # Save voice mapping
            voice_mapping_file = (
                output_dir / f"{transcript_file.stem}_voice_mapping.json"
            )
            with open(voice_mapping_file, "w") as f:
                json.dump(dialogue.metadata.get("voice_mapping", {}), f, indent=2)
            results["files_generated"]["voice_mapping"] = str(voice_mapping_file)

            # Step 5: Test API availability
            if self.config.verbose:
                print("🔍 Testing ElevenLabs Text to Dialogue API availability...")

            api_status = self.generator.test_dialogue_api_availability()
            results["api_status"] = api_status

            # Step 6: Generate audio (if API available)
            if api_status.get("available", False):
                if self.config.verbose:
                    print("🎵 Generating dialogue audio...")

                audio_file = output_dir / f"{transcript_file.stem}_dialogue.mp3"

                # Get cost estimate
                cost_estimate = self.generator.estimate_generation_cost(dialogue)
                results["cost_estimate"] = cost_estimate

                if self.config.verbose:
                    print(f"💰 Estimated cost: ${cost_estimate['estimated_cost_usd']}")
                    print(
                        f"⏱️  Estimated time: {cost_estimate['estimated_time_minutes']} minutes"
                    )

                # Generate audio
                audio_success = self.generator.generate_dialogue_with_fallback(
                    dialogue=dialogue, output_file=audio_file, use_fallback=True
                )

                if audio_success:
                    results["files_generated"]["audio"] = str(audio_file)
                    if self.config.verbose:
                        print(f"✅ Audio generated: {audio_file}")
                else:
                    results["errors"].append("Audio generation failed")
            else:
                if self.config.verbose:
                    print(
                        f"⚠️  Text to Dialogue API not available: {api_status.get('status', 'Unknown')}"
                    )
                results["errors"].append(
                    f"API not available: {api_status.get('status', 'Unknown')}"
                )

            # Step 7: Save character profiles if created
            if use_character_profiles:
                profiles_file = (
                    output_dir / f"{transcript_file.stem}_character_profiles.json"
                )
                self.character_manager.save_profiles(profiles_file)
                results["files_generated"]["character_profiles"] = str(profiles_file)

            results["success"] = True

            if self.config.verbose:
                print("✅ Dialogue workflow completed successfully")

        except Exception as e:
            error_msg = f"Dialogue workflow failed: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            if self.config.verbose:
                print(f"❌ {error_msg}")

        return results

    def preview_dialogue_generation(
        self,
        transcript_file: Path,
        language: str = "en",
        custom_voice_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Preview dialogue generation without actually generating audio

        Args:
            transcript_file: Path to STT transcript file
            language: Target language for voice selection
            custom_voice_mapping: Optional custom speaker -> voice mapping

        Returns:
            Preview information
        """
        if not self._check_components():
            raise ValueError("Dialogue workflow components not initialized")

        try:
            # Merge custom voice mapping with environment voice mapping
            from ..audio.voice import get_voice_mapping

            env_voice_mapping = get_voice_mapping()

            final_voice_mapping = {}
            if env_voice_mapping:
                final_voice_mapping.update(env_voice_mapping)
            if custom_voice_mapping:
                final_voice_mapping.update(custom_voice_mapping)

            # Convert transcript to dialogue
            dialogue = self.converter.convert_transcript_to_dialogue(
                transcript_source=str(transcript_file),
                language=language,
                custom_voice_mapping=final_voice_mapping,
                auto_assign=True,
            )

            # Get preview
            preview = self.generator.preview_dialogue_generation(dialogue)

            # Add analysis
            analysis = self.converter.analyze_dialogue_complexity(dialogue)
            preview["complexity_analysis"] = analysis

            # Add voice compatibility analysis
            voice_ids = [seg.voice_id for seg in dialogue.segments if seg.voice_id]
            if voice_ids:
                compatibility = self.voice_manager.analyze_voice_compatibility(
                    voice_ids
                )
                preview["voice_compatibility"] = compatibility

            return preview

        except Exception as e:
            logger.error("Preview generation failed: %s", e)
            return {"error": str(e)}

    def interactive_voice_assignment(
        self, transcript_file: Path, language: str = "en"
    ) -> Dict[str, str]:
        """
        Interactive voice assignment for speakers

        Args:
            transcript_file: Path to STT transcript file
            language: Target language for voice selection

        Returns:
            Selected voice mapping
        """
        if not self._check_components():
            raise ValueError("Dialogue workflow components not initialized")

        # Parse transcript to get speakers
        segments = self.converter.parse_transcript_file(transcript_file)
        speakers = list(set(seg.speaker_id for seg in segments))

        # Get available voices for language
        available_voices = self.voice_manager.get_voices_by_language(language)

        if not available_voices:
            available_voices = self.voice_manager.get_all_voices()

        voice_mapping = {}

        print(f"\n🎭 Interactive Voice Assignment")
        print(f"Language: {language}")
        print(f"Speakers found: {len(speakers)}")
        print(f"Available voices: {len(available_voices)}")
        print("\n" + "=" * 50)

        for i, speaker in enumerate(speakers):
            print(f"\n👤 Speaker: {speaker}")

            # Show voice options
            print("Available voices:")
            for j, voice in enumerate(available_voices[:10]):  # Show top 10
                gender = voice.get("labels", {}).get("gender", "unknown")
                name = voice.get("name", "Unknown")
                voice_id = voice.get("voice_id", "")
                print(f"  {j+1}. {name} ({gender}) - {voice_id[:8]}...")

            # Get user selection
            while True:
                try:
                    choice = input(
                        f"Select voice for {speaker} (1-{min(10, len(available_voices))}): "
                    )
                    choice_idx = int(choice) - 1

                    if 0 <= choice_idx < min(10, len(available_voices)):
                        selected_voice = available_voices[choice_idx]
                        voice_mapping[speaker] = selected_voice["voice_id"]
                        print(f"✅ Assigned {selected_voice['name']} to {speaker}")
                        break
                    else:
                        print("❌ Invalid selection. Please try again.")
                except (ValueError, KeyboardInterrupt):
                    print("\n❌ Voice assignment cancelled")
                    return {}

        print(f"\n✅ Voice assignment completed!")
        print("Final mapping:")
        for speaker, voice_id in voice_mapping.items():
            voice_info = self.voice_manager.get_voice_info(voice_id)
            voice_name = voice_info.get("name", "Unknown") if voice_info else "Unknown"
            print(f"  {speaker} -> {voice_name} ({voice_id[:8]}...)")

        return voice_mapping

    def _check_components(self) -> bool:
        """Check if workflow components are initialized"""
        return all(
            [self.voice_manager, self.character_manager, self.converter, self.generator]
        )

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get status of workflow components"""
        return {
            "components_initialized": self._check_components(),
            "voice_manager_available": self.voice_manager is not None,
            "character_manager_available": self.character_manager is not None,
            "converter_available": self.converter is not None,
            "generator_available": self.generator is not None,
            "api_key_configured": hasattr(self.config, "elevenlabs_api_key")
            and self.config.elevenlabs_api_key is not None,
        }

    def create_sample_character_profiles(
        self, output_file: Path, language: str = "en"
    ) -> bool:
        """
        Create sample character profiles for testing

        Args:
            output_file: Output file for character profiles
            language: Language for character voices

        Returns:
            True if successful
        """
        if not self._check_components():
            return False

        try:
            # Get some voices for the language
            voices = self.voice_manager.get_voices_by_language(language)[:4]

            if not voices:
                voices = self.voice_manager.get_all_voices()[:4]

            # Create sample characters
            for i, voice in enumerate(voices):
                character_id = f"character_{i+1}"
                name = f"Character {i+1}"

                character = self.character_manager.create_character_from_voice(
                    voice_id=voice["voice_id"], character_id=character_id, name=name
                )

                # Add some sample traits
                character.personality_traits = ["friendly", "articulate"]
                character.speaking_style = [
                    "calm",
                    "energetic",
                    "authoritative",
                    "dramatic",
                ][i % 4]

            # Save profiles
            self.character_manager.save_profiles(output_file)

            if self.config.verbose:
                print(f"✅ Sample character profiles created: {output_file}")

            return True

        except Exception as e:
            logger.error("Failed to create sample character profiles: %s", e)
            return False
