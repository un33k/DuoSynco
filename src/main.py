#!/usr/bin/env python
"""
DuoSynco - Video/Audio Synchronization Tool
Main CLI entry point for processing videos with speaker isolation using AssemblyAI
"""

import click
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys
import logging

from .utils.util_files import FileHandler
from .utils.config import Config
from .video.video_sync import VideoSynchronizer


@click.command()
@click.argument('input_file', type=click.Path(path_type=Path), required=False)
@click.option('--output-dir', '-o',
              type=click.Path(path_type=Path),
              default='./output',
              help='Output directory for processed files')
@click.option('--speakers', '-s',
              type=int,
              default=2,
              help='Number of speakers to separate (default: 2)')
@click.option('--format', '-f',
              type=click.Choice(['mp4', 'avi', 'mov']),
              default='mp4',
              help='Output video format')
@click.option('--quality', '-q',
              type=click.Choice(['low', 'medium', 'high']),
              default='medium',
              help='Processing quality level')
@click.option('--language', '-l',
              type=str,
              default='en',
              help='Audio language code (default: en)')
@click.option('--enhanced-processing',
              is_flag=True,
              default=True,
              help='Enable enhanced voice separation (default: enabled)')
@click.option('--provider', '-p',
              type=click.Choice(['assemblyai', 'elevenlabs']),
              default='assemblyai',
              help='Primary provider for the operation (determines execution path, cost, and features)')
@click.option('--api-key',
              type=str,
              help='API key for the provider (or set env var)')
@click.option('--list-providers',
              is_flag=True,
              help='List available providers and exit')
@click.option('--verbose', '-v',
              is_flag=True,
              help='Enable verbose output')
@click.option('--mode',
              type=click.Choice(['diarization', 'tts', 'edit']),
              default='diarization',
              help='Processing mode: diarization (default), tts generation, or edit workflow')
@click.option('--transcript-file',
              type=click.Path(path_type=Path),
              help='Transcript file for TTS mode (JSON or TXT format)')
@click.option('--total-duration',
              type=float,
              help='Total duration in seconds for TTS mode')
@click.option('--voice-mapping',
              type=str,
              help='Voice mapping as JSON string (e.g., \'{"A": "voice-id", "B": "voice-id"}\') or "auto" to use .env-local')
@click.option('--tts-workers',
              type=int,
              default=3,
              help='Number of concurrent TTS workers (default: 3)')
@click.option('--list-voices',
              is_flag=True,
              help='List available TTS voices and exit')
@click.option('--show-config',
              is_flag=True,
              help='Show current environment configuration and exit')
@click.option('--tts-quality',
              type=click.Choice(['low', 'medium', 'high', 'ultra']),
              default='high',
              help='TTS quality level (default: high)')
@click.option('--timing-mode',
              type=click.Choice(['adaptive', 'strict']),
              default='adaptive',
              help='Timing mode: adaptive (adjust for voice speed) or strict (preserve original)')
@click.option('--gap-duration',
              type=float,
              default=0.4,
              help='Gap between speakers in seconds (default: 0.4)')
@click.option('--secondary-provider', '-sp',
              type=click.Choice(['assemblyai', 'elevenlabs']),
              help='Secondary provider for multi-step operations (edit mode only)')
@click.option('--stt-quality', '-sq',
              type=click.Choice(['low', 'medium', 'high', 'ultra']),
              default='high',
              help='STT quality level when using speech-to-text (default: high)')
@click.option('--edit-interactive', '-ei',
              is_flag=True,
              help='Enable interactive editing mode with prompts')
@click.option('--speaker-mapping', '-sm',
              type=str,
              help='Speaker mapping file (JSON) for edit mode')
@click.option('--output-transcript', '-ot',
              type=click.Path(path_type=Path),
              help='Output path for edited transcript (edit mode)')
@click.option('--list-execution-paths', '-lep',
              is_flag=True,
              help='List available execution paths for each provider and exit')
def cli(input_file: Optional[Path],
        output_dir: Path,
        speakers: int,
        format: str,
        quality: str,
        language: str,
        enhanced_processing: bool,
        provider: str,
        api_key: Optional[str],
        list_providers: bool,
        verbose: bool,
        mode: str,
        transcript_file: Optional[Path],
        total_duration: Optional[float],
        voice_mapping: Optional[str],
        tts_workers: int,
        list_voices: bool,
        show_config: bool,
        tts_quality: str,
        timing_mode: str,
        gap_duration: float,
        secondary_provider: Optional[str],
        stt_quality: str,
        edit_interactive: bool,
        speaker_mapping: Optional[str],
        output_transcript: Optional[Path],
        list_execution_paths: bool):
    """
    DuoSynco - Sync videos with isolated speaker audio tracks

    Takes an input file and creates processed output files based on the selected mode:
    - Diarization/Edit modes: Input should be audio/video file
    - TTS mode: Input should be transcript file (JSON/TXT format)

    PROVIDER-BASED EXECUTION PATHS:
    - assemblyai: Professional diarization workflow (cost-effective)
    - elevenlabs: Premium STT/TTS with advanced features

    INPUT_FILE: Path to the input file (audio/video for diarization/edit, transcript for TTS)
    """

    # Set up logging
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(levelname)s: %(message)s'
    )

    # Handle list-execution-paths option
    if list_execution_paths:
        show_execution_paths()
        return

    # Handle list-providers option
    if list_providers:
        from .audio.audio_diarization import SpeakerDiarizer
        providers = SpeakerDiarizer.list_providers()
        click.echo("🔧 Available Speaker Diarization Providers:")
        for provider_name, info in providers.items():
            status = "✅ Available" if info['available'] else "❌ Not Available"
            api_key_req = " (requires API key)" if info.get('requires_api_key') else ""
            click.echo(f"  {provider_name}: {status}{api_key_req}")
            if not info['available'] and 'error' in info:
                click.echo(f"    Error: {info['error']}")
        return

    # Handle list-voices option
    if list_voices:
        from .audio.audio_tts import TTSAudioGenerator
        try:
            tts_generator = TTSAudioGenerator(provider='elevenlabs', api_key=api_key)
            voices_info = tts_generator.list_available_voices()
            
            click.echo("🗣️  Available ElevenLabs Voices:")
            if voices_info['total_voices'] > 0:
                for voice in voices_info['available_voices'][:10]:  # Show first 10
                    name = voice.get('name', 'Unknown')
                    voice_id = voice.get('voice_id', 'Unknown')
                    gender = voice.get('labels', {}).get('gender', 'Unknown')
                    click.echo(f"  {name} ({gender}): {voice_id}")
                
                if voices_info['total_voices'] > 10:
                    click.echo(f"  ... and {voices_info['total_voices'] - 10} more voices")
                    
                click.echo("\n🎯 Default Voice Mapping:")
                for voice_id, info in voices_info['default_voices'].items():
                    name = info.get('name', 'Unknown')
                    click.echo(f"  {name}: {voice_id}")
            else:
                click.echo("  No voices available or API key invalid")
                
        except Exception as e:
            click.echo(f"❌ Error retrieving voices: {e}", err=True)
        return

    # Handle show-config option
    if show_config:
        from .utils.util_env import env
        env.print_config()
        
        # Show voice mapping if available
        voice_map = env.get_voice_mapping()
        if voice_map:
            click.echo("\n🗣️  Voice Mapping:")
            for speaker, voice_id in voice_map.items():
                click.echo(f"  {speaker}: {voice_id}")
        else:
            click.echo("\n🗣️  No voice mapping configured")
            click.echo("  Set VOICE_SPEAKER_A and VOICE_SPEAKER_B in .env-local")
        return

    # Handle TTS mode
    if mode == 'tts':
        # For TTS mode, input_file should be the transcript file
        if input_file is None:
            click.echo("❌ Error: Transcript file is required for TTS mode.", err=True)
            click.echo("Usage: python -m src.main transcript.json -p elevenlabs --mode tts", err=True)
            sys.exit(1)
        
        return handle_tts_mode(
            input_file, total_duration, output_dir, api_key,
            voice_mapping, tts_workers, tts_quality, timing_mode, gap_duration, verbose
        )
    
    # Handle Edit mode
    if mode == 'edit':
        return handle_edit_mode(
            input_file, output_dir, speakers, language, provider, secondary_provider,
            stt_quality, edit_interactive, speaker_mapping, output_transcript,
            api_key, verbose
        )

    # Validate input file is provided for diarization mode
    if input_file is None:
        click.echo("❌ Error: INPUT_FILE is required for diarization mode.", err=True)
        click.echo("Use --help for usage information.", err=True)
        sys.exit(1)

    # Check if input file exists
    if not input_file.exists():
        click.echo(f"❌ Error: Input file '{input_file}' does not exist.", err=True)
        sys.exit(1)

    # Initialize configuration
    config = Config(quality=quality, output_format=format, verbose=verbose)
    file_handler = FileHandler(config)

    try:
        # Validate input file
        if not file_handler.validate_input_file(input_file):
            click.echo(f"❌ Error: Invalid input file {input_file}", err=True)
            sys.exit(1)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            click.echo(f"📁 Processing: {input_file}")
            click.echo(f"📁 Output directory: {output_dir}")
            click.echo(f"👥 Expected speakers: {speakers}")
            click.echo(f"🌍 Language: {language}")
            click.echo(f"⚡ Enhanced processing: {enhanced_processing}")

        # Step 1: Speaker Diarization
        click.echo(f"🔍 Analyzing speakers with {provider}...")

        try:
            from .audio.audio_diarization import SpeakerDiarizer
            diarizer = SpeakerDiarizer(provider=provider, api_key=api_key)
        except ValueError as e:
            click.echo(f"❌ Error: {e}", err=True)
            if provider.lower() == 'assemblyai':
                click.echo(
                    "💡 Get your API key from: https://www.assemblyai.com/",
                    err=True
                )
                click.echo(
                    "💡 Set it with: export ASSEMBLYAI_API_KEY=your_key",
                    err=True
                )
            sys.exit(1)

        # Perform speaker separation
        result = diarizer.separate_speakers(
            audio_file=str(input_file),
            output_dir=str(output_dir),
            speakers_expected=speakers,
            language=language,
            enhanced_processing=enhanced_processing,
            base_filename=input_file.stem
        )

        # Display results
        stats = result['stats']
        click.echo("✅ Speaker separation completed!")
        click.echo(
            f"📊 Coverage: {stats['total_coverage']:.1f}% "
            f"({stats['total_speaker_duration']:.1f}s / "
            f"{stats['original_duration']:.1f}s)"
        )

        for speaker, speaker_stats in stats['speakers'].items():
            click.echo(
                f"  {speaker}: {speaker_stats['duration']:.1f}s "
                f"({speaker_stats['coverage']:.1f}%)"
            )

        # Check if input is audio-only or video
        file_info = file_handler.get_file_info(input_file)

        if file_info and file_info.is_audio and not file_info.is_video:
            # Audio-only processing - we already have the separated files
            click.echo("🎵 Audio-only processing completed!")
            click.echo(f"📄 Transcript: {result['transcript_file']}")
            click.echo("🎵 Separated audio files:")
            for audio_file in result['speaker_files']:
                click.echo(f"  - {audio_file}")
        else:
            # Step 2: Video Processing - Sync videos with separated audio
            click.echo("🎬 Synchronizing videos with separated audio...")

            synchronizer = VideoSynchronizer(config)

            # Generate synchronized video files for each speaker
            video_files = []
            for i, audio_file in enumerate(result['speaker_files']):
                speaker_name = (
                    result['speakers'][i] if i < len(result['speakers'])
                    else f"speaker_{i+1}"
                )
                output_file = (
                    output_dir /
                    f"{input_file.stem}_{speaker_name.lower()}.{format}"
                )

                if verbose:
                    click.echo(f"🎬 Creating video: {output_file}")

                # Synchronize video with isolated audio
                synchronizer.sync_video_audio(
                    input_file, Path(audio_file), output_file
                )

                video_files.append(output_file)

            click.echo("✅ Video synchronization completed!")
            click.echo(f"📄 Transcript: {result['transcript_file']}")
            click.echo("🎵 Separated audio files:")
            for audio_file in result['speaker_files']:
                click.echo(f"  - {audio_file}")
            click.echo("🎬 Synchronized video files:")
            for video_file in video_files:
                click.echo(f"  - {video_file}")

    except Exception as e:
        click.echo(f"❌ Error processing file: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def handle_tts_mode(
    transcript_file: Path,
    total_duration: Optional[float],
    output_dir: Path,
    api_key: Optional[str],
    voice_mapping: Optional[str],
    tts_workers: int,
    tts_quality: str,
    timing_mode: str,
    gap_duration: float,
    verbose: bool
) -> None:
    """Handle TTS generation mode"""
    import json
    
    # Validate required parameters for TTS mode
    if total_duration is None:
        click.echo("❌ Error: --total-duration is required for TTS mode.", err=True)
        sys.exit(1)
        
    if not transcript_file.exists():
        click.echo(f"❌ Error: Transcript file '{transcript_file}' does not exist.", err=True)
        sys.exit(1)
        
    # Parse voice mapping
    parsed_voice_mapping = None
    if voice_mapping:
        if voice_mapping.lower() == "auto":
            # Load from environment
            from .utils.util_env import get_voice_mapping
            parsed_voice_mapping = get_voice_mapping()
            if not parsed_voice_mapping:
                click.echo("❌ Error: No voice mapping found in .env-local", err=True)
                click.echo("💡 Set VOICE_SPEAKER_A and VOICE_SPEAKER_B in .env-local", err=True)
                sys.exit(1)
        else:
            try:
                parsed_voice_mapping = json.loads(voice_mapping)
            except json.JSONDecodeError as e:
                click.echo(f"❌ Error: Invalid voice mapping JSON: {e}", err=True)
                sys.exit(1)
    
    try:
        # Load transcript segments
        transcript_segments = load_transcript_file(transcript_file)
        
        if verbose:
            click.echo(f"📄 Loaded {len(transcript_segments)} transcript segments")
            click.echo(f"⏱️  Total duration: {total_duration}s")
            click.echo(f"🎯 TTS quality: {tts_quality}")
            click.echo(f"⌚ Timing mode: {timing_mode}")
            click.echo(f"⏸️  Speaker gap: {gap_duration}s")
            if parsed_voice_mapping:
                click.echo(f"🗣️  Voice mapping: {parsed_voice_mapping}")
        
        # Initialize TTS generator
        from .audio.audio_tts import TTSAudioGenerator
        tts_generator = TTSAudioGenerator(provider='elevenlabs', api_key=api_key)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate audio tracks
        click.echo(f"🗣️  Generating {tts_quality} quality TTS audio tracks ({timing_mode} timing)...")
        result = tts_generator.generate_audio_tracks(
            transcript_segments=transcript_segments,
            total_duration=total_duration,
            output_dir=str(output_dir),
            base_filename=transcript_file.stem,
            voice_mapping=parsed_voice_mapping,
            max_workers=tts_workers,
            quality=tts_quality,
            timing_mode=timing_mode,
            gap_duration=gap_duration
        )
        
        # Display results
        stats = result['stats']
        click.echo("✅ TTS generation completed!")
        click.echo(f"📊 Generated {len(result['audio_files'])} audio tracks")
        click.echo(f"👥 Speakers: {', '.join(result['speakers'])}")
        click.echo(f"📝 Total segments: {stats['total_segments']}")
        click.echo(f"⏱️  Total speech duration: {stats['total_speech_duration']:.1f}s")
        click.echo(f"🔤 Total characters: {stats['total_characters']}")
        
        click.echo("\n🎵 Generated audio files:")
        for audio_file in result['audio_files']:
            click.echo(f"  - {audio_file}")
            
        click.echo(f"\n🗣️  Voice mapping used:")
        for speaker, voice_id in result['voice_mapping'].items():
            click.echo(f"  {speaker}: {voice_id}")
            
    except Exception as e:
        click.echo(f"❌ Error in TTS generation: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def load_transcript_file(file_path: Path) -> List[Dict]:
    """Load transcript segments from file"""
    import json
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Try to parse as JSON first
        try:
            data = json.loads(content)
            
            # Handle different JSON formats
            if isinstance(data, list):
                # Direct list of segments
                return data
            elif isinstance(data, dict):
                # Check for common keys
                if 'segments' in data:
                    return data['segments']
                elif 'utterances' in data:
                    return data['utterances']
                else:
                    # Assume it's a single segment
                    return [data]
            else:
                raise ValueError("Unsupported JSON format")
                
        except json.JSONDecodeError:
            # Try to parse as simple text format
            # Format: "SPEAKER_ID (start-end): text"
            segments = []
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('='):
                    continue
                    
                # Parse format: "A (1.23s-5.67s): Hello world"
                if ':' in line and '(' in line and ')' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        speaker_part = parts[0].strip()
                        text = parts[1].strip()
                        
                        # Extract speaker and timing
                        if '(' in speaker_part and ')' in speaker_part:
                            speaker = speaker_part.split('(')[0].strip()
                            timing_part = speaker_part.split('(')[1].split(')')[0]
                            
                            if '-' in timing_part:
                                start_str, end_str = timing_part.split('-')
                                start = float(start_str.replace('s', ''))
                                end = float(end_str.replace('s', ''))
                                
                                segments.append({
                                    'speaker': speaker,
                                    'start': start,
                                    'end': end,
                                    'text': text
                                })
            
            if segments:
                return segments
            else:
                raise ValueError("Could not parse transcript file format")
                
    except Exception as e:
        raise ValueError(f"Failed to load transcript file: {e}")


def handle_edit_mode(
    input_file: Optional[Path],
    output_dir: Path,
    speakers: int,
    language: str,
    primary_provider: str,
    secondary_provider: Optional[str],
    stt_quality: str,
    edit_interactive: bool,
    speaker_mapping: Optional[str],
    output_transcript: Optional[Path],
    api_key: Optional[str],
    verbose: bool
) -> None:
    """Handle Edit workflow mode with provider-based execution paths"""
    
    # Determine execution path based on provider
    execution_path = determine_execution_path(primary_provider, secondary_provider)
    
    # Validate required parameters for edit mode
    if input_file is None:
        click.echo("❌ Error: INPUT_FILE is required for edit mode.", err=True)
        sys.exit(1)
        
    if not input_file.exists():
        click.echo(f"❌ Error: Input file '{input_file}' does not exist.", err=True)
        sys.exit(1)
    
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            click.echo(f"🎤 Edit Mode Workflow Starting")
            click.echo(f"📁 Input file: {input_file}")
            click.echo(f"📁 Output directory: {output_dir}")
            click.echo(f"🛤️  Execution Path: {execution_path['name']}")
            click.echo(f"🗣️  STT Provider: {execution_path['stt_provider']}")
            click.echo(f"🔧 Final Provider: {execution_path['final_provider']}")
            click.echo(f"⚡ STT Quality: {stt_quality}")
            click.echo(f"💰 Cost Profile: {execution_path['cost_profile']}")
            click.echo(f"🎯 Features: {', '.join(execution_path['features'])}")
        
        # Step 1: Speech-to-Text with Speaker Diarization
        stt_provider = execution_path['stt_provider']
        click.echo(f"🎤 Step 1: Transcribing audio with {stt_provider}...")
        
        from .audio.audio_stt import STTAudioTranscriber
        
        try:
            stt_transcriber = STTAudioTranscriber(provider=stt_provider, api_key=api_key)
        except ValueError as e:
            click.echo(f"❌ Error: {e}", err=True)
            if stt_provider.lower() == 'elevenlabs-stt':
                click.echo("💡 Get your API key from: https://elevenlabs.io/", err=True)
                click.echo("💡 Set it with: export ELEVENLABS_API_KEY=your_key", err=True)
            sys.exit(1)
        
        # Perform STT transcription
        stt_result = stt_transcriber.transcribe_audio_file(
            audio_file=str(input_file),
            output_dir=str(output_dir),
            base_filename=f"{input_file.stem}_stt",
            speakers_expected=speakers,
            language=language,
            quality=stt_quality,
            enhanced_processing=True,
            save_results=True
        )
        
        click.echo(f"✅ STT completed: {len(stt_result['utterances'])} utterances, {len(stt_result['speakers'])} speakers")
        
        # Step 2: Text Editing and Speaker Replacement
        click.echo("✏️  Step 2: Text editing and speaker replacement...")
        
        from .text import TranscriptEditor, SpeakerReplacer
        
        # Initialize editor and replacer
        editor = TranscriptEditor()
        replacer = SpeakerReplacer(editor)
        
        # Load the STT results as transcript data
        transcript_data = {
            'utterances': stt_result['utterances'],
            'speakers': stt_result['speakers'],
            'duration': stt_result['duration'],
            'language': stt_result['language'],
            'provider': stt_result['provider']
        }
        editor.transcript_data = transcript_data
        editor.original_data = transcript_data.copy()
        
        # Load speaker mapping rules if provided
        if speaker_mapping and Path(speaker_mapping).exists():
            click.echo(f"📋 Loading speaker mapping rules: {speaker_mapping}")
            replacer.load_replacement_rules(speaker_mapping)
            
            # Apply replacement rules
            replacement_results = replacer.apply_replacement_rules()
            total_replaced = sum(replacement_results.values())
            
            if total_replaced > 0:
                click.echo(f"✅ Applied speaker replacements: {total_replaced} utterances modified")
                for old_speaker, count in replacement_results.items():
                    if count > 0:
                        click.echo(f"  {old_speaker}: {count} utterances")
            else:
                click.echo("ℹ️  No speaker replacements applied")
        else:
            # Detect patterns and suggest improvements
            patterns = replacer.detect_speaker_patterns()
            
            if patterns.get('suggestions'):
                click.echo("💡 Speaker analysis suggestions:")
                for suggestion in patterns['suggestions']:
                    click.echo(f"  - {suggestion}")
            
            # Interactive editing if requested
            if edit_interactive:
                click.echo("\n🎯 Interactive editing mode:")
                handle_interactive_editing(editor, replacer)
        
        # Save edited transcript
        edited_transcript_file = output_transcript or (output_dir / f"{input_file.stem}_edited_transcript.json")
        editor.save_transcript(str(edited_transcript_file), format="json", backup_original=False)
        
        click.echo(f"💾 Saved edited transcript: {edited_transcript_file}")
        
        # Step 3: Final Diarization
        final_provider = execution_path['final_provider']
        click.echo(f"🎯 Step 3: Final audio separation with {final_provider}...")
        
        from .audio.audio_diarization import SpeakerDiarizer
        
        try:
            final_diarizer = SpeakerDiarizer(provider=final_provider, api_key=api_key)
        except ValueError as e:
            click.echo(f"❌ Error: {e}", err=True)
            if final_provider.lower() == 'assemblyai':
                click.echo("💡 Get your API key from: https://www.assemblyai.com/", err=True)
                click.echo("💡 Set it with: export ASSEMBLYAI_API_KEY=your_key", err=True)
            sys.exit(1)
        
        # Perform final speaker separation
        final_result = final_diarizer.separate_speakers(
            audio_file=str(input_file),
            output_dir=str(output_dir),
            speakers_expected=len(editor.transcript_data['speakers']),
            language=language,
            enhanced_processing=True,
            base_filename=f"{input_file.stem}_final"
        )
        
        # Display final results
        click.echo("✅ Edit workflow completed successfully!")
        click.echo("\n📊 Results Summary:")
        click.echo(f"  🎤 STT Transcription: {len(stt_result['utterances'])} utterances")
        click.echo(f"  ✏️  Edited Transcript: {edited_transcript_file}")
        
        stats = final_result['stats']
        click.echo(f"  🎯 Final Separation: {stats['total_coverage']:.1f}% coverage")
        
        for speaker, speaker_stats in stats['speakers'].items():
            click.echo(f"    {speaker}: {speaker_stats['duration']:.1f}s ({speaker_stats['coverage']:.1f}%)")
        
        click.echo(f"\n📄 Final transcript: {final_result['transcript_file']}")
        click.echo("🎵 Final separated audio files:")
        for audio_file in final_result['speaker_files']:
            click.echo(f"  - {audio_file}")
        
    except Exception as e:
        click.echo(f"❌ Error in edit workflow: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def handle_interactive_editing(editor: 'TranscriptEditor', replacer: 'SpeakerReplacer') -> None:
    """Handle interactive editing session"""
    
    click.echo("🎯 Interactive Editing Session")
    click.echo("Commands: list, stats, replace, edit, save, help, quit")
    
    while True:
        try:
            command = click.prompt("\nEdit command", type=str).strip().lower()
            
            if command in ['quit', 'q', 'exit']:
                break
            elif command in ['help', 'h']:
                click.echo("Available commands:")
                click.echo("  list - Show all speakers")
                click.echo("  stats - Show transcript statistics")
                click.echo("  replace <old> <new> - Replace speaker ID")
                click.echo("  edit <index> - Edit utterance text")
                click.echo("  save - Save current changes")
                click.echo("  quit - Exit interactive mode")
            elif command == 'list':
                speakers = editor.transcript_data.get('speakers', [])
                click.echo(f"Current speakers: {', '.join(speakers)}")
            elif command == 'stats':
                stats = editor.get_statistics()
                click.echo(f"Total utterances: {stats.get('total_utterances', 0)}")
                click.echo(f"Total duration: {stats.get('total_duration', 0):.1f}s")
                click.echo(f"Speakers: {stats.get('speaker_count', 0)}")
                for speaker, speaker_stats in stats.get('speakers', {}).items():
                    click.echo(f"  {speaker}: {speaker_stats['utterances']} utterances, {speaker_stats['total_duration']:.1f}s")
            elif command.startswith('replace '):
                parts = command.split(' ', 2)
                if len(parts) >= 3:
                    old_speaker = parts[1]
                    new_speaker = parts[2]
                    count = editor.replace_speaker_id(old_speaker, new_speaker)
                    click.echo(f"Replaced '{old_speaker}' -> '{new_speaker}' in {count} utterances")
                else:
                    click.echo("Usage: replace <old_speaker> <new_speaker>")
            elif command.startswith('edit '):
                parts = command.split(' ', 1)
                if len(parts) >= 2:
                    try:
                        index = int(parts[1])
                        utterances = editor.transcript_data.get('utterances', [])
                        if 0 <= index < len(utterances):
                            utterance = utterances[index]
                            click.echo(f"Current text: {utterance.get('text', '')}")
                            new_text = click.prompt("New text", default=utterance.get('text', ''))
                            editor.edit_utterance_text(index, new_text)
                            click.echo("✅ Utterance updated")
                        else:
                            click.echo(f"Invalid index. Range: 0-{len(utterances)-1}")
                    except ValueError:
                        click.echo("Invalid index number")
                else:
                    click.echo("Usage: edit <utterance_index>")
            else:
                click.echo(f"Unknown command: {command}. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            click.echo("\n🛑 Interactive editing interrupted")
            break
        except Exception as e:
            click.echo(f"❌ Error: {e}")


def show_execution_paths() -> None:
    """Display available execution paths for each provider"""
    click.echo("🛤️  Available Execution Paths:")
    click.echo("=" * 50)
    
    paths = get_all_execution_paths()
    
    for path_key, path_info in paths.items():
        click.echo(f"\n🎯 {path_info['name']}")
        click.echo(f"   Provider: {path_key}")
        click.echo(f"   Modes: {', '.join(path_info['modes'])}")
        click.echo(f"   Cost: {path_info['cost_profile']}")
        click.echo(f"   Features: {', '.join(path_info['features'])}")
        click.echo(f"   Use Case: {path_info['use_case']}")
        
        # Show example commands for different modes
        for mode in path_info['modes']:
            if mode == 'diarization':
                example_cmd = f"python -m src.main input.mp4 -p {path_key}"
            elif mode == 'edit':
                example_cmd = f"python -m src.main input.mp4 -p {path_key} --mode edit"
            elif mode == 'tts':
                example_cmd = f"python -m src.main transcript.json -p {path_key} --mode tts"
            click.echo(f"   {mode.title()}: {example_cmd}")


def determine_execution_path(primary_provider: str, secondary_provider: Optional[str] = None) -> Dict[str, Any]:
    """
    Determine execution path based on provider selection
    
    Args:
        primary_provider: Primary provider choice
        secondary_provider: Optional secondary provider for multi-step workflows
        
    Returns:
        Dictionary with execution path details
    """
    paths = get_all_execution_paths()
    
    # Handle secondary provider override for edit mode
    if secondary_provider:
        # Create custom path combining providers
        primary_path = paths.get(primary_provider, paths['assemblyai'])
        secondary_path = paths.get(secondary_provider, paths['assemblyai'])
        
        return {
            'name': f"Custom: {primary_provider} → {secondary_provider}",
            'stt_provider': primary_path['stt_provider'],
            'final_provider': secondary_path.get('final_provider', secondary_path['stt_provider']),
            'cost_profile': f"{primary_path['cost_profile']} + {secondary_path['cost_profile']}",
            'features': list(set(primary_path['features'] + secondary_path['features'])),
            'use_case': f"Custom workflow combining {primary_provider} and {secondary_provider}"
        }
    
    return paths.get(primary_provider, paths['assemblyai'])


def get_all_execution_paths() -> Dict[str, Dict[str, Any]]:
    """Get all available execution paths with their characteristics"""
    return {
        'assemblyai': {
            'name': 'AssemblyAI Professional Path',
            'stt_provider': 'assemblyai',
            'final_provider': 'assemblyai',
            'cost_profile': 'Low ($0.37/hour)',
            'features': ['Professional diarization', 'High accuracy', 'Fast processing', 'Custom vocabulary'],
            'use_case': 'Cost-effective solution for most use cases, excellent accuracy-to-cost ratio',
            'modes': ['diarization', 'edit'],
            'api_docs': 'https://www.assemblyai.com/docs/'
        },
        'elevenlabs': {
            'name': 'ElevenLabs Premium Path',
            'stt_provider': 'elevenlabs-stt',  # When using STT feature
            'final_provider': 'elevenlabs-stt',  # When using STT feature
            'cost_profile': 'Medium ($0.40/hour STT) + High (TTS per character)',
            'features': ['Premium STT quality', 'Speaker diarization', '99 languages', 'Audio events detection', 'Premium voice synthesis', 'Multiple voice options', 'Adaptive timing'],
            'use_case': 'High-quality transcription with detailed analysis OR voice synthesis',
            'modes': ['diarization', 'edit', 'tts'],
            'api_docs': 'https://elevenlabs.io/docs/'
        }
    }


if __name__ == '__main__':
    cli()
