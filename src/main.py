#!/usr/bin/env python
"""
DuoSynco - Video/Audio Synchronization Tool
Main CLI entry point for processing videos with speaker isolation using AssemblyAI
"""

import click
from pathlib import Path
from typing import Optional, List, Dict
import sys
import logging

from .utils.file_handler import FileHandler
from .utils.config import Config
from .video.synchronizer import VideoSynchronizer


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
              type=str,
              default='assemblyai',
              help='Speaker diarization provider (default: assemblyai)')
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
              type=click.Choice(['diarization', 'tts']),
              default='diarization',
              help='Processing mode: diarization (default) or tts generation')
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
        gap_duration: float):
    """
    DuoSynco - Sync videos with isolated speaker audio tracks

    Takes an input video file and creates separate output files,
    each containing only one speaker's audio while maintaining
    video synchronization. Supports multiple providers for
    speaker diarization.

    INPUT_FILE: Path to the input video file to process
    """

    # Set up logging
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(levelname)s: %(message)s'
    )

    # Handle list-providers option
    if list_providers:
        from .audio.diarization import SpeakerDiarizer
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
        from .audio.tts_generator import TTSAudioGenerator
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
        from .utils.env_loader import env
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
        return handle_tts_mode(
            transcript_file, total_duration, output_dir, api_key,
            voice_mapping, tts_workers, tts_quality, timing_mode, gap_duration, verbose
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
            from .audio.diarization import SpeakerDiarizer
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
    transcript_file: Optional[Path],
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
    if transcript_file is None:
        click.echo("❌ Error: --transcript-file is required for TTS mode.", err=True)
        sys.exit(1)
        
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
            from .utils.env_loader import get_voice_mapping
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
        from .audio.tts_generator import TTSAudioGenerator
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


if __name__ == '__main__':
    cli()
