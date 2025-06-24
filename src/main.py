#!/usr/bin/env python
"""
DuoSynco - Video/Audio Synchronization Tool
Main CLI entry point for processing videos with speaker isolation using AssemblyAI
"""

import click
from pathlib import Path
from typing import Optional
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
        verbose: bool):
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

    # Validate input file is provided
    if input_file is None:
        click.echo("❌ Error: INPUT_FILE is required.", err=True)
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


if __name__ == '__main__':
    cli()
