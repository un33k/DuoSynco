#!/usr/bin/env python3
"""
DuoSynco - Video/Audio Synchronization Tool
Main CLI entry point for processing videos with speaker isolation
"""

import click
from pathlib import Path
from typing import Optional
import sys
import os

from .utils.file_handler import FileHandler
from .utils.config import Config
from .audio.diarization import SpeakerDiarizer
from .audio.isolator import VoiceIsolator
from .video.processor import VideoProcessor
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
@click.option('--backend', '-b',
              type=click.Choice(['ffmpeg', 'speechbrain', 'demucs', 'spectral', 'whisperx']),
              default='speechbrain',
              help='Voice separation backend (default: speechbrain)')
@click.option('--list-backends',
              is_flag=True,
              help='List available backends and exit')
@click.option('--verbose', '-v',
              is_flag=True,
              help='Enable verbose output')
def cli(input_file: Optional[Path], 
        output_dir: Path, 
        speakers: int, 
        format: str, 
        quality: str, 
        backend: str,
        list_backends: bool,
        verbose: bool):
    """
    DuoSynco - Sync two videos with isolated audio tracks
    
    Takes an input video file and creates separate output files,
    each containing only one speaker's audio while maintaining
    video synchronization.
    
    INPUT_FILE: Path to the input video file to process
    """
    
    # Handle list-backends option
    if list_backends:
        available_backends = Config.get_available_backends()
        click.echo("🔧 Available Voice Separation Backends:")
        for backend_name, is_available in available_backends.items():
            status = "✅ Available" if is_available else "❌ Not Available"
            click.echo(f"  {backend_name}: {status}")
        return
    
    # Validate input file is provided for normal operations
    if input_file is None:
        click.echo("❌ Error: INPUT_FILE is required.", err=True)
        click.echo("Use --help for usage information.", err=True)
        sys.exit(1)
    
    # Check if input file exists
    if not input_file.exists():
        click.echo(f"❌ Error: Input file '{input_file}' does not exist.", err=True)
        sys.exit(1)
    
    # Configure logging
    if verbose:
        click.echo("Verbose mode enabled")
    
    # Initialize components
    config = Config(quality=quality, output_format=format, backend=backend, verbose=verbose)
    
    # Validate backend availability
    if not config.validate_backend_availability():
        available_backends = Config.get_available_backends()
        available_list = [name for name, avail in available_backends.items() if avail]
        click.echo(f"❌ Error: Backend '{backend}' is not available.", err=True)
        click.echo(f"Available backends: {', '.join(available_list)}", err=True)
        click.echo("Use --list-backends to see detailed availability.", err=True)
        sys.exit(1)
    
    file_handler = FileHandler(config)
    
    try:
        # Validate input file
        if not file_handler.validate_input_file(input_file):
            click.echo(f"Error: Invalid input file {input_file}", err=True)
            sys.exit(1)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            click.echo(f"Processing {input_file}")
            click.echo(f"Output directory: {output_dir}")
            click.echo(f"Expected speakers: {speakers}")
            click.echo(f"Using backend: {backend}")
        
        # Step 1: Speaker Diarization
        click.echo("🔍 Analyzing speakers...")
        diarizer = SpeakerDiarizer(config)
        speaker_segments = diarizer.diarize(input_file, num_speakers=speakers)
        
        if verbose:
            click.echo(f"Found {len(speaker_segments)} speaker segments")
        
        # Step 2: Audio Isolation
        click.echo("🎵 Isolating audio tracks...")
        isolator = VoiceIsolator(config)
        isolated_tracks = isolator.isolate_speakers(input_file, speaker_segments)
        
        # Check if input is audio-only or video
        file_info = file_handler.get_file_info(input_file)
        
        if file_info and file_info.is_audio and not file_info.is_video:
            # Audio-only processing - convert to MP3 for smaller size
            click.echo("🎵 Processing audio-only file...")
            output_files = []
            for i, (_, audio_track) in enumerate(isolated_tracks.items()):
                output_file = output_dir / f"{input_file.stem}_speaker_{i+1}.mp3"
                
                if verbose:
                    click.echo(f"Creating {output_file}")
                
                # Convert isolated audio track to MP3 for smaller size
                file_handler.convert_to_mp3(audio_track, output_file)
                output_files.append(output_file)
        else:
            # Step 3: Video Processing
            click.echo("🎬 Processing video...")
            video_processor = VideoProcessor(config)
            synchronizer = VideoSynchronizer(config)
            
            # Generate output files for each speaker
            output_files = []
            for i, (_, audio_track) in enumerate(isolated_tracks.items()):
                output_file = output_dir / f"{input_file.stem}_speaker_{i+1}.{format}"
                
                if verbose:
                    click.echo(f"Creating {output_file}")
                
                # Synchronize video with isolated audio
                synchronizer.sync_video_audio(
                    input_file, audio_track, output_file
                )
                
                output_files.append(output_file)
        
        # Success message
        click.echo("✅ Processing complete!")
        click.echo(f"Generated {len(output_files)} files:")
        for file in output_files:
            click.echo(f"  - {file}")
            
    except Exception as e:
        click.echo(f"❌ Error processing file: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    cli()