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
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
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
@click.option('--verbose', '-v',
              is_flag=True,
              help='Enable verbose output')
def cli(input_file: Path, 
        output_dir: Path, 
        speakers: int, 
        format: str, 
        quality: str, 
        verbose: bool):
    """
    DuoSynco - Sync two videos with isolated audio tracks
    
    Takes an input video file and creates separate output files,
    each containing only one speaker's audio while maintaining
    video synchronization.
    
    INPUT_FILE: Path to the input video file to process
    """
    
    # Configure logging
    if verbose:
        click.echo("Verbose mode enabled")
    
    # Initialize components
    config = Config(quality=quality, output_format=format, verbose=verbose)
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
        
        # Step 3: Video Processing
        click.echo("🎬 Processing video...")
        video_processor = VideoProcessor(config)
        synchronizer = VideoSynchronizer(config)
        
        # Generate output files for each speaker
        output_files = []
        for i, (speaker_id, audio_track) in enumerate(isolated_tracks.items()):
            output_file = output_dir / f"{input_file.stem}_speaker_{i+1}.{format}"
            
            if verbose:
                click.echo(f"Creating {output_file}")
            
            # Synchronize video with isolated audio
            synchronized_video = synchronizer.sync_video_audio(
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