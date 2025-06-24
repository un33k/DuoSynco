"""
Ali command-line interface
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .config import AliConfig
from .commands import AliCommands


def show_help():
    """Display help information"""
    print("ali - AI Line Interpreter for DuoSynco")
    print("")
    print("Usage:")
    print("  ali stt <audio_file> [options]    # Speech-to-text with speaker diarization")
    print("  ali tts <transcript_file> [options] # Text-to-speech from transcript")
    print("  ali clone <transcript_file> [options] # TTS with voice cloning")
    print("  ali edit <audio_file> [options]   # Interactive editing workflow")
    print("  ali voices [options]              # List available voices")
    print("  ali config                        # Show current configuration")
    print("")
    print("Common Options:")
    print("  -p, --provider PROVIDER          # Provider: elevenlabs, assemblyai (default: elevenlabs)")
    print("  -l, --language LANG              # Language code (default: fa)")
    print("  -o, --output-dir DIR             # Output directory (default: ./output)")
    print("  -q, --quality QUALITY            # Quality: low, medium, high (default: high)")
    print("  -m, --model MODEL                # Model ID (e.g., eleven_flash_v2_5, eleven_v3)")
    print("  -v, --verbose                    # Enable verbose output")
    print("  --no-verbose                     # Disable verbose output")
    print("")
    print("Examples:")
    print("  ali stt sample_data/annunaki-fa.mp3")
    print("  ali tts output/transcript.json -p elevenlabs -l en")
    print("  ali tts output/transcript.json -m eleven_flash_v2_5")
    print("  ali tts output/transcript.json -m eleven_v3  # Alpha access required")
    print("  ali clone output/transcript.json --quality medium")
    print("")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with subcommands"""
    parser = argparse.ArgumentParser(
        prog='ali',
        description='AI Line Interpreter for DuoSynco',
        add_help=False
    )
    
    # Add help manually to control formatting
    parser.add_argument('-h', '--help', action='store_true', help='Show help message')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments for all commands
    def add_common_args(subparser):
        subparser.add_argument('-p', '--provider', choices=['elevenlabs', 'assemblyai'],
                             help='Provider (default: elevenlabs)')
        subparser.add_argument('-l', '--language', help='Language code (default: fa)')
        subparser.add_argument('-o', '--output-dir', help='Output directory (default: ./output)')
        subparser.add_argument('-q', '--quality', choices=['low', 'medium', 'high', 'ultra'],
                             help='Quality level (default: high)')
        subparser.add_argument('-m', '--model', help='Model ID (e.g., eleven_flash_v2_5, eleven_v3)')
        subparser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
        subparser.add_argument('--no-verbose', action='store_true', help='Disable verbose output')
    
    # STT command
    stt_parser = subparsers.add_parser('stt', help='Speech-to-text with speaker diarization')
    stt_parser.add_argument('audio_file', help='Path to audio file')
    add_common_args(stt_parser)
    
    # TTS command
    tts_parser = subparsers.add_parser('tts', help='Text-to-speech from transcript')
    tts_parser.add_argument('transcript_file', help='Path to transcript file')
    tts_parser.add_argument('--voice-mapping', help='Voice mapping mode (default: auto)')
    add_common_args(tts_parser)
    
    # Clone command
    clone_parser = subparsers.add_parser('clone', help='TTS with voice cloning')
    clone_parser.add_argument('transcript_file', help='Path to transcript file')
    add_common_args(clone_parser)
    
    # Edit command
    edit_parser = subparsers.add_parser('edit', help='Interactive editing workflow')
    edit_parser.add_argument('audio_file', help='Path to audio file')
    add_common_args(edit_parser)
    
    # Voices command
    voices_parser = subparsers.add_parser('voices', help='List available voices')
    voices_parser.add_argument('-p', '--provider', choices=['elevenlabs', 'assemblyai'],
                              help='Provider (default: elevenlabs)')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show current configuration')
    
    return parser


def main(args: List[str]):
    """
    Main CLI entry point
    
    Args:
        args: Command line arguments (excluding script name)
    """
    parser = create_parser()
    
    # Handle empty args or help
    if not args or (len(args) == 1 and args[0] in ['help', '-h', '--help']):
        show_help()
        return 0
    
    try:
        # Parse arguments
        parsed_args = parser.parse_args(args)
        
        # Show help if requested
        if hasattr(parsed_args, 'help') and parsed_args.help:
            show_help()
            return 0
        
        # Check if command was provided
        if not parsed_args.command:
            show_help()
            return 0
        
        # Initialize configuration with overrides from CLI args
        config = AliConfig()
        
        # Override config with CLI arguments
        if hasattr(parsed_args, 'provider') and parsed_args.provider:
            config.defaults['provider'] = parsed_args.provider
        if hasattr(parsed_args, 'language') and parsed_args.language:
            config.defaults['language'] = parsed_args.language
        if hasattr(parsed_args, 'output_dir') and parsed_args.output_dir:
            config.defaults['output_dir'] = parsed_args.output_dir
        if hasattr(parsed_args, 'quality') and parsed_args.quality:
            config.defaults['tts_quality'] = parsed_args.quality
        if hasattr(parsed_args, 'model') and parsed_args.model:
            config.defaults['model_id'] = parsed_args.model
        if hasattr(parsed_args, 'verbose') and parsed_args.verbose:
            config.defaults['verbose'] = True
        if hasattr(parsed_args, 'no_verbose') and parsed_args.no_verbose:
            config.defaults['verbose'] = False
        if hasattr(parsed_args, 'voice_mapping') and parsed_args.voice_mapping:
            config.defaults['voice_mapping'] = parsed_args.voice_mapping
        
        # Initialize commands with updated config
        commands = AliCommands(config)
        
        # Execute command
        if parsed_args.command == 'stt':
            return commands.stt(parsed_args.audio_file)
        elif parsed_args.command == 'tts':
            return commands.tts(parsed_args.transcript_file)
        elif parsed_args.command == 'clone':
            return commands.clone(parsed_args.transcript_file)
        elif parsed_args.command == 'edit':
            return commands.edit(parsed_args.audio_file)
        elif parsed_args.command == 'voices':
            return commands.voices()
        elif parsed_args.command == 'config':
            return commands.config_info()
        else:
            print(f"❌ Error: Unknown command '{parsed_args.command}'")
            return 1
    
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 130
    
    except SystemExit as e:
        # Handle argparse errors gracefully
        if e.code != 0:
            print("\n💡 Use 'ali --help' for usage information")
        return e.code
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main(sys.argv[1:])
    sys.exit(exit_code)