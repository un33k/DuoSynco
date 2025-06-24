"""
Ali command implementations
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .config import AliConfig
from .transcript_converter import TranscriptConverter


class AliCommands:
    """Implementation of Ali commands"""
    
    def __init__(self, config: AliConfig):
        self.config = config
        self.converter = TranscriptConverter(config)
    
    def _run_duosynco(self, args: List[str]) -> int:
        """
        Run DuoSynco main module with given arguments
        
        Args:
            args: Command line arguments
            
        Returns:
            Exit code
        """
        # Change to project root
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(self.config.project_root)
            
            # Run the command
            cmd = [sys.executable, '-m', 'src.main'] + args
            result = subprocess.run(cmd, capture_output=False)
            return result.returncode
            
        finally:
            import os
            os.chdir(original_cwd)
    
    def _get_common_args(self) -> List[str]:
        """Get common arguments from config"""
        args = []
        
        # Add verbose flag
        if self.config.defaults.get('verbose', True):
            args.append('--verbose')
        
        return args
    
    def stt(self, audio_file: str) -> int:
        """
        Speech-to-text with speaker diarization
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Exit code
        """
        print("🎤 Running Speech-to-Text with speaker diarization...")
        
        args = [
            audio_file,
            '-p', self.config.defaults['provider'],
            '--mode', 'edit',
            '--language', self.config.defaults['language'],
            '--output-dir', self.config.defaults['output_dir'],
            '--stt-quality', self.config.defaults.get('tts_quality', 'high')
        ] + self._get_common_args()
        
        return self._run_duosynco(args)
    
    def tts(self, transcript_file: str) -> int:
        """
        Text-to-speech from transcript
        
        Args:
            transcript_file: Path to transcript file
            
        Returns:
            Exit code
        """
        transcript_path = Path(transcript_file)
        if not transcript_path.exists():
            print(f"❌ Error: Transcript file '{transcript_file}' does not exist")
            return 1
        
        # Convert transcript to TTS format
        tts_file = self.converter.to_tts_format(transcript_path)
        
        # Calculate total duration
        total_duration = self.converter.calculate_duration(tts_file)
        
        print("🗣️  Running Text-to-Speech...")
        
        args = [
            str(tts_file),
            '-p', self.config.defaults['provider'],
            '--mode', self.config.defaults['mode'],
            '--output-dir', self.config.defaults['output_dir'],
            '--total-duration', str(total_duration),
            '--language', self.config.defaults['language'],
            '--voice-mapping', self.config.defaults['voice_mapping'],
            '--tts-quality', self.config.defaults.get('tts_quality', 'high')
        ]
        
        # Add model_id if specified and different from default
        if (self.config.defaults.get('model_id') and 
            self.config.defaults['model_id'] != 'eleven_multilingual_v2'):
            args.extend(['--model-id', self.config.defaults['model_id']])
        
        args += self._get_common_args()
        
        return self._run_duosynco(args)
    
    def clone(self, transcript_file: str) -> int:
        """
        Text-to-speech with voice cloning
        
        Args:
            transcript_file: Path to transcript file
            
        Returns:
            Exit code
        """
        transcript_path = Path(transcript_file)
        if not transcript_path.exists():
            print(f"❌ Error: Transcript file '{transcript_file}' does not exist")
            return 1
        
        # Convert transcript to TTS format
        tts_file = self.converter.to_tts_format(transcript_path)
        
        # Calculate total duration
        total_duration = self.converter.calculate_duration(tts_file)
        
        print("🎭 Running Text-to-Speech with voice cloning...")
        
        args = [
            str(tts_file),
            '-p', self.config.defaults['provider'],
            '--mode', self.config.defaults['mode'],
            '--output-dir', self.config.defaults['output_dir'],
            '--total-duration', str(total_duration),
            '--language', self.config.defaults['language'],
            '--tts-quality', self.config.defaults.get('tts_quality', 'high'),
            '--clone-voices'
        ]
        
        # Add model_id if specified and different from default
        if (self.config.defaults.get('model_id') and 
            self.config.defaults['model_id'] != 'eleven_multilingual_v2'):
            args.extend(['--model-id', self.config.defaults['model_id']])
        
        args += self._get_common_args()
        
        return self._run_duosynco(args)
    
    def edit(self, audio_file: str) -> int:
        """
        Interactive editing workflow
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Exit code
        """
        print("✏️  Running interactive editing workflow...")
        
        args = [
            audio_file,
            '-p', self.config.defaults['provider'],
            '--mode', 'edit',
            '--language', self.config.defaults['language'],
            '--output-dir', self.config.defaults['output_dir'],
            '--stt-quality', self.config.defaults.get('tts_quality', 'high'),
            '--edit-interactive'
        ] + self._get_common_args()
        
        return self._run_duosynco(args)
    
    def voices(self) -> int:
        """
        List available voices
        
        Returns:
            Exit code
        """
        print("🗣️  Listing available voices...")
        
        args = [
            '--list-voices',
            '-p', self.config.defaults['provider']
        ]
        
        return self._run_duosynco(args)
    
    def config_info(self) -> int:
        """
        Show current configuration
        
        Returns:
            Exit code
        """
        print("⚙️  Current configuration...")
        
        args = ['--show-config']
        
        return self._run_duosynco(args)