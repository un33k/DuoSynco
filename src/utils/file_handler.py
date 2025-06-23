"""
File Handling Utilities
Manages file operations, validation, and path handling
"""

from pathlib import Path
from typing import List, Optional
import os
import shutil
import mimetypes
import subprocess
from dataclasses import dataclass

from .config import Config


@dataclass
class FileInfo:
    """Information about a file"""
    path: Path
    size_mb: float
    format: str
    mime_type: Optional[str]
    is_video: bool
    is_audio: bool


class FileHandler:
    """
    Handles file operations and validations for DuoSynco
    """
    
    # Supported file formats
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    SUPPORTED_AUDIO_FORMATS = {'.mp3', '.wav', '.aac', '.m4a', '.ogg', '.flac'}
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize mimetypes
        mimetypes.init()
    
    def validate_input_file(self, file_path: Path) -> bool:
        """
        Validate an input file for processing
        
        Args:
            file_path: Path to input file
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            # Check if file exists
            if not file_path.exists():
                if self.config.verbose:
                    print(f"❌ File not found: {file_path}")
                return False
            
            # Check if it's a file (not directory)
            if not file_path.is_file():
                if self.config.verbose:
                    print(f"❌ Not a file: {file_path}")
                return False
            
            # Check file format
            if not self._is_supported_format(file_path):
                if self.config.verbose:
                    print(f"❌ Unsupported format: {file_path.suffix}")
                return False
            
            # Check file size
            if not self._is_reasonable_size(file_path):
                if self.config.verbose:
                    print(f"❌ File too large: {file_path}")
                return False
            
            # Check if file is readable
            if not os.access(file_path, os.R_OK):
                if self.config.verbose:
                    print(f"❌ File not readable: {file_path}")
                return False
            
            return True
        
        except Exception as e:
            if self.config.verbose:
                print(f"❌ File validation error: {e}")
            return False
    
    def _is_supported_format(self, file_path: Path) -> bool:
        """Check if file format is supported"""
        extension = file_path.suffix.lower()
        return (extension in self.SUPPORTED_VIDEO_FORMATS or 
                extension in self.SUPPORTED_AUDIO_FORMATS)
    
    def _is_reasonable_size(self, file_path: Path) -> bool:
        """Check if file size is reasonable for processing"""
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # Reasonable limits (can be adjusted)
            max_size_mb = 5000  # 5GB
            
            return size_mb <= max_size_mb
        
        except Exception:
            return False
    
    def get_file_info(self, file_path: Path) -> Optional[FileInfo]:
        """
        Get detailed information about a file
        
        Args:
            file_path: Path to file
            
        Returns:
            FileInfo object or None if error
        """
        try:
            if not file_path.exists():
                return None
            
            # Get file size
            size_bytes = file_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            
            # Get format and mime type
            extension = file_path.suffix.lower()
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            # Determine file type
            is_video = extension in self.SUPPORTED_VIDEO_FORMATS
            is_audio = extension in self.SUPPORTED_AUDIO_FORMATS
            
            return FileInfo(
                path=file_path,
                size_mb=size_mb,
                format=extension,
                mime_type=mime_type,
                is_video=is_video,
                is_audio=is_audio
            )
        
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Could not get file info: {e}")
            return None
    
    def create_output_directory(self, output_path: Path) -> bool:
        """
        Create output directory if it doesn't exist
        
        Args:
            output_path: Path to output directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Check if directory is writable
            if not os.access(output_path, os.W_OK):
                if self.config.verbose:
                    print(f"❌ Output directory not writable: {output_path}")
                return False
            
            return True
        
        except Exception as e:
            if self.config.verbose:
                print(f"❌ Could not create output directory: {e}")
            return False
    
    def generate_output_filename(self, 
                                input_path: Path, 
                                suffix: str = "", 
                                extension: Optional[str] = None) -> str:
        """
        Generate output filename based on input file
        
        Args:
            input_path: Input file path
            suffix: Suffix to add to filename
            extension: New extension (use original if None)
            
        Returns:
            Generated filename
        """
        base_name = input_path.stem
        
        if suffix:
            base_name += f"_{suffix}"
        
        if extension:
            if not extension.startswith('.'):
                extension = f".{extension}"
            return f"{base_name}{extension}"
        else:
            return f"{base_name}{input_path.suffix}"
    
    def ensure_unique_filename(self, file_path: Path) -> Path:
        """
        Ensure filename is unique by adding number if needed
        
        Args:
            file_path: Desired file path
            
        Returns:
            Unique file path
        """
        if not file_path.exists():
            return file_path
        
        counter = 1
        original_stem = file_path.stem
        extension = file_path.suffix
        parent = file_path.parent
        
        while True:
            new_name = f"{original_stem}_{counter}{extension}"
            new_path = parent / new_name
            
            if not new_path.exists():
                return new_path
            
            counter += 1
            
            # Prevent infinite loop
            if counter > 1000:
                raise ValueError(f"Could not generate unique filename for {file_path}")
    
    def copy_file(self, source: Path, destination: Path) -> bool:
        """
        Copy file from source to destination
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure destination directory exists
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(source, destination)
            
            if self.config.verbose:
                print(f"📄 Copied: {source.name} -> {destination.name}")
            
            return True
        
        except Exception as e:
            if self.config.verbose:
                print(f"❌ Copy failed: {e}")
            return False
    
    def move_file(self, source: Path, destination: Path) -> bool:
        """
        Move file from source to destination
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure destination directory exists
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.move(str(source), str(destination))
            
            if self.config.verbose:
                print(f"📁 Moved: {source.name} -> {destination.name}")
            
            return True
        
        except Exception as e:
            if self.config.verbose:
                print(f"❌ Move failed: {e}")
            return False
    
    def delete_file(self, file_path: Path) -> bool:
        """
        Delete a file
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if file_path.exists():
                file_path.unlink()
                
                if self.config.verbose:
                    print(f"🗑️  Deleted: {file_path.name}")
                
                return True
            else:
                return True  # File doesn't exist, consider it "deleted"
        
        except Exception as e:
            if self.config.verbose:
                print(f"❌ Delete failed: {e}")
            return False
    
    def get_available_space(self, directory: Path) -> Optional[float]:
        """
        Get available disk space in directory (in MB)
        
        Args:
            directory: Directory to check
            
        Returns:
            Available space in MB, or None if error
        """
        try:
            statvfs = os.statvfs(directory)
            available_bytes = statvfs.f_bavail * statvfs.f_frsize
            available_mb = available_bytes / (1024 * 1024)
            
            return available_mb
        
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Could not get disk space: {e}")
            return None
    
    def check_sufficient_space(self, 
                              directory: Path, 
                              required_mb: float) -> bool:
        """
        Check if directory has sufficient space
        
        Args:
            directory: Directory to check
            required_mb: Required space in MB
            
        Returns:
            True if sufficient space, False otherwise
        """
        available_mb = self.get_available_space(directory)
        
        if available_mb is None:
            return False  # Can't determine, assume insufficient
        
        # Add 10% buffer
        required_with_buffer = required_mb * 1.1
        
        return available_mb >= required_with_buffer
    
    def list_files_in_directory(self, 
                               directory: Path, 
                               pattern: str = "*") -> List[Path]:
        """
        List files in directory matching pattern
        
        Args:
            directory: Directory to search
            pattern: Glob pattern to match
            
        Returns:
            List of matching file paths
        """
        try:
            if not directory.exists() or not directory.is_dir():
                return []
            
            return list(directory.glob(pattern))
        
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Could not list files: {e}")
            return []
    
    def cleanup_directory(self, directory: Path, keep_files: Optional[List[Path]] = None) -> None:
        """
        Clean up directory, optionally keeping specific files
        
        Args:
            directory: Directory to clean
            keep_files: List of files to keep (optional)
        """
        try:
            if not directory.exists():
                return
            
            keep_files = keep_files or []
            keep_file_names = {f.name for f in keep_files}
            
            for file_path in directory.iterdir():
                if file_path.is_file() and file_path.name not in keep_file_names:
                    self.delete_file(file_path)
        
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Directory cleanup error: {e}")
    
    def convert_to_mp3(self, source: Path, destination: Path, bitrate: str = "128k") -> bool:
        """
        Convert audio file to MP3 format using FFmpeg
        
        Args:
            source: Source audio file path
            destination: Destination MP3 file path
            bitrate: MP3 bitrate (default: 128k for good quality/size balance)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure destination directory exists
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Build FFmpeg command
            cmd = [
                'ffmpeg',
                '-i', str(source),
                '-c:a', 'libmp3lame',
                '-b:a', bitrate,
                '-y',  # Overwrite output file
                str(destination)
            ]
            
            # Run conversion
            if self.config.verbose:
                print(f"🔄 Converting {source.name} to MP3...")
                result = subprocess.run(cmd, capture_output=True, text=True)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True, 
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if result.returncode == 0:
                if self.config.verbose:
                    print(f"✅ Converted: {source.name} -> {destination.name}")
                return True
            else:
                if self.config.verbose:
                    print(f"❌ Conversion failed: {result.stderr}")
                return False
        
        except Exception as e:
            if self.config.verbose:
                print(f"❌ Conversion error: {e}")
            return False