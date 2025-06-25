"""
File Management Utilities
Enhanced file management with progress tracking, cleanup, and organization
"""

import logging
import shutil
import tempfile
from typing import Dict, Optional, Any, Callable, Set
from pathlib import Path
from datetime import datetime
import json
import hashlib

logger = logging.getLogger(__name__)


class FileManager:
    """
    Enhanced file manager for DuoSynco operations
    Handles temporary files, intermediate results, and cleanup
    """

    def __init__(
        self,
        base_output_dir: str,
        temp_dir: Optional[str] = None,
        auto_cleanup: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        """
        Initialize file manager

        Args:
            base_output_dir: Base directory for all output files
            temp_dir: Custom temporary directory (uses system temp if None)
            auto_cleanup: Automatically cleanup temporary files
            progress_callback: Optional progress callback
        """
        self.base_output_dir = Path(base_output_dir)
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.auto_cleanup = auto_cleanup
        self.progress_callback = progress_callback

        # File tracking
        self.created_files: Set[str] = set()
        self.temp_files: Set[str] = set()
        self.intermediate_files: Set[str] = set()
        self.final_files: Set[str] = set()

        # Ensure directories exist
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "FileManager initialized: output=%s, temp=%s",
            self.base_output_dir,
            self.temp_dir,
        )

    def create_temp_file(
        self, suffix: str = "", prefix: str = "duosynco_", content: Optional[str] = None
    ) -> str:
        """
        Create a temporary file

        Args:
            suffix: File suffix/extension
            prefix: File prefix
            content: Optional initial content

        Returns:
            Path to created temporary file
        """
        temp_file = tempfile.NamedTemporaryFile(
            suffix=suffix, prefix=prefix, dir=self.temp_dir, delete=False
        )

        if content:
            temp_file.write(content.encode("utf-8"))

        temp_file.close()
        temp_path = temp_file.name

        self.temp_files.add(temp_path)
        self.created_files.add(temp_path)

        logger.debug("Created temp file: %s", temp_path)
        return temp_path

    def create_output_file(
        self, filename: str, subdir: Optional[str] = None, is_intermediate: bool = False
    ) -> str:
        """
        Create an output file path in the managed directory structure

        Args:
            filename: Base filename
            subdir: Optional subdirectory
            is_intermediate: Whether this is an intermediate file

        Returns:
            Full path to the output file location
        """
        if subdir:
            output_path = self.base_output_dir / subdir
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = self.base_output_dir

        file_path = str(output_path / filename)

        if is_intermediate:
            self.intermediate_files.add(file_path)
        else:
            self.final_files.add(file_path)

        self.created_files.add(file_path)

        logger.debug("Created output file path: %s", file_path)
        return file_path

    def save_json(
        self,
        data: Dict[str, Any],
        filename: str,
        subdir: Optional[str] = None,
        is_intermediate: bool = False,
        pretty: bool = True,
    ) -> str:
        """
        Save data as JSON file

        Args:
            data: Data to save
            filename: Output filename
            subdir: Optional subdirectory
            is_intermediate: Whether this is an intermediate file
            pretty: Whether to format JSON nicely

        Returns:
            Path to saved file
        """
        file_path = self.create_output_file(filename, subdir, is_intermediate)

        with open(file_path, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)

        logger.info("Saved JSON file: %s", file_path)
        return file_path

    def save_text(
        self,
        content: str,
        filename: str,
        subdir: Optional[str] = None,
        is_intermediate: bool = False,
    ) -> str:
        """
        Save text content to file

        Args:
            content: Text content to save
            filename: Output filename
            subdir: Optional subdirectory
            is_intermediate: Whether this is an intermediate file

        Returns:
            Path to saved file
        """
        file_path = self.create_output_file(filename, subdir, is_intermediate)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info("Saved text file: %s", file_path)
        return file_path

    def copy_file(self, source: str, destination: str, is_intermediate: bool = False) -> str:
        """
        Copy file to managed location

        Args:
            source: Source file path
            destination: Destination filename (relative to output dir)
            is_intermediate: Whether this is an intermediate file

        Returns:
            Path to copied file
        """
        dest_path = self.create_output_file(destination, is_intermediate=is_intermediate)
        shutil.copy2(source, dest_path)

        logger.info("Copied file: %s -> %s", source, dest_path)
        return dest_path

    def move_file(self, source: str, destination: str, is_intermediate: bool = False) -> str:
        """
        Move file to managed location

        Args:
            source: Source file path
            destination: Destination filename (relative to output dir)
            is_intermediate: Whether this is an intermediate file

        Returns:
            Path to moved file
        """
        dest_path = self.create_output_file(destination, is_intermediate=is_intermediate)
        shutil.move(source, dest_path)

        # Update tracking if source was tracked
        if source in self.temp_files:
            self.temp_files.remove(source)
        if source in self.created_files:
            self.created_files.remove(source)

        logger.info("Moved file: %s -> %s", source, dest_path)
        return dest_path

    def organize_files(self, organization_map: Dict[str, str]) -> Dict[str, str]:
        """
        Organize files into subdirectories

        Args:
            organization_map: Mapping of file patterns to subdirectories

        Returns:
            Mapping of old paths to new paths
        """
        moved_files = {}

        for file_path in list(self.created_files):
            if not Path(file_path).exists():
                continue

            filename = Path(file_path).name

            # Find matching pattern
            target_subdir = None
            for pattern, subdir in organization_map.items():
                if pattern in filename:
                    target_subdir = subdir
                    break

            if target_subdir:
                new_path = self.create_output_file(filename, target_subdir)
                if file_path != new_path:
                    shutil.move(file_path, new_path)
                    moved_files[file_path] = new_path

                    # Update tracking
                    self.created_files.remove(file_path)
                    self.created_files.add(new_path)

                    if file_path in self.intermediate_files:
                        self.intermediate_files.remove(file_path)
                        self.intermediate_files.add(new_path)

                    if file_path in self.final_files:
                        self.final_files.remove(file_path)
                        self.final_files.add(new_path)

        logger.info("Organized %d files into subdirectories", len(moved_files))
        return moved_files

    def cleanup_temp_files(self) -> int:
        """
        Clean up temporary files

        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0

        for temp_file in list(self.temp_files):
            try:
                if Path(temp_file).exists():
                    Path(temp_file).unlink()
                    cleaned_count += 1
                self.temp_files.remove(temp_file)
                self.created_files.discard(temp_file)
            except Exception as e:
                logger.warning("Failed to cleanup temp file %s: %s", temp_file, e)

        logger.info("Cleaned up %d temporary files", cleaned_count)
        return cleaned_count

    def cleanup_intermediate_files(self) -> int:
        """
        Clean up intermediate files

        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0

        for intermediate_file in list(self.intermediate_files):
            try:
                if Path(intermediate_file).exists():
                    Path(intermediate_file).unlink()
                    cleaned_count += 1
                self.intermediate_files.remove(intermediate_file)
                self.created_files.discard(intermediate_file)
            except Exception as e:
                logger.warning("Failed to cleanup intermediate file %s: %s", intermediate_file, e)

        logger.info("Cleaned up %d intermediate files", cleaned_count)
        return cleaned_count

    def cleanup_all(self) -> int:
        """
        Clean up all managed files

        Returns:
            Number of files cleaned up
        """
        total_cleaned = 0
        total_cleaned += self.cleanup_temp_files()
        total_cleaned += self.cleanup_intermediate_files()

        logger.info("Total files cleaned up: %d", total_cleaned)
        return total_cleaned

    def get_file_info(self) -> Dict[str, Any]:
        """
        Get information about managed files

        Returns:
            Dictionary with file information
        """
        info = {
            "total_files": len(self.created_files),
            "temp_files": len(self.temp_files),
            "intermediate_files": len(self.intermediate_files),
            "final_files": len(self.final_files),
            "file_breakdown": {
                "temp": list(self.temp_files),
                "intermediate": list(self.intermediate_files),
                "final": list(self.final_files),
            },
        }

        # Calculate total size
        total_size = 0
        for file_path in self.created_files:
            try:
                if Path(file_path).exists():
                    total_size += Path(file_path).stat().st_size
            except Exception:
                pass

        info["total_size_mb"] = total_size / (1024 * 1024)

        return info

    def create_file_manifest(self, filename: str = "file_manifest.json") -> str:
        """
        Create a manifest of all managed files

        Args:
            filename: Manifest filename

        Returns:
            Path to manifest file
        """
        manifest = {
            "created_at": datetime.now().isoformat(),
            "base_output_dir": str(self.base_output_dir),
            "file_info": self.get_file_info(),
            "files": {},
        }

        # Add detailed file information
        for file_path in self.created_files:
            try:
                if Path(file_path).exists():
                    stat = Path(file_path).stat()
                    manifest["files"][file_path] = {
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "type": self._get_file_type(file_path),
                        "hash": self._calculate_file_hash(file_path),
                    }
                else:
                    manifest["files"][file_path] = {"status": "missing"}
            except Exception as e:
                manifest["files"][file_path] = {"status": "error", "error": str(e)}

        manifest_path = self.save_json(manifest, filename, is_intermediate=False)
        return manifest_path

    def _get_file_type(self, file_path: str) -> str:
        """Determine file type category"""
        if file_path in self.temp_files:
            return "temp"
        elif file_path in self.intermediate_files:
            return "intermediate"
        elif file_path in self.final_files:
            return "final"
        else:
            return "unknown"

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate file hash for verification"""
        try:
            hasher = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""

    def _report_progress(self, message: str, progress: float) -> None:
        """Report progress if callback available"""
        if self.progress_callback:
            try:
                self.progress_callback(message, progress)
            except Exception as e:
                logger.warning("Progress callback failed: %s", e)

    def __del__(self):
        """Cleanup on destruction if auto_cleanup is enabled"""
        if self.auto_cleanup:
            try:
                self.cleanup_temp_files()
            except Exception:
                pass
