"""
Video Synchronization Module
Handles synchronization between video and isolated audio tracks
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import tempfile
import subprocess

from .processor import VideoProcessor
from ..utils.config import Config


class VideoSynchronizer:
    """
    Synchronizes video with isolated audio tracks
    Ensures timing alignment and handles audio/video sync
    """

    def __init__(self, config: Config):
        self.config = config
        self.video_processor = VideoProcessor(config)

    def sync_video_audio(
        self, original_video: Path, isolated_audio: Path, output_path: Path
    ) -> bool:
        """
        Synchronize video with isolated audio track

        Args:
            original_video: Original video file
            isolated_audio: Isolated audio track
            output_path: Output synchronized video file

        Returns:
            True if successful, False otherwise
        """
        if self.config.verbose:
            print(f"🔄 Synchronizing {output_path.name}")

        try:
            # Get video and audio information
            video_info = self.video_processor.get_video_info(original_video)

            if not video_info:
                if self.config.verbose:
                    print(f"❌ Could not get video info for {original_video}")
                return False

            # Check if isolated audio file exists and is valid
            if not isolated_audio.exists():
                if self.config.verbose:
                    print(f"❌ Isolated audio file not found: {isolated_audio}")
                return False

            # Perform synchronization
            success = self._perform_synchronization(
                original_video, isolated_audio, output_path, video_info
            )

            if success and self.config.verbose:
                print(f"✅ Successfully synchronized: {output_path.name}")

            return success

        except Exception as e:
            if self.config.verbose:
                print(f"❌ Synchronization failed: {e}")
            return False

    def _perform_synchronization(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        video_info: Dict[str, Any],
    ) -> bool:
        """
        Perform the actual video-audio synchronization
        """
        try:
            # Method 1: Direct replacement (fastest, most reliable)
            if self._sync_by_replacement(video_path, audio_path, output_path):
                return True

            # Method 2: Alignment with timing correction (fallback)
            if self.config.verbose:
                print("🔄 Trying alignment method...")

            return self._sync_by_alignment(
                video_path, audio_path, output_path, video_info
            )

        except Exception as e:
            if self.config.verbose:
                print(f"❌ Synchronization method failed: {e}")
            return False

    def _sync_by_replacement(
        self, video_path: Path, audio_path: Path, output_path: Path
    ) -> bool:
        """
        Simple audio replacement - fastest method
        """
        try:
            return self.video_processor.replace_audio_in_video(
                video_path, audio_path, output_path
            )
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Direct replacement failed: {e}")
            return False

    def _sync_by_alignment(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        video_info: Dict[str, Any],
    ) -> bool:
        """
        Advanced alignment with timing correction
        """
        try:
            # This method would implement more sophisticated sync
            # For now, we'll use a basic approach with FFmpeg

            if not self.video_processor.ffmpeg_available:
                return False

            # Use FFmpeg with advanced sync options
            cmd = [
                "ffmpeg",
                "-i",
                str(video_path),  # Video input
                "-i",
                str(audio_path),  # Audio input
                "-c:v",
                "copy",  # Copy video stream
                "-c:a",
                "aac",  # Audio codec
                "-map",
                "0:v:0",  # Map video from first input
                "-map",
                "1:a:0",  # Map audio from second input
                "-async",
                "1",  # Audio sync method
                "-vsync",
                "cfr",  # Video sync method
                "-shortest",  # End when shortest stream ends
                "-avoid_negative_ts",
                "make_zero",  # Handle timing issues
                "-y",  # Overwrite output
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True)

            if result.returncode != 0 and self.config.verbose:
                print(f"⚠️  FFmpeg sync error: {result.stderr.decode()}")

            return result.returncode == 0

        except Exception as e:
            if self.config.verbose:
                print(f"❌ Alignment method failed: {e}")
            return False

    def verify_synchronization(
        self, output_video: Path, expected_duration: Optional[float] = None
    ) -> bool:
        """
        Verify that the synchronized video is valid

        Args:
            output_video: Path to synchronized video
            expected_duration: Expected duration in seconds

        Returns:
            True if verification passes, False otherwise
        """
        try:
            if not output_video.exists():
                if self.config.verbose:
                    print(f"❌ Output video not found: {output_video}")
                return False

            # Check if video is readable
            if not self.video_processor.validate_video_file(output_video):
                if self.config.verbose:
                    print(f"❌ Output video is not valid: {output_video}")
                return False

            # Get info about the synchronized video
            video_info = self.video_processor.get_video_info(output_video)

            # Check duration if expected
            if expected_duration is not None:
                duration_diff = abs(video_info["duration"] - expected_duration)
                if duration_diff > 1.0:  # Allow 1 second tolerance
                    if self.config.verbose:
                        print(
                            f"⚠️  Duration mismatch: expected {expected_duration}s, "
                            f"got {video_info['duration']:.1f}s"
                        )
                    return False

            # Check if video has audio
            if not video_info["audio"]:
                if self.config.verbose:
                    print(f"⚠️  Synchronized video has no audio: {output_video}")
                return False

            if self.config.verbose:
                print(
                    f"✅ Synchronization verified: {output_video.name} "
                    f"({video_info['duration']:.1f}s)"
                )

            return True

        except Exception as e:
            if self.config.verbose:
                print(f"❌ Synchronization verification failed: {e}")
            return False

    def batch_synchronize(
        self, original_video: Path, isolated_tracks: Dict[str, Path], output_dir: Path
    ) -> Dict[str, Path]:
        """
        Synchronize multiple isolated tracks with the original video

        Args:
            original_video: Original video file
            isolated_tracks: Dictionary of speaker_id -> isolated audio path
            output_dir: Output directory for synchronized videos

        Returns:
            Dictionary mapping speaker_id to synchronized video path
        """
        synchronized_videos = {}

        # Get original video info for verification
        video_info = self.video_processor.get_video_info(original_video)
        expected_duration = video_info.get("duration", 0.0)

        for speaker_id, audio_path in isolated_tracks.items():
            # Generate output path
            output_name = (
                f"{original_video.stem}_{speaker_id}.{self.config.output_format}"
            )
            output_path = output_dir / output_name

            if self.config.verbose:
                print(f"🎬 Processing {speaker_id}...")

            # Perform synchronization
            success = self.sync_video_audio(original_video, audio_path, output_path)

            if success:
                # Verify the result
                if self.verify_synchronization(output_path, expected_duration):
                    synchronized_videos[speaker_id] = output_path
                else:
                    if self.config.verbose:
                        print(f"⚠️  Verification failed for {speaker_id}")
            else:
                if self.config.verbose:
                    print(f"❌ Synchronization failed for {speaker_id}")

        return synchronized_videos

    def optimize_synchronized_videos(
        self, synchronized_videos: Dict[str, Path]
    ) -> Dict[str, Path]:
        """
        Optimize synchronized videos for final output

        Args:
            synchronized_videos: Dictionary of synchronized video paths

        Returns:
            Dictionary of optimized video paths
        """
        optimized_videos = {}

        for speaker_id, video_path in synchronized_videos.items():
            # Create optimized filename
            optimized_path = (
                video_path.parent / f"{video_path.stem}_optimized{video_path.suffix}"
            )

            if self.config.verbose:
                print(f"⚡ Optimizing {speaker_id}...")

            # Optimize the video
            success = self.video_processor.optimize_video_for_output(
                video_path, optimized_path
            )

            if success:
                optimized_videos[speaker_id] = optimized_path

                # Remove original if optimization succeeded
                try:
                    video_path.unlink()
                except Exception:
                    pass  # Ignore cleanup errors
            else:
                # Keep original if optimization failed
                optimized_videos[speaker_id] = video_path

        return optimized_videos
