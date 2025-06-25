"""
Video Processing Module
Handles video file operations, format conversion, and basic video manipulation
"""

from pathlib import Path
from typing import Dict, Any
import subprocess

try:
    from moviepy.editor import VideoFileClip, AudioFileClip

    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

from ..utils.config import Config


class VideoProcessor:
    """
    Handles video processing operations using MoviePy and FFmpeg
    """

    def __init__(self, config: Config):
        self.config = config
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if required video processing tools are available"""
        # Check for FFmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            self.ffmpeg_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.ffmpeg_available = False
            if self.config.verbose:
                print("⚠️  FFmpeg not found - some features may be limited")

        if not MOVIEPY_AVAILABLE and not self.ffmpeg_available:
            raise ImportError(
                "Either MoviePy or FFmpeg is required for video processing. "
                "Install with: pip install moviepy or install FFmpeg"
            )

    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """
        Get information about a video file

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video information
        """
        try:
            if MOVIEPY_AVAILABLE:
                with VideoFileClip(str(video_path)) as clip:
                    return {
                        "duration": clip.duration,
                        "fps": clip.fps,
                        "size": clip.size,
                        "audio": clip.audio is not None,
                        "format": video_path.suffix.lower(),
                    }

            elif self.ffmpeg_available:
                # Use FFprobe to get video info
                cmd = [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    str(video_path),
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                import json

                info = json.loads(result.stdout)

                video_stream = None
                audio_stream = None

                for stream in info["streams"]:
                    if stream["codec_type"] == "video" and video_stream is None:
                        video_stream = stream
                    elif stream["codec_type"] == "audio" and audio_stream is None:
                        audio_stream = stream

                duration = (
                    float(info["format"]["duration"]) if "duration" in info["format"] else 0.0
                )

                return {
                    "duration": duration,
                    "fps": (
                        eval(video_stream["r_frame_rate"])
                        if video_stream and "r_frame_rate" in video_stream
                        else 30.0
                    ),
                    "size": (
                        (int(video_stream["width"]), int(video_stream["height"]))
                        if video_stream
                        else (1920, 1080)
                    ),
                    "audio": audio_stream is not None,
                    "format": video_path.suffix.lower(),
                }

        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Could not get video info: {e}")

            return {
                "duration": 0.0,
                "fps": 30.0,
                "size": (1920, 1080),
                "audio": True,
                "format": video_path.suffix.lower(),
            }

    def extract_audio_from_video(self, video_path: Path, output_path: Path) -> bool:
        """
        Extract audio track from video file

        Args:
            video_path: Input video file
            output_path: Output audio file path

        Returns:
            True if successful, False otherwise
        """
        try:
            if MOVIEPY_AVAILABLE:
                with VideoFileClip(str(video_path)) as video:
                    if video.audio is not None:
                        video.audio.write_audiofile(str(output_path), verbose=False, logger=None)
                        return True
                    else:
                        if self.config.verbose:
                            print("⚠️  Video has no audio track")
                        return False

            elif self.ffmpeg_available:
                cmd = [
                    "ffmpeg",
                    "-i",
                    str(video_path),
                    "-vn",  # No video
                    "-acodec",
                    "pcm_s16le",  # Audio codec
                    "-ar",
                    "44100",  # Sample rate
                    "-ac",
                    "2",  # Stereo
                    "-y",  # Overwrite output
                    str(output_path),
                ]

                result = subprocess.run(cmd, capture_output=True)
                return result.returncode == 0

        except Exception as e:
            if self.config.verbose:
                print(f"❌ Audio extraction failed: {e}")
            return False

    def replace_audio_in_video(self, video_path: Path, audio_path: Path, output_path: Path) -> bool:
        """
        Replace the audio track in a video file

        Args:
            video_path: Input video file
            audio_path: New audio file
            output_path: Output video file

        Returns:
            True if successful, False otherwise
        """
        try:
            if MOVIEPY_AVAILABLE:
                with VideoFileClip(str(video_path)) as video:
                    with AudioFileClip(str(audio_path)) as audio:
                        # Ensure audio duration matches video duration
                        if audio.duration > video.duration:
                            audio = audio.subclip(0, video.duration)
                        elif audio.duration < video.duration:
                            # Loop audio if it's shorter
                            loops_needed = int(video.duration / audio.duration) + 1
                            audio = audio.loop(n=loops_needed).subclip(0, video.duration)

                        final_video = video.set_audio(audio)
                        final_video.write_videofile(
                            str(output_path),
                            codec="libx264",
                            audio_codec="aac",
                            verbose=False,
                            logger=None,
                        )
                        return True

            elif self.ffmpeg_available:
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
                    "-shortest",  # End when shortest stream ends
                    "-y",  # Overwrite output
                    str(output_path),
                ]

                result = subprocess.run(cmd, capture_output=True)
                return result.returncode == 0

        except Exception as e:
            if self.config.verbose:
                print(f"❌ Audio replacement failed: {e}")
            return False

    def create_video_with_isolated_audio(
        self, original_video: Path, isolated_audio: Path, output_video: Path
    ) -> bool:
        """
        Create a new video with original video and isolated audio

        Args:
            original_video: Original video file
            isolated_audio: Isolated audio track
            output_video: Output video file

        Returns:
            True if successful, False otherwise
        """
        if self.config.verbose:
            print(f"🎬 Creating video with isolated audio: {output_video.name}")

        return self.replace_audio_in_video(original_video, isolated_audio, output_video)

    def get_video_quality_settings(self) -> Dict[str, Any]:
        """
        Get video encoding settings based on quality level

        Returns:
            Dictionary with encoding parameters
        """
        quality_settings = {
            "low": {
                "crf": 28,
                "preset": "fast",
                "scale": None,  # Keep original size
                "bitrate": "1000k",
            },
            "medium": {
                "crf": 23,
                "preset": "medium",
                "scale": None,
                "bitrate": "2000k",
            },
            "high": {"crf": 18, "preset": "slow", "scale": None, "bitrate": "4000k"},
        }

        return quality_settings.get(self.config.quality, quality_settings["medium"])

    def optimize_video_for_output(self, input_path: Path, output_path: Path) -> bool:
        """
        Optimize video file for final output using quality settings

        Args:
            input_path: Input video file
            output_path: Optimized output file

        Returns:
            True if successful, False otherwise
        """
        try:
            settings = self.get_video_quality_settings()

            if self.ffmpeg_available:
                cmd = [
                    "ffmpeg",
                    "-i",
                    str(input_path),
                    "-c:v",
                    "libx264",
                    "-crf",
                    str(settings["crf"]),
                    "-preset",
                    settings["preset"],
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-y",
                    str(output_path),
                ]

                result = subprocess.run(cmd, capture_output=True)
                return result.returncode == 0

            elif MOVIEPY_AVAILABLE:
                # Basic optimization with MoviePy
                with VideoFileClip(str(input_path)) as clip:
                    clip.write_videofile(
                        str(output_path),
                        codec="libx264",
                        audio_codec="aac",
                        verbose=False,
                        logger=None,
                    )
                return True

        except Exception as e:
            if self.config.verbose:
                print(f"❌ Video optimization failed: {e}")
            return False

    def validate_video_file(self, video_path: Path) -> bool:
        """
        Validate that a video file is readable and has expected properties

        Args:
            video_path: Path to video file

        Returns:
            True if valid, False otherwise
        """
        try:
            info = self.get_video_info(video_path)

            # Basic validation checks
            if info["duration"] <= 0:
                if self.config.verbose:
                    print(f"⚠️  Video has zero duration: {video_path}")
                return False

            if info["size"][0] <= 0 or info["size"][1] <= 0:
                if self.config.verbose:
                    print(f"⚠️  Video has invalid dimensions: {video_path}")
                return False

            return True

        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Video validation failed: {e}")
            return False
