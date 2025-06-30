"""
Base class for lipsync provider integrations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LipsyncProvider(ABC):
    """Abstract base class for lipsync providers"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize lipsync provider
        
        Args:
            api_key: API key for the provider
        """
        self.api_key = api_key
        self.provider_name = self.__class__.__name__
        self.requires_api_key = True
        
    @abstractmethod
    def create_video(
        self,
        audio_file: Path,
        template_id: str,
        output_file: Path,
        aspect_ratio: str = "16:9",
        speakers: Dict[str, str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create lipsynced video from audio
        
        Args:
            audio_file: Path to input audio file
            template_id: Template identifier for the video
            output_file: Path for output video
            aspect_ratio: Video aspect ratio (16:9 or 9:16)
            speakers: Speaker mapping for multi-speaker videos
            **kwargs: Provider-specific parameters
            
        Returns:
            Dictionary with video metadata and status
        """
        pass
        
    @abstractmethod
    def list_templates(
        self,
        template_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available templates
        
        Args:
            template_type: Filter by template type (single/duo/multi)
            
        Returns:
            List of available templates
        """
        pass
        
    @abstractmethod
    def get_template_info(self, template_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a template
        
        Args:
            template_id: Template identifier
            
        Returns:
            Template details including speaker slots, formats, etc.
        """
        pass
        
    @abstractmethod
    def estimate_cost(
        self,
        audio_duration: float,
        template_id: str,
        **kwargs
    ) -> Dict[str, float]:
        """
        Estimate cost for video generation
        
        Args:
            audio_duration: Duration of audio in seconds
            template_id: Template to use
            **kwargs: Additional cost factors
            
        Returns:
            Cost breakdown
        """
        pass
        
    def validate_audio(self, audio_file: Path) -> bool:
        """
        Validate audio file meets provider requirements
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            True if valid
        """
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
        # Basic validation - providers can override for specific requirements
        valid_extensions = ['.mp3', '.wav', '.m4a', '.aac']
        if audio_file.suffix.lower() not in valid_extensions:
            raise ValueError(f"Unsupported audio format: {audio_file.suffix}")
            
        return True
        
    def get_aspect_ratio_dimensions(self, aspect_ratio: str) -> tuple:
        """
        Get video dimensions for aspect ratio
        
        Args:
            aspect_ratio: Aspect ratio string (16:9 or 9:16)
            
        Returns:
            Tuple of (width, height)
        """
        dimensions = {
            "16:9": (1920, 1080),
            "9:16": (1080, 1920),
            "1:1": (1080, 1080),
            "4:3": (1440, 1080)
        }
        
        return dimensions.get(aspect_ratio, (1920, 1080))