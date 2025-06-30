"""
Base class for content generators
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ContentGenerator(ABC):
    """Abstract base class for content generators"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize content generator
        
        Args:
            config: Generator-specific configuration
        """
        self.config = config or {}
        self.generator_type = self.__class__.__name__
        
    @abstractmethod
    def generate_script(
        self,
        input_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate content script
        
        Args:
            input_data: Input data (text, URL, etc.)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated script with metadata
        """
        pass
        
    @abstractmethod
    def assign_speakers(
        self,
        script: Dict[str, Any],
        speaker_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assign speakers to script segments
        
        Args:
            script: Generated script
            speaker_preferences: Optional speaker preferences
            
        Returns:
            Script with speaker assignments
        """
        pass
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data
        
        Args:
            input_data: Input to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If input is invalid
        """
        if not input_data:
            raise ValueError("Input data is empty")
            
        return True
        
    def format_for_tts(
        self,
        script: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Format script for TTS processing
        
        Args:
            script: Script with speaker assignments
            
        Returns:
            List of TTS segments
        """
        segments = []
        
        for segment in script.get("segments", []):
            tts_segment = {
                "speaker": segment.get("speaker"),
                "text": segment.get("text"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "metadata": segment.get("metadata", {})
            }
            segments.append(tts_segment)
            
        return segments