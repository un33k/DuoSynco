"""
HeyGen lipsync provider implementation
"""

import requests
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import time

from .base import LipsyncProvider
from ...utils.util_env import get_env

logger = logging.getLogger(__name__)


class HeyGenProvider(LipsyncProvider):
    """HeyGen API integration for lipsync video generation"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize HeyGen provider"""
        super().__init__(api_key or get_env('HEYGEN_API_KEY'))
        self.base_url = "https://api.heygen.com/v1"
        self.provider_name = "HeyGen"
        
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
        Create lipsynced video using HeyGen
        
        Args:
            audio_file: Path to input audio file
            template_id: HeyGen avatar/template ID
            output_file: Path for output video
            aspect_ratio: Video aspect ratio
            speakers: Speaker mapping for avatars
            **kwargs: Additional HeyGen parameters
            
        Returns:
            Video generation result
        """
        self.validate_audio(audio_file)
        
        # Upload audio
        audio_url = self._upload_audio(audio_file)
        
        # Prepare video request
        width, height = self.get_aspect_ratio_dimensions(aspect_ratio)
        
        request_data = {
            "avatar_id": template_id,
            "audio_url": audio_url,
            "dimension": {
                "width": width,
                "height": height
            }
        }
        
        # Add background if specified
        if kwargs.get('background'):
            request_data['background'] = kwargs['background']
            
        # Handle multi-speaker scenarios
        if speakers and len(speakers) > 1:
            # HeyGen uses scenes for multi-speaker videos
            request_data = self._prepare_multi_speaker_request(
                speakers, audio_url, width, height, kwargs
            )
            
        # Create video
        video_id = self._create_video(request_data)
        
        # Wait for completion
        video_url = self._wait_for_video(video_id)
        
        # Download video
        self._download_video(video_url, output_file)
        
        return {
            "status": "completed",
            "output_file": str(output_file),
            "video_id": video_id,
            "video_url": video_url,
            "provider": "HeyGen"
        }
        
    def list_templates(
        self,
        template_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List available HeyGen avatars/templates"""
        headers = {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Get avatars
        response = requests.get(
            f"{self.base_url}/avatars",
            headers=headers
        )
        response.raise_for_status()
        
        avatars = response.json().get('avatars', [])
        
        # Format as templates
        formatted_templates = []
        for avatar in avatars:
            # Filter by type if specified
            if template_type:
                if template_type == "single" and avatar.get('type') != 'single':
                    continue
                elif template_type == "duo" and avatar.get('type') == 'single':
                    continue
                    
            formatted_templates.append({
                "id": avatar['avatar_id'],
                "name": avatar['name'],
                "type": "single",  # HeyGen avatars are typically single
                "preview_url": avatar.get('preview_url'),
                "gender": avatar.get('gender'),
                "style": avatar.get('style', 'professional')
            })
            
        return formatted_templates
        
    def get_template_info(self, template_id: str) -> Dict[str, Any]:
        """Get detailed avatar information"""
        headers = {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            f"{self.base_url}/avatars/{template_id}",
            headers=headers
        )
        response.raise_for_status()
        
        return response.json()
        
    def estimate_cost(
        self,
        audio_duration: float,
        template_id: str,
        **kwargs
    ) -> Dict[str, float]:
        """Estimate cost for video generation"""
        # HeyGen pricing model
        seconds = audio_duration
        
        # Base rate per second (varies by plan)
        base_rate = 0.05  # $0.05 per second
        
        # Avatar type affects pricing
        avatar_info = self.get_template_info(template_id)
        if avatar_info.get('premium', False):
            base_rate *= 1.5
            
        base_cost = seconds * base_rate
        
        # Additional features
        hd_multiplier = 1.2 if kwargs.get('hd', True) else 1.0
        background_cost = 2.0 if kwargs.get('custom_background') else 0.0
        
        total_cost = (base_cost * hd_multiplier) + background_cost
        
        return {
            "base_cost": base_cost,
            "hd_multiplier": hd_multiplier - 1.0,
            "background_cost": background_cost,
            "total_cost": total_cost,
            "currency": "USD"
        }
        
    def _upload_audio(self, audio_file: Path) -> str:
        """Upload audio and get URL"""
        headers = {
            "X-Api-Key": self.api_key
        }
        
        with open(audio_file, 'rb') as f:
            files = {'file': (audio_file.name, f, 'audio/mpeg')}
            response = requests.post(
                f"{self.base_url}/upload",
                headers=headers,
                files=files
            )
            
        response.raise_for_status()
        return response.json()['url']
        
    def _create_video(self, request_data: Dict[str, Any]) -> str:
        """Create video generation request"""
        headers = {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{self.base_url}/videos",
            headers=headers,
            json=request_data
        )
        response.raise_for_status()
        
        return response.json()['video_id']
        
    def _wait_for_video(self, video_id: str, timeout: int = 1200) -> str:
        """Wait for video to be ready"""
        headers = {
            "X-Api-Key": self.api_key
        }
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = requests.get(
                f"{self.base_url}/videos/{video_id}",
                headers=headers
            )
            response.raise_for_status()
            
            video_data = response.json()
            status = video_data.get('status')
            
            if status == 'completed':
                return video_data['video_url']
            elif status == 'failed':
                raise Exception(f"Video generation failed: {video_data.get('error')}")
                
            time.sleep(5)
            
        raise TimeoutError(f"Video generation timed out after {timeout} seconds")
        
    def _download_video(self, video_url: str, output_file: Path) -> None:
        """Download the generated video"""
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
    def _prepare_multi_speaker_request(
        self,
        speakers: Dict[str, str],
        audio_url: str,
        width: int,
        height: int,
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare request for multi-speaker video"""
        # HeyGen uses scenes for multi-speaker videos
        scenes = []
        
        # For now, create sequential scenes
        # In future, could support split-screen with custom backgrounds
        for speaker_id, avatar_id in speakers.items():
            scene = {
                "avatar_id": avatar_id,
                "audio_url": audio_url,  # Would need per-speaker audio in real implementation
                "duration": kwargs.get(f"{speaker_id}_duration", 10)
            }
            scenes.append(scene)
            
        return {
            "scenes": scenes,
            "dimension": {
                "width": width,
                "height": height
            },
            "transition": kwargs.get('transition', 'fade')
        }