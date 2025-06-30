"""
Collossyan lipsync provider implementation
"""

import requests
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import time

from .base import LipsyncProvider
from ...utils.util_env import get_env

logger = logging.getLogger(__name__)


class CollossyanProvider(LipsyncProvider):
    """Collossyan API integration for lipsync video generation"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Collossyan provider"""
        super().__init__(api_key or get_env('COLLOSSYAN_API_KEY'))
        self.base_url = "https://api.collossyan.com/v1"
        self.provider_name = "Collossyan"
        
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
        Create lipsynced video using Collossyan
        
        Args:
            audio_file: Path to input audio file
            template_id: Collossyan template ID
            output_file: Path for output video
            aspect_ratio: Video aspect ratio
            speakers: Speaker mapping
            **kwargs: Additional Collossyan parameters
            
        Returns:
            Video generation result
        """
        self.validate_audio(audio_file)
        
        # Upload audio file
        upload_result = self._upload_audio(audio_file)
        audio_url = upload_result.get('url')
        
        # Prepare video request
        width, height = self.get_aspect_ratio_dimensions(aspect_ratio)
        
        request_data = {
            "template_id": template_id,
            "audio_url": audio_url,
            "resolution": {
                "width": width,
                "height": height
            },
            "speakers": speakers or {},
            **kwargs
        }
        
        # Create video job
        job_result = self._create_video_job(request_data)
        job_id = job_result.get('job_id')
        
        # Poll for completion
        video_url = self._wait_for_completion(job_id)
        
        # Download video
        self._download_video(video_url, output_file)
        
        return {
            "status": "completed",
            "output_file": str(output_file),
            "job_id": job_id,
            "video_url": video_url,
            "duration": job_result.get('duration'),
            "cost": job_result.get('cost')
        }
        
    def list_templates(
        self,
        template_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List available Collossyan templates"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        params = {}
        if template_type:
            params['type'] = template_type
            
        response = requests.get(
            f"{self.base_url}/templates",
            headers=headers,
            params=params
        )
        response.raise_for_status()
        
        templates = response.json().get('templates', [])
        
        # Format template information
        formatted_templates = []
        for template in templates:
            formatted_templates.append({
                "id": template['id'],
                "name": template['name'],
                "type": template['type'],
                "speakers": template.get('speaker_count', 1),
                "description": template.get('description', ''),
                "preview_url": template.get('preview_url'),
                "supported_aspects": template.get('supported_aspects', ['16:9'])
            })
            
        return formatted_templates
        
    def get_template_info(self, template_id: str) -> Dict[str, Any]:
        """Get detailed template information"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            f"{self.base_url}/templates/{template_id}",
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
        # Collossyan typically charges per minute
        minutes = audio_duration / 60.0
        
        # Get template info for pricing tier
        template_info = self.get_template_info(template_id)
        base_rate = template_info.get('price_per_minute', 2.0)
        
        # Calculate costs
        base_cost = minutes * base_rate
        
        # Additional costs
        rush_multiplier = kwargs.get('rush_processing', 1.0)
        hd_addon = 0.5 if kwargs.get('hd_quality', False) else 0.0
        
        total_cost = (base_cost * rush_multiplier) + hd_addon
        
        return {
            "base_cost": base_cost,
            "rush_fee": base_cost * (rush_multiplier - 1),
            "hd_addon": hd_addon,
            "total_cost": total_cost,
            "currency": "USD"
        }
        
    def _upload_audio(self, audio_file: Path) -> Dict[str, Any]:
        """Upload audio file to Collossyan"""
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        with open(audio_file, 'rb') as f:
            files = {'audio': (audio_file.name, f, 'audio/mpeg')}
            response = requests.post(
                f"{self.base_url}/upload/audio",
                headers=headers,
                files=files
            )
            
        response.raise_for_status()
        return response.json()
        
    def _create_video_job(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create video generation job"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{self.base_url}/videos",
            headers=headers,
            json=request_data
        )
        response.raise_for_status()
        
        return response.json()
        
    def _wait_for_completion(self, job_id: str, timeout: int = 1800) -> str:
        """Wait for video generation to complete"""
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = requests.get(
                f"{self.base_url}/videos/{job_id}",
                headers=headers
            )
            response.raise_for_status()
            
            status_data = response.json()
            status = status_data.get('status')
            
            if status == 'completed':
                return status_data.get('video_url')
            elif status == 'failed':
                raise Exception(f"Video generation failed: {status_data.get('error')}")
                
            # Wait before next poll
            time.sleep(10)
            
        raise TimeoutError(f"Video generation timed out after {timeout} seconds")
        
    def _download_video(self, video_url: str, output_file: Path) -> None:
        """Download generated video"""
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)