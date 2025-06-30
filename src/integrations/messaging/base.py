"""
Base class for messaging platform integrations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MessageReceiver(ABC):
    """Abstract base class for message receivers"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize message receiver
        
        Args:
            config: Platform-specific configuration
        """
        self.config = config or {}
        self.platform_name = self.__class__.__name__
        self.message_handlers: List[Callable] = []
        
    @abstractmethod
    def start(self) -> None:
        """Start receiving messages"""
        pass
        
    @abstractmethod
    def stop(self) -> None:
        """Stop receiving messages"""
        pass
        
    @abstractmethod
    def send_message(
        self,
        recipient: str,
        message: str,
        attachments: Optional[List[Path]] = None,
        **kwargs
    ) -> bool:
        """
        Send a message
        
        Args:
            recipient: Recipient identifier
            message: Message text
            attachments: Optional file attachments
            **kwargs: Platform-specific parameters
            
        Returns:
            True if sent successfully
        """
        pass
        
    def register_handler(self, handler: Callable) -> None:
        """
        Register a message handler
        
        Args:
            handler: Function to handle incoming messages
                     Should accept (message: Dict[str, Any]) -> None
        """
        self.message_handlers.append(handler)
        
    def process_message(self, message: Dict[str, Any]) -> None:
        """
        Process incoming message through registered handlers
        
        Args:
            message: Standardized message format
        """
        # Standardize message format
        standardized = self._standardize_message(message)
        
        # Call all registered handlers
        for handler in self.message_handlers:
            try:
                handler(standardized)
            except Exception as e:
                logger.error(f"Handler error: {e}")
                
    def _standardize_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert platform-specific message to standard format
        
        Args:
            message: Platform-specific message
            
        Returns:
            Standardized message format
        """
        return {
            "platform": self.platform_name,
            "sender_id": message.get("sender_id", "unknown"),
            "sender_name": message.get("sender_name", ""),
            "text": message.get("text", ""),
            "timestamp": message.get("timestamp"),
            "attachments": message.get("attachments", []),
            "metadata": message.get("metadata", {}),
            "raw": message
        }