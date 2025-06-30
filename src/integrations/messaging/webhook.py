"""
Webhook receiver for n8n and other automation platforms
"""

from typing import Dict, Any, Optional, Callable
from pathlib import Path
import logging
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

from .base import MessageReceiver

logger = logging.getLogger(__name__)


class WebhookReceiver(MessageReceiver):
    """HTTP webhook receiver for automation platforms like n8n"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize webhook receiver"""
        super().__init__(config)
        self.platform_name = "Webhook"
        self.port = config.get('port', 8080) if config else 8080
        self.path = config.get('path', '/webhook') if config else '/webhook'
        self.auth_token = config.get('auth_token') if config else None
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        
    def start(self) -> None:
        """Start webhook server"""
        logger.info(f"Starting webhook receiver on port {self.port}")
        
        # Create request handler with access to self
        handler = self._create_handler()
        
        # Start server
        self.server = HTTPServer(('', self.port), handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        logger.info(f"Webhook receiver listening on http://localhost:{self.port}{self.path}")
        
    def stop(self) -> None:
        """Stop webhook server"""
        logger.info("Stopping webhook receiver")
        
        if self.server:
            self.server.shutdown()
            self.server_thread.join()
            
    def send_message(
        self,
        recipient: str,
        message: str,
        attachments: Optional[List[Path]] = None,
        **kwargs
    ) -> bool:
        """Webhooks typically don't send messages back"""
        logger.warning("Webhook receiver does not support sending messages")
        return False
        
    def _create_handler(self) -> type:
        """Create HTTP request handler class"""
        receiver = self
        
        class WebhookHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                # Check path
                if self.path != receiver.path:
                    self.send_error(404)
                    return
                    
                # Check auth if configured
                if receiver.auth_token:
                    auth_header = self.headers.get('Authorization')
                    if auth_header != f"Bearer {receiver.auth_token}":
                        self.send_error(401)
                        return
                        
                # Read body
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                
                try:
                    # Parse JSON body
                    data = json.loads(body.decode('utf-8'))
                    
                    # Process message
                    receiver._handle_webhook(data)
                    
                    # Send success response
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok"}).encode())
                    
                except Exception as e:
                    logger.error(f"Webhook processing error: {e}")
                    self.send_error(500)
                    
            def log_message(self, format, *args):
                # Suppress default logging
                pass
                
        return WebhookHandler
        
    def _handle_webhook(self, data: Dict[str, Any]) -> None:
        """Handle incoming webhook data"""
        # Extract message from webhook payload
        # Format depends on the sending platform (n8n, Zapier, etc.)
        
        message_data = {
            "sender_id": data.get("sender_id", "webhook"),
            "sender_name": data.get("sender_name", "Webhook"),
            "text": data.get("text", ""),
            "timestamp": data.get("timestamp"),
            "webhook_data": data
        }
        
        # Handle different webhook formats
        if "message" in data:
            # n8n format
            message_data["text"] = data["message"]
        elif "content" in data:
            # Alternative format
            message_data["text"] = data["content"]
            
        # Extract URLs if present
        if "url" in data:
            message_data["metadata"] = {"url": data["url"]}
        elif "link" in data:
            message_data["metadata"] = {"url": data["link"]}
            
        # Process through handlers
        self.process_message(message_data)