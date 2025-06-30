"""
MCP Server implementation for n8n integration
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import logging
from typing import Dict, Any
import json

from ..ai.llm.factory import LLMFactory
from ..ai.prompts.debate import DebatePrompts
from ..ai.prompts.news import NewsPrompts
from ..ai.prompts.translate import TranslatePrompts

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP-compatible server for n8n workflows"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MCP server"""
        self.config = config
        self.app = Flask(__name__)
        
        # Configure CORS
        CORS(self.app, origins=config.get('allowed_origins', ['*']))
        
        # Setup routes
        self._setup_routes()
        
        # Initialize LLM providers
        self.llm_config = config.get('llm_config', {})
        
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/mcp/status', methods=['GET'])
        def status():
            """Server status endpoint"""
            return jsonify({
                'status': 'running',
                'version': '0.2.0',
                'capabilities': [
                    'generate_debate',
                    'generate_news', 
                    'translate',
                    'analyze'
                ],
                'max_requests': self.config.get('max_requests', 100)
            })
            
        @self.app.route('/mcp/generate', methods=['POST'])
        def generate():
            """Content generation endpoint"""
            try:
                data = request.json
                content_type = data.get('type', 'debate')
                provider = data.get('provider', 'openai')
                
                # Initialize LLM
                llm = LLMFactory.create_provider(
                    provider,
                    config=self.llm_config.get(provider, {})
                )
                
                # Generate based on type
                if content_type == 'debate':
                    result = self._generate_debate(llm, data)
                elif content_type == 'news':
                    result = self._generate_news(llm, data)
                elif content_type == 'translate':
                    result = self._generate_translation(llm, data)
                else:
                    return jsonify({'error': f'Unknown content type: {content_type}'}), 400
                    
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/mcp/analyze', methods=['POST'])
        def analyze():
            """Content analysis endpoint"""
            try:
                data = request.json
                content = data.get('content', '')
                format = data.get('format', 'summary')
                provider = data.get('provider', 'openai')
                
                # Initialize LLM
                llm = LLMFactory.create_provider(
                    provider,
                    config=self.llm_config.get(provider, {})
                )
                
                # Create analysis prompt
                prompts = {
                    "summary": "Summarize this content in 3-5 key points",
                    "pros-cons": "Analyze the pros and cons",
                    "key-points": "Extract the key points",
                    "sentiment": "Analyze the sentiment and tone"
                }
                
                prompt = prompts.get(format, prompts['summary'])
                result = llm.generate(f"{prompt}:\n\n{content}")
                
                return jsonify({
                    'format': format,
                    'analysis': result,
                    'provider': provider
                })
                
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                return jsonify({'error': str(e)}), 500
                
    def _generate_debate(self, llm, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate debate content"""
        prompts = DebatePrompts()
        
        system_prompt, user_prompt = prompts.get_debate_prompt(
            topic=data.get('topic', 'Unknown topic'),
            style=data.get('style', 'balanced'),
            duration=data.get('duration', 300)
        )
        
        result = llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt
        )
        
        return prompts.parse_debate_response(result)
        
    def _generate_news(self, llm, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate news content"""
        prompts = NewsPrompts()
        
        system_prompt, user_prompt = prompts.get_news_prompt(
            content=data.get('content', ''),
            style=data.get('style', 'standard'),
            anchor_gender=data.get('anchor', 'female')
        )
        
        result = llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt
        )
        
        return prompts.parse_news_response(result)
        
    def _generate_translation(self, llm, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate translation"""
        prompts = TranslatePrompts()
        
        subtitles = data.get('subtitles', [])
        translated_subtitles = []
        
        for sub in subtitles:
            system_prompt, user_prompt = prompts.get_translation_prompt(
                text=sub.get('text', ''),
                source_lang=data.get('source_lang', 'en'),
                target_lang=data.get('target_lang', 'es'),
                preserve_length=data.get('preserve_timing', True)
            )
            
            translated_text = llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt
            )
            
            translated_subtitles.append({
                **sub,
                'original_text': sub.get('text'),
                'text': translated_text.strip()
            })
            
        return {
            'source_language': data.get('source_lang'),
            'target_language': data.get('target_lang'),
            'subtitles': translated_subtitles
        }
        
    def run(self):
        """Start the server"""
        self.app.run(
            host=self.config.get('host', '0.0.0.0'),
            port=self.config.get('port', 3000),
            debug=self.config.get('debug', False)
        )