"""
MCP server command
"""

import click
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@click.command(name='serve')
@click.option('--port', '-p', type=int, default=3000, help='Server port')
@click.option('--host', '-h', default='0.0.0.0', help='Server host')
@click.option('--mcp', is_flag=True, default=True, help='Enable MCP protocol')
@click.option('--allowed-origins', '-ao', multiple=True, help='Allowed CORS origins')
@click.option('--max-requests', '-mr', type=int, default=100, help='Max concurrent requests')
@click.option('--config', '-c', type=click.Path(exists=True), help='Server configuration file')
@click.pass_context
def serve_command(ctx, port, host, mcp, allowed_origins, max_requests, config):
    """Start MCP server for n8n integration"""
    click.echo(f"🚀 Starting MCP server on {host}:{port}")
    
    # Import server module
    try:
        from ...mcp.server import MCPServer
        
        # Load configuration
        server_config = {
            'host': host,
            'port': port,
            'max_requests': max_requests,
            'allowed_origins': list(allowed_origins) if allowed_origins else ['*'],
            'llm_config': ctx.obj.get('config', {}).get('llm', {})
        }
        
        if config:
            import json
            with open(config) as f:
                server_config.update(json.load(f))
                
        # Create and start server
        server = MCPServer(server_config)
        
        click.echo(f"✅ Server configuration:")
        click.echo(f"   Host: {host}")
        click.echo(f"   Port: {port}")
        click.echo(f"   Max requests: {max_requests}")
        click.echo(f"   CORS origins: {server_config['allowed_origins']}")
        click.echo(f"   MCP Protocol: {'Enabled' if mcp else 'Disabled'}")
        click.echo("\n📡 Endpoints:")
        click.echo("   POST /mcp/generate - Generate content")
        click.echo("   POST /mcp/analyze - Analyze content")
        click.echo("   POST /mcp/extract - Extract from URLs")
        click.echo("   GET  /mcp/status - Server status")
        click.echo("   WS   /mcp/stream - WebSocket streaming")
        click.echo("\nPress Ctrl+C to stop the server")
        
        # Start server
        server.run()
        
    except ImportError:
        click.echo("❌ MCP server module not yet implemented", err=True)
        click.echo("This feature will be available in the next update", err=True)
    except KeyboardInterrupt:
        click.echo("\n👋 Server stopped")
    except Exception as e:
        click.echo(f"❌ Server error: {e}", err=True)
        raise click.Abort()