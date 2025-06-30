"""
Main CLI entry point for DuoSynco AI
"""

import click
import logging
from pathlib import Path
from typing import Optional

# Import commands
from .commands.create import create_group
from .commands.chat import chat_command
from .commands.analyze import analyze_command
from .commands.serve import serve_command

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@click.group()
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--config', type=click.Path(exists=True), help='Configuration file')
@click.pass_context
def cli(ctx, debug, config):
    """
    DuoSynco AI - LLM-powered content generation
    
    Use this tool to generate debates, news, translations, and more
    using various LLM providers (OpenAI, Anthropic, etc.)
    """
    ctx.ensure_object(dict)
    
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration if provided
    if config:
        import json
        with open(config) as f:
            ctx.obj['config'] = json.load(f)
    else:
        ctx.obj['config'] = {}
        
    click.echo("🤖 DuoSynco AI CLI")


# Add command groups
cli.add_command(create_group)
cli.add_command(chat_command)
cli.add_command(analyze_command)
cli.add_command(serve_command)


# Convenience commands at top level
@cli.command()
@click.option('--provider', '-p', default='openai', help='LLM provider to use')
def list_providers(provider):
    """List available LLM providers"""
    from ..ai.llm.factory import LLMFactory
    
    click.echo("Available LLM Providers:")
    providers = LLMFactory.list_providers()
    
    for name, info in providers.items():
        status = "✅" if info.get('available', False) else "❌"
        model = info.get('default_model', 'Unknown')
        click.echo(f"  {status} {name}: {model}")


@cli.command()
def version():
    """Show version information"""
    click.echo("DuoSynco AI v0.2.0")
    click.echo("LLM Integration for Content Generation")


def main():
    """Main entry point"""
    cli(obj={})