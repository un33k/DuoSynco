"""
Content analysis command
"""

import click
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@click.command(name='analyze')
@click.option('--link', '-l', help='URL to analyze')
@click.option('--file', '-f', type=click.Path(exists=True), help='File to analyze')
@click.option('--format', '-fmt', 
              type=click.Choice(['summary', 'pros-cons', 'key-points', 'sentiment']), 
              default='summary')
@click.option('--provider', '-p', default='openai', help='LLM provider')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.pass_context
def analyze_command(ctx, link, file, format, provider, output):
    """Analyze content from URL or file"""
    if not link and not file:
        click.echo("❌ Error: Provide either --link or --file", err=True)
        raise click.Abort()
        
    click.echo(f"🔍 Analyzing content ({format})")
    
    # Initialize LLM provider
    from ...ai.llm.factory import LLMFactory
    
    config = ctx.obj.get('config', {})
    llm_config = config.get('llm', {}).get(provider, {})
    
    try:
        llm = LLMFactory.create_provider(provider, config=llm_config)
        
        # Get content
        content = ""
        if file:
            with open(file, 'r') as f:
                content = f.read()
            click.echo(f"📄 Analyzing file: {file}")
        elif link:
            click.echo(f"🔗 Analyzing URL: {link}")
            # In production, fetch actual content
            content = f"Content from {link}"
            
        # Create analysis prompt based on format
        prompts = {
            "summary": {
                "system": "You are an expert content analyst who creates concise, insightful summaries.",
                "user": f"Summarize the following content in 3-5 key points:\n\n{content}"
            },
            "pros-cons": {
                "system": "You are an analytical expert who identifies advantages and disadvantages objectively.",
                "user": f"Analyze the pros and cons of the following:\n\n{content}\n\nProvide as JSON with 'pros' and 'cons' arrays."
            },
            "key-points": {
                "system": "You are an expert at extracting and organizing key information.",
                "user": f"Extract the key points from this content:\n\n{content}\n\nProvide as a numbered list."
            },
            "sentiment": {
                "system": "You are a sentiment analysis expert who identifies emotional tone and bias.",
                "user": f"Analyze the sentiment and tone of this content:\n\n{content}\n\nProvide overall sentiment, tone, and any bias detected."
            }
        }
        
        prompt_config = prompts[format]
        
        # Generate analysis
        click.echo("🤔 Analyzing...")
        result = llm.generate(
            prompt=prompt_config['user'],
            system_prompt=prompt_config['system']
        )
        
        # Format output
        analysis = {
            "source": link or str(file),
            "format": format,
            "analysis": result,
            "word_count": len(content.split()),
            "provider": provider
        }
        
        # Output
        if output:
            output_path = Path(output)
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            click.echo(f"✅ Saved to: {output_path}")
        else:
            click.echo("\n" + "="*50)
            click.echo(f"Analysis ({format}):")
            click.echo("="*50)
            click.echo(result)
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()