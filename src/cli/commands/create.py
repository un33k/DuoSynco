"""
Content creation commands
"""

import click
import json
from pathlib import Path
from typing import Optional
import logging

from ...ai.llm.factory import LLMFactory
from ...ai.prompts.debate import DebatePrompts
from ...ai.prompts.news import NewsPrompts
from ...ai.prompts.translate import TranslatePrompts

logger = logging.getLogger(__name__)


@click.group(name='create')
def create_group():
    """Create content using LLM"""
    pass


@create_group.command()
@click.option('--topic', '-t', required=True, help='Debate topic')
@click.option('--style', '-s', type=click.Choice(['balanced', 'heated', 'academic']), default='balanced')
@click.option('--duration', '-d', type=int, default=300, help='Target duration in seconds')
@click.option('--provider', '-p', default='openai', help='LLM provider')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', '-f', type=click.Choice(['json', 'text']), default='json')
@click.pass_context
def debate(ctx, topic, style, duration, provider, output, format):
    """Generate a debate on a topic"""
    click.echo(f"🎭 Generating {style} debate on: {topic}")
    
    # Initialize LLM provider
    config = ctx.obj.get('config', {})
    llm_config = config.get('llm', {}).get(provider, {})
    
    try:
        llm = LLMFactory.create_provider(provider, config=llm_config)
        
        # Get debate prompt
        prompts = DebatePrompts()
        system_prompt, user_prompt = prompts.get_debate_prompt(
            topic=topic,
            style=style,
            duration=duration
        )
        
        # Generate content
        click.echo("🤔 Thinking...")
        result = llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt
        )
        
        # Parse and format result
        debate_data = prompts.parse_debate_response(result)
        
        # Output
        if output:
            output_path = Path(output)
            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(debate_data, f, indent=2)
            else:
                with open(output_path, 'w') as f:
                    f.write(prompts.format_as_text(debate_data))
            click.echo(f"✅ Saved to: {output_path}")
        else:
            # Print to console
            if format == 'json':
                click.echo(json.dumps(debate_data, indent=2))
            else:
                click.echo(prompts.format_as_text(debate_data))
                
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@create_group.command()
@click.option('--link', '-l', help='Article or content URL')
@click.option('--text', '-t', help='Direct text input')
@click.option('--anchor', '-a', type=click.Choice(['male', 'female']), default='female')
@click.option('--style', '-s', type=click.Choice(['standard', 'breaking', 'feature']), default='standard')
@click.option('--provider', '-p', default='openai', help='LLM provider')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.pass_context
def news(ctx, link, text, anchor, style, provider, output):
    """Generate news format content"""
    if not link and not text:
        click.echo("❌ Error: Provide either --link or --text", err=True)
        raise click.Abort()
        
    click.echo(f"📰 Generating {style} news with {anchor} anchor")
    
    # Initialize LLM provider
    config = ctx.obj.get('config', {})
    llm_config = config.get('llm', {}).get(provider, {})
    
    try:
        llm = LLMFactory.create_provider(provider, config=llm_config)
        
        # Fetch content if link provided
        content = text
        if link:
            click.echo(f"🔗 Fetching content from: {link}")
            # In production, use actual web scraping
            content = f"Content from {link}"
            
        # Get news prompt
        prompts = NewsPrompts()
        system_prompt, user_prompt = prompts.get_news_prompt(
            content=content,
            style=style,
            anchor_gender=anchor
        )
        
        # Generate content
        click.echo("📝 Writing news script...")
        result = llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt
        )
        
        # Parse and format result
        news_data = prompts.parse_news_response(result)
        news_data['anchor'] = anchor
        
        # Output
        if output:
            output_path = Path(output)
            with open(output_path, 'w') as f:
                json.dump(news_data, f, indent=2)
            click.echo(f"✅ Saved to: {output_path}")
        else:
            click.echo(json.dumps(news_data, indent=2))
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@create_group.command()
@click.option('--youtube', '-y', help='YouTube video URL')
@click.option('--file', '-f', type=click.Path(exists=True), help='Subtitle file')
@click.option('--from-lang', '-fl', default='en', help='Source language')
@click.option('--to-lang', '-tl', required=True, help='Target language')
@click.option('--preserve-timing', is_flag=True, default=True, help='Preserve original timing')
@click.option('--provider', '-p', default='openai', help='LLM provider')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.pass_context
def translate(ctx, youtube, file, from_lang, to_lang, preserve_timing, provider, output):
    """Translate video subtitles"""
    if not youtube and not file:
        click.echo("❌ Error: Provide either --youtube or --file", err=True)
        raise click.Abort()
        
    click.echo(f"🌍 Translating from {from_lang} to {to_lang}")
    
    # Initialize LLM provider
    config = ctx.obj.get('config', {})
    llm_config = config.get('llm', {}).get(provider, {})
    
    try:
        llm = LLMFactory.create_provider(provider, config=llm_config)
        
        # Get subtitles
        subtitles = []
        if youtube:
            click.echo(f"📹 Extracting subtitles from: {youtube}")
            # In production, use youtube-dl or API
            subtitles = [
                {"text": "Hello world", "start": 0, "end": 3},
                {"text": "This is a test", "start": 3, "end": 6}
            ]
        elif file:
            with open(file) as f:
                subtitles = json.load(f)
                
        # Get translation prompts
        prompts = TranslatePrompts()
        
        # Translate each subtitle
        translated_subtitles = []
        for i, sub in enumerate(subtitles):
            click.echo(f"Translating {i+1}/{len(subtitles)}...", nl=False)
            
            system_prompt, user_prompt = prompts.get_translation_prompt(
                text=sub['text'],
                source_lang=from_lang,
                target_lang=to_lang,
                preserve_length=preserve_timing
            )
            
            translated_text = llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt
            )
            
            translated_subtitles.append({
                **sub,
                "original_text": sub['text'],
                "text": translated_text.strip()
            })
            click.echo(" ✓")
            
        # Output
        result = {
            "source_language": from_lang,
            "target_language": to_lang,
            "subtitles": translated_subtitles
        }
        
        if output:
            output_path = Path(output)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            click.echo(f"✅ Saved to: {output_path}")
        else:
            click.echo(json.dumps(result, indent=2, ensure_ascii=False))
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()