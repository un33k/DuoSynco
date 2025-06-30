"""
Interactive chat command
"""

import click
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


@click.command(name='chat')
@click.option('--provider', '-p', default='openai', help='LLM provider')
@click.option('--model', '-m', help='Specific model to use')
@click.option('--system', '-s', help='System prompt')
@click.pass_context
def chat_command(ctx, provider, model, system):
    """Interactive chat with LLM"""
    from ...ai.llm.factory import LLMFactory
    
    click.echo("💬 Starting interactive chat session")
    click.echo("Type 'exit' or 'quit' to end the session")
    click.echo("Type 'clear' to clear conversation history")
    click.echo("-" * 50)
    
    # Initialize LLM provider
    config = ctx.obj.get('config', {})
    llm_config = config.get('llm', {}).get(provider, {})
    
    if model:
        llm_config['model'] = model
        
    try:
        llm = LLMFactory.create_provider(provider, config=llm_config)
        click.echo(f"Using {provider} ({llm.model})")
        
        # Initialize conversation
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
            
        # Chat loop
        while True:
            try:
                # Get user input
                user_input = click.prompt("\nYou", type=str)
                
                # Handle special commands
                if user_input.lower() in ['exit', 'quit']:
                    click.echo("👋 Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    messages = []
                    if system:
                        messages.append({"role": "system", "content": system})
                    click.echo("🧹 Conversation cleared")
                    continue
                    
                # Add user message
                messages.append({"role": "user", "content": user_input})
                
                # Get AI response
                click.echo("\nAI: ", nl=False)
                response = llm.chat(messages)
                click.echo(response)
                
                # Add to conversation
                messages.append({"role": "assistant", "content": response})
                
            except KeyboardInterrupt:
                click.echo("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                click.echo(f"\n❌ Error: {e}", err=True)
                
    except Exception as e:
        click.echo(f"❌ Failed to initialize chat: {e}", err=True)
        raise click.Abort()