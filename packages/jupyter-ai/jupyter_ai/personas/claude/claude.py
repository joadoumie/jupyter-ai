from typing import Optional
import time
from jupyterlab_chat.models import Message, NewMessage

from ..base_persona import BasePersona, PersonaDefaults
from .prompt_template import CLAUDE_SYSTEM_PROMPT_TEMPLATE, ClaudeSystemPromptArgs
from .streaming_processor import StreamingProcessor


class ClaudeAgentPersona(BasePersona):
    """
    Claude Agent SDK powered persona for Jupyter AI.
    Uses Claude Agent SDK directly with Sonnet 4.5 as the default model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def defaults(self):
        return PersonaDefaults(
            name="Claude",
            avatar_path="/api/ai/static/claude.svg",
            description="Agent powered by the Claude Agent SDK with full tool support.",
            system_prompt="...",
            model_uid="claude-sonnet-4-5-20250929",  # Default to Sonnet 4.5
        )

    async def process_message(self, message: Message) -> None:
        # Build system prompt
        system_prompt = self._build_system_prompt(message)

        # Get MCP configuration from Jupyter AI (from .jupyter/mcp_config.json)
        mcp_config = self.get_mcp_config()

        # Initialize streaming processor
        processor = StreamingProcessor(
            ychat=self.ychat,
            persona_id=self.id,
            logger=self.log,
            awareness=self.awareness,
        )

        # Process the message with Claude Agent SDK
        # The SDK will use ANTHROPIC_API_KEY from environment and its built-in tools
        await processor.process_with_agent_sdk(
            prompt=message.body,
            system_prompt=system_prompt,
            model="claude-sonnet-4-5-20250929",
            mcp_config=mcp_config,
        )

    def _build_system_prompt(self, message: Message) -> str:
        context = self.process_attachments(message)
        format_args = ClaudeSystemPromptArgs(
            persona_name=self.name,
            context=context,
        )
        system_prompt = CLAUDE_SYSTEM_PROMPT_TEMPLATE.render(format_args.model_dump())
        return system_prompt
