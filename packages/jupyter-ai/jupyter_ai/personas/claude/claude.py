from typing import Optional
from jupyterlab_chat.models import Message, NewMessage
from claude_agent_sdk import ClaudeSDKClient

from ..base_persona import BasePersona, PersonaDefaults
from .prompt_template import CLAUDE_SYSTEM_PROMPT_TEMPLATE, ClaudeSystemPromptArgs
from .streaming_processor import StreamingProcessor, create_agent_options


class ClaudeAgentPersona(BasePersona):
    """
    Claude Agent SDK powered persona for Jupyter AI.
    Uses ClaudeSDKClient to maintain conversation history across messages.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sdk_client: Optional[ClaudeSDKClient] = None

    @property
    def defaults(self):
        return PersonaDefaults(
            name="Claude",
            avatar_path="/api/ai/static/claude.svg",
            description="Agent powered by the Claude Agent SDK with full tool support.",
            system_prompt="...",
            model_uid="claude-sonnet-4-5-20250929",  # Default to Sonnet 4.5
        )

    async def _get_or_create_client(self, mcp_config: dict, system_prompt: str) -> ClaudeSDKClient:
        """Get existing client or create a new one."""
        if self._sdk_client is None:
            self.log.info("Creating new Claude SDK client for conversation")
            options = create_agent_options(
                system_prompt=system_prompt,
                model="claude-sonnet-4-5-20250929",
                mcp_config=mcp_config,
                logger=self.log
            )
            self._sdk_client = ClaudeSDKClient(options=options)
            await self._sdk_client.connect()

        return self._sdk_client

    async def process_message(self, message: Message) -> None:
        # Build system prompt (only used on first client creation)
        system_prompt = self._build_system_prompt(message)

        # Get MCP configuration from Jupyter AI (from .jupyter/mcp_config.json)
        mcp_config = self.get_mcp_config()

        # Get or create persistent client (maintains conversation history)
        sdk_client = await self._get_or_create_client(mcp_config, system_prompt)

        # Build user prompt with attachment context
        user_prompt = self._build_user_prompt(message)

        # Initialize streaming processor
        processor = StreamingProcessor(
            ychat=self.ychat,
            persona_id=self.id,
            logger=self.log,
            awareness=self.awareness,
        )

        # Process message with persistent client
        await processor.process_with_persistent_client(
            sdk_client=sdk_client,
            prompt=user_prompt,
        )

    def _build_system_prompt(self, message: Message) -> str:
        """Build system prompt (only used when creating new client)."""
        format_args = ClaudeSystemPromptArgs(
            persona_name=self.name,
        )
        system_prompt = CLAUDE_SYSTEM_PROMPT_TEMPLATE.render(format_args.model_dump())
        return system_prompt

    def _build_user_prompt(self, message: Message) -> str:
        """Build user prompt with attachment file paths included."""
        if not message.attachments:
            return message.body

        # Get file paths for attachments
        file_paths = []
        for attachment_id in message.attachments:
            file_path = self.resolve_attachment_to_path(attachment_id)
            if file_path:
                file_paths.append(file_path)

        if not file_paths:
            return message.body

        # Build prompt with file references
        files_list = "\n".join([f"- {path}" for path in file_paths])
        prompt = (
            f"The user has attached the following file(s):\n"
            f"{files_list}\n\n"
            f"{message.body}"
        )
        self.log.info(f"Built user prompt with {len(file_paths)} file attachment(s)")
        return prompt
