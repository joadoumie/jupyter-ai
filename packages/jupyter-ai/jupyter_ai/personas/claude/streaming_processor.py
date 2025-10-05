"""
Streaming processor for Claude Agent SDK responses.

Uses ClaudeSDKClient for continuous conversations with maintained context.
The SDK automatically handles tool execution and streams messages via receive_response().
"""
import time
import json
from typing import Optional, Dict, Any
from logging import Logger
from jupyterlab_chat.models import Message, NewMessage
from jupyterlab_chat.ychat import YChat
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from claude_agent_sdk.types import AssistantMessage, TextBlock, ToolUseBlock, ToolResultBlock

from ...personas.persona_awareness import PersonaAwareness
from .tool_adapter import render_tool_calls


def create_agent_options(
    system_prompt: str,
    model: str,
    mcp_config: Optional[dict],
    logger: Logger,
) -> ClaudeAgentOptions:
    """Create ClaudeAgentOptions with the given configuration."""
    if mcp_config:
        mcp_servers = mcp_config
        logger.info(f"Using MCP servers from config: {list(mcp_servers.keys())}")
    else:
        mcp_servers = {
            "demo": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-everything"]
            }
        }
        logger.info(f"No MCP config found, using demo server")

    return ClaudeAgentOptions(
        system_prompt=system_prompt,
        model=model,
        mcp_servers=mcp_servers,
        permission_mode='bypassPermissions',
    )


class StreamingProcessor:
    """
    Processes streaming responses from Claude Agent SDK and updates YChat.
    Handles continuous conversations with maintained context across messages.
    """

    def __init__(
        self,
        ychat: YChat,
        persona_id: str,
        logger: Logger,
        awareness: PersonaAwareness,
    ):
        self.ychat = ychat
        self.persona_id = persona_id
        self.logger = logger
        self.awareness = awareness

        # Current message state - track elements in order
        self.message_id: Optional[str] = None
        self.message_elements: list = []  # Ordered list of ('text', content) or ('tool', tool_id)
        self.tool_calls: Dict[str, Dict[str, Any]] = {}  # tool_use_id -> tool data

    async def process_with_persistent_client(
        self,
        sdk_client: ClaudeSDKClient,
        prompt: str,
    ) -> None:
        """
        Process a user prompt using a persistent ClaudeSDKClient.

        The client maintains conversation history across multiple queries.
        Uses receive_response() which naturally completes when the response is done.
        """
        try:
            self.awareness.set_local_state_field("isWriting", True)

            # Reset state for new response
            self.message_id = None
            self.message_elements = []
            self.tool_calls = {}

            # Send query to SDK
            await sdk_client.query(prompt=prompt)

            # Stream response messages - loop ends naturally when response is complete
            async for message in sdk_client.receive_response():
                await self._handle_message(message)

        except Exception as e:
            self.logger.exception(f"Error processing with Claude Agent SDK: {e}")
            self.ychat.add_message(NewMessage(
                sender=self.persona_id,
                body=f"An error occurred: {str(e)}"
            ))
        finally:
            self.awareness.set_local_state_field("isWriting", False)

    async def _handle_message(self, message: Any) -> None:
        """Handle a message from the SDK stream."""
        self.logger.info(f"Received message type: {type(message).__name__}")

        if isinstance(message, AssistantMessage):
            await self._handle_assistant_message(message)
        elif hasattr(message, 'content') and isinstance(message.content, list):
            # Handle UserMessage or any message with content blocks
            await self._handle_content_blocks(message.content)
        else:
            self.logger.info(f"Unhandled message type: {type(message).__name__}, content: {message}")

    async def _handle_assistant_message(self, message: AssistantMessage) -> None:
        """Handle AssistantMessage - delegates to content block handler."""
        await self._handle_content_blocks(message.content)

    async def _handle_content_blocks(self, content_blocks: list) -> None:
        """
        Handle content blocks from any message type.

        Blocks arrive in order and we need to preserve that order in the UI.
        Text and tool calls should appear exactly as they stream in.
        """
        self.logger.info(f"Processing {len(content_blocks)} content blocks")

        for block in content_blocks:
            self.logger.info(f"  Block type: {type(block).__name__}")

            if isinstance(block, TextBlock):
                # Add text element
                self.message_elements.append(('text', block.text))
                self.logger.info(f"  TextBlock: {repr(block.text[:100])}")

            elif isinstance(block, ToolUseBlock):
                self.logger.info(f"  ToolUseBlock: {block.name} (id={block.id})")
                # Add tool call if not already tracked
                if block.id not in self.tool_calls:
                    self.tool_calls[block.id] = {
                        'tool_id': block.id,
                        'index': len(self.tool_calls),
                        'type': 'function',
                        'function_name': block.name,
                        'function_args': json.dumps(block.input),
                    }
                    # Add tool element marker
                    self.message_elements.append(('tool', block.id))
                    self.logger.info(f"  Added tool call {block.id}")

            elif isinstance(block, ToolResultBlock):
                self.logger.info(f"  ToolResultBlock for tool_use_id={block.tool_use_id}")
                # Tool result - add output to existing tool call
                if block.tool_use_id in self.tool_calls:
                    output_obj = {
                        'tool_call_id': block.tool_use_id,
                        'role': 'tool',
                        'name': self.tool_calls[block.tool_use_id]['function_name'],
                        'content': str(block.content)
                    }
                    self.tool_calls[block.tool_use_id]['output'] = json.dumps(output_obj)
                    self.logger.info(f"  Added output to tool call {block.tool_use_id}")
                else:
                    self.logger.warning(f"  ToolResultBlock for unknown tool_use_id: {block.tool_use_id}")

        await self._update_ui()

    async def _update_ui(self) -> None:
        """
        Update the chat UI with elements in order.

        Render message_elements in the exact order they were received,
        inserting tool call UI where tool elements appear.
        """
        parts = []

        for element_type, element_data in self.message_elements:
            if element_type == 'text':
                parts.append(element_data)
            elif element_type == 'tool':
                # Render this specific tool call
                tool_id = element_data
                if tool_id in self.tool_calls:
                    tool_ui = render_tool_calls([self.tool_calls[tool_id]])
                    parts.append(tool_ui)

        body = "\n\n".join(parts) if parts else ""

        if not self.message_id:
            self.message_id = self.ychat.add_message(NewMessage(
                sender=self.persona_id,
                body=body
            ))
        else:
            self.ychat.update_message(Message(
                id=self.message_id,
                body=body,
                time=time.time(),
                sender=self.persona_id,
                raw_time=False,
            ))
