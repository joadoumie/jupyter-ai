"""
Streaming processor for Claude Agent SDK responses.
"""
import time
import json
from typing import Optional, List, Dict, Any
from logging import Logger
from jupyterlab_chat.models import Message, NewMessage
from jupyterlab_chat.ychat import YChat
from claude_agent_sdk import query, ClaudeAgentOptions, HookMatcher
from claude_agent_sdk.types import AssistantMessage, TextBlock, ToolUseBlock, ToolResultBlock, UserMessage, SystemMessage, ResultMessage

from ...personas.persona_awareness import PersonaAwareness
from .tool_adapter import render_tool_calls


class StreamingProcessor:
    """
    Processes streaming responses from Claude Agent SDK and updates YChat.
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

        # State tracking
        self.message_id: Optional[str] = None
        self.message_parts: List[str] = []  # Ordered list of text and tool call UI elements
        self.tool_calls: Dict[str, Dict[str, Any]] = {}  # Maps tool_use_id -> tool call data
        self.tool_call_positions: Dict[str, int] = {}  # Maps tool_use_id -> position in message_parts

    def _create_pre_tool_hook(self):
        """Create a pre-tool hook that captures the instance context."""
        async def pre_tool_hook(input_data: dict, tool_use_id: str, context: dict) -> dict:
            """Hook called before a tool is executed."""
            self.logger.info(f"PreToolUse: {input_data.get('tool_name')} with id {tool_use_id}")

            # Store tool call information
            self.tool_calls[tool_use_id] = {
                'tool_id': tool_use_id,
                'index': len(self.tool_calls),
                'type': 'function',  # Must be 'function' per JaiToolCallProps
                'function_name': input_data.get('tool_name', 'unknown'),
                'function_args': json.dumps(input_data.get('tool_input', {})),
            }

            # Update UI to show tool is being called
            await self._update_message()
            return {}

        return pre_tool_hook

    def _create_post_tool_hook(self):
        """Create a post-tool hook that captures the instance context."""
        async def post_tool_hook(input_data: dict, tool_use_id: str, context: dict) -> dict:
            """Hook called after a tool is executed."""
            self.logger.info(f"PostToolUse: {input_data.get('tool_name')} with id {tool_use_id}")
            self.logger.info(f"PostToolUse input_data keys: {input_data.keys()}")

            # Add output to the tool call
            if tool_use_id in self.tool_calls:
                # Get tool response - it might be 'tool_response' or 'tool_output'
                tool_response = input_data.get('tool_response') or input_data.get('tool_output', '')

                # Output should be a JSON string containing the output object
                output_obj = {
                    'tool_call_id': tool_use_id,
                    'role': 'tool',
                    'name': input_data.get('tool_name', 'unknown'),
                    'content': str(tool_response)
                }
                self.tool_calls[tool_use_id]['output'] = json.dumps(output_obj)

                # Update UI to show tool result
                await self._update_message()
            return {}

        return post_tool_hook

    async def process_with_agent_sdk(
        self,
        prompt: str,
        system_prompt: str,
        model: str = "claude-sonnet-4-5-20250929",
        mcp_config: Optional[dict] = None,
    ) -> None:
        """
        Process a user prompt with Claude Agent SDK and stream the response.

        Args:
            prompt: User's input message
            system_prompt: System prompt for the agent
            model: Model to use (defaults to Sonnet 4.5)
            mcp_config: MCP server configuration from Jupyter AI
        """
        try:
            self.awareness.set_local_state_field("isWriting", True)

            self.logger.info(f"=== Starting Claude Agent SDK query ===")
            self.logger.info(f"Prompt: {prompt}")
            self.logger.info(f"MCP config from Jupyter AI: {mcp_config}")

            # Create hook functions
            pre_tool_hook = self._create_pre_tool_hook()
            post_tool_hook = self._create_post_tool_hook()

            self.logger.info(f"Created hook functions")

            # Use MCP config from Jupyter AI if available, otherwise create a simple demo server
            if mcp_config:
                mcp_servers = mcp_config
                self.logger.info(f"Using MCP servers from config: {list(mcp_servers.keys())}")
            else:
                # Demo: Simple MCP server using npx to run @modelcontextprotocol/server-everything
                mcp_servers = {
                    "demo": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-everything"]
                    }
                }
                self.logger.info(f"No MCP config found, using demo server")

            options = ClaudeAgentOptions(
                system_prompt=system_prompt,
                model=model,
                mcp_servers=mcp_servers,
                permission_mode='bypassPermissions',  # Bypass permission prompts for MCP tools
                include_partial_messages=True,  # Stream partial messages for smoother text rendering
                hooks={
                    'PreToolUse': [HookMatcher(hooks=[pre_tool_hook])],
                    'PostToolUse': [HookMatcher(hooks=[post_tool_hook])],
                }
            )

            self.logger.info(f"Created ClaudeAgentOptions with hooks and MCP servers")

            # Stream messages from Claude Agent SDK
            self.logger.info(f"Starting message stream...")
            message_count = 0
            async for message in query(
                prompt=prompt,
                options=options,
            ):
                message_count += 1
                self.logger.info(f"Received message #{message_count}: {type(message).__name__}")
                await self._handle_message(message)

            self.logger.info(f"=== Finished Claude Agent SDK query ===")

        except Exception as e:
            self.logger.exception(f"Error processing with Claude Agent SDK: {e}")
            # Send error message to chat
            self.ychat.add_message(NewMessage(
                sender=self.persona_id,
                body=f"An error occurred: {str(e)}"
            ))
        finally:
            self.awareness.set_local_state_field("isWriting", False)

    async def _handle_message(self, message: Any) -> None:
        """Handle a single message from the Agent SDK stream."""
        self.logger.info(f"Handling message: {type(message).__name__}")

        if isinstance(message, AssistantMessage):
            await self._handle_assistant_message(message)
        elif isinstance(message, ResultMessage):
            await self._handle_result_message(message)
        elif isinstance(message, UserMessage):
            await self._handle_user_message(message)
        elif isinstance(message, SystemMessage):
            self.logger.info(f"SystemMessage: {message}")

    async def _handle_assistant_message(self, message: AssistantMessage) -> None:
        """Handle an AssistantMessage from the Agent SDK."""
        self.logger.info(f"AssistantMessage content blocks: {len(message.content)}")

        # Process content blocks - with include_partial_messages=True,
        # each AssistantMessage contains incremental updates
        for i, block in enumerate(message.content):
            self.logger.info(f"Block {i}: {type(block).__name__}")

            if isinstance(block, TextBlock):
                # Append text incrementally as it streams in
                self.message_parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                self.logger.info(f"ToolUseBlock found: {block.name} with id {block.id}, input: {block.input}")

                # Store tool call info if not already stored
                if block.id not in self.tool_calls:
                    self.tool_calls[block.id] = {
                        'tool_id': block.id,
                        'index': len(self.tool_calls),
                        'type': 'function',
                        'function_name': block.name,
                        'function_args': json.dumps(block.input),
                    }

                    # Render tool call UI and add to message parts
                    tool_call_ui = render_tool_calls([self.tool_calls[block.id]])
                    position = len(self.message_parts)
                    self.message_parts.append(tool_call_ui)
                    self.tool_call_positions[block.id] = position

        # Update or create message in YChat
        await self._update_message()

    async def _handle_user_message(self, message: UserMessage) -> None:
        """Handle a UserMessage from the Agent SDK - contains tool results."""
        self.logger.info(f"UserMessage content blocks: {len(message.content)}")

        # Process content blocks - looking for ToolResultBlock
        for i, block in enumerate(message.content):
            self.logger.info(f"UserMessage Block {i}: {type(block).__name__}")

            if isinstance(block, ToolResultBlock):
                self.logger.info(f"ToolResultBlock found for tool_use_id: {block.tool_use_id}, content length: {len(str(block.content))}")
                # Add output to existing tool call
                if block.tool_use_id in self.tool_calls:
                    output_obj = {
                        'tool_call_id': block.tool_use_id,
                        'role': 'tool',
                        'name': self.tool_calls[block.tool_use_id]['function_name'],
                        'content': str(block.content)
                    }
                    self.tool_calls[block.tool_use_id]['output'] = json.dumps(output_obj)
                    self.logger.info(f"Set output for tool {block.tool_use_id} in UserMessage")

                    # Re-render this specific tool call with output and update its position
                    if block.tool_use_id in self.tool_call_positions:
                        position = self.tool_call_positions[block.tool_use_id]
                        updated_tool_ui = render_tool_calls([self.tool_calls[block.tool_use_id]])
                        self.message_parts[position] = updated_tool_ui

                    # Update UI with tool result
                    await self._update_message()
                else:
                    self.logger.warning(f"Received ToolResultBlock in UserMessage for unknown tool_use_id: {block.tool_use_id}")

    async def _handle_result_message(self, message: ResultMessage) -> None:
        """Handle a ResultMessage from the Agent SDK."""
        self.logger.info(f"ResultMessage: {message}")

    async def _update_message(self) -> None:
        """Update the message in YChat with current content and tool calls."""
        # Join all message parts in order (text and tool calls inline)
        message_body = "".join(self.message_parts).strip()

        # Debug logging
        self.logger.info(f"Updating message - parts count: {len(self.message_parts)}, tool_calls: {len(self.tool_calls)}")

        if not self.message_id:
            # Create new message
            self.message_id = self.ychat.add_message(NewMessage(
                sender=self.persona_id,
                body=message_body
            ))
        else:
            # Update existing message
            self.ychat.update_message(
                Message(
                    id=self.message_id,
                    body=message_body,
                    time=time.time(),
                    sender=self.persona_id,
                    raw_time=False,
                )
            )
