import asyncio
import os
from pathlib import Path
from typing import Any, Optional

from jupyterlab_chat.models import Message
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from ...history import YChatHistory
from ..base_persona import BasePersona, PersonaDefaults
from .prompt_template import CLAUDE_PROMPT_TEMPLATE, ClaudeVariables


class ClaudePersona(BasePersona):
    """
    The Claude persona that integrates with Claude Code for multi-step conversations
    and advanced code assistance capabilities.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._claude_code_available = None
        self._claude_cli_command = 'claude'  # Default CLI command
        self._conversation_state = {}

    @property
    def defaults(self):
        return PersonaDefaults(
            name="Claude",
            avatar_path="/api/ai/static/claude.svg",
            description="Claude Code integration for advanced multi-step conversations and code assistance.",
            system_prompt="You are Claude, an AI assistant created by Anthropic. You have access to Claude Code capabilities for advanced code assistance, file operations, and multi-step problem solving.",
        )

    async def process_message(self, message: Message) -> None:
        """Process a message using Claude Code SDK for enhanced capabilities."""
        if not self.config_manager.lm_provider:
            self.send_message(
                "No language model provider configured. Please set one in the Jupyter AI settings."
            )
            return

        try:
            # Check if claude-code-sdk is available
            if await self._is_claude_code_available():
                # Use Claude Code SDK for enhanced processing
                await self._process_with_claude_code(message)
            else:
                # Fallback to standard LLM processing
                await self._process_with_standard_llm(message)
                
        except Exception as e:
            self.log.error(f"Error processing message in Claude persona: {e}")
            self.send_message(
                f"I encountered an error while processing your message: {str(e)}"
            )

    async def _is_claude_code_available(self) -> bool:
        """Check if claude-code-sdk is available and can be imported."""
        if self._claude_code_available is not None:
            return self._claude_code_available
            
        try:
            # Try to import claude-code-sdk
            import claude_code_sdk
            self.log.info("✅ Claude Code SDK Python package is available")
            
            # Also check if the CLI is available
            import subprocess
            import shutil
            
            # Log environment info for debugging
            self.log.info(f"🔍 Node.js version check...")
            try:
                node_result = subprocess.run(['node', '--version'], 
                                           capture_output=True, text=True, timeout=5)
                if node_result.returncode == 0:
                    self.log.info(f"✅ Node.js available: {node_result.stdout.strip()}")
                else:
                    self.log.warning("❌ Node.js not available")
            except Exception as e:
                self.log.warning(f"❌ Node.js check failed: {e}")
            
            # Check if claude CLI is in PATH (the actual executable name)
            claude_cli_path = shutil.which('claude')
            claude_code_cli_path = shutil.which('claude-code')
            
            if claude_cli_path:
                self.log.info(f"✅ Claude CLI found at: {claude_cli_path}")
                self._claude_cli_command = 'claude'
            elif claude_code_cli_path:
                self.log.info(f"✅ Claude Code CLI found at: {claude_code_cli_path}")
                self._claude_cli_command = 'claude-code'
            else:
                self.log.warning("❌ Neither 'claude' nor 'claude-code' CLI found in PATH")
                self.log.info(f"Current PATH: {os.environ.get('PATH', 'Not set')[:500]}...")
                
                # Try some common npm global locations for both commands
                potential_paths = [
                    os.path.expanduser("~/.nvm/versions/node/*/bin/claude"),
                    os.path.expanduser("~/.nvm/versions/node/*/bin/claude-code"),
                    "/usr/local/bin/claude",
                    "/usr/local/bin/claude-code",
                    os.path.expanduser("~/node_modules/.bin/claude"),
                    os.path.expanduser("~/node_modules/.bin/claude-code"),
                    os.path.expanduser("~/.local/bin/claude"),
                    os.path.expanduser("~/.local/bin/claude-code")
                ]
                
                import glob
                for path_pattern in potential_paths:
                    matches = glob.glob(path_pattern)
                    if matches:
                        self.log.info(f"🔍 Found potential Claude CLI at: {matches}")
                        # Try to use the first match we find
                        if 'claude-code' in path_pattern:
                            self._claude_cli_command = matches[0]
                        else:
                            self._claude_cli_command = matches[0]
                        break
                else:
                    self._claude_cli_command = 'claude'  # Default fallback
                
            # Try to run the CLI to verify it works
            cli_command = getattr(self, '_claude_cli_command', 'claude')
            try:
                result = subprocess.run([cli_command, '--version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    self.log.info(f"✅ Claude CLI is working: {result.stdout.strip()}")
                    self._claude_code_available = True
                    return True
                else:
                    self.log.warning(f"❌ Claude CLI failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                self.log.warning("❌ Claude CLI command timed out")
            except FileNotFoundError:
                self.log.warning(f"❌ Claude CLI '{cli_command}' not found when trying to execute")
            except Exception as e:
                self.log.warning(f"❌ Error testing Claude CLI: {e}")
            
            # Fall back to Python SDK only mode
            self.log.info("Falling back to standard LLM processing (Claude Code CLI not available)")
            self._claude_code_available = False
            return False
            
        except ImportError:
            self._claude_code_available = False
            self.log.info(
                "claude-code-sdk not available. Install it with: pip install claude-code-sdk"
            )
            return False

    async def _process_with_claude_code(self, message: Message) -> None:
        """Process message using Claude Code SDK for enhanced capabilities."""
        try:
            from claude_code_sdk import query, ClaudeCodeOptions
            
            # Process file attachments
            context = self.process_attachments(message)
            
            # Build enhanced system prompt for Claude Code
            enhanced_system_prompt = self._build_enhanced_system_prompt()
            
            # Build the prompt with context if available
            full_prompt = message.body
            if context:
                full_prompt = f"Context from attached files:\n{context}\n\nUser request:\n{message.body}"
                
            # Add conversation history context if available
            chat_path = self.get_chat_path()
            if chat_path in self._conversation_state and self._conversation_state[chat_path]:
                recent_context = self._get_recent_conversation_context()
                if recent_context:
                    full_prompt = f"Recent conversation context:\n{recent_context}\n\n{full_prompt}"
            
            # Configure Claude Code options with enhanced settings
            workspace_dir = Path(self.get_workspace_dir())
            options = ClaudeCodeOptions(
                max_turns=5,  # Allow more turns for complex tasks
                cwd=workspace_dir,
                system_prompt=enhanced_system_prompt,
                # Enable tools that are safe and useful for Jupyter AI environment
                allowed_tools=["Read", "Write", "Grep", "Bash", "LS", "Edit"],
                permission_mode="acceptEdits"  # Allow Claude Code to make edits
            )
            
            # Create an async generator that yields message content for streaming
            async def claude_code_stream():
                messages = []
                full_response = ""
                message_count = 0
                
                try:
                    self.log.info("🚀 Starting Claude Code streaming...")
                    async for msg in query(prompt=full_prompt, options=options):
                        message_count += 1
                        self.log.debug(f"📨 Received message #{message_count}: {type(msg).__name__}")
                        messages.append(msg)
                        
                        # Extract content from Claude Code SDK message objects
                        content = self._parse_claude_code_message(msg)
                        
                        if content:
                            self.log.debug(f"📝 Yielding content (length: {len(content)}): {content[:100]}...")
                            full_response += content
                            yield content
                        else:
                            self.log.debug(f"⚪ No content extracted from message #{message_count}")
                            # For debugging: let's see what this message actually contains
                            self.log.debug(f"🔍 Message details: {str(msg)[:200]}...")
                            # Yield a progress indicator to keep the stream alive
                            if message_count % 5 == 0:  # Every 5th empty message
                                yield "."
                    
                    self.log.info(f"✅ Claude Code streaming completed. Total messages: {message_count}")
                
                except Exception as e:
                    self.log.error(f"Error during Claude Code streaming: {e}")
                    # Yield error message if something goes wrong during streaming
                    error_msg = f"\n\n[Error occurred during Claude Code processing: {str(e)}]"
                    yield error_msg
                    full_response += error_msg
                
                finally:
                    # Store conversation state for future reference
                    if messages:
                        self._update_conversation_state(messages, full_response)
            
            # Stream the response
            await self.stream_message(claude_code_stream())
            
        except Exception as e:
            self.log.error(f"Error in Claude Code processing: {e}")
            # Fallback to standard processing
            await self._process_with_standard_llm(message)

    async def _process_with_standard_llm(self, message: Message) -> None:
        """Fallback processing using standard LLM without Claude Code SDK."""
        provider_name = self.config_manager.lm_provider.name
        model_id = self.config_manager.lm_provider_params["model_id"]

        # Process file attachments and include their content in the context
        context = self.process_attachments(message)

        runnable = self.build_runnable()
        variables = ClaudeVariables(
            input=message.body,
            model_id=model_id,
            provider_name=provider_name,
            persona_name=self.name,
            context=context,
        )
        variables_dict = variables.model_dump()
        reply_stream = runnable.astream(variables_dict)
        await self.stream_message(reply_stream)

    def build_runnable(self) -> Any:
        """Build the runnable chain for standard LLM processing."""
        llm = self.config_manager.lm_provider(**self.config_manager.lm_provider_params)
        runnable = CLAUDE_PROMPT_TEMPLATE | llm | StrOutputParser()

        runnable = RunnableWithMessageHistory(
            runnable=runnable,
            get_session_history=lambda: YChatHistory(ychat=self.ychat, k=2),
            input_messages_key="input",
            history_messages_key="history",
        )

        return runnable

    def _build_enhanced_system_prompt(self) -> str:
        """Build an enhanced system prompt for Claude Code with Jupyter AI context."""
        base_prompt = self.system_prompt
        workspace_dir = self.get_workspace_dir()
        
        enhanced_prompt = f"""{base_prompt}

You are operating within a Jupyter AI environment with the following context:
- Workspace directory: {workspace_dir}
- You have access to enhanced Claude Code capabilities for file operations, code analysis, and multi-step problem solving
- You can read, write, and modify files in the workspace
- You can run shell commands safely within the workspace context
- You should provide helpful, accurate responses while being mindful of the user's development environment

When working with code or files:
1. Always understand the project structure first
2. Make targeted, helpful changes
3. Explain your reasoning and approach
4. Test changes when possible
5. Be careful with destructive operations

Remember: You are in a collaborative coding environment. Be helpful, thorough, and safe."""
        
        return enhanced_prompt

    def _parse_claude_code_message(self, msg) -> str:
        """Parse Claude Code SDK message objects to extract displayable content."""
        try:
            # Log the message type and structure for debugging
            msg_type = type(msg).__name__
            self.log.debug(f"🔍 Parsing message type: {msg_type}")
            
            # Handle different message types from Claude Code SDK
            
            # Handle string messages directly
            if isinstance(msg, str):
                self.log.debug("✅ Handled as string message")
                return msg
            
            # Handle messages with a result attribute (like ResultMessage)
            if hasattr(msg, 'result') and msg.result:
                self.log.debug("✅ Handled as ResultMessage")
                return str(msg.result)
            
            # Handle messages with content list (like messages containing TextBlocks)
            if hasattr(msg, 'content') and isinstance(msg.content, list):
                self.log.debug(f"✅ Handled as content list with {len(msg.content)} blocks")
                content_parts = []
                for block in msg.content:
                    if hasattr(block, 'text'):
                        content_parts.append(str(block.text))
                    elif hasattr(block, 'content'):
                        content_parts.append(str(block.content))
                    else:
                        content_parts.append(str(block))
                return ''.join(content_parts)
            
            # Handle messages with direct content attribute
            if hasattr(msg, 'content') and msg.content:
                self.log.debug("✅ Handled as direct content")
                return str(msg.content)
            
            # Handle messages with text attribute
            if hasattr(msg, 'text') and msg.text:
                self.log.debug("✅ Handled as text attribute")
                return str(msg.text)
                
            # Handle SystemMessage and other structured messages - be more permissive
            if hasattr(msg, 'data') and isinstance(msg.data, dict):
                # Only skip pure system init messages, allow others through
                if msg.data.get('type') == 'system' and msg.data.get('subtype') == 'init':
                    self.log.debug("⚪ Skipped system init message")
                    return ""
                    
            # Handle messages that might have nested structure
            if hasattr(msg, '__dict__'):
                # Look for common content fields
                for field in ['message', 'body', 'value', 'output']:
                    if hasattr(msg, field):
                        content = getattr(msg, field)
                        if content and isinstance(content, (str, int, float)):
                            self.log.debug(f"✅ Handled as nested field: {field}")
                            return str(content)
            
            # More permissive fallback: convert to string but only filter truly internal messages
            msg_str = str(msg)
            
            # Only skip very specific internal system messages
            if ('SystemMessage' in msg_str and 'subtype=\'init\'' in msg_str and 'session_id' in msg_str):
                self.log.debug("⚪ Skipped specific system init message")
                return ""
            
            # Allow all other messages through, even if they're long
            self.log.debug("✅ Handled as fallback string conversion")
            return msg_str
            
        except Exception as e:
            self.log.warning(f"❌ Error parsing Claude Code message: {e}")
            # Fallback to basic string conversion - be permissive in error cases
            try:
                return str(msg)
            except:
                return ""

    def _get_recent_conversation_context(self) -> Optional[str]:
        """Get recent conversation context for better continuity."""
        chat_path = self.get_chat_path()
        if chat_path not in self._conversation_state or not self._conversation_state[chat_path]:
            return None
        
        # Get the last few conversation entries
        recent_messages = self._conversation_state[chat_path][-3:]  # Last 3 exchanges
        if not recent_messages:
            return None
        
        context_parts = []
        for i, entry in enumerate(recent_messages):
            if isinstance(entry, dict) and 'response' in entry:
                context_parts.append(f"Previous exchange {i+1}: {entry['response'][:200]}...")
            elif isinstance(entry, str):
                context_parts.append(f"Previous message {i+1}: {entry[:200]}...")
        
        return "\n".join(context_parts) if context_parts else None

    def _update_conversation_state(self, messages, full_response: str = "") -> None:
        """Update conversation state with new messages for context."""
        chat_path = self.get_chat_path()
        if chat_path not in self._conversation_state:
            self._conversation_state[chat_path] = []
        
        # Store a summary of this conversation turn
        conversation_entry = {
            'messages': messages,
            'response': full_response,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        self._conversation_state[chat_path].append(conversation_entry)
        
        # Keep only the last few conversation turns to avoid memory issues
        if len(self._conversation_state[chat_path]) > 5:
            self._conversation_state[chat_path] = self._conversation_state[chat_path][-5:]

    def shutdown(self) -> None:
        """Shutdown the Claude persona and clean up resources."""        
        # Clear conversation state
        self._conversation_state.clear()
        
        # Reset availability check
        self._claude_code_available = None
        
        # Call parent shutdown
        super().shutdown()