import asyncio
import json
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
            
            # Extract active context metadata
            active_context = self._extract_active_context_metadata(message)
            
            # Debug logging for active context
            if active_context:
                self.log.info(f"🎯 Active context captured: {active_context.get('activeFile', {}).get('relativePath', 'No active file')}")
                if active_context.get('activeFile', {}).get('isNotebook'):
                    active_cell = active_context.get('activeFile', {}).get('activeCell')
                    if active_cell:
                        self.log.info(f"📓 Active notebook cell: #{active_cell.get('cellIndex', -1) + 1} ({active_cell.get('cellType', 'unknown')})")
            else:
                self.log.info("⚠️ No active context metadata found")
            
            # Build enhanced system prompt for Claude Code
            enhanced_system_prompt = self._build_enhanced_system_prompt(active_context)
            
            # Debug logging for system prompt
            self.log.info(f"🔧 Enhanced system prompt length: {len(enhanced_system_prompt)}")
            if "ACTIVE FILE CONTEXT:" in enhanced_system_prompt:
                context_start = enhanced_system_prompt.find("ACTIVE FILE CONTEXT:")
                context_section = enhanced_system_prompt[context_start:context_start + 300]
                self.log.info(f"📋 Active context section: {context_section}...")
            
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
                allowed_tools=["Read", "Write", "Grep", "Bash", "LS", "Edit", "NotebookRead", "NotebookEdit"],
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
                            # Only yield progress indicator if we haven't had any content for a while
                            if message_count % 20 == 0 and not full_response:  # Much less frequent
                                yield "..."
                    
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

    def _build_enhanced_system_prompt(self, active_context: Optional[dict] = None) -> str:
        """Build an enhanced system prompt for Claude Code with Jupyter AI context."""
        base_prompt = self.system_prompt
        workspace_dir = self.get_workspace_dir()
        
        # Build active file context section
        active_context_section = self._format_active_context(active_context)
        
        enhanced_prompt = f"""{base_prompt}

You are operating within a Jupyter AI environment with the following context:
- Workspace directory: {workspace_dir}
- You have access to enhanced Claude Code capabilities for file operations, code analysis, and multi-step problem solving
- You can read, write, and modify files in the workspace using your Read, Write, Edit, Grep, LS, and Bash tools
- You can run shell commands safely within the workspace context
- You should provide helpful, accurate responses while being mindful of the user's development environment

{active_context_section}

When working with code or files:
1. Always understand the project structure first using your LS and Grep tools
2. Use your Read tool to examine file contents when needed - don't assume file contents
3. When users refer to "this file", "current file", or similar context-dependent terms, they likely refer to the currently active file
4. CONTEXT RELEVANCE: Use judgment to determine when to acknowledge active file/notebook context:
   - ACKNOWLEDGE when relevant: "fix this", "debug this code", "what's wrong", "improve this", "explain this function", vague requests that could relate to current work
   - SKIP when clearly unrelated: "create a new library", "what is Python?", "help with a different project"
   - When in doubt, lean toward mentioning context briefly - it shows awareness and helps users
5. For notebook cells, use NotebookRead to get full context and NotebookEdit to make changes
6. Make targeted, helpful changes using your Edit or Write tools
7. Explain your reasoning and approach
8. Test changes when possible using your Bash tool
9. Be careful with destructive operations

Remember: You are in a collaborative coding environment. Be helpful, thorough, and safe. Use good judgment about when active context is relevant to the user's request."""
        
        return enhanced_prompt

    def _extract_active_context_metadata(self, message: Message) -> Optional[dict]:
        """Extract active context metadata from message attachments."""
        try:
            if not message.attachments:
                return None
            
            for attachment_id in message.attachments:
                attachment_data = self.ychat.get_attachments().get(attachment_id)
                
                if (attachment_data and 
                    isinstance(attachment_data, dict) and 
                    attachment_data.get('type') == 'file'):
                    
                    value = attachment_data.get('value', '')
                    # Check if this is our special active context attachment
                    if value.startswith('__active_context__:'):
                        context_json = value[len('__active_context__:'):]
                        return json.loads(context_json)
            
            return None
            
        except Exception as e:
            self.log.warning(f"Failed to extract active context metadata: {e}")
            return None

    def _format_active_context(self, active_context: Optional[dict]) -> str:
        """Format active context information for the system prompt."""
        if not active_context:
            return "ACTIVE FILE CONTEXT: No active file context available."
        
        try:
            context_parts = ["ACTIVE FILE CONTEXT:"]
            
            # Format currently active file
            active_file = active_context.get('activeFile')
            if active_file:
                cursor_info = ""
                if active_file.get('cursorPosition'):
                    cursor = active_file['cursorPosition']
                    cursor_info = f" (cursor at line {cursor['line']}, column {cursor['column']})"
                
                size_info = ""
                if active_file.get('size'):
                    size_kb = active_file['size'] / 1024
                    size_info = f" ({size_kb:.1f}KB)"
                
                language_info = ""
                if active_file.get('language'):
                    language_info = f" [{active_file['language']}]"
                
                # Check if this is a notebook
                if active_file.get('isNotebook'):
                    notebook_info = self._format_notebook_context(active_file)
                    context_parts.append(
                        f"• Currently active notebook: {active_file['relativePath']}{language_info}{size_info}"
                    )
                    context_parts.extend(notebook_info)
                else:
                    context_parts.append(
                        f"• Currently active file: {active_file['relativePath']}{language_info}{size_info}{cursor_info}"
                    )
            else:
                context_parts.append("• No file is currently active in the editor")
            
            # Format open tabs
            open_tabs = active_context.get('openTabs', [])
            if open_tabs:
                non_active_tabs = [tab for tab in open_tabs if not tab.get('isActive', False)]
                if non_active_tabs:
                    context_parts.append(f"• Other open tabs ({len(non_active_tabs)}):")
                    for tab in non_active_tabs[:5]:  # Limit to first 5 to avoid clutter
                        language_info = f" [{tab['language']}]" if tab.get('language') else ""
                        size_info = f" ({tab['size']/1024:.1f}KB)" if tab.get('size') else ""
                        notebook_info = " [notebook]" if tab.get('isNotebook') else ""
                        context_parts.append(f"  - {tab['relativePath']}{language_info}{size_info}{notebook_info}")
                    
                    if len(non_active_tabs) > 5:
                        context_parts.append(f"  ... and {len(non_active_tabs) - 5} more files")
            
            # Add workspace context
            workspace_root = active_context.get('workspaceRoot', '')
            if workspace_root:
                context_parts.append(f"• Workspace root: {workspace_root}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            self.log.warning(f"Error formatting active context: {e}")
            return "ACTIVE FILE CONTEXT: Error formatting context information."

    def _format_notebook_context(self, active_file: dict) -> list:
        """Format notebook-specific context information."""
        context_parts = []
        
        try:
            total_cells = active_file.get('totalCells', 0)
            context_parts.append(f"  - Total cells: {total_cells}")
            
            active_cell = active_file.get('activeCell')
            if active_cell:
                cell_type = active_cell.get('cellType', 'unknown')
                cell_index = active_cell.get('cellIndex', -1)
                
                cell_info = f"  - Active cell: #{cell_index + 1} ({cell_type})"
                
                # Add execution info for code cells
                if cell_type == 'code':
                    execution_count = active_cell.get('executionCount')
                    has_output = active_cell.get('hasOutput', False)
                    
                    exec_info = []
                    if execution_count is not None:
                        exec_info.append(f"exec count: {execution_count}")
                    if has_output:
                        exec_info.append("has output")
                    
                    if exec_info:
                        cell_info += f" [{', '.join(exec_info)}]"
                
                context_parts.append(cell_info)
                
                # Add source preview if available
                source = active_cell.get('source')
                if source:
                    source_preview = source[:100].replace('\n', ' ').strip()
                    if len(source) > 100:
                        source_preview += "..."
                    context_parts.append(f"  - Cell content: {source_preview}")
            
            return context_parts
            
        except Exception as e:
            self.log.warning(f"Error formatting notebook context: {e}")
            return ["  - Error formatting notebook context"]

    def _parse_claude_code_message(self, msg) -> str:
        """Parse Claude Code SDK message objects to extract displayable content."""
        try:
            msg_str = str(msg)
            
            # First, aggressively filter out tool use and metadata patterns
            skip_patterns = [
                'ToolUseBlock(',
                'tool_use_id',
                'tool_result',
                'toolu_',
                "{'tool_use_id':",
                "'type': 'tool_result'",
                'SystemMessage',
                'session_id',
                'duration_ms',
                'total_cost_usd',
                'usage',
                'cache_',
                'api_ms',
                'num_turns',
                'service_tier',
                'model_id',
                'mcp_servers',
                'tools',
                'permissionMode',
                'name=\'',
                'input={',
                'NotebookRead',
                'NotebookEdit',
                'Write',
                'Read',
                'Edit',
                'Bash',
                'LS',
                'Grep',
                "'content': [",
                "'type': 'text'"
            ]
            
            # If the message contains ANY tool use or metadata patterns, skip it entirely
            if any(pattern in msg_str for pattern in skip_patterns):
                return ""
            
            # Skip very long messages that are likely metadata dumps
            if len(msg_str) > 500:
                return ""
            
            # Skip messages that look like JSON or dictionary structures
            if msg_str.strip().startswith(('{', '[')):
                return ""
            
            # Handle different message types from Claude Code SDK
            
            # Handle string messages directly (but only if they don't contain tool patterns)
            if isinstance(msg, str):
                return msg
            
            # Handle messages with a result attribute (like ResultMessage)
            if hasattr(msg, 'result') and msg.result:
                result_str = str(msg.result)
                # Double-check the result doesn't contain tool patterns
                if not any(pattern in result_str for pattern in skip_patterns):
                    return result_str
                return ""
            
            # Handle messages with content list (like messages containing TextBlocks)
            if hasattr(msg, 'content') and isinstance(msg.content, list):
                content_parts = []
                for block in msg.content:
                    if hasattr(block, 'text'):
                        text = str(block.text)
                        # Only include text that doesn't contain tool patterns
                        if not any(pattern in text for pattern in skip_patterns):
                            content_parts.append(text)
                    elif hasattr(block, 'content'):
                        content = str(block.content)
                        if not any(pattern in content for pattern in skip_patterns):
                            content_parts.append(content)
                return ''.join(content_parts)
            
            # Handle messages with direct content attribute
            if hasattr(msg, 'content') and msg.content:
                content_str = str(msg.content)
                if not any(pattern in content_str for pattern in skip_patterns):
                    return content_str
                return ""
            
            # Handle messages with text attribute
            if hasattr(msg, 'text') and msg.text:
                text_str = str(msg.text)
                if not any(pattern in text_str for pattern in skip_patterns):
                    return text_str
                return ""
                
            # Skip all system messages and metadata
            if hasattr(msg, 'data') and isinstance(msg.data, dict):
                return ""
            
            # If we get here, it's likely not user-facing content
            return ""
            
        except Exception as e:
            self.log.warning(f"Error parsing Claude Code message: {e}")
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