from jinja2 import Template
from pydantic import BaseModel
from typing import Optional


class ClaudeSystemPromptArgs(BaseModel):
    """Arguments for Claude Agent persona system prompt"""

    persona_name: str
    context: Optional[str] = None


CLAUDE_SYSTEM_PROMPT_TEMPLATE = Template(
    """
You are {{ persona_name }}, a helpful AI assistant powered by the Claude Agent SDK.
You have access to various tools to help you interact with the Jupyter environment and filesystem.

Always be helpful, accurate, and concise in your responses.

{% if context %}
## Context

The user has provided the following context:

{{ context }}
{% endif %}
""".strip()
)
