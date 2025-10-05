"""
Tool adapter to translate Claude Agent SDK tool use blocks to Jupyter AI UI format.
"""
from typing import Any, Dict, Optional
import json
from jinja2 import Template


# Template for rendering tool calls as UI elements
JAI_TOOL_CALL_TEMPLATE = Template("""
{% for props in props_list %}
<jai-tool-call {{props | xmlattr}}>
</jai-tool-call>
{% endfor %}
""".strip())


def translate_tool_use_to_ui(tool_use_block: Any, index: int, output: Optional[str] = None) -> Dict[str, Any]:
    """
    Translate a Claude Agent SDK ToolUseBlock to Jupyter AI tool call UI props.

    Args:
        tool_use_block: The ToolUseBlock from Agent SDK
        index: The index of this tool call
        output: Optional output/result from tool execution

    Returns:
        Dictionary of props for the <jai-tool-call> element
    """
    props = {
        'tool_id': tool_use_block.id,
        'index': index,
        'type': 'tool_use',
        'function_name': tool_use_block.name,
        'function_args': json.dumps(tool_use_block.input),
    }

    if output is not None:
        # Create output in the format expected by the UI
        output_dict = {
            'role': 'tool',
            'tool_call_id': tool_use_block.id,
            'content': output
        }
        props['output'] = json.dumps(output_dict)

    return props


def render_tool_calls(tool_call_props_list: list[Dict[str, Any]]) -> str:
    """
    Render a list of tool call props as <jai-tool-call> XML elements.

    Args:
        tool_call_props_list: List of tool call property dictionaries

    Returns:
        Rendered XML string with <jai-tool-call> elements
    """
    return JAI_TOOL_CALL_TEMPLATE.render({
        "props_list": tool_call_props_list
    })
