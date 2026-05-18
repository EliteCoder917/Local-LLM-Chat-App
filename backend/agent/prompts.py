"""Agent-mode system prompt builder.

Only called when agent mode is ON. For pure-chat (agent off) the loop sends
the user's Settings prompt verbatim — no framework involvement.

Appends to the user's prompt:
* Compact tool catalog — one-line signatures, no JSON schemas (~50% smaller
  than the verbose form).
* A short rules block telling the model how to emit `tool` JSON blocks.
"""
from __future__ import annotations

from typing import Dict

from ..tools.registry import Tool


def render_tool_signatures(tools: Dict[str, Tool]) -> str:
    lines = []
    for t in tools.values():
        args = list((t.schema or {}).keys())
        sig = f"{t.name}({', '.join(args)})"
        gated = " [gated]" if t.permission else ""
        lines.append(f"- `{sig}`{gated} — {t.description}")
    return "\n".join(lines)


def build_system_prompt(base_prompt: str, tools: Dict[str, Tool],
                        workspace: str, agent_mode: bool) -> str:
    # Caller (loop.py) guarantees agent_mode=True when this branch is taken.
    # Param kept for backward compatibility with any other call sites.
    if not agent_mode:
        return base_prompt
    catalog = render_tool_signatures(tools)
    workspace_line = (
        f"Workspace: {workspace}" if workspace
        else "No workspace open — file tools will fail until one is opened."
    )
    rules = (
        "## Tool use\n"
        "You have access to the tools listed below, but **default to answering "
        "without tools**. Tools are for taking concrete actions on the user's "
        "files, code, or system — not for answering questions, explaining "
        "concepts, generating code in chat, planning, or brainstorming.\n\n"
        "Use a tool ONLY when the user has explicitly asked you to do something "
        "that requires one (e.g. \"read X\", \"create file Y\", \"run this "
        "script\", \"refactor this file\"). If the user is just asking a "
        "question, having a discussion, or asking you to write code in chat, "
        "answer directly — do not call any tool.\n\n"
        "When a tool IS appropriate, emit ONE fenced JSON block tagged `tool` "
        "per turn:\n"
        "```tool\n"
        '{\"tool\": \"<name>\", \"args\": { ... }}\n'
        "```\n"
        "Wait for the `tool_result` reply before chaining another call. Stop "
        "calling tools the moment the requested task is complete."
    )
    return (
        f"{base_prompt}\n\n"
        f"{workspace_line}\n\n"
        f"## Tools available\n{catalog}\n\n"
        f"{rules}"
    )

