"""Tool registry & dispatcher.

Each tool entry declares:
    name        — the JSON id the model emits
    description — natural-language description for the system prompt
    schema      — JSON-schema-ish description of args (for prompt + validation)
    handler     — sync or async callable
    permission  — perm key required, or None if free

The dispatcher checks permissions, calls the handler, and returns a string
result. All exceptions are caught and converted to an error string.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from ..permissions import permission_manager
from . import files as files_tools
from . import exec as exec_tools
from . import memory as memory_tools
from . import web as web_tools


Handler = Callable[..., Union[Any, Awaitable[Any]]]


@dataclass
class Tool:
    name: str
    description: str
    schema: Dict[str, Any]
    handler: Handler
    permission: Optional[str]
    # Coarse grouping used to decide which tools a given chat mode exposes:
    #   file / exec / memory → Cowork mode (acting on the local system)
    #   web                  → Search mode (internet)
    # The Code tab and CONFIG.agent_mode expose everything.
    category: str = "file"


REGISTRY: Dict[str, Tool] = {}


def register(tool: Tool) -> None:
    REGISTRY[tool.name] = tool


# ─── file tools ───────────────────────────────────────────────────────
register(Tool(
    name="read_file",
    description="Read a text file. Accepts an absolute path (anywhere on the machine) or a path relative to the open workspace.",
    schema={"path": {"type": "string", "description": "Absolute path, or relative to the workspace"}},
    handler=files_tools.read_file,
    permission="file.read",
))
register(Tool(
    name="write_file",
    description="Write content to a file (creates dirs as needed). Absolute path or relative to the workspace.",
    schema={
        "path": {"type": "string"},
        "content": {"type": "string"},
    },
    handler=files_tools.write_file,
    permission="file.write",
))
register(Tool(
    name="create_folder",
    description="Create a folder (recursive).",
    schema={"path": {"type": "string"}},
    handler=files_tools.create_folder,
    permission="file.write",
))
register(Tool(
    name="delete_file",
    description="Delete a file or folder. DANGEROUS.",
    schema={"path": {"type": "string"}},
    handler=files_tools.delete_file,
    permission="file.delete",
))
register(Tool(
    name="move_file",
    description="Move a file or folder.",
    schema={"src": {"type": "string"}, "dst": {"type": "string"}},
    handler=files_tools.move_file,
    permission="file.write",
))
register(Tool(
    name="rename_file",
    description="Rename a file or folder.",
    schema={"old": {"type": "string"}, "new": {"type": "string"}},
    handler=files_tools.rename_file,
    permission="file.write",
))
register(Tool(
    name="list_dir",
    description="List entries in a directory. Accepts an absolute path or one relative to the open workspace.",
    schema={"path": {"type": "string", "default": "."}},
    handler=files_tools.list_dir,
    permission="file.read",
))
register(Tool(
    name="search_text",
    description="Naive text search (grep) inside the workspace.",
    schema={
        "query": {"type": "string"},
        "path": {"type": "string", "default": "."},
        "max_results": {"type": "integer", "default": 100},
    },
    handler=files_tools.search_text,
    permission="file.read",
))

# ─── exec tools ───────────────────────────────────────────────────────
register(Tool(
    name="run_python",
    description="Execute Python code in a sandboxed subprocess.",
    schema={
        "code": {"type": "string"},
        "timeout": {"type": "number", "default": 20},
    },
    handler=exec_tools.run_python,
    permission="exec.code",
    category="exec",
))
register(Tool(
    name="run_shell",
    description="Execute a shell command (PowerShell on Windows).",
    schema={
        "command": {"type": "string"},
        "timeout": {"type": "number", "default": 20},
    },
    handler=exec_tools.run_shell,
    permission="exec.code",
    category="exec",
))
register(Tool(
    name="run_script",
    description="Run a script file by path (.py, .ps1, .bat, .sh).",
    schema={
        "path": {"type": "string"},
        "timeout": {"type": "number", "default": 30},
    },
    handler=exec_tools.run_script,
    permission="exec.code",
    category="exec",
))
register(Tool(
    name="open_app",
    description="Open/launch an app, file, folder, or URL and leave it running. Use this to start programs (e.g. \"steam\", \"chrome\"), open a document, reveal a folder, or open a web link. Args: target (app name, path, or URL), optional args list.",
    schema={
        "target": {"type": "string", "description": "App name, file/folder path, or URL"},
        "args": {"type": "array", "description": "Optional launch arguments"},
    },
    handler=exec_tools.open_app,
    permission="system.open",
    category="exec",
))

# ─── memory tools ─────────────────────────────────────────────────────
register(Tool(
    name="get_memory",
    description="Read a value from persistent memory.",
    schema={"key": {"type": "string"}},
    handler=memory_tools.get_memory,
    permission="memory",
    category="memory",
))
register(Tool(
    name="set_memory",
    description="Store a value in persistent memory.",
    schema={"key": {"type": "string"}, "value": {}},
    handler=memory_tools.set_memory,
    permission="memory",
    category="memory",
))
register(Tool(
    name="list_memory",
    description="List all memory keys.",
    schema={},
    handler=memory_tools.list_memory,
    permission="memory",
    category="memory",
))
register(Tool(
    name="delete_memory",
    description="Delete a memory key.",
    schema={"key": {"type": "string"}},
    handler=memory_tools.delete_memory,
    permission="memory",
    category="memory",
))

# ─── web tools (Search mode) ──────────────────────────────────────────
register(Tool(
    name="web_search",
    description="Search the web (DuckDuckGo) and return titles, URLs, and snippets. Use this to find current information online.",
    schema={
        "query": {"type": "string"},
        "max_results": {"type": "integer", "default": 5},
    },
    handler=web_tools.web_search,
    permission="network",
    category="web",
))
register(Tool(
    name="web_fetch",
    description="Fetch a web page and return its readable text content. Use after web_search to read a result in full.",
    schema={
        "url": {"type": "string"},
        "max_chars": {"type": "integer", "default": 8000},
    },
    handler=web_tools.web_fetch,
    permission="network",
    category="web",
))


def list_tools() -> Dict[str, Tool]:
    return REGISTRY


# Which tool categories each chat/tool mode exposes.
_MODE_CATEGORIES = {
    "all": None,                              # everything (Code tab)
    "cowork": {"file", "exec", "memory"},     # act on the local system
    "search": {"web"},                        # internet only
}


def tools_for_mode(tool_mode: str | None) -> Dict[str, Tool]:
    """Subset of the registry a given chat mode is allowed to use.

    `None`/'all' → everything. 'cowork' → file/exec/memory. 'search' → web.
    Anything else (incl. 'normal') → no tools.
    """
    if tool_mode in (None, "all"):
        return REGISTRY
    cats = _MODE_CATEGORIES.get(tool_mode)
    if not cats:
        return {}
    return {n: t for n, t in REGISTRY.items() if t.category in cats}


async def dispatch_tool(name: str, args: Dict[str, Any]) -> str:
    """Run the named tool with `args` after a permission check.

    Returns a string (the model sees this verbatim). Errors are returned
    as `Error: ...` rather than raised.
    """
    tool = REGISTRY.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'"

    description = tool.description
    if not await permission_manager.check(name, description, args):
        return f"Error: permission denied for tool '{name}'"

    try:
        if inspect.iscoroutinefunction(tool.handler):
            result = await tool.handler(**args)
        else:
            result = await asyncio.to_thread(tool.handler, **args)
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)
    except TypeError as e:
        return f"Error: invalid args for '{name}': {e}"
    except Exception as e:   # noqa: BLE001 — tool error surfaces to model
        return f"Error: {type(e).__name__}: {e}"
