"""Extract ```tool ...``` JSON blocks from assistant output."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

_FENCE = re.compile(
    r"```(?:tool)\s*\n(?P<body>.*?)```",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class ParsedToolCall:
    tool: str
    args: dict
    raw: str


def find_tool_calls(text: str) -> List[ParsedToolCall]:
    out: List[ParsedToolCall] = []
    for m in _FENCE.finditer(text):
        body = m.group("body").strip()
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        tool = obj.get("tool")
        args = obj.get("args", {})
        if isinstance(tool, str) and isinstance(args, dict):
            out.append(ParsedToolCall(tool=tool, args=args, raw=m.group(0)))
    return out


def first_tool_call(text: str) -> Optional[ParsedToolCall]:
    calls = find_tool_calls(text)
    return calls[0] if calls else None


def format_tool_result(call_id: str, result: str) -> str:
    return f"```tool_result {{\"id\": \"{call_id}\"}}\n{result}\n```"
