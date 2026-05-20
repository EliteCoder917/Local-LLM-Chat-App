"""The plan→act→observe agent loop.

Each iteration:
    1. Stream tokens from the LLM, forwarding them as `message-delta` events.
    2. Once the assistant turn finishes, scan its content for ```tool blocks.
    3. If found:
         - emit `tool-call` event,
         - dispatch through the registry (with permission check),
         - emit `tool-result` event,
         - feed the result back as a new user message,
         - start a new assistant message.
    4. If no tool call OR max_iterations hit → finish.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Awaitable, Callable, Dict, List

from ..config import CONFIG
from ..llm import ChatMessage, loader
from ..tools.registry import dispatch_tool, list_tools
from .parser import first_tool_call, format_tool_result
from .prompts import build_system_prompt


Emit = Callable[[Dict[str, Any]], Awaitable[None]]


class AgentRunner:
    def __init__(self, emit: Emit) -> None:
        self._emit = emit
        self._cancel = asyncio.Event()

    def cancel(self) -> None:
        self._cancel.set()

    async def run(
        self,
        history: List[Dict[str, str]],
        agent_mode: bool | None = None,
    ) -> None:
        self._cancel.clear()

        # Per-request override beats the persisted CONFIG setting. The Chat
        # tab passes False (pure chat, no tools); the Code tab passes True.
        use_agent = CONFIG.agent_mode if agent_mode is None else agent_mode

        # Agent OFF: ship exactly what the user typed in Settings — no
        # catalog, no workspace line, no rules.
        # Agent ON: append the compact tool catalog + restrained rules.
        system = (
            build_system_prompt(
                CONFIG.system_prompt, list_tools(), CONFIG.workspace, True,
            )
            if use_agent
            else CONFIG.system_prompt
        )

        # Build the running message list. We don't mutate `history` directly.
        messages: List[ChatMessage] = []
        if system:
            messages.append(ChatMessage(role="system", content=system))
        for m in history:
            role = m.get("role") or "user"
            content = m.get("content") or ""
            if role in ("user", "assistant", "system", "tool"):
                # collapse "tool" role into user-visible content for engines
                # that don't natively support tool messages
                if role == "tool":
                    role = "user"
                messages.append(ChatMessage(role=role, content=content))

        max_iters = max(1, CONFIG.max_iterations) if use_agent else 1

        for step in range(max_iters):
            if self._cancel.is_set():
                await self._emit({"type": "error", "msgId": "", "message": "Cancelled"})
                return

            await self._emit({"type": "agent-step", "msgId": "", "step": step})

            msg_id = str(uuid.uuid4())
            await self._emit({"type": "message-start", "msgId": msg_id})

            assistant_text = ""
            delta_count = 0
            try:
                if not loader.is_loaded():
                    raise RuntimeError(
                        "Model is not loaded. Open Settings → Model → Load model."
                    )
                engine = loader.get()
                async for delta in engine.stream(messages):
                    if self._cancel.is_set():
                        break
                    assistant_text += delta
                    delta_count += 1
                    await self._emit({
                        "type": "message-delta",
                        "msgId": msg_id,
                        "delta": delta,
                    })
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
                # Native crashes during inference (access violations, segfaults,
                # graph-allocation failures) leave llama.cpp's internal state
                # corrupted — every subsequent call will null-pointer until the
                # engine is fully torn down and reconstructed. Force-eject so
                # the next request either reloads cleanly or surfaces a real
                # "model not loaded" error instead of cascading crashes.
                if any(s in msg.lower() for s in (
                    "access violation",
                    "exception:",
                    "failed to allocate",
                    "non-consecutive",
                    "segmentation fault",
                )):
                    try:
                        loader.unload()
                    except Exception:  # noqa: BLE001
                        pass
                    msg += "\n\n(Engine was ejected — reload the model to continue.)"
                await self._emit({
                    "type": "error",
                    "msgId": msg_id,
                    "message": msg,
                })
                await self._emit({"type": "message-end", "msgId": msg_id})
                return

            # If the model produced zero deltas (sampling immediately picked
            # EOS, prompt overflow caused empty output, etc.), the bubble
            # would be invisibly empty. Replace it with a visible explanation
            # so the user knows what happened and can retry / shorten the
            # conversation instead of staring at a blank screen.
            if delta_count == 0 and not assistant_text:
                # Honest message — the most common cause is the model sampling
                # the end-of-turn token on the first step (random sampling
                # variance, or aggressive fine-tunes biased toward brevity
                # in /no_think mode). Context overflow is RARE and we used
                # to blame that misleadingly; now we name the likely cause
                # and suggest the most useful action.
                hint = (
                    "_(no reply — the model produced no output this turn. "
                    "This sometimes happens in **Quick** mode with fine-tunes "
                    "trained for brief answers. Try sending the message "
                    "again, switching to **Smart** mode, or rephrasing.)_"
                )
                await self._emit({
                    "type": "message-delta",
                    "msgId": msg_id,
                    "delta": hint,
                })
                assistant_text = hint

            await self._emit({"type": "message-end", "msgId": msg_id})

            messages.append(ChatMessage(role="assistant", content=assistant_text))

            call = first_tool_call(assistant_text)
            if call is None or not use_agent:
                return

            call_id = str(uuid.uuid4())
            await self._emit({
                "type": "tool-call",
                "msgId": msg_id,
                "toolCall": {"id": call_id, "tool": call.tool, "args": call.args},
            })

            result = await dispatch_tool(call.tool, call.args)

            is_error = result.startswith("Error:")
            await self._emit({
                "type": "tool-result",
                "msgId": msg_id,
                "toolResult": {
                    "id": call_id,
                    "result": None if is_error else result,
                    "error": result if is_error else None,
                },
            })

            # Feed result back as a user turn so the model can iterate.
            messages.append(ChatMessage(
                role="user",
                content=format_tool_result(call_id, result),
            ))

        # max iterations hit — emit a notice
        await self._emit({
            "type": "error",
            "msgId": "",
            "message": f"Agent stopped: hit max iterations ({max_iters}).",
        })
