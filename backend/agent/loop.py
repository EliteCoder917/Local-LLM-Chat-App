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
import json
import uuid
from typing import Any, Awaitable, Callable, Dict, List

from ..config import CONFIG
from ..llm import ChatMessage, loader
from ..tools.registry import dispatch_tool, tools_for_mode
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
        tool_mode: str | None = None,
    ) -> None:
        self._cancel.clear()

        # Per-request override beats the persisted CONFIG setting. The Chat
        # tab passes False (pure chat, no tools); the Code tab passes True.
        use_agent = CONFIG.agent_mode if agent_mode is None else agent_mode

        # `tool_mode` selects WHICH tools are exposed this turn:
        #   'cowork' → file/exec/memory · 'search' → web · 'all'/None → every
        # tool (Code tab). 'normal' (or anything unknown) yields none. We only
        # consult it when tools are on at all.
        active_tools = tools_for_mode(tool_mode) if use_agent else {}
        allowed_names = set(active_tools.keys())

        # Agent OFF: ship exactly what the user typed in Settings — no
        # catalog, no workspace line, no rules.
        # Agent ON: append the compact tool catalog (filtered to this mode) +
        # restrained rules.
        system = (
            build_system_prompt(
                CONFIG.system_prompt, active_tools, CONFIG.workspace, True,
            )
            if use_agent and active_tools
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

        # Tracks (tool, args) pairs already executed this run. Weaker models
        # often treat a tool_result (fed back as a user turn) as a fresh prompt
        # and re-issue the SAME call forever. When we see a duplicate we stop
        # dispatching and force the model to answer from what it already has.
        seen_calls: set[str] = set()
        force_answer = False

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
            # `force_answer` is set after a duplicate call was detected — ignore
            # any tool block on this turn and treat the text as the final reply,
            # guaranteeing the loop terminates instead of spinning.
            if call is None or not use_agent or force_answer:
                return

            # Tool not available in this mode (e.g. the model tried a file tool
            # in Search mode). Tell it, and force a final answer so it doesn't
            # keep probing for tools it can't reach.
            if call.tool not in allowed_names:
                messages.append(ChatMessage(
                    role="user",
                    content=(
                        f"[system] The tool '{call.tool}' is not available in "
                        "this mode. Answer the request directly using only the "
                        "tools listed in the system prompt (or no tools)."
                    ),
                ))
                force_answer = True
                continue

            # Repeated identical call → the model is stuck re-reading instead of
            # answering. Don't dispatch again; nudge it to answer from what it
            # has, and force the next turn to be final.
            call_key = f"{call.tool}:{json.dumps(call.args, sort_keys=True, default=str)}"
            if call_key in seen_calls:
                messages.append(ChatMessage(
                    role="user",
                    content=(
                        "[system] You have ALREADY called this tool and received "
                        "its result above. Do NOT call it again. Answer the "
                        "original request directly using the information you "
                        "already have."
                    ),
                ))
                force_answer = True
                continue
            seen_calls.add(call_key)

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

            # User denied the permission prompt → stop the loop entirely with a
            # clear, friendly notice instead of feeding the denial back and
            # letting the model thrash / retry. The user made a deliberate
            # choice; respect it and end the turn.
            if "permission denied" in result.lower():
                notice_id = str(uuid.uuid4())
                await self._emit({"type": "message-start", "msgId": notice_id})
                await self._emit({
                    "type": "message-delta",
                    "msgId": notice_id,
                    "delta": (
                        f"🚫 Permission to use **{call.tool}** was denied, so I've "
                        "stopped. If you'd like me to proceed, enable the relevant "
                        "permission in Settings → Permissions and ask again."
                    ),
                })
                await self._emit({"type": "message-end", "msgId": notice_id})
                return

            # Feed result back as a user turn so the model can iterate. The
            # trailing instruction steers weaker models away from re-calling the
            # same tool: the result is an OBSERVATION, and unless another action
            # is genuinely required they should now answer the user.
            messages.append(ChatMessage(
                role="user",
                content=(
                    format_tool_result(call_id, result)
                    + "\n\n[This is the result of your tool call. Use it to "
                    "answer the user's request now. Only call another tool if a "
                    "further action is genuinely required.]"
                ),
            ))

        # max iterations hit — emit a notice
        await self._emit({
            "type": "error",
            "msgId": "",
            "message": f"Agent stopped: hit max iterations ({max_iters}).",
        })
