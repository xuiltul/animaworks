from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from collections.abc import AsyncGenerator
from typing import Any

from core.context_tracker import ContextTracker
from core.memory import MemoryManager
from core.messenger import Messenger
from core.paths import load_prompt
from core.prompt_builder import build_system_prompt, inject_shortterm
from core.schemas import CycleResult, ModelConfig
from core.shortterm_memory import SessionState, ShortTermMemory

logger = logging.getLogger("animaworks.agent")


class AgentCore:
    """Wraps Claude Agent SDK to provide thinking/acting for a Digital Person."""

    def __init__(
        self,
        person_dir: Path,
        memory: MemoryManager,
        model_config: ModelConfig | None = None,
        messenger: Messenger | None = None,
    ) -> None:
        self.person_dir = person_dir
        self.memory = memory
        self.model_config = model_config or ModelConfig()
        self.messenger = messenger
        self._tool_registry = self._init_tool_registry()
        self._sdk_available = self._check_sdk()

        logger.info(
            "AgentCore: model=%s, api_key=%s, base_url=%s",
            self.model_config.model,
            "direct" if self.model_config.api_key else f"env:{self.model_config.api_key_env}",
            self.model_config.api_base_url or "(default)",
        )

    def _check_sdk(self) -> bool:
        try:
            from claude_agent_sdk import query  # noqa: F401

            return True
        except ImportError:
            logger.warning(
                "claude-agent-sdk not available, falling back to anthropic SDK"
            )
            return False

    def _init_tool_registry(self):
        """Initialize tool registry with tools allowed in permissions.md."""
        try:
            from core.tools import TOOL_MODULES
            # Read permissions to determine allowed tools
            permissions = self.memory.read_permissions() if self.memory else ""
            allowed = []
            if "外部ツール" in permissions:
                for tool_name in TOOL_MODULES:
                    # Check if tool is marked as OK in permissions
                    if f"{tool_name}: OK" in permissions:
                        allowed.append(tool_name)
            return allowed
        except Exception:
            logger.debug("Tool registry initialization skipped")
            return []

    async def run_cycle(
        self, prompt: str, trigger: str = "manual"
    ) -> CycleResult:
        """Run one agent cycle with autonomous memory search.

        If the context threshold is crossed, the session is externalized
        to short-term memory and automatically continued in a fresh session.
        """
        start = time.monotonic()
        sdk_label = "agent-sdk" if self._sdk_available else "anthropic"
        logger.info(
            "run_cycle START trigger=%s prompt_len=%d sdk=%s",
            trigger, len(prompt), sdk_label,
        )

        shortterm = ShortTermMemory(self.person_dir)
        tracker = ContextTracker(
            model=self.model_config.model,
            threshold=self.model_config.context_threshold,
        )

        # Build system prompt; inject short-term memory from prior session
        system_prompt = build_system_prompt(self.memory)
        logger.debug("System prompt assembled, length=%d", len(system_prompt))
        if shortterm.has_pending():
            system_prompt = inject_shortterm(system_prompt, shortterm)
            logger.info("Injected short-term memory into system prompt")

        # Run the primary session
        if self._sdk_available:
            result, result_msg = await self._run_with_agent_sdk(
                system_prompt, prompt, tracker
            )
        else:
            result = await self._run_with_anthropic_sdk(
                system_prompt, prompt, tracker, shortterm
            )
            result_msg = None

        # Session chaining: if threshold was crossed, continue in a new session
        session_chained = False
        total_turns = result_msg.num_turns if result_msg else 0
        chain_count = 0

        while (
            self._sdk_available
            and tracker.threshold_exceeded
            and chain_count < self.model_config.max_chains
        ):
            session_chained = True
            chain_count += 1
            logger.info(
                "Session chain %d/%d: context at %.1f%%",
                chain_count,
                self.model_config.max_chains,
                tracker.usage_ratio * 100,
            )

            # Always save fresh state (clear stale data first)
            shortterm.clear()
            shortterm.save(
                SessionState(
                    session_id=result_msg.session_id if result_msg else "",
                    timestamp=datetime.now().isoformat(),
                    trigger=trigger,
                    original_prompt=prompt,
                    accumulated_response=result,
                    context_usage_ratio=tracker.usage_ratio,
                    turn_count=result_msg.num_turns if result_msg else 0,
                )
            )

            # New session with restored short-term memory
            tracker.reset()
            system_prompt_2 = inject_shortterm(
                build_system_prompt(self.memory),
                shortterm,
            )
            continuation_prompt = load_prompt("session_continuation")
            try:
                result_2, result_msg_2 = await self._run_with_agent_sdk(
                    system_prompt_2, continuation_prompt, tracker
                )
            except Exception:
                logger.exception(
                    "Chained session %d failed; preserving short-term memory",
                    chain_count,
                )
                break
            result = result + "\n" + result_2
            result_msg = result_msg_2
            if result_msg_2:
                total_turns += result_msg_2.num_turns

        # Clean up short-term memory after successful completion
        shortterm.clear()

        duration_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "run_cycle END trigger=%s duration_ms=%d response_len=%d chained=%s",
            trigger, duration_ms, len(result), session_chained,
        )
        return CycleResult(
            trigger=trigger,
            action="responded",
            summary=result,
            duration_ms=duration_ms,
            context_usage_ratio=tracker.usage_ratio,
            session_chained=session_chained,
            total_turns=total_turns,
        )

    def _resolve_api_key(self) -> str | None:
        """Resolve the actual API key (direct value from config.json, then env var)."""
        if self.model_config.api_key:
            return self.model_config.api_key
        return os.environ.get(self.model_config.api_key_env)

    # ── Agent SDK path ──────────────────────────────────────

    async def _run_with_agent_sdk(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
    ) -> tuple[str, Any | None]:
        """Run a session via Claude Agent SDK with context monitoring hook.

        Returns ``(response_text, ResultMessage | None)``.
        The second element is typed as ``Any | None`` to avoid importing
        ``ResultMessage`` at module level.
        """
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            HookMatcher,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
            query,
        )
        from claude_agent_sdk.types import (
            HookContext,
            HookInput,
            PostToolUseHookSpecificOutput,
            SyncHookJSONOutput,
        )

        threshold = self.model_config.context_threshold
        _hook_fired = False

        async def _post_tool_hook(
            input_data: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> SyncHookJSONOutput:
            nonlocal _hook_fired
            transcript_path = input_data.get("transcript_path", "")
            ratio = tracker.estimate_from_transcript(transcript_path)

            if ratio >= threshold and not _hook_fired:
                _hook_fired = True
                logger.info(
                    "PostToolUse hook: context at %.1f%%, injecting save instruction",
                    ratio * 100,
                )
                return SyncHookJSONOutput(
                    hookSpecificOutput=PostToolUseHookSpecificOutput(
                        hookEventName="PostToolUse",
                        additionalContext=(
                            f"コンテキスト使用率が{ratio:.0%}に達しました。"
                            "shortterm/session_state.md に現在の作業状態を書き出してください。"
                            "内容: 何をしていたか、どこまで進んだか、次に何をすべきか。"
                            "書き出し後、作業を中断してその旨を報告してください。"
                        ),
                    )
                )
            return SyncHookJSONOutput()

        # Build env dict so the child process uses per-person credentials
        env: dict[str, str] = {}
        api_key = self._resolve_api_key()
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key
        if self.model_config.api_base_url:
            env["ANTHROPIC_BASE_URL"] = self.model_config.api_base_url

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            allowed_tools=["Read", "Write", "Edit", "Bash", "Grep", "Glob"],
            permission_mode="acceptEdits",
            cwd=str(self.person_dir),
            max_turns=self.model_config.max_turns,
            model=self.model_config.model,
            env=env,
            hooks={
                "PostToolUse": [HookMatcher(matcher=None, hooks=[_post_tool_hook])],
            },
        )

        response_text: list[str] = []
        result_message: ResultMessage | None = None
        message_count = 0
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage):
                result_message = message
                tracker.update_from_result_message(message.usage)
            elif isinstance(message, AssistantMessage):
                message_count += 1
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text.append(block.text)

        logger.debug(
            "Agent SDK completed, messages=%d text_blocks=%d",
            message_count, len(response_text),
        )
        return "\n".join(response_text) or "(no response)", result_message

    # ── Agent SDK streaming path ─────────────────────────────

    async def _run_with_agent_sdk_streaming(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
    ) -> AsyncGenerator[dict, None]:
        """Stream events from Claude Agent SDK.

        Yields dicts:
            {"type": "text_delta", "text": "..."}
            {"type": "tool_start", "tool_name": "...", "tool_id": "..."}
            {"type": "tool_end", "tool_id": "...", "tool_name": "..."}
            {"type": "done", "full_text": "...", "result_message": ...}
        """
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            HookMatcher,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
            query,
        )
        from claude_agent_sdk.types import (
            HookContext,
            HookInput,
            PostToolUseHookSpecificOutput,
            StreamEvent,
            SyncHookJSONOutput,
        )

        threshold = self.model_config.context_threshold
        _hook_fired = False

        async def _post_tool_hook(
            input_data: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> SyncHookJSONOutput:
            nonlocal _hook_fired
            transcript_path = input_data.get("transcript_path", "")
            ratio = tracker.estimate_from_transcript(transcript_path)

            if ratio >= threshold and not _hook_fired:
                _hook_fired = True
                logger.info(
                    "PostToolUse hook (stream): context at %.1f%%",
                    ratio * 100,
                )
                return SyncHookJSONOutput(
                    hookSpecificOutput=PostToolUseHookSpecificOutput(
                        hookEventName="PostToolUse",
                        additionalContext=(
                            f"コンテキスト使用率が{ratio:.0%}に達しました。"
                            "shortterm/session_state.md に現在の作業状態を書き出してください。"
                            "内容: 何をしていたか、どこまで進んだか、次に何をすべきか。"
                            "書き出し後、作業を中断してその旨を報告してください。"
                        ),
                    )
                )
            return SyncHookJSONOutput()

        env: dict[str, str] = {}
        api_key = self._resolve_api_key()
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key
        if self.model_config.api_base_url:
            env["ANTHROPIC_BASE_URL"] = self.model_config.api_base_url

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            allowed_tools=["Read", "Write", "Edit", "Bash", "Grep", "Glob"],
            permission_mode="acceptEdits",
            cwd=str(self.person_dir),
            max_turns=self.model_config.max_turns,
            model=self.model_config.model,
            env=env,
            include_partial_messages=True,
            hooks={
                "PostToolUse": [HookMatcher(matcher=None, hooks=[_post_tool_hook])],
            },
        )

        response_text: list[str] = []
        result_message: ResultMessage | None = None
        active_tool_ids: set[str] = set()
        message_count = 0

        async for message in query(prompt=prompt, options=options):
            if isinstance(message, StreamEvent):
                event = message.event
                event_type = event.get("type", "")

                if event_type == "content_block_start":
                    block = event.get("content_block", {})
                    if block.get("type") == "tool_use":
                        tool_id = block.get("id", "")
                        active_tool_ids.add(tool_id)
                        yield {
                            "type": "tool_start",
                            "tool_name": block.get("name", ""),
                            "tool_id": tool_id,
                        }

                elif event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            yield {"type": "text_delta", "text": text}

            elif isinstance(message, AssistantMessage):
                message_count += 1
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        if block.id in active_tool_ids:
                            active_tool_ids.discard(block.id)
                            yield {
                                "type": "tool_end",
                                "tool_id": block.id,
                                "tool_name": block.name,
                            }

            elif isinstance(message, ResultMessage):
                result_message = message
                tracker.update_from_result_message(message.usage)

        logger.debug(
            "Agent SDK streaming completed, messages=%d text_blocks=%d",
            message_count, len(response_text),
        )
        full_text = "\n".join(response_text) or "(no response)"
        yield {
            "type": "done",
            "full_text": full_text,
            "result_message": result_message,
        }

    async def run_cycle_streaming(
        self, prompt: str, trigger: str = "manual"
    ) -> AsyncGenerator[dict, None]:
        """Streaming version of run_cycle.

        Yields stream chunks. Session chaining is handled seamlessly.
        Final event is ``{"type": "cycle_done", "cycle_result": {...}}``.
        """
        start = time.monotonic()
        sdk_label = "agent-sdk" if self._sdk_available else "anthropic"
        logger.info(
            "run_cycle_streaming START trigger=%s prompt_len=%d sdk=%s",
            trigger, len(prompt), sdk_label,
        )

        shortterm = ShortTermMemory(self.person_dir)
        tracker = ContextTracker(
            model=self.model_config.model,
            threshold=self.model_config.context_threshold,
        )

        system_prompt = build_system_prompt(self.memory)
        if shortterm.has_pending():
            system_prompt = inject_shortterm(system_prompt, shortterm)

        # Fallback: no streaming, yield complete text
        if not self._sdk_available:
            result = await self._run_with_anthropic_sdk(
                system_prompt, prompt, tracker, shortterm
            )
            yield {"type": "text_delta", "text": result}
            duration_ms = int((time.monotonic() - start) * 1000)
            yield {
                "type": "cycle_done",
                "cycle_result": CycleResult(
                    trigger=trigger,
                    action="responded",
                    summary=result,
                    duration_ms=duration_ms,
                    context_usage_ratio=tracker.usage_ratio,
                    session_chained=False,
                    total_turns=0,
                ).model_dump(),
            }
            return

        # Primary session
        full_text_parts: list[str] = []
        result_message: Any = None

        async for chunk in self._run_with_agent_sdk_streaming(
            system_prompt, prompt, tracker
        ):
            if chunk["type"] == "done":
                full_text_parts.append(chunk["full_text"])
                result_message = chunk["result_message"]
            else:
                yield chunk

        # Session chaining
        session_chained = False
        total_turns = result_message.num_turns if result_message else 0
        chain_count = 0

        while (
            tracker.threshold_exceeded
            and chain_count < self.model_config.max_chains
        ):
            session_chained = True
            chain_count += 1
            logger.info(
                "Session chain (stream) %d/%d: context at %.1f%%",
                chain_count,
                self.model_config.max_chains,
                tracker.usage_ratio * 100,
            )

            yield {"type": "chain_start", "chain": chain_count}

            shortterm.clear()
            shortterm.save(
                SessionState(
                    session_id=result_message.session_id if result_message else "",
                    timestamp=datetime.now().isoformat(),
                    trigger=trigger,
                    original_prompt=prompt,
                    accumulated_response="\n".join(full_text_parts),
                    context_usage_ratio=tracker.usage_ratio,
                    turn_count=result_message.num_turns if result_message else 0,
                )
            )

            tracker.reset()
            system_prompt_2 = inject_shortterm(
                build_system_prompt(self.memory),
                shortterm,
            )
            continuation_prompt = load_prompt("session_continuation")

            try:
                async for chunk in self._run_with_agent_sdk_streaming(
                    system_prompt_2, continuation_prompt, tracker
                ):
                    if chunk["type"] == "done":
                        full_text_parts.append(chunk["full_text"])
                        result_message = chunk["result_message"]
                        if result_message:
                            total_turns += result_message.num_turns
                    else:
                        yield chunk
            except Exception:
                logger.exception(
                    "Chained session (stream) %d failed", chain_count,
                )
                yield {"type": "error", "message": f"Session chain {chain_count} failed"}
                break

        shortterm.clear()

        full_text = "\n".join(full_text_parts)
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "run_cycle_streaming END trigger=%s duration_ms=%d response_len=%d chained=%s",
            trigger, duration_ms, len(full_text), session_chained,
        )

        yield {
            "type": "cycle_done",
            "cycle_result": CycleResult(
                trigger=trigger,
                action="responded",
                summary=full_text,
                duration_ms=duration_ms,
                context_usage_ratio=tracker.usage_ratio,
                session_chained=session_chained,
                total_turns=total_turns,
            ).model_dump(),
        }

    # ── Anthropic SDK fallback path ─────────────────────────

    async def _run_with_anthropic_sdk(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        shortterm: ShortTermMemory,
    ) -> str:
        """Fallback: use anthropic SDK with tool_use for memory ops.

        Mid-conversation context monitoring: if the threshold is crossed,
        state is externalized and the conversation is restarted with
        restored short-term memory.
        """
        import anthropic

        api_key = self._resolve_api_key()
        client_kwargs: dict[str, str] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if self.model_config.api_base_url:
            client_kwargs["base_url"] = self.model_config.api_base_url
        client = anthropic.AsyncAnthropic(**client_kwargs)

        tools = self._build_anthropic_tools()
        messages: list[dict] = [{"role": "user", "content": prompt}]
        all_response_text: list[str] = []
        chain_count = 0

        for iteration in range(10):
            logger.debug(
                "API call iteration=%d messages_count=%d", iteration, len(messages),
            )
            response = await client.messages.create(
                model=self.model_config.model,
                max_tokens=self.model_config.max_tokens,
                system=system_prompt,
                messages=messages,
                tools=tools,
            )

            # Track context usage from API response
            usage_dict = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            threshold_crossed = tracker.update_from_usage(usage_dict)

            if threshold_crossed and chain_count < self.model_config.max_chains:
                chain_count += 1
                logger.info(
                    "Anthropic SDK: context threshold crossed at %.1f%%, "
                    "restarting with short-term memory (chain %d/%d)",
                    tracker.usage_ratio * 100,
                    chain_count,
                    self.model_config.max_chains,
                )
                # Collect text so far
                current_text = "\n".join(
                    b.text for b in response.content if b.type == "text"
                )
                all_response_text.append(current_text)

                # Save state
                shortterm.save(
                    SessionState(
                        session_id="anthropic-fallback",
                        timestamp=datetime.now().isoformat(),
                        trigger="anthropic_sdk",
                        original_prompt=prompt,
                        accumulated_response="\n".join(all_response_text),
                        context_usage_ratio=tracker.usage_ratio,
                        turn_count=iteration,
                    )
                )

                # Restart with fresh context + short-term memory
                tracker.reset()
                system_prompt = inject_shortterm(
                    build_system_prompt(self.memory), shortterm
                )
                messages = [
                    {"role": "user", "content": load_prompt("session_continuation")}
                ]
                shortterm.clear()
                continue

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                logger.debug("Final response received at iteration=%d", iteration)
                final_text = "\n".join(
                    b.text for b in response.content if b.type == "text"
                )
                all_response_text.append(final_text)
                return "\n".join(all_response_text)

            logger.info(
                "Tool calls at iteration=%d: %s",
                iteration, ", ".join(tu.name for tu in tool_uses),
            )
            messages.append(
                {"role": "assistant", "content": response.content}
            )
            tool_results = []
            for tu in tool_uses:
                result = self._handle_tool_call(tu.name, tu.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": result,
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        logger.warning("Max iterations (10) reached, returning fallback response")
        return "\n".join(all_response_text) or "(max iterations reached)"

    # ── Tool definitions (Anthropic SDK fallback) ───────────

    def _build_anthropic_tools(self) -> list[dict]:
        tools = [
            {
                "name": "search_memory",
                "description": "Search the person's long-term memory by keyword.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search keyword",
                        },
                        "scope": {
                            "type": "string",
                            "enum": [
                                "knowledge",
                                "episodes",
                                "procedures",
                                "all",
                            ],
                            "description": "Memory category to search",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "read_memory_file",
                "description": "Read a specific memory file by relative path.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path within person dir",
                        },
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "write_memory_file",
                "description": "Write or append to a memory file.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "mode": {
                            "type": "string",
                            "enum": ["overwrite", "append"],
                        },
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "send_message",
                "description": "Send a message to another person. The recipient will be notified immediately.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Recipient person name",
                        },
                        "content": {
                            "type": "string",
                            "description": "Message content",
                        },
                        "reply_to": {
                            "type": "string",
                            "description": "Message ID to reply to (optional)",
                        },
                        "thread_id": {
                            "type": "string",
                            "description": "Thread ID to continue a conversation (optional)",
                        },
                    },
                    "required": ["to", "content"],
                },
            },
        ]

        # External tools from registry
        if self._tool_registry:
            try:
                import importlib
                from core.tools import TOOL_MODULES
                for tool_name in self._tool_registry:
                    if tool_name in TOOL_MODULES:
                        mod = importlib.import_module(TOOL_MODULES[tool_name])
                        if hasattr(mod, "get_tool_schemas"):
                            tools.extend(mod.get_tool_schemas())
            except Exception:
                logger.debug("Failed to load external tool schemas", exc_info=True)

        return tools

    def _handle_tool_call(self, name: str, args: dict) -> str:
        logger.debug("tool_call name=%s args_keys=%s", name, list(args.keys()))

        if name == "search_memory":
            results = self.memory.search_knowledge(args.get("query", ""))
            logger.debug(
                "search_memory query=%s results=%d",
                args.get("query", ""), len(results),
            )
            if not results:
                return f"No results for '{args.get('query', '')}'"
            return "\n".join(
                f"- {fname}: {line}" for fname, line in results[:10]
            )

        if name == "read_memory_file":
            path = self.person_dir / args["path"]
            if path.exists() and path.is_file():
                logger.debug("read_memory_file path=%s", args["path"])
                return path.read_text(encoding="utf-8")
            logger.debug("read_memory_file NOT FOUND path=%s", args["path"])
            return f"File not found: {args['path']}"

        if name == "write_memory_file":
            path = self.person_dir / args["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            if args.get("mode") == "append":
                with open(path, "a", encoding="utf-8") as f:
                    f.write(args["content"])
            else:
                path.write_text(args["content"], encoding="utf-8")
            logger.info("write_memory_file path=%s mode=%s", args["path"], args.get("mode", "overwrite"))
            return f"Written to {args['path']}"

        if name == "send_message":
            if not self.messenger:
                return "Error: messenger not configured"
            msg = self.messenger.send(
                to=args["to"],
                content=args["content"],
                thread_id=args.get("thread_id", ""),
                reply_to=args.get("reply_to", ""),
            )
            logger.info("send_message to=%s thread=%s", args["to"], msg.thread_id)
            return f"Message sent to {args['to']} (id: {msg.id}, thread: {msg.thread_id})"

        # External tool dispatch
        if self._tool_registry:
            return self._handle_external_tool(name, args)

        logger.warning("Unknown tool requested: %s", name)
        return f"Unknown tool: {name}"

    def _handle_external_tool(self, name: str, args: dict) -> str:
        """Dispatch to external tool modules via direct Python calls."""
        import importlib
        import json
        from core.tools import TOOL_MODULES

        for tool_name, module_path in TOOL_MODULES.items():
            if tool_name not in self._tool_registry:
                continue
            try:
                mod = importlib.import_module(module_path)
                schemas = mod.get_tool_schemas() if hasattr(mod, "get_tool_schemas") else []
                schema_names = [s["name"] for s in schemas]
                if name not in schema_names:
                    continue

                result = self._execute_tool_function(mod, tool_name, name, args)
                if isinstance(result, (dict, list)):
                    return json.dumps(result, ensure_ascii=False, indent=2, default=str)
                return str(result) if result is not None else "(no output)"
            except Exception as e:
                logger.warning("External tool %s failed: %s", name, e)
                return f"Error executing {name}: {e}"

        return f"Unknown tool: {name}"

    def _execute_tool_function(self, mod, tool_name: str, schema_name: str, args: dict):
        """Execute the appropriate function for the given tool schema name."""
        # --- web_search ---
        if schema_name == "web_search":
            return mod.search(**args)

        # --- x_search ---
        if schema_name == "x_search":
            client = mod.XSearchClient()
            return client.search_recent(
                query=args["query"],
                max_results=args.get("max_results", 10),
                days=args.get("days", 7),
            )
        if schema_name == "x_user_tweets":
            client = mod.XSearchClient()
            return client.get_user_tweets(
                username=args["username"],
                max_results=args.get("max_results", 10),
                days=args.get("days"),
            )

        # --- chatwork ---
        if schema_name == "chatwork_send":
            client = mod.ChatworkClient()
            room_id = client.resolve_room_id(args["room"])
            return client.post_message(room_id, args["message"])
        if schema_name == "chatwork_messages":
            client = mod.ChatworkClient()
            room_id = client.resolve_room_id(args["room"])
            cache = mod.MessageCache()
            try:
                msgs = client.get_messages(room_id, force=True)
                if msgs:
                    cache.upsert_messages(room_id, msgs)
                    cache.update_sync_state(room_id)
                return cache.get_recent(room_id, limit=args.get("limit", 20))
            finally:
                cache.close()
        if schema_name == "chatwork_search":
            client = mod.ChatworkClient()
            cache = mod.MessageCache()
            try:
                room_id = None
                if args.get("room"):
                    room_id = client.resolve_room_id(args["room"])
                return cache.search(
                    args["keyword"], room_id=room_id, limit=args.get("limit", 50),
                )
            finally:
                cache.close()
        if schema_name == "chatwork_unreplied":
            client = mod.ChatworkClient()
            cache = mod.MessageCache()
            try:
                my_info = client.me()
                my_id = str(my_info["account_id"])
                return cache.find_unreplied(
                    my_id, exclude_toall=not args.get("include_toall", False),
                )
            finally:
                cache.close()
        if schema_name == "chatwork_rooms":
            client = mod.ChatworkClient()
            return client.rooms()

        # --- slack ---
        if schema_name == "slack_send":
            client = mod.SlackClient()
            channel_id = client.resolve_channel(args["channel"])
            return client.post_message(
                channel_id,
                args["message"],
                thread_ts=args.get("thread_ts"),
            )
        if schema_name == "slack_messages":
            client = mod.SlackClient()
            channel_id = client.resolve_channel(args["channel"])
            cache = mod.MessageCache()
            try:
                limit = args.get("limit", 20)
                msgs = client.channel_history(channel_id, limit=limit)
                if msgs:
                    for m in msgs:
                        uid = m.get("user", m.get("bot_id", ""))
                        if uid:
                            m["user_name"] = client.resolve_user_name(uid)
                    cache.upsert_messages(channel_id, msgs)
                    cache.update_sync_state(channel_id)
                return cache.get_recent(channel_id, limit=limit)
            finally:
                cache.close()
        if schema_name == "slack_search":
            client = mod.SlackClient()
            cache = mod.MessageCache()
            try:
                channel_id = None
                if args.get("channel"):
                    channel_id = client.resolve_channel(args["channel"])
                return cache.search(
                    args["keyword"], channel_id=channel_id, limit=args.get("limit", 50),
                )
            finally:
                cache.close()
        if schema_name == "slack_unreplied":
            client = mod.SlackClient()
            cache = mod.MessageCache()
            try:
                client.auth_test()
                return cache.find_unreplied(client.my_user_id)
            finally:
                cache.close()
        if schema_name == "slack_channels":
            client = mod.SlackClient()
            return client.channels()

        # --- gmail ---
        if schema_name == "gmail_unread":
            client = mod.GmailClient()
            emails = client.get_unread_emails(max_results=args.get("max_results", 20))
            return [
                {"id": e.id, "from": e.from_addr, "subject": e.subject, "snippet": e.snippet}
                for e in emails
            ]
        if schema_name == "gmail_read_body":
            client = mod.GmailClient()
            return client.get_email_body(args["message_id"])
        if schema_name == "gmail_draft":
            client = mod.GmailClient()
            result = client.create_draft(
                to=args["to"],
                subject=args["subject"],
                body=args["body"],
                thread_id=args.get("thread_id"),
                in_reply_to=args.get("in_reply_to"),
            )
            return {"success": result.success, "draft_id": result.draft_id, "error": result.error}

        # --- local_llm ---
        if schema_name == "local_llm_generate":
            client = mod.OllamaClient(
                server=args.get("server", "auto"),
                model=args.get("model"),
                hint=args.get("hint"),
            )
            return client.generate(
                prompt=args["prompt"],
                system=args.get("system", ""),
                temperature=args.get("temperature", 0.7),
                max_tokens=args.get("max_tokens", 4096),
                think=args.get("think", "off"),
            )
        if schema_name == "local_llm_chat":
            client = mod.OllamaClient(
                server=args.get("server", "auto"),
                model=args.get("model"),
                hint=args.get("hint"),
            )
            return client.chat(
                messages=args["messages"],
                system=args.get("system", ""),
                temperature=args.get("temperature", 0.7),
                max_tokens=args.get("max_tokens", 4096),
                think=args.get("think", "off"),
            )
        if schema_name == "local_llm_models":
            client = mod.OllamaClient(server=args.get("server", "auto"))
            return client.list_models()
        if schema_name == "local_llm_status":
            client = mod.OllamaClient()
            return client.server_status()

        # --- transcribe ---
        if schema_name == "transcribe_audio":
            return mod.process_audio(
                audio_path=args["audio_path"],
                language=args.get("language"),
                model=args.get("model"),
                raw_only=args.get("raw_only", False),
                custom_prompt=args.get("custom_prompt"),
            )

        # --- aws_collector ---
        if schema_name == "aws_ecs_status":
            collector = mod.AWSCollector(region=args.get("region"))
            return collector.get_ecs_status(args["cluster"], args["service"])
        if schema_name == "aws_error_logs":
            collector = mod.AWSCollector(region=args.get("region"))
            return collector.get_error_logs(
                log_group=args["log_group"],
                hours=args.get("hours", 1),
                patterns=args.get("patterns"),
            )
        if schema_name == "aws_metrics":
            collector = mod.AWSCollector(region=args.get("region"))
            return collector.get_metrics(
                cluster=args["cluster"],
                service=args["service"],
                metric=args.get("metric", "CPUUtilization"),
                hours=args.get("hours", 1),
            )

        # --- github ---
        if schema_name == "github_list_issues":
            client = mod.GitHubClient(repo=args.get("repo"))
            return client.list_issues(
                state=args.get("state", "open"),
                labels=args.get("labels"),
                limit=args.get("limit", 20),
            )
        if schema_name == "github_create_issue":
            client = mod.GitHubClient(repo=args.get("repo"))
            return client.create_issue(
                title=args["title"],
                body=args.get("body", ""),
                labels=args.get("labels"),
            )
        if schema_name == "github_list_prs":
            client = mod.GitHubClient(repo=args.get("repo"))
            return client.list_prs(
                state=args.get("state", "open"),
                limit=args.get("limit", 20),
            )
        if schema_name == "github_create_pr":
            client = mod.GitHubClient(repo=args.get("repo"))
            return client.create_pr(
                title=args["title"],
                body=args.get("body", ""),
                head=args["head"],
                base=args.get("base", "main"),
                draft=args.get("draft", False),
            )

        raise ValueError(f"No handler for tool schema: {schema_name}")
