from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from core.memory import MemoryManager
from core.messenger import Messenger
from core.prompt_builder import build_system_prompt
from core.schemas import CycleResult, ModelConfig

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
        self._sdk_available = self._check_sdk()

        logger.info(
            "AgentCore: model=%s, api_key_env=%s, base_url=%s",
            self.model_config.model,
            self.model_config.api_key_env,
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

    async def run_cycle(
        self, prompt: str, trigger: str = "manual"
    ) -> CycleResult:
        """Run one agent cycle with autonomous memory search."""
        start = time.monotonic()
        system_prompt = build_system_prompt(self.memory)

        if self._sdk_available:
            result = await self._run_with_agent_sdk(system_prompt, prompt)
        else:
            result = await self._run_with_anthropic_sdk(system_prompt, prompt)

        duration_ms = int((time.monotonic() - start) * 1000)
        return CycleResult(
            trigger=trigger,
            action="responded",
            summary=result,
            duration_ms=duration_ms,
        )

    def _resolve_api_key(self) -> str | None:
        """Resolve the actual API key from the configured environment variable."""
        return os.environ.get(self.model_config.api_key_env)

    async def _run_with_agent_sdk(
        self, system_prompt: str, prompt: str
    ) -> str:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            TextBlock,
            query,
        )

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
        )

        response_text: list[str] = []
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text.append(block.text)

        return "\n".join(response_text) or "(no response)"

    async def _run_with_anthropic_sdk(
        self, system_prompt: str, prompt: str
    ) -> str:
        """Fallback: use anthropic SDK with tool_use for memory ops."""
        import anthropic

        api_key = self._resolve_api_key()
        client_kwargs: dict[str, str] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if self.model_config.api_base_url:
            client_kwargs["base_url"] = self.model_config.api_base_url
        client = anthropic.AsyncAnthropic(**client_kwargs)

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

        messages: list[dict] = [{"role": "user", "content": prompt}]

        for _ in range(10):
            response = await client.messages.create(
                model=self.model_config.model,
                max_tokens=self.model_config.max_tokens,
                system=system_prompt,
                messages=messages,
                tools=tools,
            )

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                return "\n".join(
                    b.text for b in response.content if b.type == "text"
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

        return "(max iterations reached)"

    def _handle_tool_call(self, name: str, args: dict) -> str:
        if name == "search_memory":
            results = self.memory.search_knowledge(args.get("query", ""))
            if not results:
                return f"No results for '{args.get('query', '')}'"
            return "\n".join(
                f"- {fname}: {line}" for fname, line in results[:10]
            )

        if name == "read_memory_file":
            path = self.person_dir / args["path"]
            if path.exists() and path.is_file():
                return path.read_text(encoding="utf-8")
            return f"File not found: {args['path']}"

        if name == "write_memory_file":
            path = self.person_dir / args["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            if args.get("mode") == "append":
                with open(path, "a", encoding="utf-8") as f:
                    f.write(args["content"])
            else:
                path.write_text(args["content"], encoding="utf-8")
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
            return f"Message sent to {args['to']} (id: {msg.id}, thread: {msg.thread_id})"

        return f"Unknown tool: {name}"
