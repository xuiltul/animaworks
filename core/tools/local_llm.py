# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""AnimaWorks local LLM tool -- Ollama API client.

Provides intelligent server selection, retry with failover,
and a clean Python API for interacting with Ollama servers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import httpx

from core.tools._base import logger
from core.tools._retry import retry_with_backoff

# ── Execution Profile ─────────────────────────────────────

EXECUTION_PROFILE: dict[str, dict[str, object]] = {
    "generate": {"expected_seconds": 300, "background_eligible": True},
    "chat":     {"expected_seconds": 300, "background_eligible": True},
    "list":     {"expected_seconds": 5,   "background_eligible": False},
    "status":   {"expected_seconds": 5,   "background_eligible": False},
}

# ---------------------------------------------------------------------------
# Server configuration
# ---------------------------------------------------------------------------


@dataclass
class OllamaServer:
    name: str
    url: str


DEFAULT_SERVERS = {
    "local": OllamaServer("local", "http://localhost:11434"),
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OllamaClient:
    """Client for interacting with Ollama API servers."""

    def __init__(
        self,
        server: str = "auto",
        model: str | None = None,
        hint: str | None = None,
        timeout: float = 600.0,
    ):
        # Parse OLLAMA_SERVERS env var if set (format: "name=url,name=url")
        self.servers = self._load_servers()
        self._server_name = server
        self._model = model
        self._hint = hint
        self._timeout = timeout
        self._client = httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0))

    def _load_servers(self) -> dict[str, OllamaServer]:
        env = os.environ.get("OLLAMA_SERVERS")
        if not env:
            return dict(DEFAULT_SERVERS)
        servers: dict[str, OllamaServer] = {}
        for entry in env.split(","):
            if "=" in entry:
                name, url = entry.split("=", 1)
                servers[name.strip()] = OllamaServer(name.strip(), url.strip())
        return servers if servers else dict(DEFAULT_SERVERS)

    def _get_server(self) -> OllamaServer:
        if self._server_name != "auto":
            if self._server_name in self.servers:
                return self.servers[self._server_name]
            raise ValueError(f"Unknown server: {self._server_name}")
        return self.select_server()

    def select_server(self) -> OllamaServer:
        """Select the least-loaded server by querying /api/ps."""
        candidates = list(self.servers.values())
        if not candidates:
            return OllamaServer("local", "http://localhost:11434")
        if len(candidates) == 1:
            return candidates[0]

        results: dict[str, tuple[int, float]] = {}

        def check(server: OllamaServer) -> tuple[OllamaServer, int, float]:
            try:
                r = httpx.get(f"{server.url}/api/ps", timeout=5.0)
                data = r.json()
                model_count = len(data.get("models", []))
                return server, model_count, r.elapsed.total_seconds()
            except Exception:
                return server, 999, 999.0

        with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            futures = [pool.submit(check, s) for s in candidates]
            for f in as_completed(futures):
                server, count, latency = f.result()
                results[server.name] = (count, latency)

        # Pick server with fewest loaded models, then lowest latency
        best = min(results.items(), key=lambda x: (x[1][0], x[1][1]))
        selected = self.servers[best[0]]
        logger.debug(
            "Selected server: %s (models=%d, latency=%.2fs)",
            selected.name,
            best[1][0],
            best[1][1],
        )
        return selected

    def resolve_model(self, hint: str | None = None) -> str:
        """Find a model on the server, optionally filtered by hint pattern."""
        if self._model:
            return self._model
        hint = hint or self._hint
        server = self._get_server()
        r = self._client.get(f"{server.url}/api/tags")
        r.raise_for_status()
        models = r.json().get("models", [])
        if not models:
            raise RuntimeError(f"No models available on {server.name}")
        if hint:
            pattern = re.compile(hint, re.IGNORECASE)
            matched = [m for m in models if pattern.search(m.get("name", ""))]
            if matched:
                return matched[0]["name"]
        return models[0]["name"]

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        think: str = "off",
    ) -> str:
        """Generate text using /api/generate."""
        server = self._get_server()
        model = self.resolve_model()
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            body["system"] = system
        self._apply_think(body, think)
        return self._send_with_retry(server, "/api/generate", body, key="response")

    def chat(
        self,
        messages: list[dict],
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        think: str = "off",
    ) -> str:
        """Chat using /api/chat."""
        server = self._get_server()
        model = self.resolve_model()
        if system:
            messages = [{"role": "system", "content": system}] + messages
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        self._apply_think(body, think)
        return self._send_with_retry(server, "/api/chat", body, key="message.content")

    def list_models(self) -> list[dict]:
        """List available models on the selected server."""
        server = self._get_server()
        r = self._client.get(f"{server.url}/api/tags")
        r.raise_for_status()
        return r.json().get("models", [])

    def server_status(self) -> dict[str, Any]:
        """Get status (loaded models) from all servers."""
        status: dict[str, Any] = {}
        for name, server in self.servers.items():
            try:
                r = httpx.get(f"{server.url}/api/ps", timeout=5.0)
                status[name] = r.json()
            except Exception as e:
                status[name] = {"error": str(e)}
        return status

    def model_info(self, model: str | None = None) -> dict:
        """Get model details via /api/show."""
        server = self._get_server()
        model = model or self.resolve_model()
        r = self._client.post(f"{server.url}/api/show", json={"model": model})
        r.raise_for_status()
        return r.json()

    def _apply_think(self, body: dict, think: str) -> None:
        """Apply thinking configuration based on model type."""
        if think == "off":
            body["options"]["think"] = False
            return
        # Map think levels
        model = body.get("model", "")
        if "gpt-oss" in model.lower():
            body["options"]["think"] = think  # string for GPT-OSS
        else:
            body["options"]["think"] = True  # boolean for others

    def _send_with_retry(
        self,
        server: OllamaServer,
        path: str,
        body: dict,
        key: str,
        max_retries: int = 60,
        poll_interval: float = 5.0,
    ) -> str:
        """Send request with retry on server busy, failover to alternate server.

        Uses the shared :func:`retry_with_backoff` utility for the
        retry loop while adding server-failover logic via
        ``on_retry``.
        """
        alternate = self._get_alternate(server)
        # Mutable state shared between the closure and on_retry callback
        state = {"current": server, "alternate": alternate}

        class _ServerBusyError(Exception):
            """Raised when the Ollama server reports busy/running."""

        def _do_request() -> str:
            current = state["current"]
            r = self._client.post(
                f"{current.url}{path}",
                json=body,
                timeout=httpx.Timeout(self._timeout, connect=10.0),
            )
            data = r.json()
            if "error" in data and (
                "is running" in data["error"]
                or "busy" in str(data.get("error", "")).lower()
            ):
                raise _ServerBusyError(
                    f"Server {current.name} busy: {data['error']}"
                )
            r.raise_for_status()
            # Extract nested key like "message.content"
            result: Any = data
            for k in key.split("."):
                result = result[k]
            return result

        def _on_retry(exc: Exception, attempt: int, wait: float) -> None:
            current = state["current"]
            alt = state["alternate"]
            if isinstance(exc, _ServerBusyError):
                logger.debug(
                    "Server %s busy, retrying in %.0fs (attempt %d)",
                    current.name,
                    poll_interval,
                    attempt,
                )
                # Alternate servers on even attempts
                if alt and attempt % 2 == 0:
                    state["current"] = alt
            elif isinstance(exc, httpx.ConnectError):
                if alt:
                    logger.warning(
                        "Server %s unreachable, failing over to %s",
                        current.name,
                        alt.name,
                    )
                    state["current"] = alt
                    state["alternate"] = None

        return retry_with_backoff(
            _do_request,
            max_retries=max_retries,
            base_delay=poll_interval,
            max_delay=poll_interval,  # constant interval for busy polling
            retry_on=(_ServerBusyError, httpx.ConnectError),
            on_retry=_on_retry,
        )

    def _get_alternate(self, server: OllamaServer) -> OllamaServer | None:
        """Get alternate server for failover."""
        others = [s for s in self.servers.values() if s.name != server.name]
        return others[0] if others else None


# ---------------------------------------------------------------------------
# Tool schemas for agent integration
# ---------------------------------------------------------------------------


def get_tool_schemas() -> list[dict]:
    """Return JSON schemas for local LLM tools."""
    return [
        {
            "name": "local_llm_generate",
            "description": (
                "Generate text using a local Ollama LLM server. "
                "Uses /api/generate for single-turn completion."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The text prompt to send to the model.",
                    },
                    "system": {
                        "type": "string",
                        "description": "System prompt (optional).",
                        "default": "",
                    },
                    "server": {
                        "type": "string",
                        "description": "Server name or 'auto' for automatic selection. Default: auto.",
                        "default": "auto",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model name. If omitted, auto-selected from server.",
                    },
                    "hint": {
                        "type": "string",
                        "description": "Pattern to match model name (regex, case-insensitive).",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Generation temperature (0.0-2.0).",
                        "default": 0.7,
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to generate.",
                        "default": 4096,
                    },
                    "think": {
                        "type": "string",
                        "description": "Thinking effort (off|low|medium|high).",
                        "default": "off",
                    },
                },
                "required": ["prompt"],
            },
        },
        {
            "name": "local_llm_chat",
            "description": (
                "Chat with a local Ollama LLM server. "
                "Uses /api/chat for multi-turn conversation."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "enum": ["user", "assistant", "system"],
                                },
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                        "description": "Chat message history.",
                    },
                    "system": {
                        "type": "string",
                        "description": "System prompt (optional).",
                        "default": "",
                    },
                    "server": {
                        "type": "string",
                        "description": "Server name or 'auto' for automatic selection. Default: auto.",
                        "default": "auto",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model name. If omitted, auto-selected from server.",
                    },
                    "hint": {
                        "type": "string",
                        "description": "Pattern to match model name (regex, case-insensitive).",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Generation temperature (0.0-2.0).",
                        "default": 0.7,
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to generate.",
                        "default": 4096,
                    },
                    "think": {
                        "type": "string",
                        "description": "Thinking effort (off|low|medium|high).",
                        "default": "off",
                    },
                },
                "required": ["messages"],
            },
        },
        {
            "name": "local_llm_models",
            "description": "List available models on the selected Ollama server.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "Server name or 'auto' for automatic selection. Default: auto.",
                        "default": "auto",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "local_llm_status",
            "description": "Get status (loaded models, latency) from all Ollama servers.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    ]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def get_cli_guide() -> str:
    """Return CLI usage guide for local LLM tools."""
    return """\
### ローカルLLM (Ollama)
```bash
animaworks-tool local_llm generate "プロンプト"
animaworks-tool local_llm chat '{"messages":[{"role":"user","content":"質問"}]}'
animaworks-tool local_llm list -j
animaworks-tool local_llm status -j
```"""


def cli_main(argv: list[str] | None = None) -> None:
    """Standalone CLI for local LLM operations."""
    parser = argparse.ArgumentParser(
        prog="animaworks-llm",
        description="AnimaWorks local LLM tool -- Ollama API client.",
    )
    parser.add_argument(
        "-s", "--server", default="auto",
        help="Server name or 'auto' for automatic selection. Default: auto.",
    )
    parser.add_argument(
        "-m", "--model", default=None,
        help="Model name. If omitted, auto-selected from server.",
    )
    parser.add_argument(
        "--hint", default=None,
        help="Pattern to match model name (regex, case-insensitive).",
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0,
        help="Request timeout in seconds (default: 600).",
    )

    sub = parser.add_subparsers(dest="command")

    # generate
    p_gen = sub.add_parser("generate", help="Generate text (single-turn)")
    p_gen.add_argument("prompt", nargs="?", help="Prompt text (reads stdin if omitted)")
    p_gen.add_argument("-S", "--system", default="", help="System prompt")
    p_gen.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    p_gen.add_argument("--max-tokens", type=int, default=4096, help="Max tokens")
    p_gen.add_argument(
        "--think", default="off", choices=["off", "low", "medium", "high"],
        help="Thinking effort",
    )

    # chat
    p_chat = sub.add_parser("chat", help="Chat (multi-turn, JSON input)")
    p_chat.add_argument(
        "json_input", nargs="?",
        help='JSON messages array or file path (reads stdin if omitted)',
    )
    p_chat.add_argument("-S", "--system", default="", help="System prompt")
    p_chat.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    p_chat.add_argument("--max-tokens", type=int, default=4096, help="Max tokens")
    p_chat.add_argument(
        "--think", default="off", choices=["off", "low", "medium", "high"],
        help="Thinking effort",
    )

    # list
    sub.add_parser("list", help="List available models")

    # status
    sub.add_parser("status", help="Show server status")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    client = OllamaClient(
        server=args.server,
        model=args.model,
        hint=args.hint,
        timeout=args.timeout,
    )

    if args.command == "generate":
        prompt = args.prompt
        if prompt is None:
            if sys.stdin.isatty():
                print("Enter prompt (Ctrl-D to finish):", file=sys.stderr)
            prompt = sys.stdin.read().strip()
        if not prompt:
            print("Error: no prompt provided", file=sys.stderr)
            sys.exit(1)
        result = client.generate(
            prompt=prompt,
            system=args.system,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            think=args.think,
        )
        print(result)

    elif args.command == "chat":
        raw = args.json_input
        if raw is None:
            if sys.stdin.isatty():
                print("Enter JSON messages (Ctrl-D to finish):", file=sys.stderr)
            raw = sys.stdin.read().strip()
        # Try as file path
        from pathlib import Path

        if raw and Path(raw).is_file():
            raw = Path(raw).read_text(encoding="utf-8")
        try:
            messages = json.loads(raw)
        except json.JSONDecodeError:
            print("Error: invalid JSON input", file=sys.stderr)
            sys.exit(1)
        if not isinstance(messages, list):
            # Support {"messages": [...]} wrapper
            if isinstance(messages, dict) and "messages" in messages:
                messages = messages["messages"]
            else:
                print("Error: expected a JSON array of messages", file=sys.stderr)
                sys.exit(1)
        result = client.chat(
            messages=messages,
            system=args.system,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            think=args.think,
        )
        print(result)

    elif args.command == "list":
        models = client.list_models()
        if not models:
            print("No models available.")
            return
        for m in models:
            name = m.get("name", "?")
            details = m.get("details", {})
            param_size = details.get("parameter_size", "?")
            quant = details.get("quantization_level", "?")
            print(f"  {name}\t({param_size}, {quant})")

    elif args.command == "status":
        status = client.server_status()
        for name, info in status.items():
            print(f"=== {name} ===")
            if "error" in info:
                print(f"  Status: unreachable ({info['error']})")
            else:
                models = info.get("models", [])
                print(f"  Status: OK")
                print(f"  Loaded models: {len(models)}")
                for m in models:
                    vram_gb = m.get("size_vram", 0) / (1024**3)
                    print(f"  - {m.get('name', '?')} (VRAM: {vram_gb:.0f}GB)")
            print()


# ── Dispatch ──────────────────────────────────────────


def dispatch(tool_name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool call to the appropriate handler."""
    if tool_name == "local_llm_generate":
        client = OllamaClient(
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
    if tool_name == "local_llm_chat":
        client = OllamaClient(
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
    if tool_name == "local_llm_models":
        client = OllamaClient(server=args.get("server", "auto"))
        return client.list_models()
    if tool_name == "local_llm_status":
        client = OllamaClient()
        return client.server_status()
    raise ValueError(f"Unknown tool: {tool_name}")


if __name__ == "__main__":
    cli_main()