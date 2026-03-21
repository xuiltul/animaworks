# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Shared LLM helper utilities for memory-management modules."""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any

logger = logging.getLogger(__name__)

_ANTHROPIC_MODEL_RE = re.compile(
    r"^(anthropic/|bedrock/|vertex_ai/)?"
    r"([a-z]{2}\.anthropic\.)?"
    r"claude-",
)

# ── Provider → environment variable mapping ─────────────────────────────────

_PROVIDER_ENV_MAP: dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "google": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

_credentials_exported: bool = False
_credentials_lock = threading.Lock()


# ── Credential export for LiteLLM ───────────────────────────────────────────


def ensure_credentials_in_env() -> None:
    """Export config.json credentials to environment variables for LiteLLM auto-detection.

    Runs at most once per process.  Thread-safe via double-checked locking.
    Silently returns if config loading fails.
    """
    global _credentials_exported
    if _credentials_exported:
        return
    with _credentials_lock:
        if _credentials_exported:
            return

        try:
            from core.config import load_config

            cfg = load_config()
        except Exception:
            return

        for provider, cred in cfg.credentials.items():
            if not cred.api_key:
                continue
            env_key = _PROVIDER_ENV_MAP.get(provider)
            if env_key is None:
                continue
            if not os.environ.get(env_key):
                os.environ[env_key] = cred.api_key
                logger.debug("Exported credential for %s to %s", provider, env_key)

        _credentials_exported = True


# ── Consolidation LLM kwargs ─────────────────────────────────────────────────


def get_consolidation_llm_kwargs() -> dict[str, Any]:
    """Build kwargs for consolidation LLM calls (model, api_key, etc.).

    Ensures credentials are exported to env first, then resolves the consolidation
    model and its API key from config or environment.

    Returns:
        Dict with at least "model" key; "api_key" included when resolved.
    """
    return get_llm_kwargs_for_model("")


def _get_provider_for_model(model: str) -> str:
    if not model or "/" not in model:
        return ""
    return model.split("/", 1)[0].lower()


def _get_api_key_for_provider(cfg: Any, provider: str) -> str | None:
    cred = cfg.credentials.get(provider) if provider else None
    api_key = cred.api_key if cred else None
    if not api_key and provider:
        env_key = _PROVIDER_ENV_MAP.get(provider)
        if env_key:
            api_key = os.environ.get(env_key) or None
    return api_key


def get_llm_kwargs_for_model(model: str) -> dict[str, Any]:
    """Resolve LiteLLM kwargs for the requested model or the consolidation default."""
    ensure_credentials_in_env()

    from core.config import load_config

    cfg = load_config()
    resolved_model = model or cfg.consolidation.llm_model
    kwargs: dict[str, Any] = {"model": resolved_model}

    provider = _get_provider_for_model(resolved_model)
    cred = cfg.credentials.get(provider) if provider else None

    if provider == "ollama":
        base_url = None
        if cred and cred.base_url:
            base_url = cred.base_url
        elif getattr(cfg, "local_llm", None):
            base_url = getattr(cfg.local_llm, "base_url", None)
        if base_url:
            kwargs["api_base"] = base_url
        return kwargs

    api_key = _get_api_key_for_provider(cfg, provider)
    if api_key:
        kwargs["api_key"] = api_key
    if cred and cred.base_url:
        kwargs["api_base"] = cred.base_url

    return kwargs


# ── One-shot completion with fallback ─────────────────────────────────────────


def _is_anthropic_model(model: str) -> bool:
    """Return True if *model* is an Anthropic Claude model (any provider prefix)."""
    return bool(_ANTHROPIC_MODEL_RE.match(model))


def _is_codex_model(model: str) -> bool:
    """Return True if *model* is a Codex model routed via the Codex CLI."""
    return model.startswith("codex/")


def _strip_provider_prefix(model: str) -> str:
    """Strip LiteLLM provider prefix for Agent SDK (Anthropic-native name).

    Examples::

        anthropic/claude-sonnet-4-6 → claude-sonnet-4-6
        bedrock/jp.anthropic.claude-sonnet-4-6 → claude-sonnet-4-6
        vertex_ai/claude-sonnet-4-6 → claude-sonnet-4-6
    """
    return re.sub(
        r"^(anthropic|bedrock|vertex_ai)/([a-z]{2}\.anthropic\.)?",
        "",
        model,
    )


def _build_sdk_env() -> dict[str, str]:
    """Build environment variables for Agent SDK subprocess (system-level auth).

    Uses ``config.anima_defaults.mode_s_auth`` for authentication mode.
    Mirrors the logic in ``AgentSDKExecutor._build_env()`` but without
    per-Anima context.
    """
    from core.config import load_config

    cfg = load_config()
    auth = cfg.anima_defaults.mode_s_auth
    extra = {}
    api_key = ""

    cred = cfg.credentials.get("anthropic")
    if cred:
        api_key = cred.api_key or ""
        extra = cred.keys if hasattr(cred, "keys") and cred.keys else {}

    env: dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    }

    if auth == "api":
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key
        else:
            env["ANTHROPIC_API_KEY"] = ""
    elif auth == "bedrock":
        env["ANTHROPIC_API_KEY"] = ""
        env["CLAUDE_CODE_USE_BEDROCK"] = "1"
        for env_key, extra_key in (
            ("AWS_ACCESS_KEY_ID", "aws_access_key_id"),
            ("AWS_SECRET_ACCESS_KEY", "aws_secret_access_key"),
            ("AWS_SESSION_TOKEN", "aws_session_token"),
            ("AWS_REGION", "aws_region_name"),
            ("AWS_PROFILE", "aws_profile"),
        ):
            val = extra.get(extra_key) or os.environ.get(env_key)
            if val:
                env[env_key] = val
    elif auth == "vertex":
        env["ANTHROPIC_API_KEY"] = ""
        env["CLAUDE_CODE_USE_VERTEX"] = "1"
        for env_key, extra_key in (
            ("CLOUD_ML_PROJECT_ID", "vertex_project"),
            ("CLOUD_ML_REGION", "vertex_location"),
            ("GOOGLE_APPLICATION_CREDENTIALS", "vertex_credentials"),
        ):
            val = extra.get(extra_key) or os.environ.get(env_key)
            if val:
                env[env_key] = val
    else:
        env["ANTHROPIC_API_KEY"] = ""

    if cred and cred.base_url:
        env["ANTHROPIC_BASE_URL"] = cred.base_url

    return env


async def _try_litellm(
    prompt: str,
    *,
    system_prompt: str,
    model: str,
    max_tokens: int,
    llm_kwargs: dict[str, Any],
) -> str | None:
    """Attempt LLM call via LiteLLM.  Returns text or None on failure."""
    import litellm

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kw = {k: v for k, v in llm_kwargs.items() if k != "model"}
    resp = await litellm.acompletion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        **kw,
    )
    return resp.choices[0].message.content or None  # type: ignore[union-attr]


async def _try_agent_sdk(
    prompt: str,
    *,
    system_prompt: str,
    model: str,
    max_tokens: int,
) -> str | None:
    """Attempt one-shot completion via Agent SDK.  Returns text or None."""
    try:
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
    except ImportError:
        try:
            from claude_code_sdk import ClaudeAgentOptions  # type: ignore[no-redef]
            from claude_code_sdk import ClaudeCodeSDKClient as ClaudeSDKClient  # type: ignore[no-redef,assignment]
        except ImportError:
            logger.debug("Agent SDK not available for one-shot fallback")
            return None

    sdk_model = _strip_provider_prefix(model)
    env = _build_sdk_env()

    options = ClaudeAgentOptions(
        model=sdk_model,
        system_prompt=system_prompt or "",
        allowed_tools=[],
        max_turns=1,
    )

    chunks: list[str] = []
    try:
        sdk_kwargs: dict[str, Any] = {"options": options}
        try:
            async with ClaudeSDKClient(**sdk_kwargs, env=env) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    if hasattr(message, "content"):
                        for block in message.content:
                            if hasattr(block, "text"):
                                chunks.append(block.text)
        except TypeError:
            async with ClaudeSDKClient(**sdk_kwargs) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    if hasattr(message, "content"):
                        for block in message.content:
                            if hasattr(block, "text"):
                                chunks.append(block.text)
    except Exception as e:
        logger.warning("Agent SDK one-shot failed: %s", e)
        return None

    return "".join(chunks) or None


async def _try_codex_sdk(
    prompt: str,
    *,
    system_prompt: str,
    model: str,
    max_tokens: int,
    llm_kwargs: dict[str, Any],
) -> str | None:
    """Attempt one-shot completion via Codex SDK."""
    del max_tokens

    try:
        from openai_codex_sdk import Codex
    except ImportError:
        logger.debug("Codex SDK not available for one-shot fallback")
        return None

    from core.execution.codex_sdk import _default_path_env, _resolve_codex_model
    from core.platform.codex import default_home_dir, get_codex_executable

    env: dict[str, str] = {
        "PATH": _default_path_env(),
        "HOME": default_home_dir(),
    }
    if llm_kwargs.get("api_key"):
        env["OPENAI_API_KEY"] = str(llm_kwargs["api_key"])
    if llm_kwargs.get("api_base"):
        env["OPENAI_BASE_URL"] = str(llm_kwargs["api_base"])

    options: dict[str, Any] = {"env": env}
    executable = get_codex_executable()
    if executable:
        options["codexPathOverride"] = executable

    try:
        client = Codex(options)
        thread = client.start_thread(
            {
                "model": _resolve_codex_model(model),
                "sandboxMode": "read-only",
                "approvalPolicy": "never",
                "skipGitRepoCheck": True,
                "workingDirectory": os.getcwd(),
                "networkAccessEnabled": False,
            }
        )
        full_prompt = prompt if not system_prompt else f"{system_prompt}\n\nUser request:\n{prompt}"
        turn = await thread.run(full_prompt)
        return getattr(turn, "final_response", None) or None
    except Exception as e:
        logger.warning("Codex SDK one-shot failed: %s", e)
        return None


async def one_shot_completion(
    prompt: str,
    *,
    system_prompt: str = "",
    model: str = "",
    max_tokens: int = 2048,
) -> str | None:
    """Execute a one-shot LLM completion with automatic backend selection.

    Fallback chain:
      1. LiteLLM (if API key available) -- fast, no subprocess
      2. Agent SDK one-shot (if installed and Anthropic model) -- Max plan compatible
      3. Return None -- caller handles gracefully

    Args:
        prompt: User message content.
        system_prompt: Optional system prompt (used by conversation compression).
        model: LLM model identifier. Defaults to config.consolidation.llm_model.
        max_tokens: Maximum tokens for the response.

    Returns:
        Generated text, or None if all backends fail.
    """
    llm_kwargs = get_llm_kwargs_for_model(model)
    resolved_model = llm_kwargs["model"]

    # 1. Try LiteLLM
    try:
        result = await _try_litellm(
            prompt,
            system_prompt=system_prompt,
            model=resolved_model,
            max_tokens=max_tokens,
            llm_kwargs=llm_kwargs,
        )
        if result:
            return result
    except Exception as e:
        logger.warning("LiteLLM one-shot failed (%s), trying Agent SDK fallback", e)

    # 2. Try Agent SDK (Anthropic models only)
    if _is_anthropic_model(resolved_model):
        try:
            return await _try_agent_sdk(
                prompt,
                system_prompt=system_prompt,
                model=resolved_model,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.warning("Agent SDK one-shot fallback also failed: %s", e)

    if _is_codex_model(resolved_model):
        try:
            return await _try_codex_sdk(
                prompt,
                system_prompt=system_prompt,
                model=resolved_model,
                max_tokens=max_tokens,
                llm_kwargs=llm_kwargs,
            )
        except Exception as e:
            logger.warning("Codex SDK one-shot fallback also failed: %s", e)

    return None
