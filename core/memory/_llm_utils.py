# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Shared LLM helper utilities for memory-management modules."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from pathlib import Path
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


def get_llm_kwargs_for_model(model: str, *, credential: str = "") -> dict[str, Any]:
    """Resolve LiteLLM kwargs for the requested model or the consolidation default.

    Args:
        model: LiteLLM model identifier. Empty string falls back to
            ``config.consolidation.llm_model``.
        credential: Explicit credential name to use instead of provider-prefix
            resolution. When empty and *model* is also empty, falls back to
            ``config.consolidation.llm_credential`` if set.
    """
    ensure_credentials_in_env()

    from core.config import load_config

    cfg = load_config()
    resolved_model = model or cfg.consolidation.llm_model
    kwargs: dict[str, Any] = {"model": resolved_model}

    explicit_cred = credential
    consolidation_model = str(getattr(cfg.consolidation, "llm_model", "") or "")
    if not explicit_cred and not model:
        _llm_cred = getattr(cfg.consolidation, "llm_credential", None)
        explicit_cred = _llm_cred if isinstance(_llm_cred, str) and _llm_cred else ""
    elif not explicit_cred and model == consolidation_model:
        # Post-consolidation helpers pass the configured consolidation model
        # explicitly.  In vLLM/LiteLLM gateway deployments that model can still
        # have an ``openai/`` provider prefix, while the usable credential is a
        # custom configured credential such as ``vllm-lb``.  Treat an exact match
        # to the configured consolidation model the same as the implicit default.
        _llm_cred = getattr(cfg.consolidation, "llm_credential", None)
        explicit_cred = _llm_cred if isinstance(_llm_cred, str) and _llm_cred else ""
    elif not explicit_cred and model and "/" not in model:
        # Memory extraction historically allowed bare model names in status.json
        # (e.g. ``deepseek-v4-flash``).  When that bare name is the same model
        # as the configured consolidation helper, use the consolidation
        # credential instead of falling through to provider-prefix resolution.
        # Without this, LiteLLM treats the bare model as OpenAI-compatible but
        # no OpenAI credential exists in vLLM-backed deployments.
        consolidation_base = consolidation_model.split("/", 1)[-1]
        if consolidation_base == model:
            _llm_cred = getattr(cfg.consolidation, "llm_credential", None)
            explicit_cred = _llm_cred if isinstance(_llm_cred, str) and _llm_cred else ""

    if explicit_cred:
        cred = cfg.credentials.get(explicit_cred)
    else:
        provider = _get_provider_for_model(resolved_model)
        cred = cfg.credentials.get(provider) if provider else None

    provider = _get_provider_for_model(resolved_model)

    if provider == "ollama":
        base_url = None
        if cred and cred.base_url:
            base_url = cred.base_url
        elif getattr(cfg, "local_llm", None):
            base_url = getattr(cfg.local_llm, "base_url", None)
        if base_url:
            kwargs["api_base"] = base_url
        return kwargs

    if explicit_cred:
        api_key = cred.api_key if cred else None
    else:
        api_key = _get_api_key_for_provider(cfg, provider)
    if api_key:
        kwargs["api_key"] = api_key
    if cred and cred.base_url:
        kwargs["api_base"] = cred.base_url

    return kwargs


def get_memory_llm_kwargs_for_model(
    model: str,
    llm_extra: dict[str, object] | None = None,
    *,
    credential: str = "",
) -> dict[str, Any]:
    """Resolve LiteLLM kwargs for memory LLM calls.

    Memory extraction can be configured per Anima with an OpenAI-compatible
    custom endpoint. LiteLLM still requires a provider prefix for bare model
    ids in that case, so ``deepseek-v4-flash`` + ``api_base`` is normalized to
    ``openai/deepseek-v4-flash``. Existing provider-prefixed ids are preserved.
    """
    kwargs = get_llm_kwargs_for_model(model, credential=credential)
    if llm_extra:
        kwargs.update(llm_extra)

    resolved_model = str(kwargs.get("model") or model or "")
    if resolved_model and "/" not in resolved_model and kwargs.get("api_base"):
        kwargs["model"] = f"openai/{resolved_model}"

    return kwargs


def get_llm_kwargs_for_model_config(model_config: Any) -> dict[str, Any]:
    """Resolve LiteLLM kwargs from the active per-Anima ModelConfig."""
    ensure_credentials_in_env()

    resolved_model = getattr(model_config, "model", "") or ""
    kwargs = get_llm_kwargs_for_model(resolved_model)
    kwargs["model"] = resolved_model

    api_key = getattr(model_config, "api_key", None)
    api_key_env = getattr(model_config, "api_key_env", "")
    if not api_key and api_key_env:
        api_key = os.environ.get(api_key_env) or None
    if api_key:
        kwargs["api_key"] = api_key

    api_base_url = getattr(model_config, "api_base_url", None)
    if api_base_url:
        kwargs["api_base"] = api_base_url

    extra = getattr(model_config, "extra_keys", None) or {}
    model = kwargs["model"]
    if model.startswith("azure/"):
        api_version = extra.get("api_version") or os.environ.get("AZURE_API_VERSION")
        if api_version:
            kwargs["api_version"] = api_version
    elif model.startswith("vertex_ai/"):
        for key in ("vertex_project", "vertex_location", "vertex_credentials"):
            val = extra.get(key) or os.environ.get(key.upper())
            if val:
                kwargs[key] = val
    elif model.startswith("bedrock/"):
        for key in (
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "aws_region_name",
            "aws_profile",
        ):
            val = extra.get(key) or os.environ.get(key.upper())
            if val:
                kwargs[key] = val

    return kwargs


# ── One-shot completion with fallback ─────────────────────────────────────────


def _is_anthropic_model(model: str) -> bool:
    """Return True if *model* is an Anthropic Claude model (any provider prefix)."""
    return bool(_ANTHROPIC_MODEL_RE.match(model))


def _is_codex_model(model: str) -> bool:
    """Return True if *model* is a Codex model routed via the Codex CLI."""
    return model.startswith("codex/") or model.startswith("openai-codex/")


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
        # サブスクリプションログイン（~/.claude）の解決にHOMEが必要
        "HOME": os.environ.get("HOME", str(Path.home())),
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
        # max: サブスクリプションログインを使う。ANTHROPIC_API_KEYは「空でも設定されている」と
        # CLIがログインより優先して invalid key になるため、キー自体を渡さない（2026-07-03）。
        env.pop("ANTHROPIC_API_KEY", None)

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

    from core.execution._sdk_options import _resolve_sdk_cli_path

    _cli = _resolve_sdk_cli_path()
    # env は ClaudeAgentOptions.env で渡す（SDKが継承環境に上書きマージする）。
    # 旧実装の ClaudeSDKClient(env=...) は存在しないkwargでTypeError→envなしに
    # フォールバックしており、_build_sdk_env が適用されていなかった（2026-07-03修正）。
    options_kwargs: dict[str, Any] = {
        "model": sdk_model,
        "system_prompt": system_prompt or "",
        "allowed_tools": [],
        "max_turns": 1,
        "env": env,
    }
    if _cli:
        options_kwargs["cli_path"] = _cli
    try:
        options = ClaudeAgentOptions(**options_kwargs)
    except TypeError:
        options_kwargs.pop("env", None)
        options = ClaudeAgentOptions(**options_kwargs)

    chunks: list[str] = []
    try:
        async with ClaudeSDKClient(options=options) as client:
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
        from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, Sandbox
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

    executable = get_codex_executable()
    config = CodexConfig(
        codex_bin=executable,
        cwd=os.getcwd(),
        env=env,
        client_name="animaworks",
        client_title="AnimaWorks",
    )

    try:
        client = AsyncCodex(config)
        thread = await client.thread_start(
            approval_mode=ApprovalMode.deny_all,
            base_instructions=system_prompt or None,
            cwd=os.getcwd(),
            model=_resolve_codex_model(model),
            sandbox=Sandbox.read_only,
        )
        turn = await thread.run(
            prompt,
            approval_mode=ApprovalMode.deny_all,
            cwd=os.getcwd(),
            model=_resolve_codex_model(model),
            sandbox=Sandbox.read_only,
        )
        return getattr(turn, "final_response", None) or None
    except Exception as e:
        logger.warning("Codex SDK one-shot failed: %s", e)
        return None
    finally:
        if "client" in locals():
            try:
                await client.close()
            except Exception:
                logger.debug("Failed to close Codex one-shot client", exc_info=True)


# Inline retry stays on the live-response budget: a wait longer than this
# (e.g. a large Retry-After) skips the same-process retry and falls back
# immediately, while the full block is still reported to the fleet guard.
_MAX_INLINE_RETRY_WAIT_S = 15.0


async def _litellm_stage_with_guard(
    prompt: str,
    *,
    system_prompt: str,
    resolved_model: str,
    max_tokens: int,
    llm_kwargs: dict[str, Any],
    log_prefix: str,
    guard: Any,
    family: str,
) -> tuple[str, str | None]:
    """Run the LiteLLM one-shot stage under the fleet rate guard.

    Returns ``(outcome, text)`` where ``outcome`` is:
      - ``"success"`` — ``text`` is the completion.
      - ``"terminal"`` — caller must return ``None`` (content-policy block; no
        fallback, see Key Decision 7).
      - ``"fallback"`` — caller proceeds to the next backend (current
        blind-fallback behavior; used for skipped/blocked/rate/unknown paths).
    """
    from core.execution.backoff import decorrelated_jitter
    from core.execution.error_classifier import FailoverReason, classify_llm_error

    blocked = guard.blocked_remaining(family)
    if blocked > 0:
        logger.info("%s LiteLLM skipped: %s rate-guarded for %.0fs", log_prefix, family, blocked)
        return "fallback", None

    try:
        result = await _try_litellm(
            prompt,
            system_prompt=system_prompt,
            model=resolved_model,
            max_tokens=max_tokens,
            llm_kwargs=llm_kwargs,
        )
        if result:
            return "success", result
        return "fallback", None
    except Exception as exc:
        error = exc
        reason, hint = classify_llm_error(exc, provider_family=family)

    if reason is FailoverReason.CONTENT_POLICY:
        logger.warning("%s LiteLLM one-shot content-policy block, not falling back: %s", log_prefix, error)
        return "terminal", None

    if reason in (FailoverReason.RATE_LIMIT, FailoverReason.OVERLOADED):
        cfg = guard.config
        # Report the full block to the fleet (report_block clamps to the max).
        guard.report_block(family, hint.backoff_s or cfg.default_block_seconds, reason.value)
        # Compute the wait once; retry in-process only when it fits the live
        # budget, otherwise fall back immediately.
        wait_s = hint.backoff_s if hint.backoff_s is not None else decorrelated_jitter(0.0)
        if hint.retryable and wait_s <= _MAX_INLINE_RETRY_WAIT_S:
            await asyncio.sleep(wait_s)
            try:
                result = await _try_litellm(
                    prompt,
                    system_prompt=system_prompt,
                    model=resolved_model,
                    max_tokens=max_tokens,
                    llm_kwargs=llm_kwargs,
                )
                if result:
                    return "success", result
            except Exception as retry_exc:
                retry_reason, retry_hint = classify_llm_error(retry_exc, provider_family=family)
                if retry_reason in (FailoverReason.RATE_LIMIT, FailoverReason.OVERLOADED):
                    guard.report_block(
                        family,
                        retry_hint.backoff_s or cfg.default_block_seconds,
                        retry_reason.value,
                    )
                logger.warning(
                    "%s LiteLLM one-shot rate-limited on retry (%s), trying fallback",
                    log_prefix,
                    retry_reason.value,
                )
        else:
            logger.info(
                "%s LiteLLM one-shot %s; block %.0fs exceeds inline budget, trying fallback",
                log_prefix,
                reason.value,
                wait_s,
            )
        return "fallback", None

    if reason in (FailoverReason.AUTH, FailoverReason.BILLING):
        logger.error(
            "%s LiteLLM one-shot %s — human attention required: %s",
            log_prefix,
            reason.value,
            error,
        )
        return "fallback", None

    logger.warning("%s LiteLLM one-shot failed (%s: %s), trying fallback", log_prefix, reason.value, error)
    return "fallback", None


def _sdk_stage_guarded(guard: Any, family: str, log_prefix: str, backend: str) -> bool:
    """Return True when the SDK *backend* should be skipped (family guarded).

    Aligns the one-shot chain with the "all backends guarded → None" edge case:
    an Anthropic 429 must not be retried against the same shared credential via
    the Agent SDK, nor an OpenAI block via the Codex SDK.
    """
    blocked = guard.blocked_remaining(family)
    if blocked > 0:
        logger.info("%s %s skipped: %s rate-guarded for %.0fs", log_prefix, backend, family, blocked)
        return True
    return False


async def one_shot_completion(
    prompt: str,
    *,
    system_prompt: str = "",
    model: str = "",
    credential: str = "",
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
        credential: Optional credential name to use for the LiteLLM call.
        max_tokens: Maximum tokens for the response.

    Returns:
        Generated text, or None if all backends fail.
    """
    llm_kwargs = get_llm_kwargs_for_model(model, credential=credential)
    resolved_model = llm_kwargs["model"]

    from core.execution.error_classifier import provider_family_of
    from core.execution.rate_guard import get_rate_guard

    guard = get_rate_guard()
    family = provider_family_of(resolved_model)

    # 1. Try LiteLLM (under the fleet rate guard)
    outcome, text = await _litellm_stage_with_guard(
        prompt,
        system_prompt=system_prompt,
        resolved_model=resolved_model,
        max_tokens=max_tokens,
        llm_kwargs=llm_kwargs,
        log_prefix="one-shot",
        guard=guard,
        family=family,
    )
    if outcome == "success":
        return text
    if outcome == "terminal":
        return None

    # 2. Try Agent SDK (Anthropic models only), unless the family is guarded.
    if _is_anthropic_model(resolved_model) and not _sdk_stage_guarded(guard, family, "one-shot", "Agent SDK"):
        try:
            return await _try_agent_sdk(
                prompt,
                system_prompt=system_prompt,
                model=resolved_model,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.warning("Agent SDK one-shot fallback also failed: %s", e)

    if _is_codex_model(resolved_model) and not _sdk_stage_guarded(guard, family, "one-shot", "Codex SDK"):
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


async def one_shot_completion_with_model_config(
    prompt: str,
    *,
    system_prompt: str = "",
    model_config: Any,
    max_tokens: int = 2048,
) -> str | None:
    """Execute one-shot completion with the active Anima model configuration."""
    llm_kwargs = get_llm_kwargs_for_model_config(model_config)
    resolved_model = llm_kwargs["model"]

    from core.execution.error_classifier import provider_family_of
    from core.execution.rate_guard import get_rate_guard

    guard = get_rate_guard()
    family = provider_family_of(resolved_model)

    outcome, text = await _litellm_stage_with_guard(
        prompt,
        system_prompt=system_prompt,
        resolved_model=resolved_model,
        max_tokens=max_tokens,
        llm_kwargs=llm_kwargs,
        log_prefix="active-model one-shot",
        guard=guard,
        family=family,
    )
    if outcome == "success":
        return text
    if outcome == "terminal":
        return None

    if _is_anthropic_model(resolved_model) and not _sdk_stage_guarded(
        guard, family, "active-model one-shot", "Agent SDK"
    ):
        try:
            return await _try_agent_sdk(
                prompt,
                system_prompt=system_prompt,
                model=resolved_model,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.warning("Agent SDK active-model one-shot fallback also failed: %s", e)

    if _is_codex_model(resolved_model) and not _sdk_stage_guarded(
        guard, family, "active-model one-shot", "Codex SDK"
    ):
        try:
            return await _try_codex_sdk(
                prompt,
                system_prompt=system_prompt,
                model=resolved_model,
                max_tokens=max_tokens,
                llm_kwargs=llm_kwargs,
            )
        except Exception as e:
            logger.warning("Codex SDK active-model one-shot fallback also failed: %s", e)

    return None
