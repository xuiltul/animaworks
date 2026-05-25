from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""LoCoMo answer LLM resolution — LiteLLM proxy on trserveru (vllm-lb)."""

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ANSWER_MODEL = "deepseek-v4-flash"
DEFAULT_LLM_CREDENTIAL = "vllm-lb"
_HOST_CONFIG_PATH = Path.home() / ".animaworks" / "config.json"


def _load_host_config() -> dict[str, Any]:
    """Load real host config (not LoCoMo temp ``ANIMAWORKS_DATA_DIR``)."""
    try:
        if _HOST_CONFIG_PATH.is_file():
            data = json.loads(_HOST_CONFIG_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        logger.debug("Could not read host config at %s", _HOST_CONFIG_PATH, exc_info=True)
    return {}


def _host_credential(name: str) -> dict[str, Any]:
    creds = _load_host_config().get("credentials", {})
    if not isinstance(creds, dict):
        return {}
    cred = creds.get(name)
    return cred if isinstance(cred, dict) else {}


def default_answer_model() -> str:
    """Default LoCoMo answer model (env → consolidation config → fallback)."""
    env_model = os.environ.get("LOCOMO_ANSWER_MODEL", "").strip()
    if env_model:
        return env_model
    cfg_model = str(_load_host_config().get("consolidation", {}).get("llm_model", "") or "").strip()
    if cfg_model:
        if cfg_model.startswith("openai/"):
            return cfg_model.split("/", 1)[1]
        return cfg_model
    return DEFAULT_ANSWER_MODEL


def default_llm_credential() -> str:
    """Credential name for OpenAI-compatible LiteLLM proxy."""
    env_cred = os.environ.get("LOCOMO_LLM_CREDENTIAL", "").strip()
    if env_cred:
        return env_cred
    cfg_cred = _load_host_config().get("consolidation", {}).get("llm_credential")
    if isinstance(cfg_cred, str) and cfg_cred.strip():
        return cfg_cred.strip()
    return DEFAULT_LLM_CREDENTIAL


def resolve_locomo_litellm_kwargs(model: str) -> tuple[str, dict[str, Any]]:
    """Resolve ``(litellm_model, kwargs)`` for LoCoMo answer generation.

    Resolution order:
      1. ``OPENAI_API_BASE`` / ``OPENAI_API_KEY`` env override
      2. ``LOCOMO_LLM_CREDENTIAL`` or ``config.consolidation.llm_credential`` (default ``vllm-lb``)
      3. Bare model + ``api_base`` → ``openai/{model}`` via ``get_memory_llm_kwargs_for_model``
    """
    from core.memory._llm_utils import get_memory_llm_kwargs_for_model

    model_lower = model.lower()
    extras: dict[str, Any] = {}
    if "qwen" in model_lower:
        extras["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    env_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_base:
        merged = {"api_base": env_base, "api_key": env_key or "dummy", **extras}
        kwargs = get_memory_llm_kwargs_for_model(model, merged)
        litellm_model = str(kwargs.pop("model"))
        return litellm_model, kwargs

    cred = _host_credential(default_llm_credential())
    base_url = str(cred.get("base_url") or "").strip()
    if not base_url:
        raise RuntimeError(
            f"LoCoMo LLM credential {default_llm_credential()!r} has no base_url in {_HOST_CONFIG_PATH}",
        )
    merged = {
        "api_base": base_url,
        "api_key": cred.get("api_key") or "dummy",
        **extras,
    }
    kwargs = get_memory_llm_kwargs_for_model(model, merged)
    litellm_model = str(kwargs.pop("model"))
    return litellm_model, kwargs


def llm_routing_configured(model: str | None = None) -> bool:
    """Return True when answer LLM routing can be resolved."""
    if os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL"):
        return True
    try:
        _, kwargs = resolve_locomo_litellm_kwargs(model or default_answer_model())
    except Exception:
        return False
    return bool(kwargs.get("api_base"))
