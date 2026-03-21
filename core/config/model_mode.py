# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Model execution mode resolution (S/C/D/G/A/B) from model name patterns."""

from __future__ import annotations

import fnmatch
import json
import logging

from core.config.schemas import AnimaWorksConfig

logger = logging.getLogger("animaworks.config")

# Default model_modes with wildcard pattern support.
# Patterns use fnmatch syntax (*, ?, [seq]).
# Order matters for specificity — more specific patterns should appear first,
# but the resolver sorts by specificity automatically.
#
# Mode values: S = SDK (Agent SDK / Claude Code), C = Codex (Codex CLI wrapper),
#              D = Cursor Agent CLI, G = Gemini CLI,
#              A = Autonomous (tool_use), B = Basic (no tool_use)
#
# IMPORTANT: When status.json omits "execution_mode", resolve_execution_mode()
# falls through to these patterns to determine the mode from the model name.
# For example, "claude-sonnet-4-6" matches "claude-*" → Mode S.
# An anima can override this by setting execution_mode explicitly in status.json
# (e.g. bedrock/* defaults to A, but mei uses execution_mode="S" to force Mode S).
DEFAULT_MODEL_MODE_PATTERNS: dict[str, str] = {
    # ── S: Claude Agent SDK ──────────────────────────────
    "claude-*": "S",
    # ── C: Codex SDK (Codex CLI wrapper) ─────────────────
    "codex/*": "C",
    # ── D: Cursor Agent CLI ──────────────────────────────
    "cursor/*": "D",
    # ── G: Gemini CLI ────────────────────────────────────
    "gemini/*": "G",
    # ── A: Cloud API providers (LiteLLM + tool_use) ──────
    "openai/*": "A",
    "azure/*": "A",
    "bedrock/*": "A",
    "google/*": "A",
    "vertex_ai/*": "A",
    "mistral/*": "A",
    "xai/*": "A",
    "cohere/*": "A",
    "zai/*": "A",
    "minimax/*": "A",
    "moonshot/*": "A",
    "deepseek/deepseek-chat": "A",
    # ── A: Ollama models with reliable tool_use ──────────
    "ollama/qwen3.5*": "A",
    "ollama/qwen3:14b": "A",
    "ollama/qwen3:30b": "A",
    "ollama/qwen3:32b": "A",
    "ollama/qwen3:235b": "A",
    "ollama/qwen3-coder:*": "A",
    "ollama/llama4:*": "A",
    "ollama/mistral-small3.2:*": "A",
    "ollama/devstral*": "A",
    "ollama/glm-4.7*": "A",
    "ollama/glm-5*": "A",
    "ollama/minimax*": "A",
    "ollama/kimi-k2*": "A",
    "ollama/gpt-oss*": "A",
    # ── B: No reliable tool_use ───────────────────────────
    "ollama/qwen3:0.6b": "B",
    "ollama/qwen3:1.7b": "B",
    "ollama/qwen3:4b": "B",
    "ollama/qwen3:8b": "B",
    "ollama/gemma3*": "B",
    "ollama/deepseek-r1*": "B",
    "ollama/deepseek-v3*": "B",
    "ollama/phi4*": "B",
    "ollama/*": "B",
}

# Backward-compatible alias
DEFAULT_MODEL_MODES = DEFAULT_MODEL_MODE_PATTERNS

# ── Known model catalog ──────────────────────────────────────────────────────
# Concrete model names for reference. Used by SUPERVISOR_TOOLS description.
# Mode is determined by DEFAULT_MODEL_MODE_PATTERNS at runtime; this list is
# informational and does NOT restrict which models can be used.
KNOWN_MODELS: list[dict[str, str]] = [
    # ── Claude / Anthropic (Mode S) ──────────────────────────────────────────
    {"name": "claude-opus-4-6", "mode": "S", "note": "最高性能・推奨"},
    {"name": "claude-sonnet-4-6", "mode": "S", "note": "バランス型・推奨"},
    {"name": "claude-haiku-4-5-20251001", "mode": "S", "note": "軽量・高速"},
    # Legacy (still available)
    {"name": "claude-opus-4-5-20251101", "mode": "S", "note": "旧フラッグシップ"},
    {"name": "claude-opus-4-1-20250805", "mode": "S", "note": "旧Opus"},
    {"name": "claude-sonnet-4-5-20250929", "mode": "S", "note": "旧Sonnet"},
    {"name": "claude-sonnet-4-20250514", "mode": "S", "note": "旧Sonnet4"},
    # ── OpenAI (Mode A) ──────────────────────────────────────────────────────
    {"name": "openai/gpt-4.1", "mode": "A", "note": "最新・コーディング強"},
    {"name": "openai/gpt-4.1-mini", "mode": "A", "note": "高速・低コスト"},
    {"name": "openai/gpt-4.1-nano", "mode": "A", "note": "最軽量"},
    {"name": "openai/gpt-4o", "mode": "A", "note": "音声対応・レガシー"},
    {"name": "openai/o3-2025-04-16", "mode": "A", "note": "推論特化"},
    {"name": "openai/o4-mini-2025-04-16", "mode": "A", "note": "推論・低コスト"},
    # ── Azure OpenAI (Mode A) ──────────────────────────────────────────────────
    {"name": "azure/gpt-4.1-mini", "mode": "A", "note": "Azure OpenAI 4.1-mini"},
    {"name": "azure/gpt-4.1", "mode": "A", "note": "Azure OpenAI 4.1"},
    # ── Google Gemini (Mode A) ────────────────────────────────────────────────
    {"name": "google/gemini-2.5-pro", "mode": "A", "note": "最高性能"},
    {"name": "google/gemini-2.5-flash", "mode": "A", "note": "高速バランス"},
    {"name": "google/gemini-2.5-flash-lite", "mode": "A", "note": "軽量・高スループット"},
    # ── Vertex AI (Mode A) ────────────────────────────────────────────────────
    {"name": "vertex_ai/gemini-2.5-flash", "mode": "A", "note": "Vertex AI Flash"},
    {"name": "vertex_ai/gemini-2.5-pro", "mode": "A", "note": "Vertex AI Pro"},
    # ── xAI Grok (Mode A) ─────────────────────────────────────────────────────
    {"name": "xai/grok-4", "mode": "A", "note": "最新Grok"},
    {"name": "xai/grok-3-beta", "mode": "A", "note": "安定版"},
    {"name": "xai/grok-3-mini-beta", "mode": "A", "note": "軽量Grok"},
    # ── Ollama Local (Mode A: tool_use 対応) ─────────────────────────────────
    {"name": "ollama/qwen3.5:9b", "mode": "A", "note": "GDN hybrid 9B・高効率"},
    {"name": "ollama/glm-4.7", "mode": "A", "note": "ローカル・tool_use対応"},
    {"name": "ollama/qwen3:14b", "mode": "A", "note": "ローカル中型"},
    {"name": "ollama/qwen3:32b", "mode": "A", "note": "ローカル大型"},
    # ── Codex (Mode C) ──────────────────────────────────────────────────────
    {"name": "codex/o4-mini", "mode": "C", "note": "Codex CLI経由・高速"},
    {"name": "codex/o3", "mode": "C", "note": "Codex CLI経由・推論"},
    {"name": "codex/gpt-4.1", "mode": "C", "note": "Codex CLI経由・コーディング"},
    # ── Ollama Local (Mode B: tool_use 非対応) ────────────────────────────────
    {"name": "ollama/gemma3:4b", "mode": "B", "note": "軽量ローカル"},
    {"name": "ollama/gemma3:12b", "mode": "B", "note": "中型ローカル"},
]

# ── Legacy mode value mapping ──────────────────────────────
# Maps legacy A1/A1F/A2 and text-based values to canonical S/C/D/G/A/B scheme.
_LEGACY_MODE_MAP: dict[str, str] = {
    "autonomous": "A",
    "assisted": "B",
    "a1": "S",
    "a1f": "A",
    "a1_fallback": "A",
    "a2": "A",
}

# ── models.json cache ─────────────────────────────────────
_models_json_cache: dict[str, dict] | None = None
_models_json_mtime: float = 0.0


def _load_models_json() -> dict[str, dict]:
    """Load the user-editable models.json from the runtime data directory.

    Reads ``~/.animaworks/models.json`` (resolved via
    ``core.paths.get_data_dir``).  The result is cached at module level and
    automatically reloaded when the file's mtime changes.

    Returns:
        A dict mapping model-name patterns to entry dicts containing
        ``"mode"`` and ``"context_window"`` keys.  Returns an empty dict
        if the file is missing or cannot be parsed.
    """
    global _models_json_cache, _models_json_mtime

    from core.paths import get_data_dir

    models_path = get_data_dir() / "models.json"

    # Fast path: return cache if mtime unchanged
    if _models_json_cache is not None:
        try:
            disk_mtime = models_path.stat().st_mtime
        except OSError:
            disk_mtime = 0.0
        if disk_mtime == _models_json_mtime:
            return _models_json_cache

    # Capture mtime before reading to avoid TOCTOU race
    try:
        file_mtime = models_path.stat().st_mtime
    except OSError:
        logger.debug("models.json not found at %s; skipping", models_path)
        _models_json_cache = {}
        _models_json_mtime = 0.0
        return _models_json_cache

    try:
        raw = json.loads(models_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load models.json from %s: %s", models_path, exc)
        _models_json_cache = {}
        _models_json_mtime = 0.0
        return _models_json_cache

    if not isinstance(raw, dict):
        logger.warning("models.json is not a JSON object; ignoring")
        _models_json_cache = {}
        _models_json_mtime = 0.0
        return _models_json_cache

    # Filter to entries that are dicts (skip comment keys, etc.)
    result: dict[str, dict] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            result[key] = value

    _models_json_cache = result
    _models_json_mtime = file_mtime

    logger.debug("Loaded models.json with %d entries", len(result))
    return _models_json_cache


def invalidate_models_json_cache() -> None:
    """Reset the models.json module-level cache."""
    global _models_json_cache, _models_json_mtime
    _models_json_cache = None
    _models_json_mtime = 0.0


def _pattern_specificity(pattern: str) -> tuple[int, int, int]:
    """Return a sort key so more-specific patterns match first.

    Ranking (lower = matched first):
      - Exact match (no wildcard chars): (0, 0, -len)
      - Wildcard pattern: (1, -prefix_len, -total_len)
        where prefix_len is the length before the first wildcard char.
    """
    if not any(c in pattern for c in ("*", "?", "[")):
        # Exact match — highest priority
        return (0, 0, -len(pattern))
    # Find prefix length before first wildcard character
    prefix_len = len(pattern)
    for i, ch in enumerate(pattern):
        if ch in ("*", "?", "["):
            prefix_len = i
            break
    return (1, -prefix_len, -len(pattern))


def _match_pattern_table(
    model_name: str,
    table: dict[str, str],
) -> str | None:
    """Match *model_name* against a pattern table.

    Phase 1: O(1) exact dict lookup.
    Phase 2: fnmatch scan in specificity-descending order.

    Returns the mode string (e.g. ``"S"``) or ``None`` if no match.
    """
    # Phase 1: exact match
    if model_name in table:
        return table[model_name].upper()

    # Phase 2: wildcard patterns sorted by specificity
    wildcard_patterns = [p for p in table if any(c in p for c in ("*", "?", "["))]
    wildcard_patterns.sort(key=_pattern_specificity)

    for pattern in wildcard_patterns:
        if fnmatch.fnmatch(model_name, pattern):
            return table[pattern].upper()

    return None


def _normalise_mode(raw: str) -> str:
    """Normalise a mode value to S/C/D/G/A/B, applying legacy mapping if needed.

    Accepts legacy values (``"A1"``, ``"A2"``, ``"autonomous"``, etc.) and
    canonical values (``"S"``, ``"C"``, ``"D"``, ``"G"``, ``"A"``, ``"B"``).
    """
    lower = raw.strip().lower()
    mapped = _LEGACY_MODE_MAP.get(lower)
    if mapped:
        return mapped
    upper = raw.strip().upper()
    if upper in ("S", "C", "A", "B", "D", "G"):
        return upper
    # Unrecognised — return as-is (upper) for forward compat
    logger.warning("Unrecognised execution mode '%s'; passing through as '%s'", raw, upper)
    return upper


def _match_models_json(model_name: str) -> dict | None:
    """Match *model_name* against models.json entries.

    Returns the matched entry dict (with ``"mode"`` and ``"context_window"``
    keys) or ``None`` if no match.  Uses specificity-sorted pattern matching.
    """
    table = _load_models_json()
    if not table:
        return None

    # Phase 1: exact match
    if model_name in table:
        return table[model_name]

    # Phase 2: wildcard patterns sorted by specificity
    wildcard_patterns = [p for p in table if any(c in p for c in ("*", "?", "["))]
    wildcard_patterns.sort(key=_pattern_specificity)

    for pattern in wildcard_patterns:
        if fnmatch.fnmatch(model_name, pattern):
            return table[pattern]

    return None


def resolve_execution_mode(
    config: AnimaWorksConfig,
    model_name: str,
    explicit_override: str | None = None,
) -> str:
    """Resolve execution mode from model name with wildcard pattern support.

    When ``status.json`` omits ``execution_mode``, this function determines
    the mode automatically from the model name.  For example,
    ``claude-sonnet-4-6`` matches the ``"claude-*": "S"`` pattern and runs
    in Mode S without an explicit setting.

    Priority:
      1. Per-anima explicit override (``status.json`` ``execution_mode``)
      2. models.json user table (``~/.animaworks/models.json``)
      3. config.json model_modes (deprecated fallback, with legacy mapping)
      4. DEFAULT_MODEL_MODE_PATTERNS (code defaults, e.g. ``"claude-*"`` → S)
      5. Default ``"B"`` (safe side)

    Args:
        config: Global AnimaWorks configuration.
        model_name: Model identifier (e.g. ``"claude-sonnet-4-6"``,
            ``"bedrock/jp.anthropic.claude-sonnet-4-6"``).
        explicit_override: Per-anima ``execution_mode`` from ``status.json``.
            When set, takes highest priority.

    Returns:
        One of ``"S"`` (SDK), ``"C"`` (Codex), ``"D"`` (Cursor Agent),
        ``"G"`` (Gemini CLI), ``"A"`` (Autonomous), or ``"B"`` (Basic).
    """
    # 1. Per-anima explicit override
    if explicit_override:
        return _normalise_mode(explicit_override)

    # 2. models.json user table
    entry = _match_models_json(model_name)
    if entry is not None:
        mode_val = entry.get("mode")
        if mode_val:
            return _normalise_mode(str(mode_val))

    # 3. config.json model_modes (deprecated fallback)
    user_table = config.model_modes or {}
    if user_table:
        result = _match_pattern_table(model_name, user_table)
        if result is not None:
            return _normalise_mode(result)

    # 4. Code defaults
    result = _match_pattern_table(model_name, DEFAULT_MODEL_MODE_PATTERNS)
    if result is not None:
        return result  # Already S/C/D/G/A/B in the table

    return "B"  # unknown model → safe side


__all__ = [
    "DEFAULT_MODEL_MODE_PATTERNS",
    "DEFAULT_MODEL_MODES",
    "KNOWN_MODELS",
    "_LEGACY_MODE_MAP",
    "_pattern_specificity",
    "_match_pattern_table",
    "_normalise_mode",
    "resolve_execution_mode",
    "_load_models_json",
    "invalidate_models_json_cache",
    "_match_models_json",
]
