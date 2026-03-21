# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Centralized path resolution for AnimaWorks.

All modules import directory paths from here instead of computing them ad-hoc.
Runtime data directory can be overridden via ANIMAWORKS_DATA_DIR environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root: where the code lives (immutable, git-tracked)
PROJECT_DIR = Path(__file__).resolve().parent.parent


def _resolve_templates_dir() -> Path:
    """Resolve templates directory for both development and installed modes.

    Resolution order:
      1. ANIMAWORKS_TEMPLATES_DIR env var (explicit override, e.g. Docker)
      2. PROJECT_DIR / "templates" (development / editable install)
    """
    env = os.environ.get("ANIMAWORKS_TEMPLATES_DIR")
    if env:
        return Path(env).resolve()
    return PROJECT_DIR / "templates"


TEMPLATES_DIR = _resolve_templates_dir()

# Default runtime data directory
_DEFAULT_DATA_DIR = Path.home() / ".animaworks"


def get_data_dir() -> Path:
    """Return the runtime data directory, respecting ANIMAWORKS_DATA_DIR env var."""
    env_val = os.environ.get("ANIMAWORKS_DATA_DIR")
    if env_val:
        return Path(env_val).expanduser().resolve()
    return _DEFAULT_DATA_DIR


def get_animas_dir() -> Path:
    return get_data_dir() / "animas"


def get_shared_dir() -> Path:
    return get_data_dir() / "shared"


def get_company_dir() -> Path:
    return get_data_dir() / "company"


def get_common_skills_dir() -> Path:
    return get_data_dir() / "common_skills"


def get_common_knowledge_dir() -> Path:
    return get_data_dir() / "common_knowledge"


def get_reference_dir() -> Path:
    return get_data_dir() / "reference"


def get_tmp_dir() -> Path:
    return get_data_dir() / "tmp"


def get_global_permissions_path() -> Path:
    return get_data_dir() / "permissions.global.json"


def get_anima_vectordb_dir(anima_name: str) -> Path:
    """Return per-anima vectordb directory: {data_dir}/animas/{anima_name}/vectordb."""
    return get_animas_dir() / anima_name / "vectordb"


# --- Prompt templates ---

# Cache loaded templates to avoid repeated disk reads
_prompt_cache: dict[tuple[str, str], str] = {}


class _SafeFormatDict(dict):
    """Dict that returns ``{key}`` for missing keys during format_map.

    This ensures ``{{`` always resolves to ``{`` (double-brace escaping)
    even when no kwargs are passed, while leaving unknown ``{placeholder}``
    patterns intact in the output.
    """

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _get_locale() -> str:
    """Get locale from config lazily to avoid circular imports."""
    try:
        from core.config.models import load_config

        loc = load_config().locale
        if isinstance(loc, str) and loc:
            return loc
        return "ja"
    except Exception:
        return "ja"


def _unique(seq: list[str]) -> list[str]:
    """Return unique elements preserving order."""
    seen: set[str] = set()
    return [x for x in seq if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]


def resolve_template_path(
    category: str,
    filename: str,
    locale: str | None = None,
) -> Path:
    """Resolve template path with fallback chain: locale -> en -> ja.

    Args:
        category: Template category (e.g. "prompts", "company", "roles/engineer").
        filename: File name within the category (e.g. "environment.md").
        locale: Override locale. If None, uses config.locale.

    Returns:
        Path to the resolved template file.

    Raises:
        FileNotFoundError: If the template is not found in any fallback.
    """
    loc = locale or _get_locale()
    if ".." in category or ".." in filename:
        raise ValueError(f"Path traversal not allowed: {category}/{filename}")
    for fallback in _unique([loc, "en", "ja"]):
        path = TEMPLATES_DIR / fallback / category / filename
        if path.exists():
            return path
    shared = TEMPLATES_DIR / "_shared" / category / filename
    if shared.exists():
        return shared
    raise FileNotFoundError(f"Template not found: {category}/{filename} (tried locales: {loc}, en, ja and _shared)")


def load_prompt(name: str, *, locale: str | None = None, **kwargs: object) -> str:
    """Load a prompt template and format it with locale-aware resolution.

    Templates use Python str.format_map() placeholders like {anima_dir}.
    Literal braces in templates should be doubled: {{ and }}.

    Subdirectory paths are supported: ``load_prompt("memory/daily_consolidation")``
    resolves to ``templates/{locale}/prompts/memory/daily_consolidation.md``.

    Args:
        name: Template file name without extension. May include subdirectory
              (e.g. ``"behavior_rules"``, ``"memory/daily_consolidation"``).
        locale: Override locale. If None, uses config.locale.
        **kwargs: Values to substitute into the template placeholders.
    """
    loc = locale or _get_locale()
    key = (loc, name)
    if key not in _prompt_cache:
        path = resolve_template_path("prompts", f"{name}.md", loc)
        _prompt_cache[key] = path.read_text(encoding="utf-8")
    template = _prompt_cache[key]
    return template.format_map(_SafeFormatDict(kwargs))
