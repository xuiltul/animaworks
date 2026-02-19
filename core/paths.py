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

# Templates shipped with the project
TEMPLATES_DIR = PROJECT_DIR / "templates"

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


def get_tmp_dir() -> Path:
    return get_data_dir() / "tmp"


# --- Prompt templates ---

PROMPTS_DIR = TEMPLATES_DIR / "prompts"

# Cache loaded templates to avoid repeated disk reads
_prompt_cache: dict[str, str] = {}


def load_prompt(name: str, **kwargs: object) -> str:
    """Load a prompt template from templates/prompts/{name}.md and format it.

    Templates use Python str.format_map() placeholders like {anima_dir}.
    Literal braces in templates should be doubled: {{ and }}.

    Args:
        name: Template file name without extension (e.g. "behavior_rules").
        **kwargs: Values to substitute into the template placeholders.
    """
    if name not in _prompt_cache:
        path = PROMPTS_DIR / f"{name}.md"
        _prompt_cache[name] = path.read_text(encoding="utf-8")
    template = _prompt_cache[name]
    if kwargs:
        return template.format_map(kwargs)
    return template