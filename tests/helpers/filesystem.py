# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Filesystem scaffolding helpers for tests.

Creates isolated AnimaWorks runtime data directories so that each test
runs against its own temporary filesystem without touching real data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Minimal valid config.json for tests.
DEFAULT_TEST_CONFIG: dict[str, Any] = {
    "version": 1,
    "system": {"mode": "server", "log_level": "DEBUG"},
    "credentials": {
        "anthropic": {"api_key": "", "base_url": None},
        "openai": {"api_key": "", "base_url": None},
        "azure": {"api_key": "", "base_url": None},
        "ollama": {"api_key": "", "base_url": "http://localhost:11434"},
    },
    "anima_defaults": {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "max_turns": 5,
        "credential": "anthropic",
        "context_threshold": 0.50,
        "max_chains": 2,
        "conversation_history_threshold": 0.30,
    },
    "animas": {},
}

# Subdirectories required inside each anima directory.
_ANIMA_SUBDIRS = [
    "episodes",
    "knowledge",
    "procedures",
    "skills",
    "state",
    "shortterm",
    "shortterm/archive",
    "transcripts",
]


def create_test_data_dir(base: Path) -> Path:
    """Create the ``~/.animaworks/``-like directory tree under *base*.

    Returns the data directory path (``base / ".animaworks"``).
    """
    data_dir = base / ".animaworks"
    data_dir.mkdir()

    (data_dir / "animas").mkdir()
    (data_dir / "shared" / "inbox").mkdir(parents=True)
    (data_dir / "shared" / "users").mkdir(parents=True)
    (data_dir / "company").mkdir()
    (data_dir / "company" / "vision.md").write_text(
        "# Company Vision\nTest company.", encoding="utf-8"
    )
    (data_dir / "common_skills").mkdir()
    (data_dir / "common_knowledge").mkdir()
    (data_dir / "tmp" / "attachments").mkdir(parents=True)

    # Write default config.json
    config_path = data_dir / "config.json"
    config_path.write_text(
        json.dumps(DEFAULT_TEST_CONFIG, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return data_dir


def create_anima_dir(
    data_dir: Path,
    name: str,
    *,
    identity: str = "# Test Anima\nA test digital anima for E2E testing.",
    injection: str = "",
    permissions: str = "",
    model: str = "claude-sonnet-4-20250514",
    execution_mode: str | None = None,
    credential: str | None = None,
    supervisor: str | None = None,
    speciality: str | None = None,
    api_key: str | None = None,
    api_key_env: str = "ANTHROPIC_API_KEY",
    api_base_url: str | None = None,
    context_threshold: float = 0.50,
    max_chains: int = 2,
    max_turns: int = 5,
    conversation_history_threshold: float = 0.30,
) -> Path:
    """Create an anima directory with all required files and subdirectories.

    Also updates ``config.json`` with the anima's model configuration.
    Returns the anima directory path.
    """
    anima_dir = data_dir / "animas" / name
    anima_dir.mkdir(parents=True, exist_ok=True)

    # Write identity and config files
    (anima_dir / "identity.md").write_text(identity, encoding="utf-8")
    if injection:
        (anima_dir / "injection.md").write_text(injection, encoding="utf-8")
    if permissions:
        (anima_dir / "permissions.md").write_text(permissions, encoding="utf-8")

    # Create required subdirectories
    for sub in _ANIMA_SUBDIRS:
        (anima_dir / sub).mkdir(parents=True, exist_ok=True)

    # Initialize state files
    (anima_dir / "state" / "current_task.md").write_text(
        "status: idle\n", encoding="utf-8"
    )
    (anima_dir / "state" / "pending.md").write_text("", encoding="utf-8")

    # Update config.json with anima entry
    config_path = data_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    anima_config: dict[str, Any] = {"model": model}
    if execution_mode is not None:
        anima_config["execution_mode"] = execution_mode
    if credential is not None:
        anima_config["credential"] = credential
    if supervisor is not None:
        anima_config["supervisor"] = supervisor
    if speciality is not None:
        anima_config["speciality"] = speciality
    anima_config["context_threshold"] = context_threshold
    anima_config["max_chains"] = max_chains
    anima_config["max_turns"] = max_turns
    anima_config["conversation_history_threshold"] = conversation_history_threshold

    config["animas"][name] = anima_config

    # Update credentials for the appropriate credential type
    cred_name = credential or "anthropic"
    if cred_name not in config["credentials"]:
        config["credentials"][cred_name] = {"api_key": "", "base_url": None}
    if api_key:
        config["credentials"][cred_name]["api_key"] = api_key
    if api_base_url:
        config["credentials"][cred_name]["base_url"] = api_base_url

    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return anima_dir
