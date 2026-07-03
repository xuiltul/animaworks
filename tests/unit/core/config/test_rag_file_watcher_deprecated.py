from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""F17: the file watcher was removed (2026-07) but enable_file_watcher stays
in the schema so existing config.json files still validate."""

from core.config.schemas import RAGConfig


def test_enable_file_watcher_default_true() -> None:
    """Field remains for config.json backward compatibility."""
    assert RAGConfig().enable_file_watcher is True


def test_enable_file_watcher_accepts_true_from_config() -> None:
    """A config.json carrying enable_file_watcher: true passes validation."""
    rag = RAGConfig.model_validate({"enable_file_watcher": True})
    assert rag.enable_file_watcher is True


def test_enable_file_watcher_accepts_false_from_config() -> None:
    rag = RAGConfig.model_validate({"enable_file_watcher": False})
    assert rag.enable_file_watcher is False
