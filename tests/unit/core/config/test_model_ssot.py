"""Unit tests for core/config/models.py — Model Config SSoT helpers.

Tests update_status_model.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config.models import update_status_model

# ── update_status_model ────────────────────────────────────


class TestUpdateStatusModel:
    def test_updates_model_in_status_json(self, tmp_path: Path) -> None:
        """Create status.json with model field, update model, verify."""
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        status_path = anima_dir / "status.json"
        status_path.write_text(
            json.dumps({"model": "claude-sonnet", "enabled": True}),
            encoding="utf-8",
        )
        update_status_model(anima_dir, model="openai/gpt-4o")
        data = json.loads(status_path.read_text(encoding="utf-8"))
        assert data["model"] == "openai/gpt-4o"
        assert data["enabled"] is True

    def test_updates_credential(self, tmp_path: Path) -> None:
        """Update credential only."""
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        status_path = anima_dir / "status.json"
        status_path.write_text(
            json.dumps({"model": "claude-sonnet", "credential": "anthropic"}),
            encoding="utf-8",
        )
        update_status_model(anima_dir, credential="openai")
        data = json.loads(status_path.read_text(encoding="utf-8"))
        assert data["model"] == "claude-sonnet"
        assert data["credential"] == "openai"

    def test_updates_both(self, tmp_path: Path) -> None:
        """Update both model and credential."""
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        status_path = anima_dir / "status.json"
        status_path.write_text(
            json.dumps({"model": "old", "credential": "old_cred"}),
            encoding="utf-8",
        )
        update_status_model(anima_dir, model="new-model", credential="new_cred")
        data = json.loads(status_path.read_text(encoding="utf-8"))
        assert data["model"] == "new-model"
        assert data["credential"] == "new_cred"

    def test_no_status_json_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError when no status.json."""
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="status.json not found"):
            update_status_model(anima_dir, model="new-model")

    def test_preserves_other_fields(self, tmp_path: Path) -> None:
        """Other fields (enabled, role, etc.) remain intact."""
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        status_path = anima_dir / "status.json"
        status_path.write_text(
            json.dumps({
                "model": "old-model",
                "enabled": True,
                "role": "engineer",
                "supervisor": "bob",
            }),
            encoding="utf-8",
        )
        update_status_model(anima_dir, model="new-model")
        data = json.loads(status_path.read_text(encoding="utf-8"))
        assert data["model"] == "new-model"
        assert data["enabled"] is True
        assert data["role"] == "engineer"
        assert data["supervisor"] == "bob"
