from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tests for _handle_set_subordinate_model and _handle_restart_subordinate.

Covers:
- set_subordinate_model: config.json update, KNOWN_MODELS warning,
  missing argument validation, and non-subordinate permission check.
- restart_subordinate: restart_requested sentinel flag in status.json,
  missing-status.json creation, missing argument validation,
  and non-subordinate permission check.
- KNOWN_MODELS: structural integrity (name/mode/note fields, valid modes,
  no duplicate names).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ──────────────────────────────────────────────


def _make_handler(tmp_path: Path, anima_name: str = "supervisor"):
    """Create a ToolHandler with minimal mocked dependencies.

    The ``load_config()`` call inside ``__init__`` is patched to return an
    empty config so no real filesystem is needed for initialisation.
    """
    from core.tooling.handler import ToolHandler

    memory = MagicMock()
    # Patch load_config used inside __init__ for subordinate-path caching
    with patch("core.tooling.handler.ToolHandler.__init__", lambda self, **kw: None):
        handler = ToolHandler.__new__(ToolHandler)

    # Manually populate the attributes that the handlers need
    handler._anima_dir = tmp_path / "animas" / anima_name
    handler._anima_dir.mkdir(parents=True, exist_ok=True)
    handler._anima_name = anima_name
    handler._memory = memory
    handler._messenger = None
    handler._on_message_sent = None
    handler._on_schedule_changed = None
    handler._human_notifier = None
    handler._background_manager = None
    handler._pending_notifications = []
    handler._replied_to = {"chat": set(), "background": set()}
    handler._subordinate_activity_dirs = []
    handler._subordinate_management_files = []

    import uuid
    handler._session_id = uuid.uuid4().hex[:12]

    from core.memory.activity import ActivityLogger
    handler._activity = MagicMock(spec=ActivityLogger)

    from core.tooling.dispatch import ExternalToolDispatcher
    handler._external = MagicMock(spec=ExternalToolDispatcher)

    return handler


def _parse_error(result: str) -> dict:
    """Parse a JSON error result from a tool handler."""
    return json.loads(result)


# ── set_subordinate_model tests ───────────────────────────


class TestSetSubordinateModel:
    """Tests for ToolHandler._handle_set_subordinate_model()."""

    def _animas_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "animas"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_set_model_success(self, tmp_path: Path):
        """Known model name updates config.json correctly."""
        handler = _make_handler(tmp_path, "manager")

        # Use a known model name (first entry in KNOWN_MODELS)
        from core.config.models import KNOWN_MODELS, AnimaModelConfig, AnimaWorksConfig

        known_model = KNOWN_MODELS[0]["name"]

        # Build a config with "engineer" as a direct subordinate
        cfg = AnimaWorksConfig()
        cfg.animas["engineer"] = AnimaModelConfig(supervisor="manager")

        saved: list = []

        def fake_save(c):
            saved.append(c)

        with (
            patch(
                "core.tooling.handler.ToolHandler._check_subordinate",
                return_value=None,
            ),
            patch(
                "core.config.models.load_config",
                return_value=cfg,
            ),
            patch(
                "core.config.models.save_config",
                side_effect=fake_save,
            ),
        ):
            result = handler._handle_set_subordinate_model(
                {"name": "engineer", "model": known_model}
            )

        # Should succeed (no error JSON)
        assert "error" not in result.lower() or "警告" in result
        assert known_model in result
        # config was saved with the new model
        assert len(saved) == 1
        assert saved[0].animas["engineer"].model == known_model

    def test_set_model_unknown_model_warns(self, tmp_path: Path, caplog):
        """Models outside KNOWN_MODELS succeed but emit a warning."""
        import logging

        handler = _make_handler(tmp_path, "manager")

        from core.config.models import AnimaModelConfig, AnimaWorksConfig

        cfg = AnimaWorksConfig()
        cfg.animas["engineer"] = AnimaModelConfig(supervisor="manager")

        with (
            patch(
                "core.tooling.handler.ToolHandler._check_subordinate",
                return_value=None,
            ),
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.config.models.save_config"),
            caplog.at_level(logging.WARNING),
        ):
            result = handler._handle_set_subordinate_model(
                {"name": "engineer", "model": "unknown/totally-fake-model"}
            )

        # Should still return success (not an error)
        try:
            parsed = json.loads(result)
            assert parsed.get("status") != "error", f"Unexpected error: {result}"
        except json.JSONDecodeError:
            pass  # Plain text result is also fine

        # Warning text must appear in the result
        assert "警告" in result
        # Warning should also appear in logger output
        assert any("unknown model" in r.message.lower() or "unknown" in r.message for r in caplog.records)

    def test_set_model_missing_name(self, tmp_path: Path):
        """Empty name returns InvalidArguments error."""
        handler = _make_handler(tmp_path, "manager")

        result = handler._handle_set_subordinate_model(
            {"name": "", "model": "claude-sonnet-4-6"}
        )

        parsed = _parse_error(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "InvalidArguments"

    def test_set_model_missing_model(self, tmp_path: Path):
        """Empty model returns InvalidArguments error."""
        handler = _make_handler(tmp_path, "manager")

        result = handler._handle_set_subordinate_model(
            {"name": "engineer", "model": ""}
        )

        parsed = _parse_error(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "InvalidArguments"

    def test_set_model_non_subordinate(self, tmp_path: Path):
        """Non-subordinate target returns PermissionDenied error."""
        handler = _make_handler(tmp_path, "manager")

        perm_denied = json.dumps({
            "status": "error",
            "error_type": "PermissionDenied",
            "message": "'stranger' はあなたの直属部下ではありません",
        }, ensure_ascii=False)

        with patch(
            "core.tooling.handler.ToolHandler._check_subordinate",
            return_value=perm_denied,
        ):
            result = handler._handle_set_subordinate_model(
                {"name": "stranger", "model": "claude-sonnet-4-6"}
            )

        # Result must be the PermissionDenied error unchanged
        parsed = _parse_error(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "PermissionDenied"


# ── restart_subordinate tests ─────────────────────────────


class TestRestartSubordinate:
    """Tests for ToolHandler._handle_restart_subordinate()."""

    def _make_target_dir(self, tmp_path: Path, name: str) -> Path:
        """Create a target anima directory under tmp_path/animas/{name}."""
        d = tmp_path / "animas" / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_restart_success(self, tmp_path: Path):
        """restart_requested: true is written to an existing status.json."""
        handler = _make_handler(tmp_path, "manager")

        target_dir = self._make_target_dir(tmp_path, "engineer")
        status_file = target_dir / "status.json"
        status_file.write_text(
            json.dumps({"enabled": True, "model": "claude-sonnet-4-6"}),
            encoding="utf-8",
        )

        with (
            patch(
                "core.tooling.handler.ToolHandler._check_subordinate",
                return_value=None,
            ),
            patch(
                "core.tooling.handler._handle_restart_subordinate.__module__",
                create=True,
            ) if False else patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        ):
            result = handler._handle_restart_subordinate({"name": "engineer"})

        # Should not be an error
        try:
            parsed = json.loads(result)
            assert parsed.get("status") != "error", f"Unexpected error: {result}"
        except json.JSONDecodeError:
            pass  # Plain text success message is fine

        # status.json must have restart_requested = True
        updated = json.loads(status_file.read_text(encoding="utf-8"))
        assert updated["restart_requested"] is True
        # Existing fields must be preserved
        assert updated["enabled"] is True
        assert updated["model"] == "claude-sonnet-4-6"

    def test_restart_creates_status_if_missing(self, tmp_path: Path):
        """If status.json does not exist, it is created with restart_requested: true."""
        handler = _make_handler(tmp_path, "manager")

        target_dir = self._make_target_dir(tmp_path, "newbie")
        status_file = target_dir / "status.json"
        assert not status_file.exists()

        with (
            patch(
                "core.tooling.handler.ToolHandler._check_subordinate",
                return_value=None,
            ),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        ):
            result = handler._handle_restart_subordinate({"name": "newbie"})

        assert status_file.exists()
        updated = json.loads(status_file.read_text(encoding="utf-8"))
        assert updated["restart_requested"] is True

    def test_restart_missing_name(self, tmp_path: Path):
        """Empty name returns InvalidArguments error."""
        handler = _make_handler(tmp_path, "manager")

        result = handler._handle_restart_subordinate({"name": ""})

        parsed = _parse_error(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "InvalidArguments"

    def test_restart_non_subordinate(self, tmp_path: Path):
        """Non-subordinate target returns PermissionDenied error."""
        handler = _make_handler(tmp_path, "manager")

        perm_denied = json.dumps({
            "status": "error",
            "error_type": "PermissionDenied",
            "message": "'outsider' はあなたの直属部下ではありません",
        }, ensure_ascii=False)

        with patch(
            "core.tooling.handler.ToolHandler._check_subordinate",
            return_value=perm_denied,
        ):
            result = handler._handle_restart_subordinate({"name": "outsider"})

        parsed = _parse_error(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "PermissionDenied"


# ── KNOWN_MODELS structural tests ────────────────────────


class TestKnownModels:
    """Tests for KNOWN_MODELS catalog structural integrity."""

    def _get_known_models(self):
        from core.config.models import KNOWN_MODELS
        return KNOWN_MODELS

    def test_known_models_structure(self):
        """All entries have name/mode/note fields and mode is S, A, or B."""
        models = self._get_known_models()
        assert len(models) > 0, "KNOWN_MODELS must not be empty"

        valid_modes = {"S", "A", "B"}
        for entry in models:
            assert "name" in entry, f"Missing 'name' in entry: {entry}"
            assert "mode" in entry, f"Missing 'mode' in entry: {entry}"
            assert "note" in entry, f"Missing 'note' in entry: {entry}"
            assert isinstance(entry["name"], str) and entry["name"], \
                f"'name' must be a non-empty string: {entry}"
            assert entry["mode"] in valid_modes, \
                f"'mode' must be S, A, or B, got '{entry['mode']}': {entry}"
            assert isinstance(entry["note"], str), \
                f"'note' must be a string: {entry}"

    def test_known_models_no_duplicates(self):
        """No duplicate names in KNOWN_MODELS."""
        models = self._get_known_models()
        names = [m["name"] for m in models]
        seen: set[str] = set()
        duplicates: list[str] = []
        for name in names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        assert not duplicates, f"Duplicate model names found: {duplicates}"
