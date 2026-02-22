"""Unit tests for disable_subordinate / enable_subordinate tool handlers."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.tooling.handler import ToolHandler


def _make_handler(tmp_path: Path, anima_name: str = "sakura") -> ToolHandler:
    """Create a ToolHandler with minimal mocked dependencies."""
    anima_dir = tmp_path / "animas" / anima_name
    anima_dir.mkdir(parents=True)
    (anima_dir / "permissions.md").write_text("", encoding="utf-8")

    memory = MagicMock()
    memory.read_permissions.return_value = ""

    handler = ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        messenger=None,
        tool_registry=[],
    )
    return handler


def _setup_subordinate(
    tmp_path: Path,
    name: str,
    supervisor: str,
    *,
    enabled: bool = True,
) -> Path:
    """Create a subordinate anima directory with status.json."""
    anima_dir = tmp_path / "animas" / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "enabled": enabled,
        "supervisor": supervisor,
        "model": "claude-sonnet-4-20250514",
        "role": "general",
    }
    (anima_dir / "status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8",
    )
    return anima_dir


def _mock_config(tmp_path: Path, animas: dict[str, dict]) -> MagicMock:
    """Build a mock config with AnimaModelConfig entries."""
    from core.config.models import AnimaModelConfig

    config = MagicMock()
    config.animas = {
        name: AnimaModelConfig(**fields)
        for name, fields in animas.items()
    }
    return config


class TestDisableSubordinate:
    """Tests for _handle_disable_subordinate."""

    def test_disable_direct_subordinate(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura", enabled=True)

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        ):
            result = handler.handle("disable_subordinate", {"name": "hinata", "reason": "test"})

        assert "休止" in result
        status = json.loads((tmp_path / "animas" / "hinata" / "status.json").read_text())
        assert status["enabled"] is False
        # Other fields preserved
        assert status["supervisor"] == "sakura"
        assert status["model"] == "claude-sonnet-4-20250514"
        assert status["role"] == "general"

    def test_disable_non_subordinate_rejected(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "mio", supervisor="taka", enabled=True)

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "mio": {"supervisor": "taka"},
        })

        with patch("core.config.models.load_config", return_value=mock_cfg):
            result = handler.handle("disable_subordinate", {"name": "mio"})

        assert "PermissionDenied" in result
        assert "直属部下ではありません" in result

    def test_disable_self_rejected(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")

        # No config mock needed - self-check happens before config load
        result = handler.handle("disable_subordinate", {"name": "sakura"})
        assert "自分自身" in result

    def test_disable_nonexistent_anima(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")

        mock_cfg = _mock_config(tmp_path, {"sakura": {}})

        with patch("core.config.models.load_config", return_value=mock_cfg):
            result = handler.handle("disable_subordinate", {"name": "nobody"})

        assert "AnimaNotFound" in result

    def test_disable_already_disabled(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura", enabled=False)

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        ):
            result = handler.handle("disable_subordinate", {"name": "hinata"})

        assert "既に休止中" in result

    def test_disable_missing_name(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        result = handler.handle("disable_subordinate", {})
        assert "InvalidArguments" in result


class TestEnableSubordinate:
    """Tests for _handle_enable_subordinate."""

    def test_enable_disabled_subordinate(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura", enabled=False)

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        ):
            result = handler.handle("enable_subordinate", {"name": "hinata"})

        assert "有効" in result
        status = json.loads((tmp_path / "animas" / "hinata" / "status.json").read_text())
        assert status["enabled"] is True
        # Other fields preserved
        assert status["supervisor"] == "sakura"
        assert status["model"] == "claude-sonnet-4-20250514"

    def test_enable_already_enabled(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        _setup_subordinate(tmp_path, "hinata", supervisor="sakura", enabled=True)

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        ):
            result = handler.handle("enable_subordinate", {"name": "hinata"})

        assert "既に有効" in result

    def test_enable_non_subordinate_rejected(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "mio": {"supervisor": "taka"},
        })

        with patch("core.config.models.load_config", return_value=mock_cfg):
            result = handler.handle("enable_subordinate", {"name": "mio"})

        assert "PermissionDenied" in result


class TestEdgeCases:
    """Tests for edge cases: config failure, invalid JSON, missing status.json."""

    def test_config_load_failure(self, tmp_path):
        """When load_config() raises, return ConfigError."""
        handler = _make_handler(tmp_path, "sakura")

        with patch(
            "core.config.models.load_config",
            side_effect=RuntimeError("config broken"),
        ):
            result = handler.handle("disable_subordinate", {"name": "hinata"})

        assert "ConfigError" in result

    def test_disable_with_invalid_status_json(self, tmp_path):
        """When status.json contains invalid JSON, treat as empty and proceed."""
        handler = _make_handler(tmp_path, "sakura")
        anima_dir = tmp_path / "animas" / "hinata"
        anima_dir.mkdir(parents=True, exist_ok=True)
        (anima_dir / "status.json").write_text("not valid json{{{", encoding="utf-8")

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        ):
            result = handler.handle("disable_subordinate", {"name": "hinata"})

        assert "休止" in result
        status = json.loads((anima_dir / "status.json").read_text())
        assert status["enabled"] is False

    def test_disable_with_missing_status_json(self, tmp_path):
        """When status.json doesn't exist, create it with enabled=false."""
        handler = _make_handler(tmp_path, "sakura")
        anima_dir = tmp_path / "animas" / "hinata"
        anima_dir.mkdir(parents=True, exist_ok=True)
        # No status.json created

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        ):
            result = handler.handle("disable_subordinate", {"name": "hinata"})

        assert "休止" in result
        status = json.loads((anima_dir / "status.json").read_text())
        assert status["enabled"] is False

    def test_enable_with_missing_status_json(self, tmp_path):
        """When status.json doesn't exist, treat as enabled=true (already enabled)."""
        handler = _make_handler(tmp_path, "sakura")
        anima_dir = tmp_path / "animas" / "hinata"
        anima_dir.mkdir(parents=True, exist_ok=True)
        # No status.json

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        ):
            result = handler.handle("enable_subordinate", {"name": "hinata"})

        assert "既に有効" in result


class TestStatusJsonPreservation:
    """Verify that enable/disable preserves all existing status.json fields."""

    def test_roundtrip_preserves_fields(self, tmp_path):
        handler = _make_handler(tmp_path, "sakura")
        anima_dir = _setup_subordinate(
            tmp_path, "hinata", supervisor="sakura", enabled=True,
        )
        # Add extra fields
        status = json.loads((anima_dir / "status.json").read_text())
        status["custom_field"] = "preserve_me"
        (anima_dir / "status.json").write_text(json.dumps(status, indent=2))

        mock_cfg = _mock_config(tmp_path, {
            "sakura": {},
            "hinata": {"supervisor": "sakura"},
        })

        with (
            patch("core.config.models.load_config", return_value=mock_cfg),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        ):
            # Disable
            handler.handle("disable_subordinate", {"name": "hinata"})
            status = json.loads((anima_dir / "status.json").read_text())
            assert status["enabled"] is False
            assert status["custom_field"] == "preserve_me"
            assert status["supervisor"] == "sakura"

            # Enable
            handler.handle("enable_subordinate", {"name": "hinata"})
            status = json.loads((anima_dir / "status.json").read_text())
            assert status["enabled"] is True
            assert status["custom_field"] == "preserve_me"
            assert status["supervisor"] == "sakura"
