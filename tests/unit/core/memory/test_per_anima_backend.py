from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-Anima memory backend resolution (resolve_backend_type)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.backend.registry import resolve_backend_type


# ── resolve_backend_type ──────────────────────────────────────────────


class TestResolveBackendType:
    """Per-Anima → global config → default resolution."""

    def test_per_anima_neo4j(self, tmp_path: Path) -> None:
        """status.json memory_backend takes priority over global config."""
        status = {"enabled": True, "model": "claude-sonnet-4-6", "memory_backend": "neo4j"}
        (tmp_path / "status.json").write_text(json.dumps(status))

        result = resolve_backend_type(tmp_path)
        assert result == "neo4j"

    def test_per_anima_legacy(self, tmp_path: Path) -> None:
        """Explicit 'legacy' in status.json is respected."""
        status = {"enabled": True, "memory_backend": "legacy"}
        (tmp_path / "status.json").write_text(json.dumps(status))

        result = resolve_backend_type(tmp_path)
        assert result == "legacy"

    def test_fallback_to_global_config(self, tmp_path: Path) -> None:
        """When status.json has no memory_backend, fall back to global config."""
        status = {"enabled": True, "model": "claude-sonnet-4-6"}
        (tmp_path / "status.json").write_text(json.dumps(status))

        mock_cfg = MagicMock()
        mock_cfg.memory.backend = "neo4j"
        with patch("core.config.models.load_config", return_value=mock_cfg):
            result = resolve_backend_type(tmp_path)

        assert result == "neo4j"

    def test_fallback_to_default_legacy(self, tmp_path: Path) -> None:
        """When status.json has no field and config fails, default to 'legacy'."""
        status = {"enabled": True}
        (tmp_path / "status.json").write_text(json.dumps(status))

        with patch("core.config.models.load_config", side_effect=Exception("no config")):
            result = resolve_backend_type(tmp_path)

        assert result == "legacy"

    def test_no_status_json_uses_global(self, tmp_path: Path) -> None:
        """When status.json does not exist, use global config."""
        mock_cfg = MagicMock()
        mock_cfg.memory.backend = "neo4j"
        with patch("core.config.models.load_config", return_value=mock_cfg):
            result = resolve_backend_type(tmp_path)

        assert result == "neo4j"

    def test_corrupt_status_json_uses_global(self, tmp_path: Path) -> None:
        """When status.json is invalid JSON, fall back to global config."""
        (tmp_path / "status.json").write_text("not valid json{{{")

        mock_cfg = MagicMock()
        mock_cfg.memory.backend = "neo4j"
        with patch("core.config.models.load_config", return_value=mock_cfg):
            result = resolve_backend_type(tmp_path)

        assert result == "neo4j"

    def test_empty_memory_backend_field_uses_global(self, tmp_path: Path) -> None:
        """Empty string in memory_backend is treated as unset."""
        status = {"enabled": True, "memory_backend": ""}
        (tmp_path / "status.json").write_text(json.dumps(status))

        mock_cfg = MagicMock()
        mock_cfg.memory.backend = "neo4j"
        with patch("core.config.models.load_config", return_value=mock_cfg):
            result = resolve_backend_type(tmp_path)

        assert result == "neo4j"

    def test_null_memory_backend_field_uses_global(self, tmp_path: Path) -> None:
        """None/null in memory_backend is treated as unset."""
        status = {"enabled": True, "memory_backend": None}
        (tmp_path / "status.json").write_text(json.dumps(status))

        mock_cfg = MagicMock()
        mock_cfg.memory.backend = "legacy"
        with patch("core.config.models.load_config", return_value=mock_cfg):
            result = resolve_backend_type(tmp_path)

        assert result == "legacy"


# ── MemoryManager integration ─────────────────────────────────────────


class TestMemoryManagerBackendInit:
    """MemoryManager._init_memory_backend uses resolve_backend_type."""

    def test_uses_per_anima_setting(self, tmp_path: Path) -> None:
        """MemoryManager selects backend based on per-anima status.json."""
        status = {"enabled": True, "memory_backend": "neo4j"}
        (tmp_path / "status.json").write_text(json.dumps(status))
        (tmp_path / "state").mkdir()

        with (
            patch("core.memory.backend.registry.resolve_backend_type", return_value="neo4j"),
            patch("core.memory.backend.registry.get_backend") as mock_get,
            patch("core.paths.get_data_dir", return_value=tmp_path),
            patch("core.paths.get_common_knowledge_dir", return_value=tmp_path / "ck"),
            patch("core.paths.get_common_skills_dir", return_value=tmp_path / "cs"),
            patch("core.paths.get_company_dir", return_value=tmp_path / "co"),
        ):
            mock_backend = MagicMock()
            mock_get.return_value = mock_backend

            from core.memory.manager import MemoryManager

            mm = MemoryManager(tmp_path)
            mm._init_memory_backend()

            mock_get.assert_called_once_with(
                "neo4j",
                tmp_path,
            )


# ── update_status_model ───────────────────────────────────────────────


class TestUpdateStatusModelMemoryBackend:
    """update_status_model handles memory_backend field."""

    def test_set_memory_backend(self, tmp_path: Path) -> None:
        """Setting memory_backend writes to status.json."""
        status_path = tmp_path / "status.json"
        status_path.write_text(json.dumps({"enabled": True, "model": "test"}))

        from core.config.model_config import update_status_model

        update_status_model(tmp_path, memory_backend="neo4j")

        data = json.loads(status_path.read_text())
        assert data["memory_backend"] == "neo4j"

    def test_clear_memory_backend(self, tmp_path: Path) -> None:
        """Clearing memory_backend removes the field."""
        status_path = tmp_path / "status.json"
        status_path.write_text(json.dumps({"enabled": True, "memory_backend": "neo4j"}))

        from core.config.model_config import update_status_model

        update_status_model(tmp_path, memory_backend="")

        data = json.loads(status_path.read_text())
        assert "memory_backend" not in data

    def test_sentinel_leaves_field_unchanged(self, tmp_path: Path) -> None:
        """Default sentinel does not modify memory_backend."""
        status_path = tmp_path / "status.json"
        status_path.write_text(json.dumps({"enabled": True, "memory_backend": "neo4j"}))

        from core.config.model_config import update_status_model

        update_status_model(tmp_path, model="new-model")

        data = json.loads(status_path.read_text())
        assert data["memory_backend"] == "neo4j"
        assert data["model"] == "new-model"
