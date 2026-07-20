"""Unit tests for core/workspace.py — workspace registry and resolution."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from core.config.models import AnimaWorksConfig
from core.workspace import (
    list_workspaces,
    qualified_alias,
    register_workspace,
    remove_workspace,
    resolve_workspace,
    workspace_hash,
    workspace_info,
)

# ── Hash generation ─────────────────────────────────────────────


class TestWorkspaceHash:
    """Tests for workspace_hash()."""

    def test_returns_8_char_hex(self) -> None:
        """workspace_hash returns exactly 8 hex characters."""
        result = workspace_hash("/some/path")
        assert len(result) == 8
        assert all(c in "0123456789abcdef" for c in result)

    def test_is_deterministic(self) -> None:
        """Same path always yields same hash."""
        path = "/home/user/project"
        assert workspace_hash(path) == workspace_hash(path)

    def test_different_paths_different_hashes(self) -> None:
        """Different paths yield different hashes."""
        h1 = workspace_hash("/path/a")
        h2 = workspace_hash("/path/b")
        assert h1 != h2


# ── Qualified alias ─────────────────────────────────────────────


class TestQualifiedAlias:
    """Tests for qualified_alias()."""

    def test_returns_alias_hash_format(self) -> None:
        """qualified_alias returns 'alias#hash8' format."""
        result = qualified_alias("myproject", "/tmp/project")
        assert "#" in result
        alias_part, hash_part = result.rsplit("#", 1)
        assert alias_part == "myproject"
        assert len(hash_part) == 8
        assert hash_part == workspace_hash("/tmp/project")


# ── Resolution ───────────────────────────────────────────────────


class TestResolveWorkspace:
    """Tests for resolve_workspace()."""

    @pytest.fixture
    def mock_config(self, tmp_path: Path) -> AnimaWorksConfig:
        """Create config with a real tmp_path workspace."""
        cfg = AnimaWorksConfig()
        cfg.workspaces = {"myproject": str(tmp_path)}
        return cfg

    def test_alias_hash_exact_match_succeeds(
        self, tmp_path: Path, mock_config: AnimaWorksConfig
    ) -> None:
        """alias#hash exact match resolves to Path."""
        qa = qualified_alias("myproject", str(tmp_path))
        with patch("core.config.models.load_config", return_value=mock_config):
            result = resolve_workspace(qa)
        assert result == tmp_path.resolve()

    def test_alias_hash_wrong_hash_fails(
        self, tmp_path: Path, mock_config: AnimaWorksConfig
    ) -> None:
        """alias#wronghash raises ValueError."""
        wrong_qa = "myproject#deadbeef"  # wrong hash
        with (
            patch("core.config.models.load_config", return_value=mock_config),
            pytest.raises(ValueError) as excinfo,
        ):
            resolve_workspace(wrong_qa)
        assert "deadbeef" in str(excinfo.value) or "not found" in str(excinfo.value).lower()

    def test_alias_only_exact_match(
        self, tmp_path: Path, mock_config: AnimaWorksConfig
    ) -> None:
        """alias-only exact match resolves."""
        with patch("core.config.models.load_config", return_value=mock_config):
            result = resolve_workspace("myproject")
        assert result == tmp_path.resolve()

    def test_hash_only_resolves(
        self, tmp_path: Path, mock_config: AnimaWorksConfig
    ) -> None:
        """8-char hex hash search across registry resolves."""
        h = workspace_hash(str(tmp_path))
        with patch("core.config.models.load_config", return_value=mock_config):
            result = resolve_workspace(h)
        assert result == tmp_path.resolve()

    def test_absolute_path_direct(self, tmp_path: Path) -> None:
        """Absolute path that exists resolves directly."""
        cfg = AnimaWorksConfig()
        cfg.workspaces = {}
        with patch("core.config.models.load_config", return_value=cfg):
            result = resolve_workspace(str(tmp_path))
        assert result == tmp_path.resolve()

    def test_empty_string_raises_value_error(self) -> None:
        """Empty string input raises ValueError immediately."""
        with pytest.raises(ValueError):
            resolve_workspace("")

    def test_whitespace_only_raises_value_error(self) -> None:
        """Whitespace-only input raises ValueError immediately."""
        with pytest.raises(ValueError):
            resolve_workspace("   ")

    def test_not_found_raises_value_error(self) -> None:
        """Unknown alias/path raises ValueError."""
        cfg = AnimaWorksConfig()
        cfg.workspaces = {}
        with patch("core.config.models.load_config", return_value=cfg), pytest.raises(ValueError) as excinfo:
            resolve_workspace("nonexistent")
        assert "nonexistent" in str(excinfo.value)

    def test_resolve_workspace_suggests_similar(
        self, tmp_path: Path
    ) -> None:
        """Fuzzy match suggests similar aliases when resolution fails."""
        cfg = AnimaWorksConfig()
        cfg.workspaces = {"myproject": str(tmp_path)}
        with patch("core.config.models.load_config", return_value=cfg), pytest.raises(ValueError) as excinfo:
            resolve_workspace("my-project")
        msg = str(excinfo.value)
        assert "もしかして" in msg or "Did you mean" in msg

    def test_resolve_workspace_lists_all_when_no_match(
        self, tmp_path: Path
    ) -> None:
        """When no fuzzy match, ValueError lists all registered workspaces."""
        cfg = AnimaWorksConfig()
        cfg.workspaces = {"myproject": str(tmp_path)}
        with patch("core.config.models.load_config", return_value=cfg), pytest.raises(ValueError) as excinfo:
            resolve_workspace("zzz-completely-different")
        msg = str(excinfo.value)
        assert "myproject" in msg

    def test_resolve_workspace_empty_registry(self) -> None:
        """Empty registry raises ValueError with (none) in message."""
        cfg = AnimaWorksConfig()
        cfg.workspaces = {}
        with patch("core.config.models.load_config", return_value=cfg), pytest.raises(ValueError) as excinfo:
            resolve_workspace("anything")
        msg = str(excinfo.value)
        assert "(none)" in msg

    def test_registered_path_deleted_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        """Registered path that no longer exists raises ValueError."""
        deleted_dir = tmp_path / "deleted"
        deleted_dir.mkdir()
        cfg = AnimaWorksConfig()
        cfg.workspaces = {"gone": str(deleted_dir)}
        deleted_dir.rmdir()
        with patch("core.config.models.load_config", return_value=cfg), pytest.raises(ValueError) as excinfo:
            resolve_workspace("gone")
        err_msg = str(excinfo.value)
        assert "deleted" in err_msg or "存在" in err_msg or "exist" in err_msg.lower()


# ── Registration ───────────────────────────────────────────────


class TestRegisterWorkspace:
    """Tests for register_workspace()."""

    def test_creates_entry_returns_qualified(
        self, tmp_path: Path
    ) -> None:
        """Registration adds entry and returns qualified alias."""
        cfg = AnimaWorksConfig()
        cfg.workspaces = {}
        with (
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.config.models.save_config") as mock_save,
        ):
            result = register_workspace("proj", str(tmp_path))
        assert result == qualified_alias("proj", str(tmp_path))
        assert cfg.workspaces["proj"] == str(tmp_path.resolve())
        mock_save.assert_called_once()

    def test_nonexistent_dir_raises_value_error(self) -> None:
        """Registering non-existent directory raises ValueError."""
        cfg = AnimaWorksConfig()
        cfg.workspaces = {}
        with patch("core.config.models.load_config", return_value=cfg), pytest.raises(ValueError) as excinfo:
            register_workspace("bad", "/nonexistent/path/xyz")
        assert "nonexistent" in str(excinfo.value) or "dir" in str(excinfo.value).lower()


# ── List workspaces ──────────────────────────────────────────────


class TestListWorkspaces:
    """Tests for list_workspaces()."""

    def test_returns_qualified_aliases_as_keys(self, tmp_path: Path) -> None:
        """list_workspaces returns {qualified_alias: path}."""
        cfg = AnimaWorksConfig()
        cfg.workspaces = {"a": str(tmp_path)}
        with patch("core.config.models.load_config", return_value=cfg):
            result = list_workspaces()
        expected_key = qualified_alias("a", str(tmp_path))
        assert expected_key in result
        assert result[expected_key] == str(tmp_path)

    def test_empty_registry_returns_empty_dict(self) -> None:
        """Empty workspaces returns empty dict."""
        cfg = AnimaWorksConfig()
        cfg.workspaces = {}
        with patch("core.config.models.load_config", return_value=cfg):
            result = list_workspaces()
        assert result == {}


# ── Remove workspace ────────────────────────────────────────────


class TestRemoveWorkspace:
    """Tests for remove_workspace()."""

    def test_removes_by_alias_returns_true(self, tmp_path: Path) -> None:
        """remove_workspace removes entry and returns True."""
        cfg = AnimaWorksConfig()
        cfg.workspaces = {"proj": str(tmp_path)}
        with (
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.config.models.save_config") as mock_save,
        ):
            result = remove_workspace("proj")
        assert result is True
        assert "proj" not in cfg.workspaces
        mock_save.assert_called_once()

    def test_missing_alias_returns_false(self) -> None:
        """remove_workspace for unknown alias returns False."""
        cfg = AnimaWorksConfig()
        cfg.workspaces = {}
        with patch("core.config.models.load_config", return_value=cfg):
            result = remove_workspace("nonexistent")
        assert result is False


# ── Workspace info ────────────────────────────────────────────────


class TestWorkspaceInfo:
    """Tests for workspace_info()."""

    def test_returns_details_for_registered(self, tmp_path: Path) -> None:
        """workspace_info returns dict with alias, path, qualified, exists."""
        cfg = AnimaWorksConfig()
        cfg.workspaces = {"proj": str(tmp_path)}
        with patch("core.config.models.load_config", return_value=cfg):
            result = workspace_info("proj")
        assert result is not None
        assert result["alias"] == "proj"
        assert result["path"] == str(tmp_path.resolve())
        assert result["qualified"] == qualified_alias("proj", str(tmp_path))
        assert result["exists"] is True

    def test_returns_none_for_missing(self) -> None:
        """workspace_info returns None for unregistered alias."""
        cfg = AnimaWorksConfig()
        cfg.workspaces = {}
        with patch("core.config.models.load_config", return_value=cfg):
            result = workspace_info("missing")
        assert result is None
