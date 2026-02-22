"""Unit tests for supervisor activity_log read permissions.

Verifies that:
- A supervisor can read direct subordinate's activity_log/
- A supervisor cannot read subordinate's other directories (episodes, knowledge, etc.)
- A non-supervisor cannot read another anima's activity_log/
- Path traversal via ``..`` is blocked by resolve()
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory import MemoryManager
from core.tooling.handler import ToolHandler


# ── Helpers ───────────────────────────────────────────────


def _make_config_with_hierarchy(supervisor: str, subordinates: list[str]):
    """Create a mock config where supervisor supervises the given subordinates."""
    mock_cfg = MagicMock()
    animas: dict[str, MagicMock] = {}
    for sub in subordinates:
        sub_cfg = MagicMock()
        sub_cfg.supervisor = supervisor
        animas[sub] = sub_cfg
    # Add supervisor itself (no supervisor field)
    sup_cfg = MagicMock()
    sup_cfg.supervisor = None
    animas[supervisor] = sup_cfg
    mock_cfg.animas = animas
    return mock_cfg


def _make_handler(
    tmp_path: Path,
    anima_name: str,
    *,
    mock_cfg: MagicMock | None = None,
) -> ToolHandler:
    """Create a ToolHandler with minimal directory structure.

    If *mock_cfg* is given, ``load_config`` and ``get_animas_dir`` are patched
    during construction so that subordinate paths are cached at init time.
    """
    anima_dir = tmp_path / "animas" / anima_name
    for d in ["state", "episodes", "knowledge", "procedures", "skills", "activity_log"]:
        (anima_dir / d).mkdir(parents=True, exist_ok=True)
    memory = MemoryManager(anima_dir)

    if mock_cfg is not None:
        with patch("core.config.models.load_config", return_value=mock_cfg), \
             patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"):
            return ToolHandler(anima_dir, memory)
    return ToolHandler(anima_dir, memory)


# ── Tests ─────────────────────────────────────────────────


class TestSupervisorActivityLogPermission:
    """Supervisor can read direct subordinate's activity_log."""

    def test_supervisor_can_read_subordinate_activity_log(self, tmp_path: Path):
        """Supervisor mio can read subordinate yuki's activity_log."""
        # Create subordinate's activity_log file
        sub_activity_dir = tmp_path / "animas" / "yuki" / "activity_log"
        sub_activity_dir.mkdir(parents=True, exist_ok=True)
        log_file = sub_activity_dir / "2026-02-22.jsonl"
        log_file.write_text('{"ts": "2026-02-22T10:00:00", "type": "tool_use"}\n', encoding="utf-8")

        mock_cfg = _make_config_with_hierarchy("mio", ["yuki", "sakura"])
        handler = _make_handler(tmp_path, "mio", mock_cfg=mock_cfg)

        result = handler._check_file_permission(str(log_file), write=False)

        assert result is None, "Supervisor should be allowed to read subordinate's activity_log"

    def test_supervisor_cannot_read_subordinate_episodes(self, tmp_path: Path):
        """Supervisor mio cannot read subordinate yuki's episodes/."""
        # Create subordinate's episodes directory
        sub_episodes_dir = tmp_path / "animas" / "yuki" / "episodes"
        sub_episodes_dir.mkdir(parents=True, exist_ok=True)
        episode_file = sub_episodes_dir / "2026-02-22.md"
        episode_file.write_text("private episode content", encoding="utf-8")

        mock_cfg = _make_config_with_hierarchy("mio", ["yuki"])
        handler = _make_handler(tmp_path, "mio", mock_cfg=mock_cfg)

        result = handler._check_file_permission(str(episode_file), write=False)

        assert result is not None, "Supervisor should NOT be allowed to read subordinate's episodes"

    def test_supervisor_cannot_read_subordinate_identity(self, tmp_path: Path):
        """Supervisor mio cannot read subordinate yuki's identity.md."""
        sub_dir = tmp_path / "animas" / "yuki"
        sub_dir.mkdir(parents=True, exist_ok=True)
        identity_file = sub_dir / "identity.md"
        identity_file.write_text("private identity", encoding="utf-8")

        mock_cfg = _make_config_with_hierarchy("mio", ["yuki"])
        handler = _make_handler(tmp_path, "mio", mock_cfg=mock_cfg)

        result = handler._check_file_permission(str(identity_file), write=False)

        assert result is not None, "Supervisor should NOT be allowed to read subordinate's identity.md"

    def test_supervisor_cannot_write_subordinate_activity_log(self, tmp_path: Path):
        """Supervisor mio cannot WRITE to subordinate yuki's activity_log."""
        sub_activity_dir = tmp_path / "animas" / "yuki" / "activity_log"
        sub_activity_dir.mkdir(parents=True, exist_ok=True)
        log_file = sub_activity_dir / "2026-02-22.jsonl"
        log_file.write_text('{"ts": "2026-02-22T10:00:00"}\n', encoding="utf-8")

        mock_cfg = _make_config_with_hierarchy("mio", ["yuki"])
        handler = _make_handler(tmp_path, "mio", mock_cfg=mock_cfg)

        result = handler._check_file_permission(str(log_file), write=True)

        assert result is not None, "Supervisor should NOT be allowed to write to subordinate's activity_log"

    def test_non_supervisor_cannot_read_activity_log(self, tmp_path: Path):
        """Non-supervisor rin cannot read yuki's activity_log."""
        sub_activity_dir = tmp_path / "animas" / "yuki" / "activity_log"
        sub_activity_dir.mkdir(parents=True, exist_ok=True)
        log_file = sub_activity_dir / "2026-02-22.jsonl"
        log_file.write_text('{"ts": "2026-02-22T10:00:00"}\n', encoding="utf-8")

        # yuki's supervisor is mio, not rin
        mock_cfg = _make_config_with_hierarchy("mio", ["yuki"])
        # Add rin as a peer (no subordinates)
        rin_cfg = MagicMock()
        rin_cfg.supervisor = "mio"
        mock_cfg.animas["rin"] = rin_cfg

        handler = _make_handler(tmp_path, "rin", mock_cfg=mock_cfg)

        result = handler._check_file_permission(str(log_file), write=False)

        assert result is not None, "Non-supervisor should NOT be allowed to read another anima's activity_log"

    def test_supervisor_cannot_read_grandchild_activity_log(self, tmp_path: Path):
        """Supervisor mio cannot read grandchild (yuki's subordinate) activity_log."""
        # grandchild's activity_log
        grandchild_dir = tmp_path / "animas" / "hana" / "activity_log"
        grandchild_dir.mkdir(parents=True, exist_ok=True)
        log_file = grandchild_dir / "2026-02-22.jsonl"
        log_file.write_text('{"ts": "2026-02-22T10:00:00"}\n', encoding="utf-8")

        # mio -> yuki -> hana (grandchild)
        mock_cfg = MagicMock()
        yuki_cfg = MagicMock()
        yuki_cfg.supervisor = "mio"
        hana_cfg = MagicMock()
        hana_cfg.supervisor = "yuki"  # hana reports to yuki, not mio
        mio_cfg = MagicMock()
        mio_cfg.supervisor = None
        mock_cfg.animas = {"mio": mio_cfg, "yuki": yuki_cfg, "hana": hana_cfg}

        handler = _make_handler(tmp_path, "mio", mock_cfg=mock_cfg)

        result = handler._check_file_permission(str(log_file), write=False)

        assert result is not None, "Supervisor should NOT be allowed to read grandchild's activity_log"

    def test_config_load_failure_denies_gracefully(self, tmp_path: Path):
        """If load_config fails at init, _subordinate_activity_dirs is empty → denied."""
        sub_activity_dir = tmp_path / "animas" / "yuki" / "activity_log"
        sub_activity_dir.mkdir(parents=True, exist_ok=True)
        log_file = sub_activity_dir / "2026-02-22.jsonl"
        log_file.write_text('{"ts": "2026-02-22T10:00:00"}\n', encoding="utf-8")

        # Construct handler with broken config → empty cache
        with patch("core.config.models.load_config", side_effect=RuntimeError("config broken")), \
             patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"):
            handler = _make_handler(tmp_path, "mio")

        result = handler._check_file_permission(str(log_file), write=False)

        # Should be denied (empty cache, falls through to permissions.md)
        assert result is not None, "Config failure should result in denial"

    def test_path_traversal_via_dotdot_denied(self, tmp_path: Path):
        """Path traversal ``activity_log/../../identity.md`` is denied after resolve()."""
        # Create subordinate dirs
        sub_dir = tmp_path / "animas" / "yuki"
        (sub_dir / "activity_log").mkdir(parents=True, exist_ok=True)
        identity_file = sub_dir / "identity.md"
        identity_file.write_text("private identity", encoding="utf-8")

        mock_cfg = _make_config_with_hierarchy("mio", ["yuki"])
        handler = _make_handler(tmp_path, "mio", mock_cfg=mock_cfg)

        # Try to escape activity_log via ..
        traversal_path = str(sub_dir / "activity_log" / ".." / "identity.md")
        result = handler._check_file_permission(traversal_path, write=False)

        assert result is not None, "Path traversal via .. should be denied"

    def test_subordinate_paths_cached_at_init(self, tmp_path: Path):
        """Verify _subordinate_activity_dirs is populated at init, not per-call."""
        (tmp_path / "animas" / "yuki" / "activity_log").mkdir(parents=True, exist_ok=True)

        mock_cfg = _make_config_with_hierarchy("mio", ["yuki"])
        handler = _make_handler(tmp_path, "mio", mock_cfg=mock_cfg)

        # Cache should contain exactly one path
        assert len(handler._subordinate_activity_dirs) == 1
        assert handler._subordinate_activity_dirs[0] == (tmp_path / "animas" / "yuki" / "activity_log").resolve()
