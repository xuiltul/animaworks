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

from core.exceptions import ConfigError
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

    def test_supervisor_can_read_subordinate_identity(self, tmp_path: Path):
        """Supervisor mio can read descendant yuki's IDENTITY.md (read_subordinate_state)."""
        sub_dir = tmp_path / "animas" / "yuki"
        sub_dir.mkdir(parents=True, exist_ok=True)
        identity_file = sub_dir / "IDENTITY.md"
        identity_file.write_text("private identity", encoding="utf-8")

        mock_cfg = _make_config_with_hierarchy("mio", ["yuki"])
        handler = _make_handler(tmp_path, "mio", mock_cfg=mock_cfg)

        result = handler._check_file_permission(str(identity_file), write=False)

        assert result is None, "Supervisor should be allowed to read descendant's IDENTITY.md"

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

    def test_unrelated_anima_cannot_read_activity_log(self, tmp_path: Path):
        """Anima under a different supervisor cannot read yuki's activity_log."""
        sub_activity_dir = tmp_path / "animas" / "yuki" / "activity_log"
        sub_activity_dir.mkdir(parents=True, exist_ok=True)
        log_file = sub_activity_dir / "2026-02-22.jsonl"
        log_file.write_text('{"ts": "2026-02-22T10:00:00"}\n', encoding="utf-8")

        # yuki's supervisor is mio, rin's supervisor is sakura → different teams
        mock_cfg = _make_config_with_hierarchy("mio", ["yuki"])
        rin_cfg = MagicMock()
        rin_cfg.supervisor = "sakura"
        sakura_cfg = MagicMock()
        sakura_cfg.supervisor = None
        mock_cfg.animas["rin"] = rin_cfg
        mock_cfg.animas["sakura"] = sakura_cfg

        handler = _make_handler(tmp_path, "rin", mock_cfg=mock_cfg)

        result = handler._check_file_permission(str(log_file), write=False)

        assert result is not None, "Anima under different supervisor should NOT be allowed to read another's activity_log"

    def test_supervisor_can_read_grandchild_activity_log(self, tmp_path: Path):
        """Supervisor mio CAN read grandchild (yuki's subordinate) activity_log.

        Since the supervisor-tools-expansion feature, supervisors can read
        any descendant's activity_log (not just direct subordinates).
        """
        grandchild_dir = tmp_path / "animas" / "hana" / "activity_log"
        grandchild_dir.mkdir(parents=True, exist_ok=True)
        log_file = grandchild_dir / "2026-02-22.jsonl"
        log_file.write_text('{"ts": "2026-02-22T10:00:00"}\n', encoding="utf-8")

        # mio -> yuki -> hana (grandchild)
        mock_cfg = MagicMock()
        yuki_cfg = MagicMock()
        yuki_cfg.supervisor = "mio"
        hana_cfg = MagicMock()
        hana_cfg.supervisor = "yuki"
        mio_cfg = MagicMock()
        mio_cfg.supervisor = None
        mock_cfg.animas = {"mio": mio_cfg, "yuki": yuki_cfg, "hana": hana_cfg}

        handler = _make_handler(tmp_path, "mio", mock_cfg=mock_cfg)

        result = handler._check_file_permission(str(log_file), write=False)

        assert result is None, "Supervisor SHOULD be allowed to read grandchild's activity_log"

    def test_config_load_failure_denies_gracefully(self, tmp_path: Path):
        """If load_config fails at init, _subordinate_activity_dirs is empty → denied."""
        sub_activity_dir = tmp_path / "animas" / "yuki" / "activity_log"
        sub_activity_dir.mkdir(parents=True, exist_ok=True)
        log_file = sub_activity_dir / "2026-02-22.jsonl"
        log_file.write_text('{"ts": "2026-02-22T10:00:00"}\n', encoding="utf-8")

        # Construct handler with broken config → empty cache
        with patch("core.config.models.load_config", side_effect=ConfigError("config broken")), \
             patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"):
            handler = _make_handler(tmp_path, "mio")

        result = handler._check_file_permission(str(log_file), write=False)

        # Should be denied (empty cache, falls through to permissions.md)
        assert result is not None, "Config failure should result in denial"

    def test_path_traversal_via_dotdot_denied(self, tmp_path: Path):
        """Path traversal ``activity_log/../episodes/...`` resolves to episodes, which is denied."""
        # Create subordinate dirs (IDENTITY.md is now allowed for descendants, so target episodes/)
        sub_dir = tmp_path / "animas" / "yuki"
        (sub_dir / "activity_log").mkdir(parents=True, exist_ok=True)
        (sub_dir / "episodes").mkdir(parents=True, exist_ok=True)
        episode_file = sub_dir / "episodes" / "2026-02-22.md"
        episode_file.write_text("private episode content", encoding="utf-8")

        mock_cfg = _make_config_with_hierarchy("mio", ["yuki"])
        handler = _make_handler(tmp_path, "mio", mock_cfg=mock_cfg)

        # Try to escape activity_log via .. to reach episodes (denied for descendants)
        traversal_path = str(sub_dir / "activity_log" / ".." / "episodes" / "2026-02-22.md")
        result = handler._check_file_permission(traversal_path, write=False)

        assert result is not None, "Path traversal via .. to episodes should be denied"

    def test_subordinate_paths_cached_at_init(self, tmp_path: Path):
        """Verify _subordinate_activity_dirs is populated at init, not per-call."""
        (tmp_path / "animas" / "yuki" / "activity_log").mkdir(parents=True, exist_ok=True)

        mock_cfg = _make_config_with_hierarchy("mio", ["yuki"])
        handler = _make_handler(tmp_path, "mio", mock_cfg=mock_cfg)

        # Cache should contain exactly one path
        assert len(handler._subordinate_activity_dirs) == 1
        assert handler._subordinate_activity_dirs[0] == (tmp_path / "animas" / "yuki" / "activity_log").resolve()


class TestPeerActivityLogPermission:
    """Peers (same supervisor) can read each other's activity_log."""

    def test_peer_can_read_peer_activity_log(self, tmp_path: Path):
        """Top-level peer sakura can read ritsu's activity_log."""
        ritsu_activity_dir = tmp_path / "animas" / "ritsu" / "activity_log"
        ritsu_activity_dir.mkdir(parents=True, exist_ok=True)
        log_file = ritsu_activity_dir / "2026-03-01.jsonl"
        log_file.write_text('{"ts": "2026-03-01T10:00:00"}\n', encoding="utf-8")

        # Both sakura and ritsu are top-level (supervisor=None)
        mock_cfg = MagicMock()
        sakura_cfg = MagicMock()
        sakura_cfg.supervisor = None
        ritsu_cfg = MagicMock()
        ritsu_cfg.supervisor = None
        mock_cfg.animas = {"sakura": sakura_cfg, "ritsu": ritsu_cfg}

        handler = _make_handler(tmp_path, "sakura", mock_cfg=mock_cfg)

        result = handler._check_file_permission(str(log_file), write=False)

        assert result is None, "Peer should be allowed to read peer's activity_log"

    def test_peer_cannot_write_peer_activity_log(self, tmp_path: Path):
        """Peer sakura cannot WRITE to ritsu's activity_log."""
        ritsu_activity_dir = tmp_path / "animas" / "ritsu" / "activity_log"
        ritsu_activity_dir.mkdir(parents=True, exist_ok=True)
        log_file = ritsu_activity_dir / "2026-03-01.jsonl"
        log_file.write_text('{"ts": "2026-03-01T10:00:00"}\n', encoding="utf-8")

        mock_cfg = MagicMock()
        sakura_cfg = MagicMock()
        sakura_cfg.supervisor = None
        ritsu_cfg = MagicMock()
        ritsu_cfg.supervisor = None
        mock_cfg.animas = {"sakura": sakura_cfg, "ritsu": ritsu_cfg}

        handler = _make_handler(tmp_path, "sakura", mock_cfg=mock_cfg)

        result = handler._check_file_permission(str(log_file), write=True)

        assert result is not None, "Peer should NOT be allowed to write to peer's activity_log"

    def test_peer_cannot_read_peer_episodes(self, tmp_path: Path):
        """Peer sakura cannot read ritsu's episodes/ (only activity_log)."""
        ritsu_episodes_dir = tmp_path / "animas" / "ritsu" / "episodes"
        ritsu_episodes_dir.mkdir(parents=True, exist_ok=True)
        episode_file = ritsu_episodes_dir / "2026-03-01.md"
        episode_file.write_text("private episode", encoding="utf-8")

        mock_cfg = MagicMock()
        sakura_cfg = MagicMock()
        sakura_cfg.supervisor = None
        ritsu_cfg = MagicMock()
        ritsu_cfg.supervisor = None
        mock_cfg.animas = {"sakura": sakura_cfg, "ritsu": ritsu_cfg}

        handler = _make_handler(tmp_path, "sakura", mock_cfg=mock_cfg)

        result = handler._check_file_permission(str(episode_file), write=False)

        assert result is not None, "Peer should NOT be allowed to read peer's episodes"

    def test_non_peer_different_supervisor_denied(self, tmp_path: Path):
        """Animas with different supervisors are not peers and cannot access each other."""
        hinata_activity_dir = tmp_path / "animas" / "hinata" / "activity_log"
        hinata_activity_dir.mkdir(parents=True, exist_ok=True)
        log_file = hinata_activity_dir / "2026-03-01.jsonl"
        log_file.write_text('{"ts": "2026-03-01T10:00:00"}\n', encoding="utf-8")

        # yuki's supervisor is mio, hinata's supervisor is rin → not peers
        mock_cfg = MagicMock()
        yuki_cfg = MagicMock()
        yuki_cfg.supervisor = "mio"
        hinata_cfg = MagicMock()
        hinata_cfg.supervisor = "rin"
        mio_cfg = MagicMock()
        mio_cfg.supervisor = "sakura"
        rin_cfg = MagicMock()
        rin_cfg.supervisor = "sakura"
        mock_cfg.animas = {"yuki": yuki_cfg, "hinata": hinata_cfg, "mio": mio_cfg, "rin": rin_cfg}

        handler = _make_handler(tmp_path, "yuki", mock_cfg=mock_cfg)

        result = handler._check_file_permission(str(log_file), write=False)

        assert result is not None, "Non-peer (different supervisor) should NOT be allowed to read activity_log"

    def test_peers_under_same_supervisor(self, tmp_path: Path):
        """Non-top-level peers under the same supervisor can read each other's activity_log."""
        tsumugi_activity_dir = tmp_path / "animas" / "tsumugi" / "activity_log"
        tsumugi_activity_dir.mkdir(parents=True, exist_ok=True)
        log_file = tsumugi_activity_dir / "2026-03-01.jsonl"
        log_file.write_text('{"ts": "2026-03-01T10:00:00"}\n', encoding="utf-8")

        # Both yuki and tsumugi have supervisor=mio → peers
        mock_cfg = MagicMock()
        yuki_cfg = MagicMock()
        yuki_cfg.supervisor = "mio"
        tsumugi_cfg = MagicMock()
        tsumugi_cfg.supervisor = "mio"
        shizuku_cfg = MagicMock()
        shizuku_cfg.supervisor = "mio"
        mio_cfg = MagicMock()
        mio_cfg.supervisor = "sakura"
        mock_cfg.animas = {
            "yuki": yuki_cfg, "tsumugi": tsumugi_cfg,
            "shizuku": shizuku_cfg, "mio": mio_cfg,
        }

        handler = _make_handler(tmp_path, "yuki", mock_cfg=mock_cfg)

        result = handler._check_file_permission(str(log_file), write=False)

        assert result is None, "Peers under same supervisor should be allowed to read each other's activity_log"

    def test_peer_activity_dirs_cached_at_init(self, tmp_path: Path):
        """Verify _peer_activity_dirs is populated at init."""
        (tmp_path / "animas" / "ritsu" / "activity_log").mkdir(parents=True, exist_ok=True)
        (tmp_path / "animas" / "mei" / "activity_log").mkdir(parents=True, exist_ok=True)

        mock_cfg = MagicMock()
        sakura_cfg = MagicMock()
        sakura_cfg.supervisor = None
        ritsu_cfg = MagicMock()
        ritsu_cfg.supervisor = None
        mei_cfg = MagicMock()
        mei_cfg.supervisor = None
        mock_cfg.animas = {"sakura": sakura_cfg, "ritsu": ritsu_cfg, "mei": mei_cfg}

        handler = _make_handler(tmp_path, "sakura", mock_cfg=mock_cfg)

        # sakura has 2 peers: ritsu and mei
        assert len(handler._peer_activity_dirs) == 2
