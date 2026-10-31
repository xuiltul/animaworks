"""Unit and integration tests for `animaworks anima rename` command."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.config.models import (
    AnimaModelConfig,
    AnimaWorksConfig,
    ExternalMessagingChannelConfig,
    ExternalMessagingConfig,
    rename_anima_in_config,
    save_config,
)

# ── Helpers ──────────────────────────────────────────────────


def _setup_data_dir(tmp_path: Path) -> Path:
    """Create a minimal data directory structure for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "animas").mkdir()
    (data_dir / "shared" / "inbox").mkdir(parents=True)
    (data_dir / "shared" / "dm_logs").mkdir(parents=True)
    (data_dir / "shared" / "channels").mkdir(parents=True)
    (data_dir / "run" / "sockets").mkdir(parents=True)
    (data_dir / "run" / "animas").mkdir(parents=True)
    return data_dir


def _create_anima(data_dir: Path, name: str, *, supervisor: str | None = None) -> Path:
    """Create a minimal anima directory with IDENTITY.md and status.json."""
    anima_dir = data_dir / "animas" / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "IDENTITY.md").write_text(f"# {name}\n", encoding="utf-8")
    (anima_dir / "activity_log").mkdir(exist_ok=True)
    (anima_dir / "state").mkdir(exist_ok=True)

    status = {"enabled": True, "model": "claude-sonnet-4-6", "role": "general"}
    if supervisor:
        status["supervisor"] = supervisor
    (anima_dir / "status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return anima_dir


def _create_config(data_dir: Path, animas: dict[str, dict]) -> None:
    """Create a config.json with the given animas entries."""
    config = AnimaWorksConfig()
    for name, entry in animas.items():
        config.animas[name] = AnimaModelConfig(**entry)
    save_config(config, data_dir / "config.json")


# ── Tests for rename_anima_in_config ─────────────────────────


class TestRenameAnimaInConfig:
    """Tests for core.config.models.rename_anima_in_config()."""

    def test_basic_rename(self, tmp_path: Path) -> None:
        """config.animas key is renamed."""
        data_dir = _setup_data_dir(tmp_path)
        _create_config(data_dir, {"sakura": {"supervisor": None}})

        rename_anima_in_config(data_dir, "sakura", "hinata")

        raw = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
        assert "hinata" in raw["animas"]
        assert "sakura" not in raw["animas"]

    def test_supervisor_references_updated(self, tmp_path: Path) -> None:
        """Other animas' supervisor references are updated."""
        data_dir = _setup_data_dir(tmp_path)
        _create_config(data_dir, {
            "sakura": {"supervisor": None},
            "mei": {"supervisor": "sakura"},
            "aoi": {"supervisor": "sakura"},
            "kana": {"supervisor": "mei"},
        })

        count = rename_anima_in_config(data_dir, "sakura", "hinata")

        assert count == 2
        raw = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
        assert raw["animas"]["mei"]["supervisor"] == "hinata"
        assert raw["animas"]["aoi"]["supervisor"] == "hinata"
        assert raw["animas"]["kana"]["supervisor"] == "mei"

    def test_anima_mapping_updated(self, tmp_path: Path) -> None:
        """external_messaging.anima_mapping values are updated."""
        data_dir = _setup_data_dir(tmp_path)
        config = AnimaWorksConfig()
        config.animas["sakura"] = AnimaModelConfig()
        config.external_messaging = ExternalMessagingConfig(
            slack=ExternalMessagingChannelConfig(
                anima_mapping={"C123": "sakura", "C456": "mei"}
            ),
            chatwork=ExternalMessagingChannelConfig(
                anima_mapping={"R789": "sakura"}
            ),
        )
        save_config(config, data_dir / "config.json")

        rename_anima_in_config(data_dir, "sakura", "hinata")

        raw = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
        slack_mapping = raw["external_messaging"]["slack"]["anima_mapping"]
        assert slack_mapping["C123"] == "hinata"
        assert slack_mapping["C456"] == "mei"
        cw_mapping = raw["external_messaging"]["chatwork"]["anima_mapping"]
        assert cw_mapping["R789"] == "hinata"

    def test_app_id_mapping_updated(self, tmp_path: Path) -> None:
        """external_messaging.app_id_mapping values are updated."""
        data_dir = _setup_data_dir(tmp_path)
        config = AnimaWorksConfig()
        config.animas["sakura"] = AnimaModelConfig()
        config.external_messaging = ExternalMessagingConfig(
            slack=ExternalMessagingChannelConfig(
                app_id_mapping={"A111": "sakura", "A222": "mei"}
            ),
        )
        save_config(config, data_dir / "config.json")

        rename_anima_in_config(data_dir, "sakura", "hinata")

        raw = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
        app_mapping = raw["external_messaging"]["slack"]["app_id_mapping"]
        assert app_mapping["A111"] == "hinata"
        assert app_mapping["A222"] == "mei"

    def test_default_anima_updated(self, tmp_path: Path) -> None:
        """external_messaging default_anima is updated."""
        data_dir = _setup_data_dir(tmp_path)
        config = AnimaWorksConfig()
        config.animas["sakura"] = AnimaModelConfig()
        config.external_messaging = ExternalMessagingConfig(
            slack=ExternalMessagingChannelConfig(default_anima="sakura"),
            chatwork=ExternalMessagingChannelConfig(default_anima="mei"),
        )
        save_config(config, data_dir / "config.json")

        rename_anima_in_config(data_dir, "sakura", "hinata")

        raw = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
        assert raw["external_messaging"]["slack"]["default_anima"] == "hinata"
        assert raw["external_messaging"]["chatwork"]["default_anima"] == "mei"

    def test_nonexistent_anima_raises(self, tmp_path: Path) -> None:
        """KeyError for non-existent anima name."""
        data_dir = _setup_data_dir(tmp_path)
        _create_config(data_dir, {"mei": {}})

        with pytest.raises(KeyError, match="not found"):
            rename_anima_in_config(data_dir, "sakura", "hinata")


# ── Tests for _rename_dm_logs ─────────────────────────────────


class TestRenameDmLogs:
    """Tests for DM log file renaming."""

    def test_rename_dm_log_file(self, tmp_path: Path) -> None:
        """DM log files are correctly renamed."""
        from cli.commands.anima_mgmt import _rename_dm_logs

        shared_dir = tmp_path / "shared"
        dm_dir = shared_dir / "dm_logs"
        dm_dir.mkdir(parents=True)

        (dm_dir / "mei-sakura.jsonl").write_text('{"ts":"2026-03-01"}\n', encoding="utf-8")

        count = _rename_dm_logs(shared_dir, "sakura", "hinata")

        assert count == 1
        assert not (dm_dir / "mei-sakura.jsonl").exists()
        new_file = dm_dir / "hinata-mei.jsonl"
        assert new_file.exists()
        assert '{"ts":"2026-03-01"}' in new_file.read_text(encoding="utf-8")

    def test_rename_preserves_sort_order(self, tmp_path: Path) -> None:
        """Renamed DM log file has correct sort order in name."""
        from cli.commands.anima_mgmt import _rename_dm_logs

        shared_dir = tmp_path / "shared"
        dm_dir = shared_dir / "dm_logs"
        dm_dir.mkdir(parents=True)

        (dm_dir / "aoi-sakura.jsonl").write_text("data\n", encoding="utf-8")

        _rename_dm_logs(shared_dir, "sakura", "zzz")

        assert (dm_dir / "aoi-zzz.jsonl").exists()

    def test_merge_on_collision(self, tmp_path: Path) -> None:
        """Merges content when renamed file already exists."""
        from cli.commands.anima_mgmt import _rename_dm_logs

        shared_dir = tmp_path / "shared"
        dm_dir = shared_dir / "dm_logs"
        dm_dir.mkdir(parents=True)

        (dm_dir / "mei-sakura.jsonl").write_text("old\n", encoding="utf-8")
        (dm_dir / "hinata-mei.jsonl").write_text("existing\n", encoding="utf-8")

        count = _rename_dm_logs(shared_dir, "sakura", "hinata")

        assert count == 1
        assert not (dm_dir / "mei-sakura.jsonl").exists()
        content = (dm_dir / "hinata-mei.jsonl").read_text(encoding="utf-8")
        assert "existing" in content
        assert "old" in content

    def test_no_dm_logs_dir(self, tmp_path: Path) -> None:
        """Returns 0 when dm_logs/ does not exist."""
        from cli.commands.anima_mgmt import _rename_dm_logs

        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()

        count = _rename_dm_logs(shared_dir, "sakura", "hinata")

        assert count == 0


# ── Tests for cmd_anima_rename (integration) ──────────────────


class TestCmdAnimaRename:
    """Integration tests for the CLI command."""

    def _make_args(
        self,
        old_name: str,
        new_name: str,
        *,
        force: bool = True,
        gateway_url: str | None = None,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            old_name=old_name,
            new_name=new_name,
            force=force,
            gateway_url=gateway_url,
        )

    @patch("core.paths.get_data_dir")
    @patch("core.paths.get_animas_dir")
    def test_full_rename_offline(
        self, mock_animas_dir, mock_data_dir, tmp_path: Path,
    ) -> None:
        """Full rename with server offline."""
        from cli.commands.anima_mgmt import cmd_anima_rename

        data_dir = _setup_data_dir(tmp_path)
        _create_anima(data_dir, "sakura")
        _create_anima(data_dir, "mei", supervisor="sakura")
        _create_config(data_dir, {
            "sakura": {"supervisor": None},
            "mei": {"supervisor": "sakura"},
        })

        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = data_dir / "animas"

        args = self._make_args("sakura", "hinata")
        cmd_anima_rename(args)

        assert not (data_dir / "animas" / "sakura").exists()
        assert (data_dir / "animas" / "hinata" / "IDENTITY.md").exists()

        raw = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
        assert "hinata" in raw["animas"]
        assert "sakura" not in raw["animas"]
        assert raw["animas"]["mei"]["supervisor"] == "hinata"

    @patch("core.paths.get_data_dir")
    @patch("core.paths.get_animas_dir")
    def test_target_exists_error(
        self, mock_animas_dir, mock_data_dir, tmp_path: Path,
    ) -> None:
        """Error when target name already exists."""
        from cli.commands.anima_mgmt import cmd_anima_rename

        data_dir = _setup_data_dir(tmp_path)
        _create_anima(data_dir, "sakura")
        _create_anima(data_dir, "hinata")

        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = data_dir / "animas"

        with pytest.raises(SystemExit):
            cmd_anima_rename(self._make_args("sakura", "hinata"))

    @patch("core.paths.get_data_dir")
    @patch("core.paths.get_animas_dir")
    def test_source_not_found_error(
        self, mock_animas_dir, mock_data_dir, tmp_path: Path,
    ) -> None:
        """Error when source anima doesn't exist."""
        from cli.commands.anima_mgmt import cmd_anima_rename

        data_dir = _setup_data_dir(tmp_path)

        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = data_dir / "animas"

        with pytest.raises(SystemExit):
            cmd_anima_rename(self._make_args("sakura", "hinata"))

    @patch("core.paths.get_data_dir")
    @patch("core.paths.get_animas_dir")
    def test_invalid_new_name_error(
        self, mock_animas_dir, mock_data_dir, tmp_path: Path,
    ) -> None:
        """Error for invalid new name."""
        from cli.commands.anima_mgmt import cmd_anima_rename

        data_dir = _setup_data_dir(tmp_path)
        _create_anima(data_dir, "sakura")

        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = data_dir / "animas"

        with pytest.raises(SystemExit):
            cmd_anima_rename(self._make_args("sakura", "INVALID-NAME"))

    @patch("core.paths.get_data_dir")
    @patch("core.paths.get_animas_dir")
    def test_same_name_error(
        self, mock_animas_dir, mock_data_dir, tmp_path: Path,
    ) -> None:
        """Error when old and new names are the same."""
        from cli.commands.anima_mgmt import cmd_anima_rename

        data_dir = _setup_data_dir(tmp_path)
        _create_anima(data_dir, "sakura")

        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = data_dir / "animas"

        with pytest.raises(SystemExit):
            cmd_anima_rename(self._make_args("sakura", "sakura"))

    @patch("core.paths.get_data_dir")
    @patch("core.paths.get_animas_dir")
    def test_status_json_supervisor_updated(
        self, mock_animas_dir, mock_data_dir, tmp_path: Path,
    ) -> None:
        """Other animas' status.json supervisor refs are updated."""
        from cli.commands.anima_mgmt import cmd_anima_rename

        data_dir = _setup_data_dir(tmp_path)
        _create_anima(data_dir, "sakura")
        _create_anima(data_dir, "mei", supervisor="sakura")
        _create_anima(data_dir, "aoi", supervisor="sakura")
        _create_config(data_dir, {
            "sakura": {},
            "mei": {"supervisor": "sakura"},
            "aoi": {"supervisor": "sakura"},
        })

        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = data_dir / "animas"

        cmd_anima_rename(self._make_args("sakura", "hinata"))

        mei_status = json.loads(
            (data_dir / "animas" / "mei" / "status.json").read_text(encoding="utf-8")
        )
        assert mei_status["supervisor"] == "hinata"

        aoi_status = json.loads(
            (data_dir / "animas" / "aoi" / "status.json").read_text(encoding="utf-8")
        )
        assert aoi_status["supervisor"] == "hinata"

    @patch("core.paths.get_data_dir")
    @patch("core.paths.get_animas_dir")
    def test_inbox_renamed(
        self, mock_animas_dir, mock_data_dir, tmp_path: Path,
    ) -> None:
        """Inbox directory is renamed when it exists."""
        from cli.commands.anima_mgmt import cmd_anima_rename

        data_dir = _setup_data_dir(tmp_path)
        _create_anima(data_dir, "sakura")
        _create_config(data_dir, {"sakura": {}})
        (data_dir / "shared" / "inbox" / "sakura").mkdir()
        (data_dir / "shared" / "inbox" / "sakura" / "msg.json").write_text("{}", encoding="utf-8")

        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = data_dir / "animas"

        cmd_anima_rename(self._make_args("sakura", "hinata"))

        assert not (data_dir / "shared" / "inbox" / "sakura").exists()
        assert (data_dir / "shared" / "inbox" / "hinata" / "msg.json").exists()

    @patch("core.paths.get_data_dir")
    @patch("core.paths.get_animas_dir")
    def test_dm_logs_renamed(
        self, mock_animas_dir, mock_data_dir, tmp_path: Path,
    ) -> None:
        """DM log files are renamed during full rename."""
        from cli.commands.anima_mgmt import cmd_anima_rename

        data_dir = _setup_data_dir(tmp_path)
        _create_anima(data_dir, "sakura")
        _create_config(data_dir, {"sakura": {}})
        (data_dir / "shared" / "dm_logs" / "mei-sakura.jsonl").write_text(
            '{"test": true}\n', encoding="utf-8",
        )

        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = data_dir / "animas"

        cmd_anima_rename(self._make_args("sakura", "hinata"))

        assert not (data_dir / "shared" / "dm_logs" / "mei-sakura.jsonl").exists()
        assert (data_dir / "shared" / "dm_logs" / "hinata-mei.jsonl").exists()

    @patch("core.paths.get_data_dir")
    @patch("core.paths.get_animas_dir")
    def test_rag_index_meta_cleared(
        self, mock_animas_dir, mock_data_dir, tmp_path: Path,
    ) -> None:
        """RAG index_meta.json is cleared after rename."""
        from cli.commands.anima_mgmt import cmd_anima_rename

        data_dir = _setup_data_dir(tmp_path)
        _create_anima(data_dir, "sakura")
        _create_config(data_dir, {"sakura": {}})
        (data_dir / "animas" / "sakura" / "index_meta.json").write_text(
            '{"old": "data"}\n', encoding="utf-8",
        )
        (data_dir / "animas" / "sakura" / "shared_index_meta.json").write_text(
            '{"shared_company_name": "old-company"}\n',
            encoding="utf-8",
        )

        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = data_dir / "animas"

        cmd_anima_rename(self._make_args("sakura", "hinata"))

        meta = (data_dir / "animas" / "hinata" / "index_meta.json").read_text(encoding="utf-8")
        assert json.loads(meta) == {}
        shared_meta = (data_dir / "animas" / "hinata" / "shared_index_meta.json").read_text(encoding="utf-8")
        assert json.loads(shared_meta) == {}

    @patch("core.paths.get_data_dir")
    @patch("core.paths.get_animas_dir")
    def test_stale_socket_cleaned(
        self, mock_animas_dir, mock_data_dir, tmp_path: Path,
    ) -> None:
        """Stale socket and PID files are cleaned up."""
        from cli.commands.anima_mgmt import cmd_anima_rename

        data_dir = _setup_data_dir(tmp_path)
        _create_anima(data_dir, "sakura")
        _create_config(data_dir, {"sakura": {}})
        (data_dir / "run" / "sockets" / "sakura.sock").write_text("", encoding="utf-8")
        (data_dir / "run" / "animas" / "sakura.pid").write_text("12345", encoding="utf-8")

        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = data_dir / "animas"

        cmd_anima_rename(self._make_args("sakura", "hinata"))

        assert not (data_dir / "run" / "sockets" / "sakura.sock").exists()
        assert not (data_dir / "run" / "animas" / "sakura.pid").exists()
