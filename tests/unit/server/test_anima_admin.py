"""Shared local file operations for anima administration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


def _make_runtime(data_dir: Path) -> tuple[Path, Path]:
    animas_dir = data_dir / "animas"
    anima_dir = animas_dir / "alice"
    anima_dir.mkdir(parents=True)
    (anima_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")
    (anima_dir / "status.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
    subordinate_dir = animas_dir / "bob"
    subordinate_dir.mkdir()
    (subordinate_dir / "status.json").write_text(json.dumps({"supervisor": "alice"}), encoding="utf-8")
    (data_dir / "config.json").write_text(json.dumps({"version": 1, "animas": {"alice": {}}}), encoding="utf-8")
    return animas_dir, anima_dir


def test_delete_anima_files_archives_unregisters_and_finds_supervisor_references(tmp_path: Path) -> None:
    from server.services.anima_admin import delete_anima_files

    data_dir = tmp_path / "runtime"
    data_dir.mkdir()
    animas_dir, anima_dir = _make_runtime(data_dir)

    with patch("core.anima.roster.refresh_anima_roster"):
        result = delete_anima_files(data_dir, "alice", archive=True)

    assert result.deleted is True
    assert result.error is None
    assert result.archive_path is not None and result.archive_path.is_file()
    assert result.supervisor_references == ("bob",)
    assert not anima_dir.exists()
    config = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
    assert "alice" not in config["animas"]
    assert (animas_dir / "bob").is_dir()


def test_delete_anima_files_stops_on_rmtree_error_without_unregistering(tmp_path: Path) -> None:
    from server.services.anima_admin import delete_anima_files

    data_dir = tmp_path / "runtime"
    data_dir.mkdir()
    _, anima_dir = _make_runtime(data_dir)

    with (
        patch("core.anima.admin.shutil.rmtree", side_effect=OSError("locked")),
        patch("core.anima.roster.refresh_anima_roster"),
    ):
        result = delete_anima_files(data_dir, "alice", archive=False)

    assert result.deleted is False
    assert result.archive_path is None
    assert result.error == "locked"
    assert (anima_dir / "IDENTITY.md").is_file()
    config = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
    assert "alice" in config["animas"]
