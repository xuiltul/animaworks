"""ssh drop-ins not owned by us must be hidden from the codex sandbox."""

import os
from pathlib import Path

from core.file_access_policy import foreign_owned_ssh_config_dirs


def test_own_files_are_not_denied(tmp_path: Path) -> None:
    (tmp_path / "10-mine.conf").write_text("Host x\n")
    assert foreign_owned_ssh_config_dirs(tmp_path) == []


def test_missing_dir_is_ignored(tmp_path: Path) -> None:
    assert foreign_owned_ssh_config_dirs(tmp_path / "nope") == []


def test_root_owned_dropin_denies_dir() -> None:
    d = Path("/etc/ssh/ssh_config.d")
    foreign = [p for p in d.glob("*.conf") if p.stat().st_uid != os.getuid()] if d.is_dir() else []
    expected = [str(d)] if foreign else []
    assert foreign_owned_ssh_config_dirs(d) == expected
