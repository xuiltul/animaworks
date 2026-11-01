from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.infra.runtime_init import ensure_runtime_dir
from core.migrations.tracker import UnsupportedRuntimeVersionError


def test_fresh_install_without_migration_state_is_supported(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "runtime"
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))

    initialized = ensure_runtime_dir(skip_animas=True)

    assert initialized == data_dir
    assert (data_dir / "config.json").is_file()
    assert not (data_dir / "migration_state.json").exists()


def test_runtime_below_014_is_rejected_with_upgrade_guidance(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "runtime"
    data_dir.mkdir()
    (data_dir / "config.json").write_text("{}", encoding="utf-8")
    (data_dir / "migration_state.json").write_text(
        json.dumps({"applied_version": "0.13.0", "steps_applied": {}, "last_migrated_at": ""}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))

    with pytest.raises(UnsupportedRuntimeVersionError, match="0.14"):
        ensure_runtime_dir()

    state = json.loads((data_dir / "migration_state.json").read_text(encoding="utf-8"))
    assert state["applied_version"] == "0.13.0"


@pytest.mark.parametrize(
    ("role", "server_running", "expect_migrations"),
    [
        ("cli", True, False),
        ("cli", False, True),
        ("root", True, True),
        ("anima", False, False),
    ],
)
def test_auto_migrations_run_only_where_settings_are_writable(
    tmp_path: Path,
    monkeypatch,
    role: str,
    server_running: bool,
    expect_migrations: bool,
) -> None:
    data_dir = tmp_path / "runtime"
    data_dir.mkdir()
    (data_dir / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))
    monkeypatch.setenv("ANIMAWORKS_PROCESS_ROLE", role)
    monkeypatch.setattr("core.platform.pid.is_server_running", lambda _data_dir=None: server_running)
    calls: list[Path] = []
    monkeypatch.setattr("core.infra.runtime_init._run_auto_migrations", calls.append)
    monkeypatch.setattr("core.infra.runtime_init._sync_shared_templates", lambda _data_dir: None)

    assert ensure_runtime_dir() == data_dir

    assert calls == ([data_dir] if expect_migrations else [])
