from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.anima.settings_store import (
    update_config,
    update_status,
    write_agents,
    write_identity,
    write_injection,
    write_permissions,
    write_soul,
)
from core.config.file_access_policy import FileAccessContext, evaluate_file_access
from core.config.io import update_config as raw_update_config
from core.platform.process_role import PROCESS_ROLE_ENV
from core.platform.status_store import update_status as raw_update_status


@pytest.mark.parametrize("role", ["anima", "task_runner", "mcp"])
def test_settings_writers_reject_worker_roles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str) -> None:
    monkeypatch.setenv(PROCESS_ROLE_ENV, role)
    anima_dir = tmp_path / "animas" / "worker"
    anima_dir.mkdir(parents=True)

    writers = (
        lambda: update_status(anima_dir, lambda status: status.update(enabled=False)),
        lambda: write_identity(anima_dir, "identity\n"),
        lambda: write_soul(anima_dir, "soul\n"),
        lambda: write_agents(anima_dir, "agents\n"),
        lambda: write_injection(anima_dir, "injection\n"),
        lambda: write_permissions(anima_dir, {"version": 1}),
        lambda: update_config(lambda config: setattr(config, "activity_level", 200), tmp_path / "config.json"),
        lambda: raw_update_status(anima_dir, lambda status: status.update(enabled=False)),
        lambda: raw_update_config(lambda config: setattr(config, "activity_level", 200), tmp_path / "config.json"),
    )
    for writer in writers:
        with pytest.raises(PermissionError, match="cannot write root-owned settings"):
            writer()

    assert not list(anima_dir.iterdir())
    assert not (tmp_path / "config.json").exists()


def test_cli_cannot_write_settings_while_server_is_live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROCESS_ROLE_ENV, "cli")
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("core.platform.pid.is_server_running", lambda _data_dir: True)
    with pytest.raises(PermissionError, match="while the server is running"):
        write_identity(tmp_path / "animas" / "rin", "identity\n")
    assert not (tmp_path / "animas" / "rin").exists()


def test_root_and_offline_cli_can_write_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROCESS_ROLE_ENV, "root")
    anima_dir = tmp_path / "animas" / "rin"
    anima_dir.mkdir(parents=True)

    update_status(anima_dir, lambda status: status.update(enabled=True))
    write_identity(anima_dir, "identity\n")
    write_soul(anima_dir, "soul\n")
    write_agents(anima_dir, "agents\n")
    write_injection(anima_dir, "injection\n")
    write_permissions(anima_dir, {"version": 1, "file_roots": ["/"]})
    config_path = tmp_path / "config.json"
    update_config(lambda config: setattr(config, "activity_level", 200), config_path)

    assert json.loads((anima_dir / "status.json").read_text(encoding="utf-8")) == {"enabled": True}
    assert (anima_dir / "IDENTITY.md").read_text(encoding="utf-8") == "identity\n"
    assert (anima_dir / "SOUL.md").read_text(encoding="utf-8") == "soul\n"
    assert (anima_dir / "AGENTS.md").read_text(encoding="utf-8") == "injection\n"
    assert json.loads((anima_dir / "permissions.json").read_text(encoding="utf-8"))["file_roots"] == ["/"]
    assert json.loads(config_path.read_text(encoding="utf-8"))["activity_level"] == 200

    monkeypatch.setenv(PROCESS_ROLE_ENV, "cli")
    write_identity(anima_dir, "offline cli\n")
    assert (anima_dir / "IDENTITY.md").read_text(encoding="utf-8") == "offline cli\n"


def test_config_writer_can_preserve_sparse_migration_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROCESS_ROLE_ENV, "root")
    config_path = tmp_path / "config.json"
    original = {"server": {"port": 18500}, "legacy_extension": {"keep": True}}
    config_path.write_text(json.dumps(original), encoding="utf-8")

    update_config(lambda _config: {"server": {"port": 18500}, "legacy_extension": {"keep": False}}, config_path)

    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "server": {"port": 18500},
        "legacy_extension": {"keep": False},
    }


def test_file_tools_cannot_write_owned_settings_even_with_broad_roots(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "rin"
    anima_dir.mkdir(parents=True)
    context = FileAccessContext(
        anima_dir=anima_dir,
        data_dir=tmp_path,
        file_roots=("/",),
        write_roots=(Path("/"),),
    )

    for filename in ("IDENTITY.md", "SOUL.md", "AGENTS.md", "permissions.json", "status.json"):
        decision = evaluate_file_access(anima_dir / filename, context, write=True)
        assert not decision.allowed, filename
        assert decision.reason == "protected_file"

    root_config = tmp_path / "config.json"
    decision = evaluate_file_access(root_config, context, write=True)
    assert not decision.allowed
    assert decision.reason == "protected_file"

    superuser_context = FileAccessContext(anima_dir=anima_dir, data_dir=tmp_path, superuser=True)
    for filename in ("IDENTITY.md", "SOUL.md", "AGENTS.md", "permissions.json", "status.json"):
        assert not evaluate_file_access(anima_dir / filename, superuser_context, write=True).allowed
    assert not evaluate_file_access(root_config, superuser_context, write=True).allowed
    descendant_status = tmp_path / "animas" / "other" / "status.json"
    assert not evaluate_file_access(descendant_status, superuser_context, write=True).allowed
