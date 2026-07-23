"""Tests for open-channel company defaults migration."""

from __future__ import annotations

import json
from pathlib import Path

from core.config.schemas import AnimaWorksConfig
from core.migrations.registry import MigrationRunner
from core.migrations.steps import (
    register_all_steps,
    step_channel_company_defaults,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def test_applies_defaults_to_open_and_meta_less_channels(tmp_path: Path) -> None:
    channels = tmp_path / "shared" / "channels"
    channels.mkdir(parents=True)
    (channels / "general.jsonl").write_text("", encoding="utf-8")
    (channels / "ops.jsonl").write_text('{"from":"x","text":"keep"}\n', encoding="utf-8")
    # Existing open meta without company
    _write_json(
        channels / "legal.meta.json",
        {"members": [], "closed": False, "description": "law"},
    )
    (channels / "legal.jsonl").write_text("", encoding="utf-8")
    # Restricted — must not change
    _write_json(
        channels / "team.meta.json",
        {"members": ["alice", "bob"], "closed": False, "company": ""},
    )
    (channels / "team.jsonl").write_text("", encoding="utf-8")
    # Closed — must not change
    _write_json(
        channels / "old-board.meta.json",
        {"members": [], "closed": True},
    )
    # Already attributed open — must not overwrite
    _write_json(
        channels / "fs-ops.meta.json",
        {"members": [], "closed": False, "company": "fs"},
    )
    (channels / "fs-ops.jsonl").write_text("", encoding="utf-8")

    _write_json(
        tmp_path / "config.json",
        {"channel_company_defaults": {"general": "alpha", "ops": "alpha", "legal": "alpha"}},
    )

    result = step_channel_company_defaults(tmp_path, dry_run=False, verbose=True)

    assert result.error is None
    assert result.changed >= 3

    general = json.loads((channels / "general.meta.json").read_text(encoding="utf-8"))
    assert general["company"] == "alpha"
    assert general["members"] == []

    ops = json.loads((channels / "ops.meta.json").read_text(encoding="utf-8"))
    assert ops["company"] == "alpha"
    # history preserved
    assert (channels / "ops.jsonl").read_text(encoding="utf-8") == '{"from":"x","text":"keep"}\n'

    legal = json.loads((channels / "legal.meta.json").read_text(encoding="utf-8"))
    assert legal["company"] == "alpha"
    assert legal["description"] == "law"

    team = json.loads((channels / "team.meta.json").read_text(encoding="utf-8"))
    assert team["members"] == ["alice", "bob"]
    assert team.get("company", "") == ""

    closed = json.loads((channels / "old-board.meta.json").read_text(encoding="utf-8"))
    assert closed["closed"] is True
    assert closed.get("company", "") == ""

    fs_ops = json.loads((channels / "fs-ops.meta.json").read_text(encoding="utf-8"))
    assert fs_ops["company"] == "fs"


def test_idempotent_second_run(tmp_path: Path) -> None:
    channels = tmp_path / "shared" / "channels"
    channels.mkdir(parents=True)
    (channels / "general.jsonl").write_text("", encoding="utf-8")
    _write_json(tmp_path / "config.json", {"channel_company_defaults": {"general": "alpha"}})

    first = step_channel_company_defaults(tmp_path, dry_run=False, verbose=False)
    assert first.error is None
    assert first.changed >= 1
    meta_after_first = (channels / "general.meta.json").read_text(encoding="utf-8")

    second = step_channel_company_defaults(tmp_path, dry_run=False, verbose=False)
    assert second.error is None
    assert second.changed == 0
    assert (channels / "general.meta.json").read_text(encoding="utf-8") == meta_after_first


def test_creates_unattributed_meta_when_no_defaults(tmp_path: Path) -> None:
    channels = tmp_path / "shared" / "channels"
    channels.mkdir(parents=True)
    (channels / "general.jsonl").write_text("", encoding="utf-8")
    # empty / missing defaults

    result = step_channel_company_defaults(tmp_path, dry_run=False, verbose=False)

    assert result.error is None
    assert result.changed == 1
    meta = json.loads((channels / "general.meta.json").read_text(encoding="utf-8"))
    assert meta["members"] == []
    assert meta.get("company", "") == ""


def test_updates_existing_open_meta_with_explicit_unattributed_company(tmp_path: Path) -> None:
    channels = tmp_path / "shared" / "channels"
    channels.mkdir(parents=True)
    meta_path = channels / "legal.meta.json"
    _write_json(meta_path, {"members": [], "closed": False, "description": "keep"})

    first = step_channel_company_defaults(tmp_path, dry_run=False, verbose=False)

    assert first.error is None
    assert first.changed == 1
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["company"] == ""
    assert meta["description"] == "keep"

    second = step_channel_company_defaults(tmp_path, dry_run=False, verbose=False)
    assert second.error is None
    assert second.changed == 0


def test_non_list_members_metadata_is_preserved(tmp_path: Path) -> None:
    channels = tmp_path / "shared" / "channels"
    channels.mkdir(parents=True)
    meta_path = channels / "malformed.meta.json"
    _write_json(meta_path, {"members": "alice", "closed": False})
    before = meta_path.read_bytes()
    _write_json(
        tmp_path / "config.json",
        {"channel_company_defaults": {"malformed": "alpha"}},
    )

    result = step_channel_company_defaults(tmp_path, dry_run=False, verbose=False)

    assert result.error is None
    assert result.changed == 0
    assert meta_path.read_bytes() == before


def test_dry_run_reports_changes_without_writing(tmp_path: Path) -> None:
    channels = tmp_path / "shared" / "channels"
    channels.mkdir(parents=True)
    (channels / "general.jsonl").write_text("history\n", encoding="utf-8")
    legal_meta = channels / "legal.meta.json"
    _write_json(legal_meta, {"members": [], "closed": False})
    legal_before = legal_meta.read_bytes()
    _write_json(
        tmp_path / "config.json",
        {"channel_company_defaults": {"general": "alpha"}},
    )

    result = step_channel_company_defaults(tmp_path, dry_run=True, verbose=True)

    assert result.error is None
    assert result.changed == 2
    assert not (channels / "general.meta.json").exists()
    assert legal_meta.read_bytes() == legal_before
    assert (channels / "general.jsonl").read_text(encoding="utf-8") == "history\n"


def test_migration_step_is_registered_as_structural_before_version(tmp_path: Path) -> None:
    runner = MigrationRunner(tmp_path)
    register_all_steps(runner)
    steps = runner.list_steps()
    ids = [step["id"] for step in steps]
    registered = next(
        step for step in steps if step["id"] == "channel_company_defaults_20260723"
    )

    assert registered["category"] == "structural"
    assert ids.index("split_board_by_company_20260720") < ids.index(
        "channel_company_defaults_20260723"
    )
    assert ids.index("channel_company_defaults_20260723") < ids.index("update_version")


def test_company_config_schema_roundtrip() -> None:
    config = AnimaWorksConfig.model_validate(
        {
            "channel_company_defaults": {"general": "alpha"},
            "external_messaging": {
                "slack": {"default_channel_company": "alpha"},
            },
        }
    )

    assert config.channel_company_defaults == {"general": "alpha"}
    assert config.external_messaging.slack.default_channel_company == "alpha"
    restored = AnimaWorksConfig.model_validate(config.model_dump())
    assert restored.channel_company_defaults == {"general": "alpha"}
    assert restored.external_messaging.slack.default_channel_company == "alpha"
    defaults = AnimaWorksConfig()
    assert defaults.channel_company_defaults == {}
    assert defaults.external_messaging.slack.default_channel_company == ""
