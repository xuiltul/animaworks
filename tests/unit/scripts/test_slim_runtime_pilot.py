from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.slim_runtime_pilot import changes, validate_sandbox


def _fixture(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "credentials": {"fixture": {"api_key": "must-not-appear-in-preview"}},
                "anima_defaults": {"model": "provider/model", "background_model": "provider/other"},
                "consolidation": {"weekly_enabled": True, "daily_enabled": True, "indexing_enabled": True},
            }
        )
    )
    status = root / "animas" / "fixture" / "status.json"
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({"enabled": True, "model": "provider/model", "heartbeat_enabled": True}))
    return root


def test_preview_contains_only_whitelisted_changes_and_never_writes(tmp_path: Path):
    root = _fixture(tmp_path / "pilot")
    before = (root / "config.json").read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            "scripts/slim_runtime_pilot.py",
            "--sandbox-runtime",
            str(root),
            "--stage",
            "maintenance",
            "--maintenance-feature",
            "weekly",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "must-not-appear" not in result.stdout
    assert json.loads(result.stdout)["patches"] == {"config.json": {"consolidation": {"weekly_enabled": False}}}
    assert (root / "config.json").read_bytes() == before


def test_apply_preserves_models_permissions_and_other_maintenance(tmp_path: Path):
    root = _fixture(tmp_path / "pilot")
    subprocess.run(
        [
            sys.executable,
            "scripts/slim_runtime_pilot.py",
            "--sandbox-runtime",
            str(root),
            "--stage",
            "priming",
            "--anima",
            "fixture",
            "--apply",
        ],
        check=True,
        capture_output=True,
    )
    config = json.loads((root / "config.json").read_text())
    status = json.loads((root / "animas/fixture/status.json").read_text())
    assert config["anima_defaults"]["background_model"] == "provider/other"
    assert config["consolidation"] == {"weekly_enabled": True, "daily_enabled": True, "indexing_enabled": True}
    assert status == {
        "enabled": True,
        "model": "provider/model",
        "heartbeat_enabled": True,
        "priming_profile": "compact",
    }


def test_heartbeat_requires_persisted_acceptance_evidence(tmp_path: Path):
    root = _fixture(tmp_path / "pilot")
    with pytest.raises(FileNotFoundError):
        changes(root, "heartbeat", ["fixture"], None)
    (root / "state").mkdir()
    marker = root / "state/slim-runtime-acceptance.json"
    marker.write_text(json.dumps({"durable_task_resume": True}))
    with pytest.raises(ValueError, match="acceptance"):
        changes(root, "heartbeat", ["fixture"], None)
    marker.write_text(
        json.dumps(
            dict.fromkeys(
                ["durable_task_resume", "message_delivery", "cron_delivery", "deadline_delivery", "cancel"],
                True,
            )
        )
    )
    assert list(changes(root, "heartbeat", ["fixture"], None).values()) == [{"heartbeat_enabled": False}]


def test_refuses_production_and_escaping_config(tmp_path: Path):
    with pytest.raises(ValueError):
        validate_sandbox(Path.home() / ".animaworks")
    root = tmp_path / "pilot"
    root.mkdir()
    source = tmp_path / "outside.json"
    source.write_text("{}")
    (root / "config.json").symlink_to(source)
    with pytest.raises(ValueError, match="link"):
        validate_sandbox(root)
