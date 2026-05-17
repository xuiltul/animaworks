from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from cli.commands.import_cmd import cmd_import_openclaw
from core.skills.migration.openclaw import OpenClawImportOptions, import_openclaw


def test_openclaw_dry_run_generates_draft_plan_without_writes(tmp_path: Path) -> None:
    source = tmp_path / ".openclaw"
    source.mkdir()
    (source / "SOUL.md").write_text("I am a persistent agent.", encoding="utf-8")
    data_dir = tmp_path / "runtime"

    report = import_openclaw(OpenClawImportOptions(source_path=source, data_dir=data_dir, target_anima="mei"))

    assert report.summary == {"planned": 1}
    assert "identity_injection_draft" in report.to_markdown()
    assert not data_dir.exists()


def test_openclaw_apply_writes_only_drafts_redacts_credentials_and_preserves_identity(tmp_path: Path) -> None:
    source = tmp_path / ".openclaw"
    source.mkdir()
    (source / "SOUL.md").write_text("Name: Claw\napi_key: sk-1234567890abcdef\n", encoding="utf-8")
    (source / "permissions.json").write_text('{"allow": ["read"], "token": "tokensecret12345"}', encoding="utf-8")
    (source / "tasks.md").write_text("- review imported task\n", encoding="utf-8")
    data_dir = tmp_path / "runtime"
    anima_dir = data_dir / "animas" / "mei"
    anima_dir.mkdir(parents=True)
    (anima_dir / "identity.md").write_text("existing identity", encoding="utf-8")
    (anima_dir / "permissions.json").write_text('{"existing": true}', encoding="utf-8")

    report = import_openclaw(OpenClawImportOptions(source_path=source, data_dir=data_dir, target_anima="mei", apply=True))

    assert (anima_dir / "identity.md").read_text(encoding="utf-8") == "existing identity"
    assert (anima_dir / "permissions.json").read_text(encoding="utf-8") == '{"existing": true}'
    draft_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (anima_dir / "state" / "migrations" / "drafts").glob("*.md")
    )
    assert "sk-1234567890abcdef" not in draft_text
    assert "tokensecret12345" not in draft_text
    assert "***" in draft_text
    assert any("credential_detected" in warning for warning in report.warnings)
    assert (anima_dir / "state" / "migrations" / "proposals" / "openclaw_taskboard_import_proposal.md").is_file()
    assert report.backup_manifest_path is not None


def test_openclaw_cli_outputs_markdown_dry_run(tmp_path: Path, monkeypatch, capsys) -> None:
    data_dir = tmp_path / "runtime"
    source = tmp_path / ".openclaw"
    source.mkdir()
    (source / "SOUL.md").write_text("OpenClaw soul", encoding="utf-8")
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))

    cmd_import_openclaw(
        Namespace(
            path=str(source),
            target_anima="mei",
            apply=False,
            replace=False,
            json_output=False,
        )
    )

    output = capsys.readouterr().out
    assert "# Openclaw Migration Report" in output
    assert "identity_injection_draft" in output
    assert not data_dir.exists()
