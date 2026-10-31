from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from core.anima.workspace_files import (
    AGENTS_FILE,
    BOOTSTRAP_FILE,
    HEARTBEAT_FILE,
    IDENTITY_FILE,
    LEGACY_IDENTITY,
    SOUL_FILE,
    split_legacy_identity,
)
from core.infra.runtime_init import ensure_runtime_dir
from core.memory.manager import MemoryManager
from core.migrations import steps
from core.migrations.registry import MigrationRunner
from core.migrations.standard_files_rollback import rollback_standard_files
from core.migrations.steps import (
    STANDARD_WORKSPACE_FILES_STEP_ID,
    register_all_steps,
    step_standard_workspace_files_20261031,
)
from core.migrations.tracker import MigrationTracker
from core.prompt.builder import build_system_prompt

LEGACY_IDENTITY_TEXT = (
    "# Identity: Mei\n"
    "\n"
    "## 基本プロフィール\n"
    "\n"
    "名前: Mei\n"
    "外見: 黒髪\n"
    "\n"
    "## 性格\n"
    "\n"
    "やさしく、好奇心が強い。\n"
    "\n"
    "## 口調\n"
    "\n"
    "丁寧で短い文で話す。\n"
)
INJECTION_TEXT = "# Injection\n\nAlways verify facts before reporting.\n"
HEARTBEAT_TEXT = "# Heartbeat\n\nCheck the inbox every morning.\n"
BOOTSTRAP_TEXT = "# Bootstrap\n\nIntroduce yourself on first contact.\n"
NOTES_TEXT = (
    "# Notes\n\n"
    "See identity.md, injection.md, heartbeat.md and bootstrap.md.\n"
    "The path foo/identity.md is a different file and must stay unchanged.\n"
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _create_legacy_anima(data_dir: Path, name: str) -> Path:
    anima_dir = data_dir / "animas" / name
    _write(anima_dir / LEGACY_IDENTITY, LEGACY_IDENTITY_TEXT)
    _write(anima_dir / "injection.md", INJECTION_TEXT)
    _write(anima_dir / "heartbeat.md", HEARTBEAT_TEXT)
    _write(anima_dir / "bootstrap.md", BOOTSTRAP_TEXT)
    _write(anima_dir / "knowledge" / "notes.md", NOTES_TEXT)
    _write(anima_dir / "episodes" / "2026-01-02.md", "identity.md is preserved in episodes.\n")
    return anima_dir


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_iter_anima_dirs_accepts_legacy_and_standard_identity(tmp_path: Path) -> None:
    legacy = _create_legacy_anima(tmp_path, "legacy")
    standard = tmp_path / "animas" / "standard"
    standard.mkdir(parents=True)
    (standard / IDENTITY_FILE).write_text("standard", encoding="utf-8")
    other = tmp_path / "animas" / "other"
    other.mkdir(parents=True)
    (other / "notes.md").write_text("not an anima", encoding="utf-8")

    found = [path.name for path in steps._iter_anima_dirs(tmp_path)]

    assert found == ["legacy", "standard"]
    assert legacy.name == "legacy"


def test_migration_renames_splits_archives_and_rewrites_references(tmp_path: Path) -> None:
    anima_dir = _create_legacy_anima(tmp_path, "mei")

    result = step_standard_workspace_files_20261031(tmp_path, dry_run=False, verbose=True)

    assert result.error is None
    assert result.changed > 0
    assert not (anima_dir / LEGACY_IDENTITY).is_file()
    assert not (anima_dir / "injection.md").is_file()
    assert (anima_dir / AGENTS_FILE).read_text(encoding="utf-8") == INJECTION_TEXT
    assert (anima_dir / HEARTBEAT_FILE).read_text(encoding="utf-8") == HEARTBEAT_TEXT
    assert (anima_dir / BOOTSTRAP_FILE).read_text(encoding="utf-8") == BOOTSTRAP_TEXT
    expected_identity, expected_soul = split_legacy_identity(LEGACY_IDENTITY_TEXT)
    assert (anima_dir / IDENTITY_FILE).read_text(encoding="utf-8") == expected_identity
    assert (anima_dir / SOUL_FILE).read_text(encoding="utf-8") == expected_soul

    manifest_path = next((anima_dir / "archive" / "standard_files_migration").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["step_id"] == STANDARD_WORKSPACE_FILES_STEP_ID
    assert manifest["old_files"] == ["identity.md", "injection.md", "heartbeat.md", "bootstrap.md"]
    assert manifest["legacy_to_standard"]["identity.md"] == [IDENTITY_FILE, SOUL_FILE]
    assert manifest["source_sha256"]["identity.md"]
    assert manifest["new_sha256"][SOUL_FILE]

    archive = manifest_path.parent
    assert (archive / LEGACY_IDENTITY).read_bytes() == LEGACY_IDENTITY_TEXT.encode()
    assert (archive / "rewritten" / "knowledge" / "notes.md").is_file()
    rewritten = (anima_dir / "knowledge" / "notes.md").read_text(encoding="utf-8")
    assert "SOUL.md/IDENTITY.md" in rewritten
    assert AGENTS_FILE in rewritten
    assert HEARTBEAT_FILE in rewritten
    assert BOOTSTRAP_FILE in rewritten
    assert "foo/identity.md" in rewritten
    assert "identity.md is preserved" in (anima_dir / "episodes" / "2026-01-02.md").read_text(encoding="utf-8")


def test_dry_run_writes_nothing_and_second_run_is_idempotent(tmp_path: Path) -> None:
    anima_dir = _create_legacy_anima(tmp_path, "mei")
    before = _snapshot(tmp_path)

    dry = step_standard_workspace_files_20261031(tmp_path, dry_run=True, verbose=True)

    assert _snapshot(tmp_path) == before
    assert dry.error is None
    assert dry.changed > 0
    assert any("would rename" in detail for detail in dry.details)

    step_standard_workspace_files_20261031(tmp_path, dry_run=False, verbose=False)
    second = step_standard_workspace_files_20261031(tmp_path, dry_run=False, verbose=False)

    assert second.error is None
    assert second.changed == 0
    assert second.details == ["No legacy workspace files found"]
    assert (anima_dir / IDENTITY_FILE).is_file()


def test_collisions_preserve_existing_standard_file_and_archive_legacy(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "mei"
    _write(anima_dir / LEGACY_IDENTITY, LEGACY_IDENTITY_TEXT)
    _write(anima_dir / IDENTITY_FILE, "existing identity")
    _write(anima_dir / "injection.md", "legacy injection")
    _write(anima_dir / AGENTS_FILE, "existing agents")

    result = step_standard_workspace_files_20261031(tmp_path, dry_run=False, verbose=False)

    assert result.error is None
    assert (anima_dir / IDENTITY_FILE).read_text(encoding="utf-8") == "existing identity"
    assert (anima_dir / AGENTS_FILE).read_text(encoding="utf-8") == "existing agents"
    assert not (anima_dir / LEGACY_IDENTITY).exists()
    assert not (anima_dir / "injection.md").exists()
    manifest_path = next((anima_dir / "archive" / "standard_files_migration").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert {"source": LEGACY_IDENTITY, "existing": IDENTITY_FILE} in manifest["collisions"]
    assert {"source": "injection.md", "existing": AGENTS_FILE} in manifest["collisions"]


def test_identity_unlinks_before_writing_and_renames_via_tmp(tmp_path: Path, monkeypatch) -> None:
    anima_dir = _create_legacy_anima(tmp_path, "mei")
    events: list[tuple[str, str, str]] = []
    original_unlink = Path.unlink
    original_rename = Path.rename
    original_write = steps._standard_write_text

    def record_unlink(path: Path, *args, **kwargs):
        events.append(("unlink", path.name, ""))
        return original_unlink(path, *args, **kwargs)

    def record_rename(path: Path, target: Path):
        events.append(("rename", path.name, Path(target).name))
        return original_rename(path, target)

    def record_write(path: Path, text: str) -> None:
        events.append(("write", path.name, ""))
        original_write(path, text)

    monkeypatch.setattr(Path, "unlink", record_unlink)
    monkeypatch.setattr(Path, "rename", record_rename)
    monkeypatch.setattr(steps, "_standard_write_text", record_write)

    step_standard_workspace_files_20261031(tmp_path, dry_run=False, verbose=False)

    names = [event for event in events if event[1] in {LEGACY_IDENTITY, IDENTITY_FILE, SOUL_FILE}]
    unlink_index = names.index(("unlink", LEGACY_IDENTITY, ""))
    identity_write = names.index(("write", IDENTITY_FILE, ""))
    soul_write = names.index(("write", SOUL_FILE, ""))
    assert unlink_index < identity_write < soul_write
    assert ("rename", "injection.md", ".injection.md.standard-migration.tmp") in events
    assert ("rename", ".injection.md.standard-migration.tmp", AGENTS_FILE) in events
    assert (anima_dir / IDENTITY_FILE).is_file()


def test_full_runner_prompt_equivalence_and_rollback_byte_equality(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    ensure_runtime_dir(skip_animas=True)
    first = _create_legacy_anima(tmp_path, "first")
    second = _create_legacy_anima(tmp_path, "second")
    original = {path: path.read_bytes() for path in first.iterdir() if path.is_file()}
    original_notes = (first / "knowledge" / "notes.md").read_bytes()

    runner = MigrationRunner(tmp_path)
    register_all_steps(runner)
    report = runner.run_all(dry_run=False, verbose=False, force=False)

    assert report.errors == []
    assert [step.id for step in runner._steps][-2:] == [STANDARD_WORKSPACE_FILES_STEP_ID, "update_version"]
    tracker = MigrationTracker(tmp_path)
    assert tracker.is_step_applied(STANDARD_WORKSPACE_FILES_STEP_ID)
    for anima_dir in (first, second):
        assert not (anima_dir / LEGACY_IDENTITY).exists()
        assert (anima_dir / IDENTITY_FILE).is_file()
        assert (anima_dir / SOUL_FILE).is_file()
        assert (anima_dir / AGENTS_FILE).is_file()
        assert (anima_dir / HEARTBEAT_FILE).is_file()

    expected_identity, expected_soul = split_legacy_identity(LEGACY_IDENTITY_TEXT)
    joined = f"{expected_identity}\n\n{expected_soul}"
    old_chars = Counter(char for line in LEGACY_IDENTITY_TEXT.splitlines() if not line.startswith("#") for char in line)
    new_chars = Counter(char for line in joined.splitlines() if not line.startswith("#") for char in line)
    assert old_chars == new_chars

    memory = MemoryManager(first)
    prompt = str(build_system_prompt(memory, trigger="chat"))
    assert "名前: Mei" in prompt
    assert "やさしく、好奇心が強い。" in prompt
    assert "丁寧で短い文で話す。" in prompt

    rollback = rollback_standard_files(tmp_path)

    assert rollback.warnings == []
    assert len(rollback.animas) == 2
    assert rollback.tracker_entry_removed is True
    assert not MigrationTracker(tmp_path).is_step_applied(STANDARD_WORKSPACE_FILES_STEP_ID)
    for path, content in original.items():
        assert (first / path.name).read_bytes() == content
    assert (first / "knowledge" / "notes.md").read_bytes() == original_notes
    assert not (first / IDENTITY_FILE).exists()
    assert not (first / SOUL_FILE).exists()
    assert (first / "archive" / "standard_files_migration").is_dir()
    rolled_back = list((first / "archive" / "standard_files_migration").glob("*/rolled_back_new/*"))
    assert {path.name for path in rolled_back} >= {IDENTITY_FILE, SOUL_FILE, AGENTS_FILE}


def test_rollback_warns_when_new_file_was_edited(tmp_path: Path) -> None:
    anima_dir = _create_legacy_anima(tmp_path, "mei")
    step_standard_workspace_files_20261031(tmp_path, dry_run=False, verbose=False)
    (anima_dir / SOUL_FILE).write_text("edited after migration", encoding="utf-8")

    result = rollback_standard_files(tmp_path, anima_name="mei")

    assert any("was edited after migration" in warning for warning in result.warnings)
    assert (anima_dir / LEGACY_IDENTITY).is_file()
    assert not (anima_dir / SOUL_FILE).exists()
    archived = list((anima_dir / "archive" / "standard_files_migration").glob("*/rolled_back_new/*"))
    assert any(path.read_text(encoding="utf-8") == "edited after migration" for path in archived)


def test_rollback_dry_run_keeps_everything(tmp_path: Path) -> None:
    anima_dir = _create_legacy_anima(tmp_path, "mei")
    step_standard_workspace_files_20261031(tmp_path, dry_run=False, verbose=False)
    before = _snapshot(tmp_path)

    result = rollback_standard_files(tmp_path, dry_run=True)

    assert _snapshot(tmp_path) == before
    assert result.dry_run is True
    assert result.animas == ["mei"]
    assert len(result.moved_new_files) > 0
    assert len(result.restored_files) >= len(result.moved_new_files)
    assert (anima_dir / IDENTITY_FILE).is_file()
    assert not (anima_dir / LEGACY_IDENTITY).is_file()
    assert os.path.isdir(anima_dir)


def test_reference_regex_handles_adjacent_japanese_text() -> None:
    from core.migrations.steps import _STANDARD_REF_RE, _standard_ref_replacement

    text = "メモはidentity.mdに書く。injection.mdを見る。foo/heartbeat.md は対象外。identity.md_old も対象外"
    out = _STANDARD_REF_RE.sub(_standard_ref_replacement, text)
    assert out == "メモはSOUL.md/IDENTITY.mdに書く。AGENTS.mdを見る。foo/heartbeat.md は対象外。identity.md_old も対象外"
