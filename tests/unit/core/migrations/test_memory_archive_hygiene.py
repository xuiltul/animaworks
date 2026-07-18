from __future__ import annotations

from pathlib import Path

import pytest

from core.migrations.registry import MigrationRunner
from core.migrations.steps import (
    register_all_steps,
    step_knowledge_archive_unify,
    step_ragignore_archive_patterns,
)

SCOPED_ARCHIVE_PATTERNS = (
    "*/knowledge/archive/*",
    "*/knowledge/archived/*",
    "*/episodes/archive/*",
    "*/episodes/archived/*",
    "*/procedures/archive/*",
    "*/procedures/archived/*",
)


def _make_anima(data_dir: Path, name: str = "sora") -> Path:
    anima_dir = data_dir / "animas" / name
    anima_dir.mkdir(parents=True)
    (anima_dir / "identity.md").write_text(f"# {name}\n", encoding="utf-8")
    return anima_dir


def test_archive_unify_moves_nested_files_and_removes_legacy_dirs(tmp_path: Path) -> None:
    anima_dir = _make_anima(tmp_path)
    archived = anima_dir / "knowledge" / "archived"
    underscored = anima_dir / "knowledge" / "_archived"
    (archived / "nested").mkdir(parents=True)
    underscored.mkdir(parents=True)
    (archived / "nested" / "one.md").write_text("one", encoding="utf-8")
    (underscored / "two.md").write_text("two", encoding="utf-8")

    result = step_knowledge_archive_unify(tmp_path, dry_run=False, verbose=True)

    assert result.error is None
    assert result.changed == 2
    assert (anima_dir / "knowledge" / "archive" / "nested" / "one.md").read_text() == "one"
    assert (anima_dir / "knowledge" / "archive" / "two.md").read_text() == "two"
    assert not archived.exists()
    assert not underscored.exists()
    assert "Collision-renamed files: 0" in result.details


def test_archive_unify_renames_collisions_with_sequence(tmp_path: Path) -> None:
    anima_dir = _make_anima(tmp_path)
    archive = anima_dir / "knowledge" / "archive"
    archived = anima_dir / "knowledge" / "archived"
    archive.mkdir(parents=True)
    archived.mkdir(parents=True)
    (archive / "foo.md").write_text("canonical", encoding="utf-8")
    (archive / "foo__from_archived.md").write_text("prior", encoding="utf-8")
    (archived / "foo.md").write_text("legacy", encoding="utf-8")

    result = step_knowledge_archive_unify(tmp_path, dry_run=False, verbose=False)

    assert result.changed == 1
    assert (archive / "foo.md").read_text() == "canonical"
    assert (archive / "foo__from_archived.md").read_text() == "prior"
    assert (archive / "foo__from_archived_2.md").read_text() == "legacy"
    assert "Collision-renamed files: 1" in result.details


def test_archive_unify_handles_all_memory_roots_and_dot_archive(tmp_path: Path) -> None:
    anima_dir = _make_anima(tmp_path)
    for memory_root, legacy_name in (("episodes", ".archive"), ("procedures", "archived")):
        legacy_dir = anima_dir / memory_root / legacy_name
        legacy_dir.mkdir(parents=True)
        (legacy_dir / "old.md").write_text(memory_root, encoding="utf-8")

    result = step_knowledge_archive_unify(tmp_path, dry_run=False, verbose=False)

    assert result.changed == 2
    assert (anima_dir / "episodes" / "archive" / "old.md").read_text() == "episodes"
    assert (anima_dir / "procedures" / "archive" / "old.md").read_text() == "procedures"
    assert not (anima_dir / "episodes" / ".archive").exists()
    assert not (anima_dir / "procedures" / "archived").exists()


def test_archive_unify_dry_run_counts_without_changes(tmp_path: Path) -> None:
    anima_dir = _make_anima(tmp_path)
    archive = anima_dir / "knowledge" / "archive"
    archived = anima_dir / "knowledge" / "archived"
    archive.mkdir(parents=True)
    archived.mkdir(parents=True)
    (archive / "same.md").write_text("canonical", encoding="utf-8")
    (archived / "same.md").write_text("legacy", encoding="utf-8")

    result = step_knowledge_archive_unify(tmp_path, dry_run=True, verbose=True)

    assert result.changed == 1
    assert (archived / "same.md").read_text() == "legacy"
    assert not (archive / "same__from_archived.md").exists()
    assert "Would move 1 archived memory files" in result.details
    assert "Collision-renamed files: 1" in result.details


def test_archive_unify_is_noop_for_canonical_archive_only(tmp_path: Path) -> None:
    anima_dir = _make_anima(tmp_path)
    archive_file = anima_dir / "knowledge" / "archive" / "only.md"
    archive_file.parent.mkdir(parents=True)
    archive_file.write_text("unchanged", encoding="utf-8")

    result = step_knowledge_archive_unify(tmp_path, dry_run=False, verbose=False)

    assert result.changed == 0
    assert result.skipped == 1
    assert archive_file.read_text() == "unchanged"


def test_archive_unify_skips_symlinked_legacy_directory(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    anima_dir = _make_anima(tmp_path)
    knowledge_dir = anima_dir / "knowledge"
    knowledge_dir.mkdir()
    outside = tmp_path / "outside-archive"
    outside.mkdir()
    outside_file = outside / "old.md"
    outside_file.write_text("outside", encoding="utf-8")
    legacy_link = knowledge_dir / "archived"
    legacy_link.symlink_to(outside, target_is_directory=True)

    result = step_knowledge_archive_unify(tmp_path, dry_run=False, verbose=False)

    assert result.changed == 0
    assert legacy_link.is_symlink()
    assert outside_file.read_text(encoding="utf-8") == "outside"
    assert not (knowledge_dir / "archive").exists()
    assert "Skipping symlinked legacy archive" in caplog.text


def test_archive_unify_skips_symlinked_anima_directory(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir()
    outside_anima = tmp_path / "outside-anima"
    legacy_file = outside_anima / "knowledge" / "archived" / "old.md"
    legacy_file.parent.mkdir(parents=True)
    (outside_anima / "identity.md").write_text("# outside\n", encoding="utf-8")
    legacy_file.write_text("outside", encoding="utf-8")
    (animas_dir / "sora").symlink_to(outside_anima, target_is_directory=True)

    result = step_knowledge_archive_unify(tmp_path, dry_run=False, verbose=False)

    assert result.changed == 0
    assert legacy_file.exists()
    assert not (outside_anima / "knowledge" / "archive").exists()
    assert "Skipping symlinked anima directory" in caplog.text


def test_archive_unify_skips_symlinked_memory_root(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    anima_dir = _make_anima(tmp_path)
    outside_memory = tmp_path / "outside-memory"
    legacy_file = outside_memory / "archived" / "old.md"
    legacy_file.parent.mkdir(parents=True)
    legacy_file.write_text("outside", encoding="utf-8")
    (anima_dir / "knowledge").symlink_to(outside_memory, target_is_directory=True)

    result = step_knowledge_archive_unify(tmp_path, dry_run=False, verbose=False)

    assert result.changed == 0
    assert legacy_file.exists()
    assert not (outside_memory / "archive").exists()
    assert "Skipping symlinked memory root" in caplog.text


def test_archive_unify_skips_symlinked_canonical_archive(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    anima_dir = _make_anima(tmp_path)
    knowledge_dir = anima_dir / "knowledge"
    legacy_file = knowledge_dir / "archived" / "old.md"
    legacy_file.parent.mkdir(parents=True)
    legacy_file.write_text("legacy", encoding="utf-8")
    outside_archive = tmp_path / "outside-canonical"
    outside_archive.mkdir()
    (knowledge_dir / "archive").symlink_to(outside_archive, target_is_directory=True)

    result = step_knowledge_archive_unify(tmp_path, dry_run=False, verbose=False)

    assert result.changed == 0
    assert legacy_file.exists()
    assert list(outside_archive.iterdir()) == []
    assert "Skipping symlinked canonical archive" in caplog.text


@pytest.mark.parametrize("collision_direction", ["destination_directory", "destination_file"])
def test_archive_unify_skips_file_directory_collisions(
    tmp_path: Path,
    collision_direction: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    anima_dir = _make_anima(tmp_path)
    archive = anima_dir / "knowledge" / "archive"
    archived = anima_dir / "knowledge" / "archived"
    archive.mkdir(parents=True)
    archived.mkdir(parents=True)
    if collision_direction == "destination_directory":
        (archive / "foo").mkdir()
        source = archived / "foo"
        source.write_text("legacy file", encoding="utf-8")
    else:
        (archive / "foo").write_text("canonical file", encoding="utf-8")
        source = archived / "foo" / "nested.md"
        source.parent.mkdir()
        source.write_text("legacy nested file", encoding="utf-8")

    result = step_knowledge_archive_unify(tmp_path, dry_run=False, verbose=False)

    assert result.changed == 0
    assert source.exists()
    assert "Skipping archive migration due to file/directory collision" in caplog.text


def test_ragignore_archive_patterns_appends_missing_patterns(tmp_path: Path) -> None:
    ragignore = tmp_path / ".ragignore"
    ragignore.write_text("00_index.md\n", encoding="utf-8")

    result = step_ragignore_archive_patterns(tmp_path, dry_run=False, verbose=True)

    assert result.changed == 1
    assert ragignore.read_text(encoding="utf-8") == (
        "00_index.md\n\n"
        "# Archived memory files (unified)\n"
        + "".join(f"{pattern}\n" for pattern in SCOPED_ARCHIVE_PATTERNS)
    )


def test_ragignore_archive_patterns_adds_only_missing_line_idempotently(tmp_path: Path) -> None:
    ragignore = tmp_path / ".ragignore"
    ragignore.write_text(
        "# Archived memory files (unified)\n*/knowledge/archive/*\n",
        encoding="utf-8",
    )

    first = step_ragignore_archive_patterns(tmp_path, dry_run=False, verbose=False)
    after_first = ragignore.read_text(encoding="utf-8")
    second = step_ragignore_archive_patterns(tmp_path, dry_run=False, verbose=False)

    assert first.changed == 1
    for pattern in SCOPED_ARCHIVE_PATTERNS:
        assert after_first.splitlines().count(pattern) == 1
    assert ragignore.read_text(encoding="utf-8") == after_first
    assert second.changed == 0
    assert second.skipped == 1


def test_ragignore_archive_patterns_missing_file_is_noop(tmp_path: Path) -> None:
    result = step_ragignore_archive_patterns(tmp_path, dry_run=False, verbose=False)

    assert result.changed == 0
    assert result.skipped == 1
    assert not (tmp_path / ".ragignore").exists()


def test_ragignore_archive_patterns_skips_symlink(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    target = tmp_path / "target-ragignore"
    target.write_text("00_index.md\n", encoding="utf-8")
    ragignore = tmp_path / ".ragignore"
    ragignore.symlink_to(target)

    result = step_ragignore_archive_patterns(tmp_path, dry_run=False, verbose=False)

    assert result.changed == 0
    assert result.skipped == 1
    assert target.read_text(encoding="utf-8") == "00_index.md\n"
    assert "Skipping symlinked .ragignore" in caplog.text


def test_ragignore_archive_patterns_dry_run_is_read_only(tmp_path: Path) -> None:
    ragignore = tmp_path / ".ragignore"
    ragignore.write_text("00_index.md\n", encoding="utf-8")

    result = step_ragignore_archive_patterns(tmp_path, dry_run=True, verbose=False)

    assert result.changed == 1
    assert ragignore.read_text(encoding="utf-8") == "00_index.md\n"


def test_default_ragignore_contains_unified_and_legacy_archive_patterns(tmp_path: Path) -> None:
    from core.init import _ensure_runtime_only_dirs

    _ensure_runtime_only_dirs(tmp_path)

    lines = (tmp_path / ".ragignore").read_text(encoding="utf-8").splitlines()
    for pattern in SCOPED_ARCHIVE_PATTERNS:
        assert pattern in lines
    assert "*/archive/*" not in lines
    assert "*/archived/*" not in lines
    assert "*/.archive/*" in lines
    assert "*/_archived/*" in lines


def test_memory_hygiene_steps_are_registered_before_version(tmp_path: Path) -> None:
    runner = MigrationRunner(tmp_path)
    register_all_steps(runner)
    ids = [step["id"] for step in runner.list_steps()]

    assert ids.index("knowledge_archive_unify_20260718") < ids.index("update_version")
    assert ids.index("ragignore_archive_patterns_20260718") < ids.index("update_version")
