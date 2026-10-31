from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for write_memory_file's memory scopes and guards."""

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.frontmatter import FrontmatterService, parse_frontmatter
from core.tooling.handler_memory import MemoryToolsMixin


class _WriteHandler(MemoryToolsMixin):
    """Minimal host for the memory writer, with observable index operations."""

    def __init__(self, anima_dir: Path) -> None:
        self._anima_dir = anima_dir
        self._anima_name = "test-anima"
        self._superuser = True
        self._memory = MagicMock()
        self.indexer = MagicMock()
        self._memory._get_indexer.return_value = self.indexer
        self._memory.write_procedure_with_meta.side_effect = self._write_procedure
        self._activity = MagicMock()
        self._subordinate_management_files: list[Path] = []
        self._state_file_lock: threading.Lock | None = None
        self._on_schedule_changed = None
        self._min_trust_seen = 2
        self._runtime_session_context = None
        self._read_paths: set[str] = set()
        self.permission_error: str | None = None
        self.allow_tool_creation = True
        self._update_longterm_bm25_source = MagicMock()

    def _write_procedure(self, path: Path, content: str, metadata: dict) -> None:
        FrontmatterService(
            self._anima_dir,
            self._anima_dir / "knowledge",
            self._anima_dir / "procedures",
        ).write_procedure_with_meta(path, content, metadata)

    def _is_state_file(self, path: Path) -> bool:
        return path.is_relative_to(self._anima_dir / "state")

    def _check_tool_creation_permission(self, _kind: str) -> bool:
        return self.allow_tool_creation

    def _check_file_permission(self, _path: str, write: bool = False) -> str | None:
        return self.permission_error if write else None

    def _handle_post_channel(self, args: dict) -> str:
        self.posted_channel_args = args
        return "posted"


@pytest.fixture
def writer(tmp_path: Path) -> _WriteHandler:
    anima_dir = tmp_path / "animas" / "test-anima"
    anima_dir.mkdir(parents=True)
    for name in ("knowledge", "procedures", "episodes", "facts", "state", "skills", "shortterm"):
        (anima_dir / name).mkdir()
    return _WriteHandler(anima_dir)


@pytest.mark.parametrize(
    ("scope", "requested_path", "written_path", "frontmatter", "index_type", "updates_bm25"),
    [
        ("knowledge", "knowledge/operations.md", "knowledge/operations.md", "knowledge", "knowledge", True),
        ("procedures", "procedures/deploy.md", "procedures/deploy.md", "procedure", "procedures", True),
        ("episodes", "episodes/2026-09-30.md", "episodes/2026-09-30.md", None, None, True),
        ("facts", "facts/contact.md", "facts/contact.md", None, None, False),
        ("state", "state/current_task.md", "state/current_state.md", None, None, False),
        ("skills", "skills/guide.md", "skills/guide.md", None, "skills", True),
        ("tools", "tools/readme.md", "tools/readme.md", None, None, False),
        ("common_knowledge", "common_knowledge/shared.md", "common_knowledge/shared.md", None, None, False),
        ("common_skills", "common_skills/guide.md", "common_skills/guide.md", None, None, False),
        ("shortterm", "shortterm/session.md", "shortterm/session.md", None, None, False),
        ("root", "HEARTBEAT.md", "HEARTBEAT.md", None, None, False),
    ],
)
def test_each_memory_scope_writes_to_its_destination_and_updates_expected_indexes(
    writer: _WriteHandler,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    requested_path: str,
    written_path: str,
    frontmatter: str | None,
    index_type: str | None,
    updates_bm25: bool,
) -> None:
    shared_knowledge = tmp_path / "shared" / "common_knowledge"
    shared_skills = tmp_path / "shared" / "common_skills"
    shared_knowledge.mkdir(parents=True)
    shared_skills.mkdir(parents=True)
    monkeypatch.setattr("core.paths.get_common_knowledge_dir", lambda: shared_knowledge)
    monkeypatch.setattr("core.paths.get_common_skills_dir", lambda: shared_skills)

    content = "# Deploy safely\nKeep the existing operational behavior.\n"
    result = writer._handle_write_memory_file({"path": requested_path, "content": content})

    assert result.startswith(f"Written to {written_path}")
    if scope == "common_knowledge":
        target = shared_knowledge / "shared.md"
    elif scope == "common_skills":
        target = shared_skills / "guide.md"
    else:
        target = writer._anima_dir / written_path
    saved = target.read_text(encoding="utf-8")

    if frontmatter == "knowledge":
        metadata, body = parse_frontmatter(saved)
        assert metadata["confidence"] == 0.5
        assert metadata["source_episodes"] == 0
        assert metadata["auto_consolidated"] is False
        assert metadata["version"] == 1
        assert "created_at" in metadata and "updated_at" in metadata
        assert "Keep the existing operational behavior." in body
    elif frontmatter == "procedure":
        metadata, body = parse_frontmatter(saved)
        assert metadata == {
            "description": "Deploy safely",
            "success_count": 0,
            "failure_count": 0,
            "confidence": 0.5,
        }
        assert "Keep the existing operational behavior." in body
    else:
        assert saved == content

    if index_type is None:
        writer._memory._get_indexer.assert_not_called()
        writer.indexer.index_file.assert_not_called()
    elif index_type == "knowledge":
        writer.indexer.index_file.assert_called_once_with(
            target,
            memory_type="knowledge",
            force=True,
            origin=None,
        )
    else:
        writer.indexer.index_file.assert_called_once_with(target, memory_type=index_type, force=True)

    if updates_bm25:
        writer._update_longterm_bm25_source.assert_called_once_with(written_path)
    else:
        writer._update_longterm_bm25_source.assert_not_called()


@pytest.mark.parametrize(
    "path",
    [
        "reference/architecture.md",
        "companies/acme/overview.md",
        "external/claude/example/SKILL.md",
    ],
)
def test_read_only_scopes_are_rejected_without_writing(writer: _WriteHandler, path: str) -> None:
    result = writer._handle_write_memory_file({"path": path, "content": "must not be written"})

    assert json.loads(result)["error_type"] == "PermissionDenied"
    assert not (writer._anima_dir / path).exists()
    writer._memory._get_indexer.assert_not_called()


@pytest.mark.parametrize("prefix", ["common_knowledge", "common_skills"])
def test_shared_scope_path_traversal_is_rejected(
    writer: _WriteHandler,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prefix: str,
) -> None:
    shared_root = tmp_path / "shared" / prefix
    shared_root.mkdir(parents=True)
    monkeypatch.setattr(f"core.paths.get_{prefix}_dir", lambda: shared_root)
    outside = shared_root.parent / "outside.md"

    result = writer._handle_write_memory_file(
        {"path": f"{prefix}/../outside.md", "content": "escape"},
    )

    assert json.loads(result)["error_type"] == "PermissionDenied"
    assert not outside.exists()


def test_file_permission_denial_prevents_memory_write(writer: _WriteHandler) -> None:
    writer.permission_error = '{"error_type":"PermissionDenied","message":"denied by file roots"}'

    result = writer._handle_write_memory_file({"path": "facts/secret.md", "content": "secret"})

    assert result == writer.permission_error
    assert not (writer._anima_dir / "facts" / "secret.md").exists()
    writer._memory._get_indexer.assert_not_called()


def test_protected_path_and_path_traversal_are_rejected(writer: _WriteHandler) -> None:
    writer._superuser = False
    outside = writer._anima_dir.parent / "outside.md"

    protected = writer._handle_write_memory_file({"path": "IDENTITY.md", "content": "replacement"})
    traversed = writer._handle_write_memory_file(
        {"path": "../outside.md", "content": "escape"},
    )

    assert json.loads(protected)["error_type"] == "PermissionDenied"
    assert json.loads(traversed)["error_type"] == "PermissionDenied"
    assert not (writer._anima_dir / "IDENTITY.md").exists()
    assert not outside.exists()


def test_tool_creation_permission_is_checked_for_python_files(writer: _WriteHandler) -> None:
    writer.allow_tool_creation = False

    result = writer._handle_write_memory_file({"path": "tools/custom.py", "content": "print('no')"})

    assert json.loads(result)["error_type"] == "PermissionDenied"
    assert not (writer._anima_dir / "tools" / "custom.py").exists()


def test_overwrite_guard_requires_read_but_append_does_not(writer: _WriteHandler) -> None:
    path = writer._anima_dir / "facts" / "existing.md"
    path.write_text("original", encoding="utf-8")

    denied = writer._handle_write_memory_file(
        {"path": "facts/existing.md", "content": "replacement", "mode": "overwrite"},
    )
    appended = writer._handle_write_memory_file(
        {"path": "facts/existing.md", "content": " + appended", "mode": "append"},
    )

    assert "ReadBeforeWrite" in denied
    assert appended.startswith("Written to facts/existing.md")
    assert path.read_text(encoding="utf-8") == "original + appended"


def test_knowledge_append_preserves_body_and_downgrades_origin(writer: _WriteHandler) -> None:
    path = writer._anima_dir / "knowledge" / "source.md"
    path.write_text("---\norigin: mixed\n---\n\nTrusted body\n", encoding="utf-8")
    writer._min_trust_seen = 0

    result = writer._handle_write_memory_file(
        {"path": "knowledge/source.md", "content": "Untrusted append", "mode": "append"},
    )

    metadata, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    assert result.startswith("Written to knowledge/source.md")
    assert metadata["origin"] == "external_web"
    assert "Trusted body" in body
    assert "Untrusted append" in body
    writer.indexer.index_file.assert_called_once_with(
        path,
        memory_type="knowledge",
        force=True,
        origin="external_web",
    )


def test_existing_episode_is_archived_before_overwrite(writer: _WriteHandler) -> None:
    episode = writer._anima_dir / "episodes" / "2026-09-30.md"
    episode.write_text("original episode", encoding="utf-8")

    result = writer._handle_write_memory_file(
        {"path": "episodes/2026-09-30.md", "content": "updated episode", "mode": "overwrite"},
    )

    archived = list((writer._anima_dir / "archive" / "episodes").glob("2026-09-30_*.md"))
    assert result.startswith("Written to episodes/2026-09-30.md")
    assert episode.read_text(encoding="utf-8") == "updated episode"
    assert len(archived) == 1
    assert archived[0].read_text(encoding="utf-8") == "original episode"


def test_episode_archive_failure_does_not_overwrite_existing_content(writer: _WriteHandler) -> None:
    episode = writer._anima_dir / "episodes" / "2026-09-30.md"
    episode.write_text("original episode", encoding="utf-8")

    with patch("core.memory.io.shutil.copy2", side_effect=OSError("archive unavailable")):
        result = writer._handle_write_memory_file(
            {"path": "episodes/2026-09-30.md", "content": "updated episode", "mode": "overwrite"},
        )

    assert "WriteError" in result
    assert "archive unavailable" in result
    assert episode.read_text(encoding="utf-8") == "original episode"


def test_legacy_identity_read_concatenates_identity_and_soul(writer: _WriteHandler) -> None:
    (writer._anima_dir / "IDENTITY.md").write_text("# Identity: Alice\n\nProfile", encoding="utf-8")
    (writer._anima_dir / "SOUL.md").write_text("# Soul: Alice\n\nPersonality", encoding="utf-8")

    result = writer._handle_read_memory_file({"path": "identity.md"})

    assert "Profile" in result
    assert "Personality" in result
    assert result.index("Profile") < result.index("Personality")


@pytest.mark.parametrize(
    ("legacy", "canonical"),
    [
        ("injection.md", "AGENTS.md"),
        ("heartbeat.md", "HEARTBEAT.md"),
        ("bootstrap.md", "BOOTSTRAP.md"),
    ],
)
def test_legacy_read_redirects_to_canonical_file(writer: _WriteHandler, legacy: str, canonical: str) -> None:
    (writer._anima_dir / canonical).write_text(f"content of {canonical}", encoding="utf-8")

    result = writer._handle_read_memory_file({"path": legacy})

    assert f"content of {canonical}" in result
    assert canonical in result


@pytest.mark.parametrize(
    ("legacy", "canonical"),
    [
        ("injection.md", "AGENTS.md"),
        ("heartbeat.md", "HEARTBEAT.md"),
        ("bootstrap.md", "BOOTSTRAP.md"),
    ],
)
def test_legacy_write_redirects_to_canonical_file(
    writer: _WriteHandler,
    monkeypatch: pytest.MonkeyPatch,
    legacy: str,
    canonical: str,
) -> None:
    monkeypatch.setattr("core.anima.settings_store.settings_server_running", lambda: False)
    monkeypatch.setattr("core.platform.process_role.get_process_role", lambda: "cli")

    result = writer._handle_write_memory_file({"path": legacy, "content": f"new {canonical}"})

    assert (writer._anima_dir / canonical).read_text(encoding="utf-8") == f"new {canonical}"
    assert canonical in result


def test_legacy_identity_write_is_rejected(writer: _WriteHandler) -> None:
    result = writer._handle_write_memory_file({"path": "identity.md", "content": "should not be written"})

    assert "InvalidArguments" in result
    assert "SOUL.md" in result
    assert "IDENTITY.md" in result
    assert not (writer._anima_dir / "IDENTITY.md").exists()
