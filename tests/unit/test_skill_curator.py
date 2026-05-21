from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Skill Curator lifecycle management."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.memory.frontmatter import parse_frontmatter
from core.memory.rag.store import Document, SearchResult
from core.skills.context import build_cron_skill_context
from core.skills.curator import SkillCurator
from core.skills.index import SkillIndex
from core.skills.models import SkillLifecycleState, SkillMetadata, SkillUsageEventType
from core.skills.router import SkillRouter
from core.skills.usage import SkillUsageTracker
from core.tooling.handler_memory import MemoryToolsMixin


def _write_skill(
    anima_dir: Path,
    name: str,
    *,
    description: str = "Deploy release safely",
    extra: str = "",
) -> Path:
    skill_dir = anima_dir / "skills" / name
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "use_when: [deploy release]\n"
        "trigger_phrases: [deploy release]\n"
        "domains: [software-delivery]\n"
        f"{extra}"
        "---\n\n"
        f"# {name}\n",
        encoding="utf-8",
    )
    return skill_md


def test_replay_state_restores_latest_lifecycle_state(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    anima_dir.mkdir()
    curator = SkillCurator(anima_dir)

    curator.archive_skill("old-skill", reason="unused", actor="mei")
    curator.restore_skill("old-skill", reason="needed", actor="mei")

    replay = curator.replay_state()
    assert replay.states["old-skill"] == SkillLifecycleState.active
    assert len(replay.events) == 2


def test_usage_stats_drive_lifecycle_suggestions_and_protected_skips_archive(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    anima_dir.mkdir()
    old = SkillMetadata(name="old", last_used_at=datetime.now(UTC) - timedelta(days=181))
    protected = SkillMetadata(name="protected-old", protected=True, last_used_at=datetime.now(UTC) - timedelta(days=181))
    flaky = SkillMetadata(name="flaky")
    patched = SkillMetadata(name="patched")
    tracker = SkillUsageTracker(anima_dir)
    for _ in range(7):
        tracker.record("flaky", SkillUsageEventType.failure)
    for _ in range(3):
        tracker.record("flaky", SkillUsageEventType.success)
    for _ in range(5):
        tracker.record("patched", SkillUsageEventType.patch)

    suggestions = SkillCurator(anima_dir).suggest_lifecycle_transitions([old, protected, flaky, patched])
    by_name = {s.skill_name: s for s in suggestions}

    assert by_name["old"].suggested_state == SkillLifecycleState.archived
    assert "protected-old" not in by_name
    assert by_name["flaky"].suggested_state == SkillLifecycleState.review
    assert by_name["patched"].reason == "patch_count_consolidation"


def test_duplicate_detector_uses_routing_and_lexical_signals(tmp_path: Path) -> None:
    curator = SkillCurator(tmp_path / "alice")
    left = SkillMetadata(
        name="gmail-draft",
        description="Draft concise Gmail replies for partners",
        trigger_phrases=["gmail draft", "partner reply"],
        domains=["email"],
    )
    right = SkillMetadata(
        name="gmail-reply-drafts",
        description="Draft Gmail replies for partner email",
        trigger_phrases=["gmail draft", "partner reply"],
        domains=["email"],
    )

    duplicates = curator.detect_duplicates([left, right])

    assert duplicates
    assert "routing_metadata_overlap" in duplicates[0].signals
    assert "description_lexical_overlap" in duplicates[0].signals


def test_index_router_and_read_memory_file_block_curator_archived_skill(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir()
    _write_skill(anima_dir, "old-skill")
    _write_skill(anima_dir, "new-skill", description="Deploy release safely with new workflow")
    SkillCurator(anima_dir).archive_skill("old-skill", reason="unused", actor="curator")

    index = SkillIndex(anima_dir / "skills", common_dir, anima_dir=anima_dir)
    names = {meta.name for meta in index.all_skills}
    assert "old-skill" not in names
    assert "new-skill" in names

    all_entries = index.search("", include_blocked=True)
    routed = SkillRouter(min_score=0.1).route("deploy release", all_entries, top_k=5)
    assert "old-skill" not in {candidate.name for candidate in routed}

    mixin = MagicMock(spec=MemoryToolsMixin)
    mixin._anima_dir = anima_dir
    mixin._superuser = False
    mixin._subordinate_activity_dirs = []
    mixin._subordinate_management_files = []
    mixin._descendant_activity_dirs = []
    mixin._descendant_state_files = []
    mixin._descendant_state_dirs = []
    mixin._read_paths = set()
    mixin._is_skill_path = MemoryToolsMixin._is_skill_path
    mixin._record_skill_view_if_applicable = MagicMock()

    result = MemoryToolsMixin._handle_read_memory_file(mixin, {"path": "skills/old-skill/SKILL.md"})
    assert "SkillBlocked" in result
    assert "curator_archived" in result


def test_skill_index_invalidates_when_curator_state_changes(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir()
    _write_skill(anima_dir, "old-skill")
    _write_skill(anima_dir, "categorized-skill", description="No matching text", extra="category: operations\n")
    index = SkillIndex(anima_dir / "skills", common_dir, anima_dir=anima_dir)

    assert "old-skill" in {meta.name for meta in index.all_skills}
    assert [meta.name for meta in index.search("operations")] == ["categorized-skill"]
    SkillCurator(anima_dir).archive_skill("old-skill", reason="unused")

    assert "old-skill" not in {meta.name for meta in index.all_skills}


def test_read_memory_file_blocks_quarantined_skill(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    _write_skill(anima_dir, "draft-skill", extra="trust_level: quarantine\n")
    mixin = MagicMock(spec=MemoryToolsMixin)
    mixin._anima_dir = anima_dir
    mixin._superuser = False
    mixin._subordinate_activity_dirs = []
    mixin._subordinate_management_files = []
    mixin._descendant_activity_dirs = []
    mixin._descendant_state_files = []
    mixin._descendant_state_dirs = []
    mixin._read_paths = set()
    mixin._is_skill_path = MemoryToolsMixin._is_skill_path
    mixin._record_skill_view_if_applicable = MagicMock()

    result = MemoryToolsMixin._handle_read_memory_file(mixin, {"path": "skills/draft-skill/SKILL.md"})

    assert "SkillBlocked" in result
    assert "trust_level_quarantine" in result


def test_restore_refuses_trust_blocked_skill(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    anima_dir.mkdir()
    _write_skill(anima_dir, "danger", extra="trust_level: blocked\n")
    curator = SkillCurator(anima_dir)

    try:
        curator.restore_skill("danger", reason="try restore")
    except ValueError as exc:
        assert "cannot be restored" in str(exc)
    else:
        raise AssertionError("Expected trust-blocked skill restore to fail")


def test_archive_generates_reference_rewrite_proposal(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    anima_dir.mkdir()
    (anima_dir / "cron.md").write_text(
        "## Daily\nschedule: 0 9 * * *\nskills:\n  - old-skill\n  - new-skill\nDo work.\n",
        encoding="utf-8",
    )

    event = SkillCurator(anima_dir).archive_skill(
        "old-skill",
        reason="merged",
        absorbed_into="new-skill",
    )

    assert event.absorbed_into == "new-skill"
    assert event.proposal_path
    proposal = anima_dir / event.proposal_path
    assert proposal.exists()
    assert "- old-skill" in proposal.read_text(encoding="utf-8")


def test_curator_state_is_append_only_and_does_not_rewrite_skill_md(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    skill_md = _write_skill(anima_dir, "stable")
    before_meta, before_body = parse_frontmatter(skill_md.read_text(encoding="utf-8"))

    SkillCurator(anima_dir).archive_skill("stable", reason="unused")

    after_meta, after_body = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
    assert after_meta == before_meta
    assert after_body == before_body


def test_rag_indexer_skips_archived_skill(tmp_path: Path) -> None:
    from core.memory.rag.indexer import MemoryIndexer

    anima_dir = tmp_path / "alice"
    skill_md = _write_skill(anima_dir, "old-skill")
    SkillCurator(anima_dir).archive_skill("old-skill", reason="unused")
    vector_store = MagicMock()
    indexer = MemoryIndexer(vector_store, "alice", anima_dir, embedding_model=MagicMock())

    chunks = indexer.index_file(skill_md, memory_type="skills", force=True)

    assert chunks == 0
    vector_store.create_collection.assert_not_called()
    vector_store.upsert.assert_not_called()


def test_rag_indexer_skips_quarantined_skill(tmp_path: Path) -> None:
    from core.memory.rag.indexer import MemoryIndexer

    anima_dir = tmp_path / "alice"
    skill_md = _write_skill(anima_dir, "draft-skill", extra="trust_level: quarantine\n")
    vector_store = MagicMock()
    indexer = MemoryIndexer(vector_store, "alice", anima_dir, embedding_model=MagicMock())

    chunks = indexer.index_file(skill_md, memory_type="skills", force=True)

    assert chunks == 0
    vector_store.create_collection.assert_not_called()
    vector_store.upsert.assert_not_called()


def test_rag_indexer_deletes_existing_archived_skill_chunks(tmp_path: Path) -> None:
    from core.memory.rag.indexer import MemoryIndexer

    anima_dir = tmp_path / "alice"
    skill_md = _write_skill(anima_dir, "old-skill")
    SkillCurator(anima_dir).archive_skill("old-skill", reason="unused")
    vector_store = MagicMock()
    vector_store.get_by_metadata.return_value = [
        SearchResult(
            document=Document(id="alice/skills/old-skill/SKILL.md#0", content="old body"),
            score=1.0,
        )
    ]
    indexer = MemoryIndexer(vector_store, "alice", anima_dir, embedding_model=MagicMock())

    chunks = indexer.index_file(skill_md, memory_type="skills", force=True)

    assert chunks == 0
    vector_store.delete_documents.assert_called_once_with(
        "alice_skills",
        ["alice/skills/old-skill/SKILL.md#0"],
    )


def test_rag_retriever_filters_archived_skill_chunks_left_in_vector_store(tmp_path: Path) -> None:
    from core.memory.rag.retriever import MemoryRetriever

    anima_dir = tmp_path / "alice"
    _write_skill(anima_dir, "old-skill")
    _write_skill(anima_dir, "new-skill")
    SkillCurator(anima_dir).archive_skill("old-skill", reason="unused")
    vector_store = MagicMock()
    vector_store.query.return_value = [
        SearchResult(
            document=Document(
                id="alice/skills/old-skill/SKILL.md#0",
                content="old body",
                metadata={"source_file": "skills/old-skill/SKILL.md"},
            ),
            score=0.9,
        ),
        SearchResult(
            document=Document(
                id="alice/skills/new-skill/SKILL.md#0",
                content="new body",
                metadata={"source_file": "skills/new-skill/SKILL.md"},
            ),
            score=0.8,
        ),
    ]
    indexer = MagicMock()
    indexer.anima_dir = anima_dir
    indexer._generate_embeddings.return_value = [[0.1, 0.2]]
    retriever = MemoryRetriever(vector_store, indexer, anima_dir / "knowledge")

    results = retriever.search("deploy release", "alice", memory_type="skills", top_k=5)

    assert [result.doc_id for result in results] == ["alice/skills/new-skill/SKILL.md#0"]


def test_rag_retriever_filters_archived_common_skill_chunks_from_shared_vectors(tmp_path: Path) -> None:
    from core.memory.rag.retriever import MemoryRetriever

    anima_dir = tmp_path / "alice"
    anima_dir.mkdir()
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir()
    for name in ("old-common", "new-common"):
        skill_dir = common_dir / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            f"name: {name}\n"
            f"description: {name} shared workflow\n"
            "use_when: [shared workflow]\n"
            "---\n\n"
            f"# {name}\n",
            encoding="utf-8",
        )
    SkillCurator(anima_dir).archive_skill("old-common", reason="unused")
    vector_store = MagicMock()
    vector_store.query.side_effect = [
        [],
        [
            SearchResult(
                document=Document(
                    id="shared/common_skills/old-common/SKILL.md#0",
                    content="old body",
                    metadata={"source_file": "common_skills/old-common/SKILL.md"},
                ),
                score=0.9,
            ),
            SearchResult(
                document=Document(
                    id="shared/common_skills/new-common/SKILL.md#0",
                    content="new body",
                    metadata={"source_file": "common_skills/new-common/SKILL.md"},
                ),
                score=0.8,
            ),
        ],
    ]
    indexer = MagicMock()
    indexer.anima_dir = anima_dir
    indexer._generate_embeddings.return_value = [[0.1, 0.2]]
    retriever = MemoryRetriever(vector_store, indexer, anima_dir / "knowledge")

    with patch("core.paths.get_data_dir", return_value=tmp_path):
        results = retriever.search("shared workflow", "alice", memory_type="skills", top_k=5, include_shared=True)

    assert [result.doc_id for result in results] == ["shared/common_skills/new-common/SKILL.md#0"]


def test_cron_skill_context_attaches_allowed_and_records_rejected_reason(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir()
    _write_skill(anima_dir, "active-skill")
    _write_skill(anima_dir, "old-skill")
    SkillCurator(anima_dir).archive_skill("old-skill", reason="unused")

    with patch("core.paths.get_common_skills_dir", return_value=common_dir):
        result = build_cron_skill_context(anima_dir, ["active-skill", "old-skill", "missing-skill"])

    rendered = result.render()
    assert "## Cron Skills" in rendered
    assert "active-skill" in rendered
    assert "old-skill: curator_archived" in rendered
    assert "missing-skill: not_found" in rendered


def test_cron_skill_context_rejects_quarantined_skill_refs(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir()
    _write_skill(anima_dir, "draft-skill", extra="trust_level: quarantine\n")
    nested_dir = anima_dir / "skills" / "quarantine" / "nested-draft"
    nested_dir.mkdir(parents=True)
    (nested_dir / "SKILL.md").write_text(
        "---\nname: nested-draft\ndescription: Quarantine draft\ntrust_level: quarantine\n---\n\n# Nested Draft\n",
        encoding="utf-8",
    )

    with patch("core.paths.get_common_skills_dir", return_value=common_dir):
        result = build_cron_skill_context(
            anima_dir,
            ["draft-skill", "skills/quarantine/nested-draft/SKILL.md"],
        )

    assert not result.attachments
    assert [(item.ref, item.reason) for item in result.rejections] == [
        ("draft-skill", "trust_level_quarantine"),
        ("skills/quarantine/nested-draft/SKILL.md", "trust_level_quarantine"),
    ]


def test_cron_skill_context_handles_empty_refs_and_canonical_paths(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir()
    _write_skill(anima_dir, "active-skill")
    common_skill_dir = common_dir / "common-skill"
    common_skill_dir.mkdir()
    (common_skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: common-skill\n"
        "description: Shared workflow\n"
        "use_when: [shared workflow]\n"
        "---\n\n"
        "# Common Skill\n",
        encoding="utf-8",
    )

    empty = build_cron_skill_context(anima_dir, [])
    assert empty.render() == ""

    with (
        patch("core.paths.get_common_skills_dir", return_value=common_dir),
        patch("core.paths.get_data_dir", return_value=tmp_path),
    ):
        result = build_cron_skill_context(
            anima_dir,
            [
                "skills/active-skill/SKILL.md",
                "common_skills/common-skill/SKILL.md",
                "skills/active-skill/README.md",
            ],
        )

    assert [item.name for item in result.attachments] == ["active-skill", "common-skill"]
    assert result.attachments[0].path == "skills/active-skill/SKILL.md"
    assert result.attachments[1].path == "common_skills/common-skill/SKILL.md"
    assert [(item.ref, item.reason) for item in result.rejections] == [
        ("skills/active-skill/README.md", "not_found")
    ]


def test_cron_skill_context_reports_read_failures(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir()
    _write_skill(anima_dir, "active-skill")

    with (
        patch("core.paths.get_common_skills_dir", return_value=common_dir),
        patch("core.skills.cron_context.load_skill_body", side_effect=OSError("boom")),
    ):
        result = build_cron_skill_context(anima_dir, ["active-skill"])

    assert not result.attachments
    assert [(item.ref, item.reason) for item in result.rejections] == [("active-skill", "read_failed")]


def test_cron_skill_context_rejects_absolute_and_traversal_paths(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir()

    with patch("core.paths.get_common_skills_dir", return_value=common_dir):
        result = build_cron_skill_context(
            anima_dir,
            ["/etc/passwd", "skills/../secrets/SKILL.md", "procedures/task.md"],
        )

    assert not result.attachments
    assert [item.reason for item in result.rejections] == ["not_found", "not_found", "not_found"]


def test_rag_indexer_refreshes_curator_replay_when_state_file_changes(tmp_path: Path) -> None:
    from core.memory.rag.indexer import MemoryIndexer

    anima_dir = tmp_path / "alice"
    skill_md = _write_skill(anima_dir, "old-skill")
    vector_store = MagicMock()
    indexer = MemoryIndexer(vector_store, "alice", anima_dir, embedding_model=MagicMock())
    indexer._generate_embeddings = MagicMock(return_value=[[0.1, 0.2]])

    first = indexer.index_file(skill_md, memory_type="skills", force=True)
    SkillCurator(anima_dir).archive_skill("old-skill", reason="unused")
    second = indexer.index_file(skill_md, memory_type="skills", force=True)

    assert first == 1
    assert second == 0
