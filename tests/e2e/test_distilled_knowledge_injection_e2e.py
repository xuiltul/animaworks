# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for distilled knowledge summary injection pipeline.

Validates the complete flow from knowledge/procedures files through
summary extraction, budget computation, and system prompt injection.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from core.memory.manager import MemoryManager
from core.prompt.builder import build_system_prompt

# ── Helpers ──────────────────────────────────────────────


def _write_file_with_frontmatter(
    directory: Path,
    name: str,
    content: str,
    confidence: float,
    *,
    description: str = "",
) -> None:
    """Write a knowledge/procedure file with YAML frontmatter."""
    directory.mkdir(parents=True, exist_ok=True)
    meta: dict = {"confidence": confidence}
    if description:
        meta["description"] = description
    frontmatter = yaml.dump(meta, default_flow_style=False)
    full = f"---\n{frontmatter}---\n\n{content}"
    (directory / f"{name}.md").write_text(full, encoding="utf-8")


def _make_mock_memory(anima_dir: Path, data_dir: Path) -> MagicMock:
    """Create a MagicMock MemoryManager with standard empty returns."""
    memory = MagicMock(spec=MemoryManager)
    memory.anima_dir = anima_dir
    memory.common_skills_dir = data_dir / "common_skills"
    memory.read_company_vision.return_value = ""
    memory.read_identity.return_value = "I am TestAnima"
    memory.read_injection.return_value = ""
    memory.read_permissions.return_value = ""
    memory.read_specialty_prompt.return_value = ""
    memory.read_current_state.return_value = ""
    memory.read_pending.return_value = ""
    memory.read_bootstrap.return_value = ""
    memory.list_knowledge_files.return_value = []
    memory.list_episode_files.return_value = []
    memory.list_procedure_files.return_value = []
    memory.list_shared_users.return_value = []
    memory.load_recent_heartbeat_summary.return_value = ""
    memory.list_procedure_metas.return_value = []
    memory.list_skill_metas.return_value = []
    memory.list_common_skill_metas.return_value = []
    memory.collect_distilled_knowledge_separated.return_value = ([], [])
    memory.read_model_config.return_value = MagicMock(
        model="claude-sonnet-4-6",
        supervisor=None,
    )
    return memory


def _fake_load_prompt(name: str, **kwargs: object) -> str:
    """Minimal load_prompt mock returning empty for most templates."""
    if name == "skills_guide":
        return "## Skills Guide"
    return ""


# ── Test Cases ───────────────────────────────────────────


class TestDistilledKnowledgeSummaryInjectionE2E:
    """E2E tests for DK summary injection pipeline."""

    def test_full_summary_pipeline(self, data_dir: Path, make_anima: object) -> None:
        """Full pipeline: collect -> extract description -> inject summaries.

        Creates 5 files (3 knowledge, 2 procedures) with descriptions,
        verifies collect returns description+mtime, and build_system_prompt
        injects summaries (not full content).
        """
        anima_dir = make_anima("test-summary")
        knowledge_dir = anima_dir / "knowledge"
        procedures_dir = anima_dir / "procedures"

        _write_file_with_frontmatter(
            knowledge_dir,
            "api-patterns",
            "API patterns for REST design.",
            0.9,
            description="REST API design patterns",
        )
        _write_file_with_frontmatter(
            knowledge_dir,
            "error-handling",
            "Error handling best practices.",
            0.7,
            description="Error handling guide",
        )
        _write_file_with_frontmatter(
            knowledge_dir,
            "logging-tips",
            "Logging guidelines for services.",
            0.5,
            description="Service logging tips",
        )
        _write_file_with_frontmatter(
            procedures_dir,
            "deploy-procedure",
            "Step-by-step deploy guide.",
            0.8,
            description="Docker deployment steps",
        )
        _write_file_with_frontmatter(
            procedures_dir,
            "rollback-procedure",
            "Rollback instructions.",
            0.3,
            description="Emergency rollback",
        )

        mm = MemoryManager(anima_dir)
        procs, knows = mm.collect_distilled_knowledge_separated()
        assert len(procs) == 2
        assert len(knows) == 3

        for entry in procs + knows:
            assert "description" in entry
            assert "mtime" in entry
            assert entry["mtime"] > 0

        mock_memory = _make_mock_memory(anima_dir, data_dir)
        mock_memory.collect_distilled_knowledge_separated.return_value = (procs, knows)

        with (
            patch("core.prompt.builder.load_prompt", side_effect=_fake_load_prompt),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            result = build_system_prompt(mock_memory, message="test")

        prompt = result.system_prompt
        assert "- **deploy-procedure**: Docker deployment steps" in prompt
        assert "- **api-patterns**: REST API design patterns" in prompt

        assert "Step-by-step deploy guide." not in prompt
        assert "API patterns for REST design." not in prompt

    def test_description_fallback_to_heading(self, data_dir: Path, make_anima: object) -> None:
        """Files without frontmatter description fall back to # heading."""
        anima_dir = make_anima("test-fallback")
        procedures_dir = anima_dir / "procedures"

        _write_file_with_frontmatter(
            procedures_dir,
            "my-procedure",
            "# Custom Heading\n\nBody text here.",
            0.7,
        )

        mm = MemoryManager(anima_dir)
        procs, _ = mm.collect_distilled_knowledge_separated()
        assert len(procs) == 1
        assert procs[0]["description"] == ""

        mock_memory = _make_mock_memory(anima_dir, data_dir)
        mock_memory.collect_distilled_knowledge_separated.return_value = (procs, [])

        with (
            patch("core.prompt.builder.load_prompt", side_effect=_fake_load_prompt),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            result = build_system_prompt(mock_memory, message="test")

        assert "- **my-procedure**: Custom Heading" in result.system_prompt

    def test_overflow_with_summary_budget(self, data_dir: Path, make_anima: object) -> None:
        """Many entries exceed the 200-token knowledge budget."""
        anima_dir = make_anima("test-overflow")
        knowledge_dir = anima_dir / "knowledge"

        for i in range(50):
            _write_file_with_frontmatter(
                knowledge_dir,
                f"knowledge-{i:02d}",
                f"Content for knowledge item {i}.",
                0.6,
                description=f"Description for knowledge item {i} with enough text to consume budget",
            )

        mm = MemoryManager(anima_dir)
        _, knows = mm.collect_distilled_knowledge_separated()
        assert len(knows) == 50

        mock_memory = _make_mock_memory(anima_dir, data_dir)
        mock_memory.collect_distilled_knowledge_separated.return_value = ([], knows)

        with (
            patch("core.prompt.builder.load_prompt", side_effect=_fake_load_prompt),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            result = build_system_prompt(mock_memory, message="test")

        assert len(result.overflow_files) > 0
        assert len(result.overflow_files) + len(result.injected_knowledge_files) == 50

    def test_procedures_keyword_and_vector_search_scope(
        self,
        data_dir: Path,
        make_anima: object,
    ) -> None:
        """search_memory_text with scope='procedures' finds keyword matches."""
        anima_dir = make_anima("test-search")
        procedures_dir = anima_dir / "procedures"

        _write_file_with_frontmatter(
            procedures_dir,
            "deploy-guide",
            "Run terraform apply to deploy the infrastructure.",
            0.8,
            description="Terraform deploy",
        )
        _write_file_with_frontmatter(
            procedures_dir,
            "monitoring-setup",
            "Configure Prometheus and Grafana for monitoring.",
            0.7,
            description="Prometheus monitoring",
        )

        mm = MemoryManager(anima_dir)

        results = mm.search_memory_text("terraform", scope="procedures")
        assert len(results) > 0
        filenames = [r["source_file"] for r in results]
        assert any("deploy-guide" in f for f in filenames)

        results2 = mm.search_memory_text("Prometheus", scope="procedures")
        assert len(results2) > 0

        from core.memory.rag_search import RAGMemorySearch

        types = RAGMemorySearch._resolve_search_types("procedures")
        assert types == ["procedures"]

    def test_confidence_default_without_frontmatter(
        self,
        data_dir: Path,
        make_anima: object,
    ) -> None:
        """Files without YAML frontmatter default to confidence=0.5."""
        anima_dir = make_anima("test-no-frontmatter")
        knowledge_dir = anima_dir / "knowledge"
        procedures_dir = anima_dir / "procedures"

        knowledge_dir.mkdir(parents=True, exist_ok=True)
        (knowledge_dir / "plain-knowledge.md").write_text(
            "# Plain Knowledge\n\nSome content without frontmatter.",
            encoding="utf-8",
        )
        procedures_dir.mkdir(parents=True, exist_ok=True)
        (procedures_dir / "plain-procedure.md").write_text(
            "# Plain Procedure\n\nStep 1: Do something.",
            encoding="utf-8",
        )

        mm = MemoryManager(anima_dir)
        entries = mm.collect_distilled_knowledge()

        assert len(entries) == 2
        for entry in entries:
            assert entry["confidence"] == pytest.approx(0.5)
            assert "description" in entry
            assert "mtime" in entry

    def test_mtime_sorting(self, data_dir: Path, make_anima: object) -> None:
        """Entries with same confidence are sorted by mtime descending."""
        import time

        anima_dir = make_anima("test-mtime")
        knowledge_dir = anima_dir / "knowledge"

        _write_file_with_frontmatter(
            knowledge_dir,
            "older-file",
            "Old content.",
            0.5,
            description="Older",
        )
        time.sleep(0.05)
        _write_file_with_frontmatter(
            knowledge_dir,
            "newer-file",
            "New content.",
            0.5,
            description="Newer",
        )

        mm = MemoryManager(anima_dir)
        _, knows = mm.collect_distilled_knowledge_separated()
        assert len(knows) == 2
        assert knows[0]["name"] == "newer-file"
        assert knows[1]["name"] == "older-file"
