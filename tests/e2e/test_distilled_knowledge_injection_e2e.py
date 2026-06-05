"""E2E tests for DK prompt-injection removal.

Knowledge and procedure files remain available through memory search and
explicit reads, but their distilled summaries are no longer injected into the
fixed system prompt.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from core.memory.manager import MemoryManager
from core.prompt.builder import build_system_prompt


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


def _fake_load_prompt(name: str, **kwargs: object) -> str:
    """Minimal load_prompt mock returning empty for most templates."""
    if name == "memory_guide":
        return "knowledge={knowledge_count} procedures={procedure_count}".format(**kwargs)
    return ""


class TestDKPromptInjectionRemovalE2E:
    def test_knowledge_and_procedures_not_injected_into_system_prompt(
        self,
        data_dir: Path,
        make_anima: object,
    ) -> None:
        anima_dir = make_anima("test-dk-removal")
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
            procedures_dir,
            "deploy-procedure",
            "Step-by-step deploy guide.",
            0.8,
            description="Docker deployment steps",
        )

        memory = MemoryManager(anima_dir)

        with (
            patch("core.prompt.builder.load_prompt", side_effect=_fake_load_prompt),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            result = build_system_prompt(memory, message="test")

        prompt = result.system_prompt
        assert "dk_procedures" not in prompt
        assert "dk_knowledge" not in prompt
        assert "REST API design patterns" not in prompt
        assert "Docker deployment steps" not in prompt
        assert "API patterns for REST design." not in prompt
        assert "Step-by-step deploy guide." not in prompt
        assert "knowledge=1 procedures=1" in prompt

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

    def test_knowledge_metadata_still_readable(
        self,
        data_dir: Path,
        make_anima: object,
    ) -> None:
        """Removing DK collection does not remove knowledge metadata support."""
        anima_dir = make_anima("test-knowledge-metadata")
        knowledge_dir = anima_dir / "knowledge"

        _write_file_with_frontmatter(
            knowledge_dir,
            "plain-knowledge",
            "# Plain Knowledge\n\nSome content.",
            0.5,
            description="Plain knowledge summary",
        )

        mm = MemoryManager(anima_dir)
        path = knowledge_dir / "plain-knowledge.md"

        assert mm.read_knowledge_metadata(path)["description"] == "Plain knowledge summary"
        assert "Some content." in mm.read_knowledge_content(path)
