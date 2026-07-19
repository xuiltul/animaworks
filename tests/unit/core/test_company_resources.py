from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from core.company_resources import company_resource_pointer, get_company_resources
from core.file_access_policy import resolve_memory_source_path
from core.memory.rag.retriever import MemoryRetriever
from core.memory.rag_search import RAGMemorySearch
from core.skills.index import SkillIndex
from core.tooling.handler import ToolHandler


def _assign(anima_dir: Path, company: str | None) -> None:
    anima_dir.mkdir(parents=True, exist_ok=True)
    payload = {"company": company} if company is not None else {}
    (anima_dir / "status.json").write_text(json.dumps(payload), encoding="utf-8")


def _skill(root: Path, name: str) -> Path:
    path = root / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {name}\n---\n", encoding="utf-8")
    return path


def test_company_resources_resolve_only_safe_assigned_company(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _assign(anima_dir, "alpha")

    resources = get_company_resources(anima_dir)

    assert resources is not None
    assert resources.knowledge_dir == tmp_path / "companies" / "alpha" / "knowledge"
    assert resources.skills_dir == tmp_path / "companies" / "alpha" / "skills"

    _assign(anima_dir, "../escape")
    assert get_company_resources(anima_dir) is None


def test_skill_index_adds_own_company_and_refreshes_assignment(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _assign(anima_dir, "alpha")
    common = tmp_path / "common_skills"
    personal = anima_dir / "skills"
    procedures = anima_dir / "procedures"
    common.mkdir(parents=True)
    personal.mkdir()
    procedures.mkdir()
    _skill(common, "common")
    alpha_path = _skill(tmp_path / "companies" / "alpha" / "skills", "alpha-only")
    beta_path = _skill(tmp_path / "companies" / "beta" / "skills", "beta-only")

    index = SkillIndex(personal, common, procedures, anima_dir=anima_dir)

    assert {meta.name for meta in index.all_skills} == {"common", "alpha-only"}
    assert index.resolve_skill_reference("companies/alpha/skills/alpha-only/SKILL.md") is not None
    assert index.resolve_skill_reference("companies/beta/skills/beta-only/SKILL.md") is None
    assert company_resource_pointer(alpha_path) == "companies/alpha/skills/alpha-only/SKILL.md"

    _assign(anima_dir, "beta")
    assert {meta.name for meta in index.all_skills} == {"common", "beta-only"}
    assert index.resolve_skill_reference("companies/alpha/skills/alpha-only/SKILL.md") is None
    assert index.resolve_skill_reference("companies/beta/skills/beta-only/SKILL.md").path == beta_path


def test_unassigned_skill_index_remains_legacy_only(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _assign(anima_dir, None)
    common = tmp_path / "common_skills"
    common.mkdir()
    _skill(common, "common")
    _skill(tmp_path / "companies" / "alpha" / "skills", "company-only")

    index = SkillIndex(anima_dir / "skills", common, anima_dir=anima_dir)

    assert [meta.name for meta in index.all_skills] == ["common"]


def test_memory_source_path_allows_only_own_company(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _assign(anima_dir, "alpha")
    own = tmp_path / "companies" / "alpha" / "knowledge" / "guide.md"

    assert resolve_memory_source_path(anima_dir, "companies/alpha/knowledge/guide.md") == own
    assert resolve_memory_source_path(anima_dir, "companies/beta/knowledge/secret.md") is None
    assert resolve_memory_source_path(anima_dir, "companies/alpha/../beta/secret.md") is None


def test_read_memory_file_allows_own_company_and_rejects_other(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _assign(anima_dir, "alpha")
    (anima_dir / "permissions.json").write_text(
        '{"version":1,"file_roots":["/"],"commands":{"allow_all":true},'
        '"external_tools":{"allow_all":true}}',
        encoding="utf-8",
    )
    own = tmp_path / "companies" / "alpha" / "knowledge" / "guide.md"
    other = tmp_path / "companies" / "beta" / "knowledge" / "secret.md"
    own.parent.mkdir(parents=True)
    other.parent.mkdir(parents=True)
    own.write_text("own company guide", encoding="utf-8")
    other.write_text("other company secret", encoding="utf-8")
    memory = MagicMock()
    memory.read_permissions.return_value = ""
    handler = ToolHandler(anima_dir=anima_dir, memory=memory, messenger=None, tool_registry=[])

    assert handler.handle("read_memory_file", {"path": "companies/alpha/knowledge/guide.md"}) == "own company guide"
    denied = handler.handle("read_memory_file", {"path": "companies/beta/knowledge/secret.md"})
    assert "PermissionDenied" in denied
    assert "other company secret" not in denied


def test_vector_sources_resolve_and_filter_to_current_company(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _assign(anima_dir, "alpha")
    own_skill = _skill(tmp_path / "companies" / "alpha" / "skills", "own")
    _skill(tmp_path / "companies" / "beta" / "skills", "other")
    indexer = MagicMock()
    indexer.anima_dir = anima_dir
    retriever = MemoryRetriever(MagicMock(), indexer, anima_dir / "knowledge")
    rag = RAGMemorySearch(anima_dir, tmp_path / "common_knowledge", tmp_path / "common_skills")

    assert retriever._resolve_skill_source_file("companies/alpha/skills/own/SKILL.md") == own_skill
    assert retriever._resolve_skill_source_file("companies/beta/skills/other/SKILL.md") is None
    assert rag._company_source_is_visible("companies/alpha/knowledge/guide.md")
    assert not rag._company_source_is_visible("companies/beta/knowledge/secret.md")
