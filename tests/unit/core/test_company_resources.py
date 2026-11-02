from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.config.file_access_policy import resolve_memory_source_path
from core.memory.rag.retriever import MemoryRetriever
from core.memory.retrieval.rag_search import RAGMemorySearch
from core.org.company_resources import company_resource_pointer, get_company_resources
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
        '{"version":1,"file_roots":["/"],"commands":{"allow_all":true},"external_tools":{"allow_all":true}}',
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


def _narrow_handler(anima_dir: Path, *, file_roots_denied: list[str] | None = None) -> ToolHandler:
    """Build a handler whose file_roots grant nothing under the data directory."""
    (anima_dir / "permissions.json").write_text(
        json.dumps(
            {
                "version": 1,
                "file_roots": ["/nonexistent-work-root"],
                "file_roots_denied": file_roots_denied or [],
                "commands": {"allow_all": True},
                "external_tools": {"allow_all": True},
            }
        ),
        encoding="utf-8",
    )
    memory = MagicMock()
    memory.read_permissions.return_value = ""
    return ToolHandler(anima_dir=anima_dir, memory=memory, messenger=None, tool_registry=[])


def _company_file(tmp_path: Path, relative: str, body: str, *, company: str = "alpha") -> Path:
    path = tmp_path / "companies" / company / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.parametrize("relative", ["skills/demo/SKILL.md", "knowledge/guide.md", "shared/note.md"])
def test_file_tools_read_own_company_with_narrow_file_roots(tmp_path: Path, relative: str) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _assign(anima_dir, "alpha")
    path = _company_file(tmp_path, relative, "---\nname: demo\ndescription: d\n---\nreadable body")
    handler = _narrow_handler(anima_dir)

    assert "readable body" in handler.handle("read_memory_file", {"path": f"companies/alpha/{relative}"})
    assert "readable body" in handler.handle("read_memory_file", {"path": str(path)})
    assert "readable body" in handler.handle("read_file", {"path": str(path)})


@pytest.mark.parametrize(
    "relative",
    ["credentials/token.txt", "company.json", "animas/bob/note.md", "vision.md"],
)
def test_file_tools_deny_rest_of_own_company_with_narrow_file_roots(tmp_path: Path, relative: str) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _assign(anima_dir, "alpha")
    path = _company_file(tmp_path, relative, "private body")
    handler = _narrow_handler(anima_dir)

    for result in (
        handler.handle("read_memory_file", {"path": f"companies/alpha/{relative}"}),
        handler.handle("read_memory_file", {"path": str(path)}),
        handler.handle("read_memory_file", {"path": f"companies/alpha/shared/../{relative}"}),
        handler.handle("read_file", {"path": str(path)}),
    ):
        assert "PermissionDenied" in result
        assert "private body" not in result


def test_file_tools_deny_other_company_with_narrow_file_roots(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _assign(anima_dir, "alpha")
    _company_file(tmp_path, "shared/own.md", "own")
    other = _company_file(tmp_path, "shared/note.md", "other company body", company="beta")
    handler = _narrow_handler(anima_dir)

    for result in (
        handler.handle("read_memory_file", {"path": "companies/beta/shared/note.md"}),
        handler.handle("read_file", {"path": str(other)}),
    ):
        assert "PermissionDenied" in result
        assert "other company body" not in result


def test_file_roots_denied_overrides_company_read_roots(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _assign(anima_dir, "alpha")
    secret = _company_file(tmp_path, "shared/private/key.pem", "key body")
    readable = _company_file(tmp_path, "shared/note.md", "note body")
    handler = _narrow_handler(anima_dir, file_roots_denied=[str(secret.parent)])

    assert "note body" in handler.handle("read_file", {"path": str(readable)})
    result = handler.handle("read_file", {"path": str(secret)})
    assert "PermissionDenied" in result
    assert "key body" not in result


def test_unassigned_anima_gets_no_company_read_roots(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _assign(anima_dir, None)
    path = _company_file(tmp_path, "skills/demo/SKILL.md", "skill body")
    handler = _narrow_handler(anima_dir)

    result = handler.handle("read_file", {"path": str(path)})
    assert "PermissionDenied" in result
    assert "skill body" not in result


def test_symlinked_company_resource_root_is_not_granted(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _assign(anima_dir, "alpha")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text("outside body", encoding="utf-8")
    company = tmp_path / "companies" / "alpha"
    company.mkdir(parents=True)
    (company / "skills").symlink_to(outside, target_is_directory=True)
    handler = _narrow_handler(anima_dir)

    result = handler.handle("read_file", {"path": str(company / "skills" / "SKILL.md")})
    assert "PermissionDenied" in result
    assert "outside body" not in result


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
