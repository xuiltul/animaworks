from __future__ import annotations

import json
from pathlib import Path

from core.memory.rag_search import RAGMemorySearch
from core.skills.index import SkillIndex


def _write_skill(path: Path, name: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {text}\n---\n\n{text}\n", encoding="utf-8")


def test_assigned_anima_discovers_and_searches_only_own_company_resources(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text(json.dumps({"company": "alpha"}), encoding="utf-8")
    common_knowledge = tmp_path / "common_knowledge"
    common_skills = tmp_path / "common_skills"
    common_knowledge.mkdir()
    common_skills.mkdir()

    alpha_knowledge = tmp_path / "companies" / "alpha" / "knowledge"
    beta_knowledge = tmp_path / "companies" / "beta" / "knowledge"
    alpha_knowledge.mkdir(parents=True)
    beta_knowledge.mkdir(parents=True)
    (alpha_knowledge / "handbook.md").write_text("alpha-exclusive launch protocol", encoding="utf-8")
    (beta_knowledge / "secret.md").write_text("beta-exclusive launch protocol", encoding="utf-8")
    _write_skill(tmp_path / "companies" / "alpha" / "skills" / "alpha-tool" / "SKILL.md", "alpha-tool", "alpha company tool")
    _write_skill(tmp_path / "companies" / "beta" / "skills" / "beta-tool" / "SKILL.md", "beta-tool", "beta company tool")

    rag = RAGMemorySearch(anima_dir, common_knowledge, common_skills)
    results = rag._keyword_search_fallback(
        "exclusive launch",
        "common_knowledge",
        0,
        knowledge_dir=anima_dir / "knowledge",
        episodes_dir=anima_dir / "episodes",
        procedures_dir=anima_dir / "procedures",
        common_knowledge_dir=common_knowledge,
    )
    skills = SkillIndex(anima_dir / "skills", common_skills, anima_dir=anima_dir).all_skills

    assert [result["source_file"] for result in results] == ["companies/alpha/knowledge/handbook.md"]
    assert [skill.name for skill in skills] == ["alpha-tool"]

