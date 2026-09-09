from __future__ import annotations

from core.prompt.assembler import PromptBudget, SectionEntry, _allocate_sections
from core.prompt.builder import _skill_catalog_sections
from core.prompt.tokens import estimate_tokens


def test_large_mode_b_catalog_protects_only_bounded_ranked_candidates():
    entries = [f"- skills/ranked-{index}/SKILL.md: Short description {index}." for index in range(100)]
    sections = _skill_catalog_sections(entries, mode_b=True)
    protected = [section for section in sections if section.kind == "rigid"]
    assert len(protected) == 1
    assert estimate_tokens(protected[0].content) <= 512
    assert protected[0].content.count("- skills/") == 3
    assert entries[0] in protected[0].content
    assert entries[3] not in protected[0].content
    rendered = "\n".join(section.content for section in sections)
    assert all(rendered.count(entry + "\n") == 1 for entry in entries)
    allocated = _allocate_sections(sections, PromptBudget(target=1, ceiling=1000))
    assert allocated == protected


def test_mode_b_catalog_still_respects_hard_ceiling():
    sections = [SectionEntry("identity", 1, "rigid", "identity")]
    sections += _skill_catalog_sections(["- skills/test/SKILL.md: A test skill."], mode_b=True)
    allocated = _allocate_sections(sections, PromptBudget(target=1, ceiling=10))
    assert [section.id for section in allocated] == ["identity"]


def test_other_modes_keep_elastic_catalog_contract():
    sections = _skill_catalog_sections(["- skills/test/SKILL.md: A test skill."], mode_b=False)
    assert len(sections) == 1
    assert sections[0].kind == "elastic"
    assert _allocate_sections(sections, PromptBudget(target=1, ceiling=1000)) == []
