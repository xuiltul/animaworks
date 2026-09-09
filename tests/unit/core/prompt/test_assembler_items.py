from __future__ import annotations

from core.prompt.assembler import PromptBudget, SectionEntry, _allocate_sections


def _by_id(allocated: list[SectionEntry], section_id: str) -> SectionEntry | None:
    return next((section for section in allocated if section.id == section_id), None)


def test_priming_blocks_are_trimmed_as_complete_items() -> None:
    first = '<priming source="activity">\n' + "first " * 60 + "\n</priming>"
    second = '<priming source="knowledge">\n' + "second " * 60 + "\n</priming>"
    sections = [
        SectionEntry("identity", 1, "rigid", "identity"),
        SectionEntry("priming", 2, "elastic", f"## Recalled\n\n{first}\n\n{second}"),
    ]

    allocated = _allocate_sections(sections, PromptBudget(target=140, ceiling=1000))
    content = _by_id(allocated, "priming").content  # type: ignore[union-attr]

    assert content.count("<priming") == content.count("</priming>")
    assert first in content
    assert second not in content


def test_trim_from_head_keeps_newest_tail_item() -> None:
    sections = [
        SectionEntry("identity", 1, "rigid", "identity"),
        SectionEntry(
            "current_state",
            2,
            "elastic",
            "old " * 50 + "\n\n" + "new " * 50,
            trim_from="head",
        ),
    ]

    allocated = _allocate_sections(sections, PromptBudget(target=70, ceiling=1000))
    content = _by_id(allocated, "current_state").content  # type: ignore[union-attr]

    assert "old" not in content
    assert "new" in content


def test_lower_priority_elastic_is_dropped_first() -> None:
    sections = [
        SectionEntry("identity", 1, "rigid", "identity"),
        SectionEntry("important", 2, "elastic", "important " * 40),
        SectionEntry("optional", 3, "elastic", "optional " * 40),
    ]

    allocated = _allocate_sections(sections, PromptBudget(target=130, ceiling=1000))

    assert _by_id(allocated, "important") is not None
    assert _by_id(allocated, "optional") is None


def test_target_does_not_drop_content_that_already_fits() -> None:
    sections = [
        SectionEntry("identity", 1, "rigid", "identity"),
        SectionEntry("context", 4, "elastic", "context " * 10),
    ]

    allocated = _allocate_sections(sections, PromptBudget(target=100, ceiling=100))

    assert [(section.id, section.content) for section in allocated] == [
        (section.id, section.content) for section in sections
    ]


def test_rigid_sections_are_dropped_only_above_ceiling() -> None:
    sections = [
        SectionEntry("identity", 1, "rigid", "identity " * 10),
        SectionEntry("important", 2, "rigid", "important " * 40),
        SectionEntry("optional", 4, "rigid", "optional " * 40),
    ]

    below_ceiling = _allocate_sections(sections, PromptBudget(target=10, ceiling=300))
    above_ceiling = _allocate_sections(sections, PromptBudget(target=10, ceiling=160))

    assert {section.id for section in below_ceiling} == {"identity", "important", "optional"}
    assert "optional" not in {section.id for section in above_ceiling}
    assert "important" in {section.id for section in above_ceiling}


def test_duplicate_long_paragraph_removed_only_from_later_elastic() -> None:
    repeated = "This deliberately long paragraph is repeated exactly across several sections. " * 2
    sections = [
        SectionEntry("rigid_a", 1, "rigid", repeated),
        SectionEntry("rigid_b", 2, "rigid", repeated),
        SectionEntry("elastic", 2, "elastic", f"unique\n\n{repeated}"),
    ]

    allocated = _allocate_sections(sections, PromptBudget(target=1000, ceiling=1000))

    assert _by_id(allocated, "rigid_a").content == repeated  # type: ignore[union-attr]
    assert _by_id(allocated, "rigid_b").content == repeated  # type: ignore[union-attr]
    assert _by_id(allocated, "elastic").content == "unique"  # type: ignore[union-attr]


def test_priming_attributes_are_ignored_for_duplicate_detection() -> None:
    body = "same recalled activity paragraph " * 5
    sections = [
        SectionEntry("first", 2, "elastic", f'<priming source="a">{body}</priming>'),
        SectionEntry("second", 2, "elastic", f'<priming source="b">{body}</priming>'),
    ]

    allocated = _allocate_sections(sections, PromptBudget(target=1000, ceiling=1000))

    assert _by_id(allocated, "first") is not None
    assert _by_id(allocated, "second") is None
