from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from core._anima_lifecycle import _format_hygiene_section
from core.i18n.strings.memory import STRINGS as MEMORY_STRINGS
from core.memory.hygiene import scan_memory_hygiene


def _paths(report: dict, category: str) -> list[str]:
    return [entry["path"] for entry in report[category]]


def test_scan_detects_all_categories_and_excludes_canonical_archive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    knowledge = anima_dir / "knowledge"
    (knowledge / "nested").mkdir(parents=True)
    (knowledge / "archive").mkdir()
    (knowledge / "inherited-team").mkdir()
    (knowledge / "archived").mkdir()
    (knowledge / "_archived").mkdir()
    (knowledge / ".archive").mkdir()

    (knowledge / "_merged_root.md").write_text("root", encoding="utf-8")
    (knowledge / "nested" / "_merged_nested.md").write_text("nested", encoding="utf-8")
    (knowledge / "notes.mdc").write_text("legacy", encoding="utf-8")
    (knowledge / "large.md").write_bytes(b"x" * (32 * 1024 + 1))
    (knowledge / "archive" / "_merged_ignored.md").write_text("ignored", encoding="utf-8")
    (knowledge / "archive" / "ignored.mdc").write_text("ignored", encoding="utf-8")

    monkeypatch.setattr("core.memory.hygiene.today_local", lambda: date(2026, 7, 18))
    report = scan_memory_hygiene(anima_dir)

    assert _paths(report, "merged_leftovers") == [
        "knowledge/_merged_root.md",
        "knowledge/nested/_merged_nested.md",
    ]
    assert _paths(report, "inherited_dirs") == ["knowledge/inherited-team"]
    assert _paths(report, "mdc_files") == ["knowledge/notes.mdc"]
    assert _paths(report, "oversized_knowledge") == ["knowledge/large.md"]
    assert report["oversized_knowledge"][0]["size_bytes"] == 32 * 1024 + 1
    assert _paths(report, "noncanonical_archive_dirs") == [
        "knowledge/archived",
        "knowledge/_archived",
        "knowledge/.archive",
    ]
    assert _paths(report, "noncanonical_episodes") == []
    assert all(entry["first_seen"] == "2026-07-18" for entries in report.values() for entry in entries)
    assert json.loads((anima_dir / "state" / "memory_hygiene.json").read_text()) == report


def test_scan_detects_noncanonical_episodes(tmp_path: Path, monkeypatch) -> None:
    """Date-prefixed episodes are clean; legacy recovered_/inbox/jsonl are flagged.

    Files under episodes/archive/ are excluded (not scanned).
    """
    anima_dir = tmp_path / "alice"
    episodes = anima_dir / "episodes"
    archive = episodes / "archive"
    archive.mkdir(parents=True)

    (episodes / "2026-07-18.md").write_text("canonical day", encoding="utf-8")
    (episodes / "2026-07-18_suffix.md").write_text("canonical with suffix", encoding="utf-8")
    (episodes / "2026-07-18_recovered-093015.md").write_text("new recovery form", encoding="utf-8")
    (episodes / "recovered_2026-06-19_081003.md").write_text("legacy recovery", encoding="utf-8")
    (episodes / "inbox-note.md").write_text("inbox", encoding="utf-8")
    (episodes / "events.jsonl").write_text("{}\n", encoding="utf-8")
    (episodes / "notes.json").write_text("{}", encoding="utf-8")
    (archive / "recovered_old.md").write_text("archived legacy", encoding="utf-8")

    monkeypatch.setattr("core.memory.hygiene.today_local", lambda: date(2026, 7, 18))
    report = scan_memory_hygiene(anima_dir)

    assert set(_paths(report, "noncanonical_episodes")) == {
        "episodes/recovered_2026-06-19_081003.md",
        "episodes/inbox-note.md",
        "episodes/events.jsonl",
        "episodes/notes.json",
    }
    assert all(entry["first_seen"] == "2026-07-18" for entry in report["noncanonical_episodes"])


def test_scan_retains_first_seen_for_noncanonical_episodes(tmp_path: Path, monkeypatch) -> None:
    anima_dir = tmp_path / "alice"
    episodes = anima_dir / "episodes"
    episodes.mkdir(parents=True)
    legacy = episodes / "recovered_old.md"
    legacy.write_text("old", encoding="utf-8")

    monkeypatch.setattr("core.memory.hygiene.today_local", lambda: date(2026, 7, 1))
    first = scan_memory_hygiene(anima_dir)
    assert first["noncanonical_episodes"][0]["first_seen"] == "2026-07-01"

    monkeypatch.setattr("core.memory.hygiene.today_local", lambda: date(2026, 7, 18))
    second = scan_memory_hygiene(anima_dir)
    assert second["noncanonical_episodes"][0]["first_seen"] == "2026-07-01"


def test_scan_retains_first_seen_and_removes_resolved_items(tmp_path: Path, monkeypatch) -> None:
    anima_dir = tmp_path / "alice"
    knowledge = anima_dir / "knowledge"
    knowledge.mkdir(parents=True)
    leftover = knowledge / "_merged_old.md"
    resolved = knowledge / "legacy.mdc"
    leftover.write_text("old", encoding="utf-8")
    resolved.write_text("legacy", encoding="utf-8")

    monkeypatch.setattr("core.memory.hygiene.today_local", lambda: date(2026, 7, 1))
    first = scan_memory_hygiene(anima_dir)
    assert first["merged_leftovers"][0]["first_seen"] == "2026-07-01"

    resolved.unlink()
    (knowledge / "new.mdc").write_text("new", encoding="utf-8")
    monkeypatch.setattr("core.memory.hygiene.today_local", lambda: date(2026, 7, 18))
    second = scan_memory_hygiene(anima_dir)

    assert second["merged_leftovers"][0]["first_seen"] == "2026-07-01"
    assert _paths(second, "mdc_files") == ["knowledge/new.mdc"]
    assert second["mdc_files"][0]["first_seen"] == "2026-07-18"
    assert "knowledge/legacy.mdc" not in json.dumps(second)


def test_scan_without_existing_report_or_knowledge_dir(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"

    report = scan_memory_hygiene(anima_dir)

    assert all(entries == [] for entries in report.values())
    assert (anima_dir / "state" / "memory_hygiene.json").is_file()


def test_scan_resets_invalid_and_future_first_seen_dates(tmp_path: Path, monkeypatch) -> None:
    anima_dir = tmp_path / "alice"
    knowledge = anima_dir / "knowledge"
    knowledge.mkdir(parents=True)
    (knowledge / "_merged_invalid.md").write_text("invalid", encoding="utf-8")
    (knowledge / "_merged_future.md").write_text("future", encoding="utf-8")
    state = anima_dir / "state"
    state.mkdir()
    (state / "memory_hygiene.json").write_text(
        json.dumps(
            {
                "merged_leftovers": [
                    {"path": "knowledge/_merged_invalid.md", "first_seen": "not-a-date"},
                    {"path": "knowledge/_merged_future.md", "first_seen": "2026-07-19"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("core.memory.hygiene.today_local", lambda: date(2026, 7, 18))

    report = scan_memory_hygiene(anima_dir)

    assert {entry["first_seen"] for entry in report["merged_leftovers"]} == {"2026-07-18"}


def test_scan_recovers_from_broken_hygiene_json(tmp_path: Path, monkeypatch) -> None:
    anima_dir = tmp_path / "alice"
    knowledge = anima_dir / "knowledge"
    knowledge.mkdir(parents=True)
    (knowledge / "_merged_note.md").write_text("note", encoding="utf-8")
    state = anima_dir / "state"
    state.mkdir()
    report_path = state / "memory_hygiene.json"
    report_path.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr("core.memory.hygiene.today_local", lambda: date(2026, 7, 18))

    report = scan_memory_hygiene(anima_dir)

    assert report["merged_leftovers"][0]["first_seen"] == "2026-07-18"
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_hygiene_prompt_section_lists_items_caps_at_twenty_and_localizes() -> None:
    report = {
        "merged_leftovers": [
            {"path": f"knowledge/_merged_{index}.md", "first_seen": "2026-07-18"} for index in range(22)
        ],
        "inherited_dirs": [],
        "mdc_files": [{"path": "knowledge/legacy.mdc", "first_seen": "2026-07-18"}],
        "oversized_knowledge": [
            {
                "path": "knowledge/large.md",
                "first_seen": "2026-07-18",
                "size_bytes": 40 * 1024,
            }
        ],
        "noncanonical_archive_dirs": [],
        "noncanonical_episodes": [],
    }

    section = _format_hygiene_section(report, locale="en")

    assert "knowledge/_merged_0.md" in section
    assert "knowledge/_merged_19.md" in section
    assert "knowledge/_merged_20.md" not in section
    assert "2 more item(s)" in section
    assert "knowledge/legacy.mdc" in section
    assert "knowledge/large.md (40.0 KB)" in section
    assert _format_hygiene_section({key: [] for key in report}, locale="ja") == ""

    keys = (
        "memory_hygiene.header",
        "memory_hygiene.merged_leftovers",
        "memory_hygiene.inherited_dirs",
        "memory_hygiene.mdc_files",
        "memory_hygiene.oversized_knowledge",
        "memory_hygiene.noncanonical_archive_dirs",
        "memory_hygiene.remaining",
    )
    for locale in ("ja", "en", "ko"):
        for key in keys:
            assert key in MEMORY_STRINGS
            assert locale in MEMORY_STRINGS[key]
            assert MEMORY_STRINGS[key][locale]


def test_weekly_templates_place_hygiene_section_before_step_one() -> None:
    repository_root = Path(__file__).parents[4]
    for locale in ("ja", "en", "ko"):
        template = (
            repository_root / "templates" / locale / "prompts" / "memory" / "weekly_consolidation_instruction.md"
        ).read_text(encoding="utf-8")
        assert template.count("{hygiene_section}") == 1
        assert template.index("{hygiene_section}") < template.index("### Step 1:")
