from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from core.tooling.policy.tool_content import apply_prompt_descriptions, get_default_guide

ROOT = Path(__file__).resolve().parents[3] / "templates"


def _desc(locale: str, name: str) -> str:
    """Return the file-backed description for *name* in *locale*."""
    p = ROOT / locale / "prompts" / "tool_descriptions" / f"{name}.md"
    return p.read_text(encoding="utf-8")


def test_description_is_loaded_from_markdown_each_time(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "ja" / "prompts" / "tool_descriptions"
    prompt_dir.mkdir(parents=True)
    description = prompt_dir / "example.md"
    description.write_text("first", encoding="utf-8")
    tools = [{"name": "example", "description": "fallback", "parameters": {}}]

    with patch("core.paths.TEMPLATES_DIR", tmp_path):
        assert apply_prompt_descriptions(tools)[0]["description"] == "first"
        description.write_text("second", encoding="utf-8")
        assert apply_prompt_descriptions(tools)[0]["description"] == "second"


def test_missing_description_keeps_canonical_schema(tmp_path: Path) -> None:
    with patch("core.paths.TEMPLATES_DIR", tmp_path):
        tools = [{"name": "example", "description": "fallback", "parameters": {}}]
        assert apply_prompt_descriptions(tools) == tools


def test_unknown_guide_has_empty_last_resort_fallback() -> None:
    assert get_default_guide("unknown", locale="en") == ""


# ── Shortened description constraint preservation ────────────────────────


def test_send_message_preserves_constraints() -> None:
    """send_message keeps the per-recipient run guard and intent constraints."""
    ja = _desc("ja", "send_message")
    en = _desc("en", "send_message")
    assert "同一 run" in ja and "宛先数の上限はない" in ja
    assert "report" in ja and "question" in ja
    assert "same recipient once" in en and "no recipient-count cap" in en
    assert "'report'" in en and "'question'" in en


def test_send_message_keeps_related_tool_guidance() -> None:
    """send_message points to the Board CLI / delegate_task for other cases."""
    ja = _desc("ja", "send_message")
    en = _desc("en", "send_message")
    assert "delegate_task" in ja and "animaworks-tool board post" in ja
    # English template generation is handled separately from the ja source.
    assert "delegate_task" in en and "animaworks-tool board post" in en


def test_delegate_task_preserves_self_contained_constraint() -> None:
    """delegate_task keeps the self-contained-instruction and dedup guidance."""
    ja = _desc("ja", "delegate_task")
    en = _desc("en", "delegate_task")
    assert "自己完結" in ja and "list_tasks" in ja
    assert "self-contained" in en and "list_tasks" in en


def test_search_memory_preserves_scope() -> None:
    """search_memory keeps the memory-category scope keywords."""
    ja = _desc("ja", "search_memory")
    en = _desc("en", "search_memory")
    assert "knowledge" in ja and "activity_log" in ja
    assert "knowledge" in en and "activity_log" in en


def test_submit_tasks_preserves_resume_constraint() -> None:
    """submit_tasks keeps resume=true and delegate_task guidance."""
    ja = _desc("ja", "submit_tasks")
    en = _desc("en", "submit_tasks")
    assert "resume=true" in ja and "delegate_task" in ja
    assert "resume=true" in en and "delegate_task" in en
