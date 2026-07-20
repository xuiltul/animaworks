from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from core.prompt.tool_content import apply_prompt_descriptions, get_default_guide


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
