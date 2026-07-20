from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.prompt.builder import _build_group1, _build_group6
from core.prompt.messaging import _load_a_reflection


def test_group1_uses_file_backed_environment_and_behavior_rules(tmp_path: Path) -> None:
    memory = MagicMock()
    memory.read_identity.return_value = ""
    memory.read_injection.return_value = ""
    with (
        patch("core.prompt.builder.load_prompt") as load,
        patch("core.prompt.builder.load_prompt_text", return_value="file rules"),
    ):
        load.side_effect = lambda name, **kwargs: {
            "environment": "file environment",
            "tool_data_interpretation": "tool data",
        }[name]
        sections = _build_group1(tmp_path / "anima", tmp_path, memory, False, {})

    contents = {section.id: section.content for section in sections}
    assert contents["environment"] == "file environment"
    assert contents["behavior_rules"] == "file rules"


def test_group6_loads_reflection_and_emotion_from_markdown() -> None:
    with (
        patch("core.prompt.builder._build_emotion_instruction", return_value="file emotion"),
        patch("core.prompt.builder._load_a_reflection", return_value="file reflection"),
    ):
        sections = _build_group6("a", True, False, False, {})

    contents = {section.id: section.content for section in sections}
    assert contents["emotion_instruction"] == "file emotion"
    assert contents["a_reflection"] == "file reflection"


@pytest.mark.parametrize(
    ("locale", "heading"),
    [("ja", "情報収集の原則"), ("en", "Information Gathering Principles"), ("ko", "정보 수집 원칙")],
)
def test_reflection_template_loads_for_every_locale(locale: str, heading: str) -> None:
    with patch("core.paths._get_locale", return_value=locale):
        content = _load_a_reflection()
    assert heading in content
