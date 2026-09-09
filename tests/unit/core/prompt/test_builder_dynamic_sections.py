from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from core.prompt.builder import _collapse_superseded_notes, build_system_prompt


def _memory(tmp_path: Path, *, identity: str = "identity", state: str = "status: idle") -> MagicMock:
    anima_dir = tmp_path / "animas" / "test-anima"
    anima_dir.mkdir(parents=True)
    memory = MagicMock()
    memory.anima_dir = anima_dir
    memory.read_identity.return_value = identity
    memory.read_injection.return_value = ""
    memory.read_permissions.return_value = ""
    memory.read_specialty_prompt.return_value = ""
    memory.read_bootstrap.return_value = ""
    memory.read_company_vision.return_value = ""
    memory.read_current_state.return_value = state
    memory.read_resolutions.return_value = []
    memory.read_model_config.return_value = None
    memory.list_knowledge_files.return_value = []
    memory.list_procedure_files.return_value = []
    memory.list_shared_users.return_value = []
    return memory


def test_collapse_superseded_notes_keeps_latest_body() -> None:
    state = """preamble

## Zoom Notes API接続 進捗（9/7 16:40）
古い本文

## 別案件 進捗（9/7 16:41）
別案件の本文

## Zoom Notes API接続 続報（9/7 16:45）
最新本文

## 日時なし 続報
触らない本文
"""

    collapsed = _collapse_superseded_notes(state)

    assert "古い本文" not in collapsed
    assert "Zoom Notes API接続 進捗（9/7 16:40）（旧版。Zoom Notes API接続 続報（9/7 16:45） に統合）" in collapsed
    assert "最新本文" in collapsed
    assert "別案件の本文" in collapsed
    assert "日時なし 続報\n触らない本文" in collapsed


def test_single_word_title_is_not_collapsed() -> None:
    state = "## 単独（9/7 10:00）\nold\n\n## 単独（9/7 11:00）\nnew"

    assert _collapse_superseded_notes(state) == state


def test_shortterm_is_allocated_inside_system_budget(tmp_path: Path, data_dir: Path) -> None:
    memory = _memory(tmp_path)
    shortterm = "SHORTTERM_MARKER " + "pending context " * 3000

    with patch("core.prompt.builder.load_prompt", return_value="small section"):
        result = build_system_prompt(
            memory,
            execution_mode="a",
            trigger="chat",
            context_window=200_000,
            system_budget=2_000,
            shortterm_text=shortterm,
        )

    assert "SHORTTERM_MARKER" not in result.system_prompt


def test_dynamic_group_and_current_time_follow_static_prefix(tmp_path: Path, data_dir: Path) -> None:
    memory = _memory(tmp_path, identity="I" * 6_000)

    with patch("core.prompt.builder.load_prompt", return_value="small section"):
        result = build_system_prompt(
            memory,
            execution_mode="a",
            trigger="chat",
            context_window=200_000,
        )

    prompt = result.system_prompt
    assert prompt.index('<group_3 title="6.') > prompt.index('<group_6 title="5.')
    assert "current_time" not in prompt[:5_000]
    assert '<section name="current_time">' in prompt
