"""E2E tests for activity-log outbound context in priming."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from core.time_utils import now_jst


def _make_anima_dir(tmp_path: Path, name: str) -> Path:
    """Create a minimal anima directory with an activity log."""
    anima_dir = tmp_path / "animas" / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "IDENTITY.md").write_text(f"# {name}\n\nテスト用Anima。\n", encoding="utf-8")
    (anima_dir / "status.json").write_text("{}", encoding="utf-8")
    (anima_dir / "activity_log").mkdir(exist_ok=True)
    return anima_dir


@pytest.mark.e2e
class TestPrimingOutboundSectionE2E:
    """Write activity-log events and verify the outbound priming section."""

    @pytest.mark.asyncio
    async def test_priming_outbound_section_e2e(self, tmp_path: Path) -> None:
        from core.memory.priming import PrimingEngine

        anima_dir = _make_anima_dir(tmp_path, "alice")
        (anima_dir / "knowledge").mkdir(exist_ok=True)

        today = now_jst().strftime("%Y-%m-%d")
        log_file = anima_dir / "activity_log" / f"{today}.jsonl"
        ts1 = (now_jst() - timedelta(minutes=45)).isoformat()
        ts2 = (now_jst() - timedelta(minutes=30)).isoformat()
        ts3 = (now_jst() - timedelta(minutes=10)).isoformat()
        entries = [
            json.dumps(
                {"ts": ts1, "type": "channel_post", "content": "進捗報告です", "channel": "general"},
                ensure_ascii=False,
            ),
            json.dumps(
                {"ts": ts2, "type": "message_sent", "content": "確認お願いします", "to": "bob"},
                ensure_ascii=False,
            ),
            json.dumps(
                {"ts": ts3, "type": "message_sent", "content": "ありがとうございました", "to": "charlie"},
                ensure_ascii=False,
            ),
        ]
        log_file.write_text("\n".join(entries) + "\n", encoding="utf-8")

        result = await PrimingEngine(anima_dir)._collect_recent_outbound()

        assert "直近のアウトバウンド行動" in result
        assert "#general に投稿済み" in result
        assert "進捗報告です" in result
        assert "bob にメッセージ送信済み" in result or "charlie にメッセージ送信済み" in result
        assert "bob" in result
        assert "charlie" in result

    @pytest.mark.asyncio
    async def test_priming_outbound_section_old_entries_excluded(self, tmp_path: Path) -> None:
        from core.memory.priming import PrimingEngine

        anima_dir = _make_anima_dir(tmp_path, "alice")
        (anima_dir / "knowledge").mkdir(exist_ok=True)
        today = now_jst().strftime("%Y-%m-%d")
        log_file = anima_dir / "activity_log" / f"{today}.jsonl"
        old_ts = (now_jst() - timedelta(hours=3)).isoformat()
        log_file.write_text(
            json.dumps(
                {"ts": old_ts, "type": "channel_post", "content": "Ancient post", "channel": "general"},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        result = await PrimingEngine(anima_dir)._collect_recent_outbound()

        assert result == ""

    @pytest.mark.asyncio
    async def test_priming_outbound_section_empty_log(self, tmp_path: Path) -> None:
        from core.memory.priming import PrimingEngine

        anima_dir = _make_anima_dir(tmp_path, "alice")
        (anima_dir / "knowledge").mkdir(exist_ok=True)

        result = await PrimingEngine(anima_dir)._collect_recent_outbound()

        assert result == ""
