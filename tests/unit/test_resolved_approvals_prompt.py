# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for resolved-approval reminder injection into system prompts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.notification.interactive import InteractionRequest, InteractionResult
from core.prompt.builder import _build_resolved_approvals_section


def _make_pair(
    *,
    anima_name: str = "natsume",
    category: str = "approval",
    callback_id: str = "cb-test-1",
    decision: str = "B source-onlyで停止",
    actor: str = "xuiltul",
    source: str = "slack",
    resolved_at: datetime | None = None,
) -> tuple[InteractionRequest, InteractionResult]:
    ra = resolved_at or datetime(2026, 7, 27, 7, 59, tzinfo=UTC)  # 16:59 JST
    req = InteractionRequest(
        callback_id=callback_id,
        anima_name=anima_name,
        category=category,
        options=["A", "B"],
        allowed_users={},
        metadata={},
        created_at=datetime(2026, 7, 27, 6, 0, tzinfo=UTC),
        approval_token="tok",
        message_ts={},
    )
    result = InteractionResult(
        callback_id=callback_id,
        decision=decision,
        actor=actor,
        source=source,
        comment="",
        resolved_at=ra,
    )
    return req, result


class TestBuildResolvedApprovalsSection:
    """_build_resolved_approvals_section behaviour."""

    def test_injects_when_resolved_approvals_exist(self):
        pair = _make_pair()
        _ss = {
            "resolved_approvals_header": "## 解決済み承認（直近{hours}h）",
            "resolved_approvals_intro": "以下の承認依頼は既に人間が回答済み。",
            "resolved_approvals_item": (
                "- callback_id: {callback_id} — 「{decision}」 by {actor} ({source}, {resolved_at})"
            ),
        }
        mock_cfg = MagicMock()
        mock_cfg.heartbeat.resolved_interaction_reminder_hours = 48

        with (
            patch("core.config.load_config", return_value=mock_cfg),
            patch(
                "core.notification.interactive.list_resolved_for_anima_resilient",
                return_value=[pair],
            ) as mock_list,
        ):
            block = _build_resolved_approvals_section("natsume", _ss)
            mock_list.assert_called_once_with("natsume", within_hours=48)

        assert "解決済み承認" in block
        assert "直近48h" in block
        assert "cb-test-1" in block
        assert "B source-onlyで停止" in block
        assert "xuiltul" in block
        assert "slack" in block
        assert "回答済み" in block

    def test_empty_when_no_resolved(self):
        mock_cfg = MagicMock()
        mock_cfg.heartbeat.resolved_interaction_reminder_hours = 48
        _ss = {
            "resolved_approvals_header": "## Resolved Approvals (last {hours}h)",
            "resolved_approvals_intro": "intro",
            "resolved_approvals_item": "- {callback_id}",
        }
        with (
            patch("core.config.load_config", return_value=mock_cfg),
            patch(
                "core.notification.interactive.list_resolved_for_anima_resilient",
                return_value=[],
            ),
        ):
            block = _build_resolved_approvals_section("natsume", _ss)
        assert block == ""

    def test_disabled_when_hours_zero(self):
        mock_cfg = MagicMock()
        mock_cfg.heartbeat.resolved_interaction_reminder_hours = 0
        _ss = {
            "resolved_approvals_header": "## Resolved Approvals (last {hours}h)",
            "resolved_approvals_intro": "intro",
            "resolved_approvals_item": "- {callback_id}",
        }
        with (
            patch("core.config.load_config", return_value=mock_cfg),
            patch(
                "core.notification.interactive.list_resolved_for_anima_resilient",
            ) as mock_list,
        ):
            block = _build_resolved_approvals_section("natsume", _ss)
        assert block == ""
        mock_list.assert_not_called()

    def test_skips_non_approval_categories(self):
        pair = _make_pair(category="design_gate")
        mock_cfg = MagicMock()
        mock_cfg.heartbeat.resolved_interaction_reminder_hours = 48
        _ss = {
            "resolved_approvals_header": "## Resolved Approvals (last {hours}h)",
            "resolved_approvals_intro": "intro",
            "resolved_approvals_item": "- {callback_id}",
        }
        with (
            patch("core.config.load_config", return_value=mock_cfg),
            patch(
                "core.notification.interactive.list_resolved_for_anima_resilient",
                return_value=[pair],
            ),
        ):
            block = _build_resolved_approvals_section("natsume", _ss)
        assert block == ""


class TestResolvedApprovalsInGroup3:
    """Integration-ish: _build_group3 includes resolved_approvals section."""

    def test_group3_adds_section_when_resolved(self, tmp_path: Path):
        from core.prompt.builder import _build_group3

        anima_dir = tmp_path / "natsume"
        anima_dir.mkdir()
        (anima_dir / "state").mkdir()
        (anima_dir / "state" / "current_state.md").write_text(
            "status: working\n- P0 承認待ち\n",
            encoding="utf-8",
        )

        memory = MagicMock()
        memory.read_current_state.return_value = "status: working\n- P0 承認待ち"
        memory.read_resolutions.return_value = []
        memory.read_model_config.return_value = MagicMock()

        pair = _make_pair()
        mock_cfg = MagicMock()
        mock_cfg.heartbeat.resolved_interaction_reminder_hours = 48
        _ss = {
            "group3_header": "# 3. Current Situation",
            "current_state_header": "## Current State",
            "resolved_approvals_header": "## 解決済み承認（直近{hours}h）",
            "resolved_approvals_intro": "以下の承認依頼は既に人間が回答済み。call_humanしない。",
            "resolved_approvals_item": (
                "- callback_id: {callback_id} — 「{decision}」 by {actor} ({source}, {resolved_at})"
            ),
        }
        _fs = {"truncated": "(truncated)"}

        with (
            patch("core.config.load_config", return_value=mock_cfg),
            patch(
                "core.notification.interactive.list_resolved_for_anima_resilient",
                return_value=[pair],
            ),
            patch(
                "core.taskboard.attention_resolver.resolver_for_anima_dir",
                side_effect=Exception("skip gate"),
            ),
            patch(
                "core.prompt.builder.load_prompt",
                side_effect=lambda name, **kw: (
                    f"## Task\n{kw.get('state', '')}" if name == "builder/task_in_progress" else ""
                ),
            ),
        ):
            sections = _build_group3(
                anima_dir,
                memory,
                scale=1.0,
                priming_section="",
                pending_human_notifications="",
                execution_mode="s",
                is_heartbeat=True,
                is_chat=False,
                is_task=False,
                _ss=_ss,
                _fs=_fs,
            )

        by_id = {s.id: s for s in sections}
        assert "resolved_approvals" in by_id
        content = by_id["resolved_approvals"].content
        assert "解決済み承認" in content
        assert "cb-test-1" in content
        assert by_id["resolved_approvals"].priority == 1
        assert by_id["resolved_approvals"].kind == "rigid"

    def test_group3_omits_section_when_empty(self, tmp_path: Path):
        from core.prompt.builder import _build_group3

        anima_dir = tmp_path / "natsume"
        anima_dir.mkdir()

        memory = MagicMock()
        memory.read_current_state.return_value = "status: idle"
        memory.read_resolutions.return_value = []

        mock_cfg = MagicMock()
        mock_cfg.heartbeat.resolved_interaction_reminder_hours = 48
        _ss = {
            "group3_header": "# 3. Current Situation",
            "current_state_header": "## Current State",
            "resolved_approvals_header": "## Resolved Approvals (last {hours}h)",
            "resolved_approvals_intro": "intro",
            "resolved_approvals_item": "- {callback_id}",
        }
        _fs: dict[str, str] = {}

        with (
            patch("core.config.load_config", return_value=mock_cfg),
            patch(
                "core.notification.interactive.list_resolved_for_anima_resilient",
                return_value=[],
            ),
        ):
            sections = _build_group3(
                anima_dir,
                memory,
                scale=1.0,
                priming_section="",
                pending_human_notifications="",
                execution_mode="s",
                is_heartbeat=True,
                is_chat=False,
                is_task=False,
                _ss=_ss,
                _fs=_fs,
            )

        assert all(s.id != "resolved_approvals" for s in sections)
