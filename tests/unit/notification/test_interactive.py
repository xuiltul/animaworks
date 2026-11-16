# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`core.notification.interactive`."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def data_dir(tmp_path):
    """Provide a temporary data directory."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    return tmp_path


@pytest.fixture
def _patch_dirs(data_dir):
    """Patch get_data_dir, get_shared_dir, and auth for stable approval tokens."""
    mock_auth = MagicMock()
    mock_auth.secret_key = "unit-test-secret-key"

    with (
        patch("core.notification.interactive.get_data_dir", return_value=data_dir),
        patch("core.notification.interactive.get_shared_dir", return_value=data_dir / "shared"),
        patch("core.notification.interactive.load_auth", return_value=mock_auth),
    ):
        import core.notification.interactive as mod

        mod._router = None
        yield
        mod._router = None


class TestInteractionRouter:
    """Tests for InteractionRouter."""

    @pytest.mark.asyncio
    async def test_create_returns_request(self, _patch_dirs):
        from core.notification.interactive import get_interaction_router

        router = get_interaction_router()
        req = await router.create("test_anima", "approval", ["approve", "reject"])
        assert req.callback_id
        assert req.anima_name == "test_anima"
        assert req.category == "approval"
        assert req.options == ["approve", "reject"]
        assert req.approval_token

    @pytest.mark.asyncio
    async def test_lookup_returns_created_request(self, _patch_dirs):
        from core.notification.interactive import get_interaction_router

        router = get_interaction_router()
        req = await router.create("test_anima", "approval", ["approve", "reject"])
        found = await router.lookup(req.callback_id)
        assert found is not None
        assert found.callback_id == req.callback_id

    @pytest.mark.asyncio
    async def test_lookup_returns_none_for_unknown(self, _patch_dirs):
        from core.notification.interactive import get_interaction_router

        router = get_interaction_router()
        found = await router.lookup("nonexistent")
        assert found is None

    @pytest.mark.asyncio
    async def test_resolve_injects_into_inbox(self, _patch_dirs):
        from core.notification.interactive import get_interaction_router

        router = get_interaction_router()
        req = await router.create("test_anima", "approval", ["approve", "reject"])

        with patch("core.messaging.messenger.Messenger") as mock_messenger:
            mock_instance = MagicMock()
            mock_messenger.return_value = mock_instance

            result = await router.resolve(req.callback_id, "approve", "tester", "slack")
            assert result is not None
            assert result.decision == "approve"
            assert result.actor == "tester"
            assert result.source == "slack"
            mock_instance.receive_external.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_inbox_includes_blocker_close_instruction(self, _patch_dirs):
        """Approval resolve must instruct the anima to close blocker entries."""
        from core.i18n import t
        from core.notification.interactive import get_interaction_router

        router = get_interaction_router()
        req = await router.create("test_anima", "approval", ["approve", "reject"])

        with patch("core.messaging.messenger.Messenger") as mock_messenger:
            mock_instance = MagicMock()
            mock_messenger.return_value = mock_instance

            await router.resolve(
                req.callback_id,
                "B source-onlyで停止",
                "xuiltul",
                "slack",
            )
            mock_instance.receive_external.assert_called_once()
            content = mock_instance.receive_external.call_args.kwargs.get("content")
            if content is None:
                content = mock_instance.receive_external.call_args.args[0]
            instruction = t("interactive.blocker_close_instruction")
            assert instruction in content
            assert "current_state.md" in content
            assert "call_human" in content
            assert mock_instance.receive_external.call_args.kwargs.get("intent") == "question"

    @pytest.mark.asyncio
    async def test_resolve_non_approval_skips_blocker_instruction(self, _patch_dirs):
        """Non-approval categories should not get the blocker-close instruction."""
        from core.i18n import t
        from core.notification.interactive import get_interaction_router

        router = get_interaction_router()
        req = await router.create("test_anima", "design_gate", ["go", "stop"])

        with patch("core.messaging.messenger.Messenger") as mock_messenger:
            mock_instance = MagicMock()
            mock_messenger.return_value = mock_instance

            await router.resolve(req.callback_id, "go", "tester", "web")
            content = mock_instance.receive_external.call_args.kwargs.get("content")
            if content is None:
                content = mock_instance.receive_external.call_args.args[0]
            assert t("interactive.blocker_close_instruction") not in content

    @pytest.mark.asyncio
    async def test_resolve_returns_none_for_already_resolved(self, _patch_dirs):
        from core.notification.interactive import get_interaction_router

        router = get_interaction_router()
        req = await router.create("test_anima", "approval", ["approve", "reject"])

        with patch("core.messaging.messenger.Messenger"):
            result1 = await router.resolve(req.callback_id, "approve", "tester", "slack")
            result2 = await router.resolve(req.callback_id, "reject", "tester2", "slack")
            assert result1 is not None
            assert result2 is None

    @pytest.mark.asyncio
    async def test_resolve_logs_human_reply_activity(self, _patch_dirs, data_dir, tmp_path):
        """Successful resolve writes human_reply to the anima activity_log."""
        from core.activity.logger import ActivityLogger
        from core.notification.interactive import get_interaction_router

        anima_dir = tmp_path / "animas" / "test_anima"
        (anima_dir / "activity_log").mkdir(parents=True)

        router = get_interaction_router()
        req = await router.create("test_anima", "approval", ["approve", "reject"])

        with (
            patch("core.messaging.messenger.Messenger"),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        ):
            result = await router.resolve(
                req.callback_id,
                "approve",
                "tester",
                "web",
                comment="LGTM",
            )
            assert result is not None

        logger = ActivityLogger(anima_dir)
        view = logger.get_conversation_view(thread_id="default", limit=20)
        messages = [m for s in view["sessions"] for m in s["messages"]]
        replies = [m for m in messages if m.get("type") == "human_reply"]
        assert len(replies) == 1
        assert replies[0]["content"] == "approve: LGTM"
        assert replies[0]["from_person"] == "tester"
        assert replies[0]["via"] == "web"

    @pytest.mark.asyncio
    async def test_verify_approval_token(self, _patch_dirs):
        from core.notification.interactive import get_interaction_router

        router = get_interaction_router()
        req = await router.create("test_anima", "approval", ["approve"])

        assert router.verify_approval_token(req.callback_id, req.approval_token)
        assert not router.verify_approval_token(req.callback_id, "wrong_token")

    @pytest.mark.asyncio
    async def test_update_message_ts(self, _patch_dirs):
        from core.notification.interactive import get_interaction_router

        router = get_interaction_router()
        req = await router.create("test_anima", "approval", ["approve"])
        await router.update_message_ts(req.callback_id, "slack", "1234.5678")

        found = await router.lookup(req.callback_id)
        assert found is not None
        assert found.message_ts.get("slack") == "1234.5678"


class TestListResolvedForAnima:
    """Tests for list_resolved_for_anima and resilient wrapper."""

    @pytest.mark.asyncio
    async def test_list_resolved_filters_by_anima_window_and_status(self, _patch_dirs):
        from datetime import timedelta

        from core.notification.interactive import get_interaction_router

        router = get_interaction_router()
        req_a = await router.create("natsume", "approval", ["approve", "reject"])
        req_b = await router.create("other", "approval", ["approve", "reject"])
        req_pending = await router.create("natsume", "approval", ["approve"])

        with patch("core.messaging.messenger.Messenger"):
            await router.resolve(req_a.callback_id, "approve", "xuiltul", "slack")
            await router.resolve(req_b.callback_id, "reject", "bob", "web")

        # Pending must not appear
        listed = await router.list_resolved_for_anima("natsume", within_hours=48)
        ids = {r.callback_id for r, _ in listed}
        assert req_a.callback_id in ids
        assert req_b.callback_id not in ids
        assert req_pending.callback_id not in ids
        assert len(listed) == 1
        assert listed[0][1].decision == "approve"
        assert listed[0][1].actor == "xuiltul"

        # Outside window: rewrite resolved_at to the past
        data = router._read_all_entries()
        entry = data["entries"][req_a.callback_id]
        old = (datetime.now(UTC) - timedelta(hours=72)).isoformat()
        entry["result"]["resolved_at"] = old
        router._write_all_entries(data)

        listed_old = await router.list_resolved_for_anima("natsume", within_hours=48)
        assert listed_old == []

        listed_wide = await router.list_resolved_for_anima("natsume", within_hours=100)
        assert len(listed_wide) == 1

    def test_list_resolved_resilient_returns_empty_on_error(self, _patch_dirs):
        from core.notification import interactive as mod

        with patch.object(mod, "get_interaction_router", side_effect=RuntimeError("boom")):
            result = mod.list_resolved_for_anima_resilient("natsume", within_hours=48)
        assert result == []

    def test_list_resolved_resilient_happy_path(self, _patch_dirs):
        from core.notification import interactive as mod

        async def _setup():
            router = mod.get_interaction_router()
            req = await router.create("natsume", "approval", ["approve"])
            with patch("core.messaging.messenger.Messenger"):
                await router.resolve(req.callback_id, "approve", "a", "slack")
            return req.callback_id

        import asyncio

        cid = asyncio.run(_setup())
        result = mod.list_resolved_for_anima_resilient("natsume", within_hours=48)
        assert len(result) == 1
        assert result[0][0].callback_id == cid


class TestBuildTextFallback:
    """Tests for build_text_fallback."""

    def test_basic_fallback(self):
        from core.notification.interactive import InteractionRequest, build_text_fallback

        req = InteractionRequest(
            callback_id="test123",
            anima_name="test",
            category="approval",
            options=["approve", "reject", "comment"],
            allowed_users={},
            metadata={},
            created_at=datetime.now(tz=UTC),
            approval_token="tok",
            message_ts={},
        )
        result = build_text_fallback(req)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result
        assert "approve" in result.lower() or "Approve" in result

    def test_fallback_with_web_url(self):
        from core.notification.interactive import InteractionRequest, build_text_fallback

        req = InteractionRequest(
            callback_id="test123",
            anima_name="test",
            category="approval",
            options=["approve", "reject"],
            allowed_users={},
            metadata={},
            created_at=datetime.now(tz=UTC),
            approval_token="tok",
            message_ts={},
        )
        result = build_text_fallback(req, web_base_url="https://example.com")
        assert "https://example.com/api/approve/test123" in result
        assert "tok" in result


class TestInteractionTimelineWiring:
    @pytest.mark.asyncio
    async def test_resolve_records_approval_human_once(self, _patch_dirs):
        from types import SimpleNamespace

        from core.notification.interactive import get_interaction_router

        router = get_interaction_router()
        req = await router.create("test_anima", "approval", ["approve", "reject"])

        with (
            patch("core.messaging.messenger.Messenger") as mock_messenger,
            patch("core.hub.timeline_human.resolve_person", return_value="tester"),
            patch("core.hub.timeline_human.find_notified", return_value="tl_1234567890abcdef") as find_notified,
            patch("core.hub.timeline_human.record_human") as record_human,
        ):
            mock_messenger.return_value.receive_external.return_value = SimpleNamespace(id="msg-1")
            result = await router.resolve(req.callback_id, "approve", "tester", "slack")

        assert result is not None
        find_notified.assert_called_once_with(callback_id=req.callback_id)
        record_human.assert_called_once_with(
            person="tester",
            anima="test_anima",
            text="approve",
            via="approval",
            about="tl_1234567890abcdef",
            ref="msg-1",
        )

    @pytest.mark.asyncio
    async def test_resolve_records_approval_with_comment(self, _patch_dirs):
        from types import SimpleNamespace

        from core.notification.interactive import get_interaction_router

        router = get_interaction_router()
        req = await router.create("test_anima", "approval", ["approve", "reject"])

        with (
            patch("core.messaging.messenger.Messenger") as mock_messenger,
            patch("core.hub.timeline_human.resolve_person", return_value="tester"),
            patch("core.hub.timeline_human.find_notified", return_value=None),
            patch("core.hub.timeline_human.record_human") as record_human,
        ):
            mock_messenger.return_value.receive_external.return_value = SimpleNamespace(id="msg-2")
            await router.resolve(req.callback_id, "approve", "tester", "web", comment="この内容で進めて")

        assert record_human.call_args.kwargs["text"] == "approve: この内容で進めて"
