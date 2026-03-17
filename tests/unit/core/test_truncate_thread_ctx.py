from __future__ import annotations

"""Unit tests for _truncate_with_thread_ctx in core._anima_inbox."""

from core._anima_inbox import (
    _MSG_BODY_BUDGET,
    _RE_THREAD_CTX,
    _THREAD_CTX_BUDGET,
    _truncate_with_thread_ctx,
)


class TestTruncateWithThreadCtx:
    """Tests for _truncate_with_thread_ctx helper."""

    def test_plain_text_no_markers(self):
        """Without thread context markers, truncate to body_budget."""
        text = "Hello world"
        assert _truncate_with_thread_ctx(text) == text

    def test_plain_text_truncated_at_body_budget(self):
        """Long text without markers is truncated to body_budget."""
        text = "x" * 3000
        result = _truncate_with_thread_ctx(text)
        assert len(result) == _MSG_BODY_BUDGET

    def test_short_context_preserves_both(self):
        """Short thread context + message: both preserved in full."""
        ctx = "[Thread context — reply]\n  <@U1>: Hi\n[/Thread context]\n"
        body = "User message here"
        content = ctx + body
        result = _truncate_with_thread_ctx(content)
        assert ctx in result
        assert body in result

    def test_long_context_truncated_message_preserved(self):
        """Long thread context is truncated; user message is preserved."""
        ctx_inner = "A" * 500
        ctx = f"[Thread context — reply]\n  {ctx_inner}\n[/Thread context]\n"
        body = "User message that must survive"
        content = ctx + body
        result = _truncate_with_thread_ctx(content)
        assert "User message that must survive" in result
        assert len(result.split("...\n")[0]) <= _THREAD_CTX_BUDGET

    def test_exact_reproduction_of_reported_bug(self):
        """Reproduce the exact bug: sakura's report + taka's message."""
        thread_ctx = (
            "[Thread context — this message is a reply in a Slack thread]\n"
            "  <@unknown>: *【報告】Polymarket $40→$1800回復タスク 期限到達・未達で終了*\n"
            "taka様\n\nPolymarket回復タスクについて最終報告します。\n\n"
            "*結果*\n• *目標*: $1,800（24h期限: 23:55 JST）\n"
            "• *現在残高*: $937.52（52.1%回復）\n"
            "• *内訳*: USDC.e $761.84 + オープンポジション $175.68\n"
            "• *判定*: 未達確定\n\n"
            "*本日の経緯*\n"
            "• 23:10 taka指示「損失しているロジックを止めてモックトレードに切り替え」\n"
            "• 23:12 crypto15m_live.py（LIVE注文プロセス）を停止・DRY_RUN=True化\n"
            "• 23:22 touka: GH LIVE残存ポジション $43.12 相当を償還\n"
            "• 本日の取引フロー: 取引支出 $175.58 / 償還収入 $4.92 / 純損失 $170.66\n\n"
            "*現状*\n• *全LIVE注文プロセス*: 停止済み（DRY RUN移行完了）\n"
            "• *SportsArb*: DRY RUN稼働中（PID 1283083、正常）\n"
            "• *GH LIVE残高 $761.84*: 待機中（LIVE指示解除まで取引なし）\n\n"
            "*次のアクション（takaさんの判断次第）*\n"
            "1. GH LIVE残高 $761.84 の引き続きの扱い（待機継続 or 出金）\n"
            "2. crypto15m戦略の今後方針（DRY RUN継続 or 廃止）\n"
            "3. Polymarket USDC.e追加入金の可否（既存の判断待ち項目）\n\n"
            "sakura\n[/Thread context]\n"
        )
        user_msg = (
            "モックトレード口座1000ドルを仮想入金して、フォワードテストして金額を"
            "トラッキングしていって。モックトレードのロジックは出来ているんでしょ？\n\n"
            "モックトレードサービスを作って、そこに注文出して仮想的にトレードする"
            "ようにしてほしい。"
        )
        content = thread_ctx + user_msg

        old_result = content[:800]
        assert "モックトレードサービスを作って" not in old_result

        new_result = _truncate_with_thread_ctx(content)
        assert "モックトレード口座1000ドルを仮想入金して" in new_result
        assert "モックトレードサービスを作って" in new_result
        assert "ようにしてほしい。" in new_result

    def test_custom_body_budget(self):
        """body_budget parameter controls user message truncation."""
        ctx = "[Thread context — reply]\n  Hi\n[/Thread context]\n"
        body = "A" * 500
        result = _truncate_with_thread_ctx(ctx + body, body_budget=100)
        assert result.endswith("A" * 100)

    def test_custom_ctx_budget(self):
        """ctx_budget parameter controls thread context truncation."""
        ctx_inner = "B" * 500
        ctx = f"[Thread context — reply]\n  {ctx_inner}\n[/Thread context]\n"
        body = "User msg"
        result = _truncate_with_thread_ctx(ctx + body, ctx_budget=50)
        assert "User msg" in result
        assert "...\n" in result

    def test_empty_content(self):
        """Empty string returns empty string."""
        assert _truncate_with_thread_ctx("") == ""

    def test_context_only_no_body(self):
        """Thread context with no user message after it."""
        ctx = "[Thread context — reply]\n  Hello\n[/Thread context]\n"
        result = _truncate_with_thread_ctx(ctx)
        assert "[Thread context" in result

    def test_regex_pattern_matches_variants(self):
        """Regex matches different Thread context header variants."""
        for header in [
            "[Thread context — this message is a reply in a Slack thread]",
            "[Thread context — this is a reply to a call_human notification]",
            "[Thread context]",
        ]:
            text = f"{header}\n  content\n[/Thread context]\nuser msg"
            m = _RE_THREAD_CTX.match(text)
            assert m is not None, f"Failed to match: {header}"


class TestFetchThreadContextForReply:
    """Tests for _fetch_thread_context_for_reply summarization."""

    def test_returns_concise_summary(self, monkeypatch):
        from unittest.mock import MagicMock

        from core.notification.reply_routing import _fetch_thread_context_for_reply

        mock_client = MagicMock()
        mock_client.thread_replies.return_value = [
            {"user": "U_BOT", "text": "Long report\nWith many lines\nAnd details", "ts": "1.0"},
            {"user": "U_HUMAN", "text": "Reply 1", "ts": "2.0"},
            {"user": "U_HUMAN", "text": "Reply 2", "ts": "3.0"},
        ]
        monkeypatch.setattr(
            "core.tools.slack.SlackClient",
            lambda token: mock_client,
        )
        result = _fetch_thread_context_for_reply("xoxb-token", "C123", "1.0")
        assert "Long report With many lines And details" in result
        assert "(2 replies in thread)" in result
        assert "Reply 1" not in result
        assert "Reply 2" not in result


class TestRouteThreadReplyFallback:
    """Test that route_thread_reply fallback uses truncated notification_text."""

    def test_stored_notification_text_truncated(self, monkeypatch, tmp_path):
        from unittest.mock import MagicMock

        from core.notification import reply_routing

        long_notification = "X" * 500
        monkeypatch.setattr(
            reply_routing,
            "lookup_notification_mapping",
            lambda ts: {
                "anima": "sakura",
                "channel": "C123",
                "notification_text": long_notification,
            },
        )
        monkeypatch.setattr(
            reply_routing,
            "_fetch_thread_context_for_reply",
            lambda *a, **kw: "",
        )

        received_content = []

        fake_messenger = MagicMock()
        fake_messenger.return_value.receive_external = lambda **kw: received_content.append(kw["content"])
        monkeypatch.setattr("core.messenger.Messenger", fake_messenger)

        event = {
            "thread_ts": "123.456",
            "text": "User reply",
            "ts": "789.0",
            "user": "U_HUMAN",
            "channel": "C123",
        }
        result = reply_routing.route_thread_reply(event, tmp_path, slack_token="")
        assert result is True
        assert len(received_content) == 1
        content = received_content[0]
        assert "User reply" in content
        assert "X" * 150 in content
        assert "X" * 200 not in content
