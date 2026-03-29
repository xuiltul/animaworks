from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for core.execution._sanitize."""


from core.execution._sanitize import (
    MAX_ORIGIN_CHAIN_LENGTH,
    ORIGIN_ANIMA,
    ORIGIN_CONSOLIDATION,
    ORIGIN_EXTERNAL_PLATFORM,
    ORIGIN_EXTERNAL_WEB,
    ORIGIN_HUMAN,
    ORIGIN_SYSTEM,
    ORIGIN_TRUST_MAP,
    ORIGIN_UNKNOWN,
    TOOL_TRUST_LEVELS,
    resolve_trust,
    wrap_priming,
    wrap_tool_result,
)


# ── wrap_tool_result ──────────────────────────────────────────


def test_wrap_tool_result_trusted() -> None:
    """Known trusted tool wraps with trust="trusted"."""
    result = wrap_tool_result("search_memory", "found 3 matches")
    assert 'trust="trusted"' in result
    assert "found 3 matches" in result


def test_wrap_tool_result_medium() -> None:
    """Known medium tool wraps with trust="medium"."""
    result = wrap_tool_result("read_file", "file contents here")
    assert 'trust="medium"' in result
    assert "file contents here" in result


def test_wrap_tool_result_untrusted() -> None:
    """Known untrusted tool wraps with trust="untrusted"."""
    result = wrap_tool_result("web_search", "search results from web")
    assert 'trust="untrusted"' in result
    assert "search results from web" in result


def test_wrap_tool_result_unknown_tool() -> None:
    """Unknown tool name defaults to trust="untrusted"."""
    result = wrap_tool_result("unknown_tool_xyz", "some output")
    assert 'trust="untrusted"' in result
    assert 'tool="unknown_tool_xyz"' in result


def test_wrap_tool_result_empty_string() -> None:
    """Empty string returns empty string unchanged."""
    result = wrap_tool_result("search_memory", "")
    assert result == ""


def test_wrap_tool_result_none_returns_none() -> None:
    """None input returns None (falsy check)."""
    result = wrap_tool_result("search_memory", None)  # type: ignore[arg-type]
    assert result is None


def test_wrap_tool_result_contains_content() -> None:
    """Result string is inside the tags."""
    content = "internal data from memory"
    result = wrap_tool_result("read_memory_file", content)
    assert content in result
    assert "<tool_result" in result
    assert "</tool_result>" in result


def test_wrap_tool_result_multiline_content() -> None:
    """Multiline content is wrapped correctly."""
    content = "line1\nline2\nline3"
    result = wrap_tool_result("write_file", content)
    assert "line1\nline2\nline3" in result
    assert 'tool="write_file"' in result
    assert 'trust="medium"' in result


def test_wrap_tool_result_format() -> None:
    """Verify exact format with newlines."""
    result = wrap_tool_result("search_memory", "x")
    expected = '<tool_result tool="search_memory" trust="trusted">\nx\n</tool_result>'
    assert result == expected


# ── wrap_priming ───────────────────────────────────────────────


def test_wrap_priming_default_trust() -> None:
    """Default trust is "mixed"."""
    result = wrap_priming("sender_profile", "user info")
    assert 'trust="mixed"' in result
    assert 'source="sender_profile"' in result


def test_wrap_priming_custom_trust() -> None:
    """Custom trust level is used."""
    result = wrap_priming("recent_activity", "activity log", trust="untrusted")
    assert 'trust="untrusted"' in result
    assert 'source="recent_activity"' in result


def test_wrap_priming_empty_string() -> None:
    """Empty string returns empty string."""
    result = wrap_priming("sender_profile", "")
    assert result == ""


def test_wrap_priming_none_returns_none() -> None:
    """None returns None."""
    result = wrap_priming("sender_profile", None)  # type: ignore[arg-type]
    assert result is None


def test_wrap_priming_format() -> None:
    """Verify exact format."""
    result = wrap_priming("related_knowledge", "knowledge", trust="medium")
    expected = '<priming source="related_knowledge" trust="medium">\nknowledge\n</priming>'
    assert result == expected


# ── TOOL_TRUST_LEVELS ──────────────────────────────────────────


def test_trust_levels_cover_all_high_risk_tools() -> None:
    """All high risk tools (read_channel, web_search, slack_messages, etc.) are "untrusted"."""
    high_risk = [
        "read_channel",
        "read_dm_history",
        "web_search",
        "x_search",
        "x_user_tweets",
        "slack_messages",
        "slack_search",
        "slack_unreplied",
        "slack_channels",
        "chatwork_messages",
        "chatwork_search",
        "chatwork_unreplied",
        "chatwork_mentions",
        "chatwork_rooms",
        "gmail_unread",
        "gmail_read_body",
        "local_llm",
    ]
    for tool in high_risk:
        assert TOOL_TRUST_LEVELS.get(tool) == "untrusted", f"{tool} should be untrusted"


def test_trust_levels_medium_tools() -> None:
    """File tools are "medium"."""
    medium_tools = [
        "read_file",
        "search_code",
        "write_file",
        "edit_file",
        "execute_command",
    ]
    for tool in medium_tools:
        assert TOOL_TRUST_LEVELS.get(tool) == "medium", f"{tool} should be medium"


def test_trust_levels_trusted_tools() -> None:
    """Memory tools are "trusted"."""
    trusted_tools = [
        "search_memory",
        "read_memory_file",
        "write_memory_file",
        "archive_memory_file",
        "create_skill",
        "list_directory",
        "backlog_task",
        "update_task",
        "list_tasks",
        "post_channel",
        "send_message",
    ]
    for tool in trusted_tools:
        assert TOOL_TRUST_LEVELS.get(tool) == "trusted", f"{tool} should be trusted"


# ── Origin constants ──────────────────────────────────────────


class TestOriginConstants:
    """Verify origin category constants and trust map."""

    def test_all_seven_origins_defined(self) -> None:
        assert ORIGIN_SYSTEM == "system"
        assert ORIGIN_HUMAN == "human"
        assert ORIGIN_ANIMA == "anima"
        assert ORIGIN_EXTERNAL_PLATFORM == "external_platform"
        assert ORIGIN_EXTERNAL_WEB == "external_web"
        assert ORIGIN_CONSOLIDATION == "consolidation"
        assert ORIGIN_UNKNOWN == "unknown"

    def test_trust_map_covers_all_origins(self) -> None:
        all_origins = [
            ORIGIN_SYSTEM, ORIGIN_HUMAN, ORIGIN_ANIMA,
            ORIGIN_EXTERNAL_PLATFORM, ORIGIN_EXTERNAL_WEB,
            ORIGIN_CONSOLIDATION, ORIGIN_UNKNOWN,
        ]
        for o in all_origins:
            assert o in ORIGIN_TRUST_MAP, f"{o} missing from ORIGIN_TRUST_MAP"

    def test_trust_map_values_are_valid(self) -> None:
        valid = {"trusted", "medium", "untrusted"}
        for origin, trust in ORIGIN_TRUST_MAP.items():
            assert trust in valid, f"{origin} has invalid trust {trust}"

    def test_max_chain_length_is_10(self) -> None:
        assert MAX_ORIGIN_CHAIN_LENGTH == 10


# ── resolve_trust ─────────────────────────────────────────────


class TestResolveTrust:
    """Test provenance-aware trust resolution."""

    def test_system_is_trusted(self) -> None:
        assert resolve_trust("system") == "trusted"

    def test_human_is_medium(self) -> None:
        assert resolve_trust("human") == "medium"

    def test_anima_is_trusted(self) -> None:
        assert resolve_trust("anima") == "trusted"

    def test_external_platform_is_untrusted(self) -> None:
        assert resolve_trust("external_platform") == "untrusted"

    def test_external_web_is_untrusted(self) -> None:
        assert resolve_trust("external_web") == "untrusted"

    def test_consolidation_is_medium(self) -> None:
        assert resolve_trust("consolidation") == "medium"

    def test_unknown_is_untrusted(self) -> None:
        assert resolve_trust("unknown") == "untrusted"

    def test_none_none_returns_untrusted(self) -> None:
        """Backward compat: no origin info at all → untrusted."""
        assert resolve_trust(None, None) == "untrusted"

    def test_none_origin_with_no_chain_returns_untrusted(self) -> None:
        assert resolve_trust(None) == "untrusted"

    def test_unrecognised_origin_string(self) -> None:
        assert resolve_trust("some_future_category") == "untrusted"

    # ── Chain tests ───────────────────────────────────────────

    def test_chain_minimum_anima_via_external_platform(self) -> None:
        """anima origin relayed through external_platform → untrusted."""
        assert resolve_trust("anima", ["external_platform"]) == "untrusted"

    def test_chain_minimum_anima_via_human(self) -> None:
        """anima origin relayed through human → medium (weakest link)."""
        assert resolve_trust("anima", ["human"]) == "medium"

    def test_chain_all_trusted(self) -> None:
        assert resolve_trust("system", ["anima", "system"]) == "trusted"

    def test_chain_mixed_picks_lowest(self) -> None:
        result = resolve_trust("system", ["anima", "human", "system"])
        assert result == "medium"

    def test_chain_with_unknown_in_middle(self) -> None:
        result = resolve_trust("system", ["anima", "unknown", "system"])
        assert result == "untrusted"

    def test_chain_truncated_at_max_length(self) -> None:
        long_chain = ["system"] * 15
        result = resolve_trust("system", long_chain)
        assert result == "trusted"

    def test_chain_with_untrusted_beyond_max_is_ignored(self) -> None:
        chain = ["system"] * 10 + ["external_web"] * 5
        result = resolve_trust("system", chain)
        assert result == "trusted"

    def test_chain_none_origin_falls_to_unknown(self) -> None:
        """origin=None with chain present → origin treated as 'unknown'."""
        result = resolve_trust(None, ["system"])
        assert result == "untrusted"

    def test_empty_chain_treated_as_no_chain(self) -> None:
        assert resolve_trust("system", []) == "trusted"


# ── wrap_tool_result with origin ──────────────────────────────


class TestWrapToolResultWithOrigin:
    """Test origin/origin_chain extensions on wrap_tool_result."""

    def test_legacy_call_unchanged(self) -> None:
        """Calling without origin preserves exact legacy format."""
        result = wrap_tool_result("search_memory", "x")
        expected = '<tool_result tool="search_memory" trust="trusted">\nx\n</tool_result>'
        assert result == expected

    def test_origin_overrides_tool_trust(self) -> None:
        result = wrap_tool_result("search_memory", "data", origin="external_web")
        assert 'trust="untrusted"' in result
        assert 'origin="external_web"' in result

    def test_origin_and_chain_attributes_present(self) -> None:
        result = wrap_tool_result(
            "web_search", "res",
            origin="external_web", origin_chain=["anima"],
        )
        assert 'origin="external_web"' in result
        assert 'origin_chain="anima"' in result
        assert 'trust="untrusted"' in result

    def test_chain_comma_separated(self) -> None:
        result = wrap_tool_result(
            "send_message", "ok",
            origin="anima", origin_chain=["human", "system"],
        )
        assert 'origin_chain="human,system"' in result

    def test_chain_truncated_in_attr(self) -> None:
        long_chain = [f"o{i}" for i in range(15)]
        result = wrap_tool_result("x", "y", origin="system", origin_chain=long_chain)
        chain_attr_count = result.split('origin_chain="')[1].split('"')[0].count(",") + 1
        assert chain_attr_count == MAX_ORIGIN_CHAIN_LENGTH

    def test_empty_result_with_origin_returns_empty(self) -> None:
        assert wrap_tool_result("search_memory", "", origin="system") == ""

    def test_none_result_with_origin_returns_none(self) -> None:
        assert wrap_tool_result("search_memory", None, origin="system") is None  # type: ignore[arg-type]

    def test_origin_system_gives_trusted(self) -> None:
        result = wrap_tool_result("web_search", "data", origin="system")
        assert 'trust="trusted"' in result


# ── wrap_priming with origin ──────────────────────────────────


class TestWrapPrimingWithOrigin:
    """Test origin/origin_chain extensions on wrap_priming."""

    def test_legacy_call_unchanged(self) -> None:
        result = wrap_priming("sender_profile", "info")
        expected = '<priming source="sender_profile" trust="mixed">\ninfo\n</priming>'
        assert result == expected

    def test_legacy_custom_trust_unchanged(self) -> None:
        result = wrap_priming("recent_activity", "log", trust="untrusted")
        expected = '<priming source="recent_activity" trust="untrusted">\nlog\n</priming>'
        assert result == expected

    def test_origin_overrides_explicit_trust(self) -> None:
        result = wrap_priming(
            "recent_activity", "content",
            trust="trusted", origin="external_platform",
        )
        assert 'trust="untrusted"' in result
        assert 'origin="external_platform"' in result

    def test_origin_and_chain_in_priming(self) -> None:
        result = wrap_priming(
            "recent_activity", "content",
            origin="external_platform", origin_chain=["anima"],
        )
        assert 'trust="untrusted"' in result
        assert 'origin="external_platform"' in result
        assert 'origin_chain="anima"' in result

    def test_chain_minimum_trust_in_priming(self) -> None:
        result = wrap_priming(
            "related_knowledge", "knowledge",
            origin="anima", origin_chain=["human"],
        )
        assert 'trust="medium"' in result

    def test_empty_content_with_origin(self) -> None:
        assert wrap_priming("x", "", origin="system") == ""

    def test_none_content_with_origin(self) -> None:
        assert wrap_priming("x", None, origin="system") is None  # type: ignore[arg-type]

    def test_chain_truncated_in_priming_attr(self) -> None:
        long_chain = [f"o{i}" for i in range(15)]
        result = wrap_priming("src", "c", origin="system", origin_chain=long_chain)
        chain_attr_count = result.split('origin_chain="')[1].split('"')[0].count(",") + 1
        assert chain_attr_count == MAX_ORIGIN_CHAIN_LENGTH
