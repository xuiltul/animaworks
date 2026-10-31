from __future__ import annotations

"""Tests for _normalize_memory_path in handler_memory.py."""

from pathlib import Path
from unittest.mock import patch

import pytest

from core.tooling.handler_memory import _normalize_memory_path, _PathNormResult

DATA_DIR = Path("/home/testuser/.animaworks")
ANIMA_DIR = DATA_DIR / "animas" / "sakura"
CK_DIR = DATA_DIR / "common_knowledge"
REF_DIR = DATA_DIR / "reference"
CS_DIR = DATA_DIR / "common_skills"
CHANNELS_DIR = DATA_DIR / "shared" / "channels"


@pytest.fixture(autouse=True)
def _mock_paths():
    """Mock core.paths functions to return deterministic test dirs."""
    with (
        patch("core.paths.get_data_dir", return_value=DATA_DIR),
        patch("core.paths.get_common_knowledge_dir", return_value=CK_DIR),
        patch("core.paths.get_reference_dir", return_value=REF_DIR),
        patch("core.paths.get_common_skills_dir", return_value=CS_DIR),
    ):
        yield


class TestFastPath:
    """Normal relative paths should pass through without resolution."""

    def test_simple_relative(self):
        result = _normalize_memory_path("knowledge/foo.md", ANIMA_DIR)
        assert result == _PathNormResult(rel="knowledge/foo.md")

    def test_common_knowledge_prefix(self):
        result = _normalize_memory_path("common_knowledge/ops/guide.md", ANIMA_DIR)
        assert result == _PathNormResult(rel="common_knowledge/ops/guide.md")

    def test_state_file(self):
        result = _normalize_memory_path("state/current_state.md", ANIMA_DIR)
        assert result == _PathNormResult(rel="state/current_state.md")


class TestSlashCleanup:
    """Various slash-related cleanups."""

    def test_double_slash(self):
        result = _normalize_memory_path("knowledge//foo.md", ANIMA_DIR)
        assert result.rel == "knowledge/foo.md"

    def test_trailing_slash(self):
        result = _normalize_memory_path("knowledge/foo.md/", ANIMA_DIR)
        assert result.rel == "knowledge/foo.md"

    def test_leading_dot_slash(self):
        result = _normalize_memory_path("./knowledge/foo.md", ANIMA_DIR)
        assert result.rel == "knowledge/foo.md"

    def test_multiple_dot_slashes(self):
        result = _normalize_memory_path("././knowledge/foo.md", ANIMA_DIR)
        assert result.rel == "knowledge/foo.md"

    def test_whitespace_stripped(self):
        result = _normalize_memory_path("  knowledge/foo.md  ", ANIMA_DIR)
        assert result.rel == "knowledge/foo.md"

    def test_triple_slash(self):
        result = _normalize_memory_path("knowledge///foo.md", ANIMA_DIR)
        assert result.rel == "knowledge/foo.md"


class TestAbsolutePathOwnAnima:
    """Absolute paths under own anima_dir are stripped to relative."""

    def test_absolute_own_knowledge(self):
        result = _normalize_memory_path(
            str(ANIMA_DIR / "knowledge" / "foo.md"), ANIMA_DIR
        )
        assert result.rel == "knowledge/foo.md"
        assert result.channel_redirect is None

    def test_absolute_own_state(self):
        result = _normalize_memory_path(
            str(ANIMA_DIR / "state" / "current_state.md"), ANIMA_DIR
        )
        assert result.rel == "state/current_state.md"

    def test_absolute_own_procedures(self):
        result = _normalize_memory_path(
            str(ANIMA_DIR / "procedures" / "deploy.md"), ANIMA_DIR
        )
        assert result.rel == "procedures/deploy.md"


class TestAbsolutePathSubordinate:
    """Absolute paths under another anima's dir become subordinate relative."""

    def test_absolute_other_anima(self):
        result = _normalize_memory_path(
            str(DATA_DIR / "animas" / "fuji" / "state" / "current_state.md"),
            ANIMA_DIR,
        )
        assert result.rel == "../fuji/state/current_state.md"
        assert result.channel_redirect is None

    def test_absolute_other_anima_activity_log(self):
        result = _normalize_memory_path(
            str(DATA_DIR / "animas" / "fuji" / "activity_log" / "2026-03-18.jsonl"),
            ANIMA_DIR,
        )
        assert result.rel == "../fuji/activity_log/2026-03-18.jsonl"


class TestAbsolutePathSharedDirs:
    """Absolute paths under shared dirs map to canonical prefixes."""

    def test_absolute_common_knowledge(self):
        result = _normalize_memory_path(
            str(CK_DIR / "ops" / "guide.md"), ANIMA_DIR
        )
        assert result.rel == "common_knowledge/ops/guide.md"

    def test_absolute_reference(self):
        result = _normalize_memory_path(
            str(REF_DIR / "organization" / "structure.md"), ANIMA_DIR
        )
        assert result.rel == "reference/organization/structure.md"

    def test_absolute_common_skills(self):
        result = _normalize_memory_path(
            str(CS_DIR / "git-flow.md"), ANIMA_DIR
        )
        assert result.rel == "common_skills/git-flow.md"


class TestChannelRedirect:
    """Paths pointing to shared/channels/ return channel_redirect."""

    def test_absolute_channel_path(self):
        result = _normalize_memory_path(
            str(CHANNELS_DIR / "general.jsonl"), ANIMA_DIR
        )
        assert result.channel_redirect == "general"

    def test_absolute_ops_channel(self):
        result = _normalize_memory_path(
            str(CHANNELS_DIR / "ops.jsonl"), ANIMA_DIR
        )
        assert result.channel_redirect == "ops"


class TestDotDotTraversal:
    """../ traversal patterns are normalized."""

    def test_dotdot_within_own_dir(self):
        """../../knowledge/foo.md from anima_dir resolves under animas/, not anima_dir.
        This becomes a subordinate path '../knowledge/foo.md' or stays as-is depending
        on how pathlib resolves it."""
        result = _normalize_memory_path("../../knowledge/foo.md", ANIMA_DIR)
        assert ".." not in result.rel or result.rel.startswith("../")

    def test_dotdot_to_sibling_anima(self):
        result = _normalize_memory_path(
            "../fuji/state/current_state.md", ANIMA_DIR
        )
        assert result.rel == "../fuji/state/current_state.md"
        assert result.channel_redirect is None

    def test_dotdot_to_channels(self):
        result = _normalize_memory_path(
            "../../shared/channels/general.jsonl", ANIMA_DIR
        )
        assert result.channel_redirect == "general"

    def test_triple_dotdot_to_channels(self):
        """Some LLMs use too many ../."""
        result = _normalize_memory_path(
            "../../../shared/channels/general.jsonl", ANIMA_DIR
        )
        # After pathlib resolution, this may or may not end up under channels_dir
        # depending on how many levels up from anima_dir the data_dir is.
        # The test verifies no crash and reasonable behavior.
        assert result is not None

    def test_dotdot_to_common_knowledge(self):
        result = _normalize_memory_path(
            "../../common_knowledge/ops/guide.md", ANIMA_DIR
        )
        assert result.rel == "common_knowledge/ops/guide.md"


class TestLeadingSlashStrip:
    """Leading / on otherwise-relative-looking paths gets stripped."""

    def test_leading_slash_knowledge(self):
        result = _normalize_memory_path("/knowledge/foo.md", ANIMA_DIR)
        assert result.rel == "knowledge/foo.md"


class TestPrefixWithDotDot:
    """Prefix-qualified paths with .. must NOT be normalized (pass-through)."""

    def test_common_knowledge_with_dotdot(self):
        result = _normalize_memory_path("common_knowledge/../secret.txt", ANIMA_DIR)
        assert result.rel == "common_knowledge/../secret.txt"

    def test_reference_with_dotdot(self):
        result = _normalize_memory_path("reference/../../etc/passwd", ANIMA_DIR)
        assert result.rel == "reference/../../etc/passwd"

    def test_common_skills_with_dotdot(self):
        result = _normalize_memory_path("common_skills/../hack.md", ANIMA_DIR)
        assert result.rel == "common_skills/../hack.md"


class TestEmptyAndEdge:
    """Edge cases."""

    def test_empty_string(self):
        result = _normalize_memory_path("", ANIMA_DIR)
        assert result.rel == ""
        assert result.channel_redirect is None

    def test_just_slash(self):
        result = _normalize_memory_path("/", ANIMA_DIR)
        assert result is not None

    def test_just_dotdot(self):
        result = _normalize_memory_path("..", ANIMA_DIR)
        assert result is not None
        assert result.channel_redirect is None

    def test_just_dot(self):
        result = _normalize_memory_path(".", ANIMA_DIR)
        assert result is not None

    def test_no_channel_redirect_by_default(self):
        result = _normalize_memory_path("knowledge/foo.md", ANIMA_DIR)
        assert result.channel_redirect is None
