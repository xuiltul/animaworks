"""Unit tests for tiered system prompt — context_window-based prompt scaling."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.prompt.builder import (
    TIER_FULL,
    TIER_LIGHT,
    TIER_MINIMAL,
    TIER_STANDARD,
    BuildResult,
    build_system_prompt,
    resolve_prompt_tier,
)


# ── resolve_prompt_tier ──────────────────────────────────


class TestResolvePromptTier:
    """Boundary-value tests for tier resolution."""

    @pytest.mark.parametrize(
        "ctx_window, expected",
        [
            (200_000, TIER_FULL),
            (128_000, TIER_FULL),
            (128_001, TIER_FULL),
            (127_999, TIER_STANDARD),
            (64_000, TIER_STANDARD),
            (32_000, TIER_STANDARD),
            (31_999, TIER_LIGHT),
            (16_000, TIER_LIGHT),
            (15_999, TIER_MINIMAL),
            (8_000, TIER_MINIMAL),
            (4_000, TIER_MINIMAL),
            (0, TIER_MINIMAL),
        ],
    )
    def test_tier_boundaries(self, ctx_window: int, expected: str):
        assert resolve_prompt_tier(ctx_window) == expected


# ── Helper: mock MemoryManager ───────────────────────────


def _make_mock_memory(
    tmp_path: Path,
    data_dir: Path,
    *,
    identity: str = "I am TestAnima",
    injection: str = "行動指針テスト",
    heartbeat: str = "- check 1\n- check 2",
    permissions: str = "",
    specialty: str = "専門テスト",
    bootstrap: str = "Bootstrap初回指示",
    vision: str = "Company Vision",
) -> MagicMock:
    """Create a mock MemoryManager with typical return values."""
    anima_dir = tmp_path / "animas" / "test-anima"
    anima_dir.mkdir(parents=True)
    (anima_dir / "identity.md").write_text(identity, encoding="utf-8")
    (anima_dir / "heartbeat.md").write_text(heartbeat, encoding="utf-8")

    memory = MagicMock()
    memory.anima_dir = anima_dir
    memory.read_identity.return_value = identity
    memory.read_injection.return_value = injection
    memory.read_permissions.return_value = permissions
    memory.read_specialty_prompt.return_value = specialty
    memory.read_bootstrap.return_value = bootstrap
    memory.read_company_vision.return_value = vision
    memory.read_current_state.return_value = "status: idle"
    memory.read_pending.return_value = ""
    memory.list_knowledge_files.return_value = []
    memory.list_episode_files.return_value = []
    memory.list_procedure_files.return_value = []
    memory.list_skill_metas.return_value = []
    memory.list_common_skill_metas.return_value = []
    memory.list_procedure_metas.return_value = []
    memory.list_shared_users.return_value = []
    memory.collect_distilled_knowledge.return_value = []
    memory.collect_distilled_knowledge_separated.return_value = ([], [])
    return memory


# ── build_system_prompt tier tests ───────────────────────


class TestBuildSystemPromptTiers:
    """Verify prompt composition per tier."""

    def _build(
        self,
        tmp_path: Path,
        data_dir: Path,
        context_window: int,
        **memory_kwargs,
    ) -> BuildResult:
        memory = _make_mock_memory(tmp_path, data_dir, **memory_kwargs)

        def _load_prompt_section(name: str, *args: object, **kwargs: object) -> str:
            if name == "builder/light_tier_org":
                anima_name = kwargs.get("anima_name", "test-anima")
                return f"あなたは{anima_name}です。他のアニマとはsend_messageで通信できます。\nメッセージはsend_messageツールで送信してください。"
            return "section"

        with patch("core.prompt.builder.load_prompt", side_effect=_load_prompt_section):
            return build_system_prompt(
                memory,
                execution_mode="a",
                context_window=context_window,
            )

    # ── T1: identity / injection / heartbeat preserved ────

    def test_t1_contains_identity_injection(self, tmp_path, data_dir):
        result = self._build(tmp_path, data_dir, 200_000)
        assert "I am TestAnima" in result
        assert "行動指針テスト" in result

    def test_t1_contains_bootstrap_and_vision(self, tmp_path, data_dir):
        result = self._build(tmp_path, data_dir, 200_000)
        assert "Bootstrap初回指示" in result
        assert "Company Vision" in result

    # ── T4: identity / injection preserved, optional components stripped ──

    def test_t4_contains_identity_injection(self, tmp_path, data_dir):
        result = self._build(tmp_path, data_dir, 8_000)
        assert "I am TestAnima" in result
        assert "行動指針テスト" in result

    def test_t4_omits_bootstrap(self, tmp_path, data_dir):
        result = self._build(tmp_path, data_dir, 8_000)
        assert "Bootstrap初回指示" not in result

    def test_t4_omits_vision(self, tmp_path, data_dir):
        result = self._build(tmp_path, data_dir, 8_000)
        assert "Company Vision" not in result

    def test_t4_omits_specialty(self, tmp_path, data_dir):
        result = self._build(tmp_path, data_dir, 8_000)
        assert "専門テスト" not in result

    def test_t4_omits_permissions(self, tmp_path, data_dir):
        result = self._build(
            tmp_path, data_dir, 8_000,
            permissions="## 権限\n- web_search: OK",
        )
        assert "web_search" not in result

    # ── T3: Light tier ────────────────────────────────

    def test_t3_contains_identity(self, tmp_path, data_dir):
        result = self._build(tmp_path, data_dir, 16_000)
        assert "I am TestAnima" in result

    def test_t3_omits_bootstrap(self, tmp_path, data_dir):
        result = self._build(tmp_path, data_dir, 16_000)
        assert "Bootstrap初回指示" not in result

    def test_t3_keeps_permissions(self, tmp_path, data_dir):
        result = self._build(
            tmp_path, data_dir, 16_000,
            permissions="## 権限\n- web_search: OK",
        )
        assert "web_search" in result

    def test_t3_has_simplified_org(self, tmp_path, data_dir):
        result = self._build(tmp_path, data_dir, 16_000)
        assert "send_message" in result

    # ── T2: Standard tier ─────────────────────────────

    def test_t2_contains_all_group2_components(self, tmp_path, data_dir):
        result = self._build(tmp_path, data_dir, 64_000)
        assert "I am TestAnima" in result
        assert "行動指針テスト" in result
        assert "Bootstrap初回指示" in result
        assert "Company Vision" in result
        assert "専門テスト" in result


class TestTierPromptSizes:
    """Verify that smaller tiers produce smaller prompts."""

    def _build_size(
        self, tmp_path: Path, data_dir: Path, context_window: int,
        suffix: str = "",
    ) -> int:
        sub = tmp_path / f"sz{suffix}"
        sub.mkdir(exist_ok=True)
        memory = _make_mock_memory(sub, data_dir)
        with patch("core.prompt.builder.load_prompt", return_value="section"):
            result = build_system_prompt(
                memory,
                execution_mode="a",
                context_window=context_window,
            )
        return len(result.system_prompt)

    def test_t4_smaller_than_t1(self, tmp_path, data_dir):
        size_t1 = self._build_size(tmp_path, data_dir, 200_000, "t1")
        size_t4 = self._build_size(tmp_path, data_dir, 8_000, "t4")
        assert size_t4 < size_t1

    def test_t3_smaller_than_t2(self, tmp_path, data_dir):
        size_t2 = self._build_size(tmp_path, data_dir, 64_000, "t2")
        size_t3 = self._build_size(tmp_path, data_dir, 16_000, "t3")
        assert size_t3 < size_t2

    def test_monotonic_decrease(self, tmp_path, data_dir):
        sizes = [
            self._build_size(tmp_path, data_dir, cw, f"m{i}")
            for i, cw in enumerate([200_000, 64_000, 16_000, 8_000])
        ]
        for i in range(len(sizes) - 1):
            assert sizes[i] >= sizes[i + 1], (
                f"T{i+1} ({sizes[i]}) should be >= T{i+2} ({sizes[i+1]})"
            )


class TestDefaultContextWindowBackwardCompat:
    """Default context_window=200000 must produce identical output to omission."""

    def test_default_equals_explicit_200k(self, tmp_path, data_dir):
        memory = _make_mock_memory(tmp_path, data_dir)
        with patch("core.prompt.builder.load_prompt", return_value="section"):
            result_default = build_system_prompt(memory, execution_mode="a")
            result_explicit = build_system_prompt(
                memory, execution_mode="a", context_window=200_000,
            )
        assert result_default.system_prompt == result_explicit.system_prompt


class TestDistilledKnowledgeTierBudget:
    """Verify DK injection is controlled by tier."""

    def test_t1_injects_dk(self, tmp_path, data_dir):
        memory = _make_mock_memory(tmp_path, data_dir)
        memory.collect_distilled_knowledge_separated.return_value = (
            [],
            [
                {
                    "name": "test_knowledge",
                    "content": "knowledge content here",
                    "confidence": 0.8,
                    "path": "/tmp/knowledge/test_knowledge.md",
                },
            ],
        )
        with patch("core.prompt.builder.load_prompt", return_value="section"):
            result = build_system_prompt(
                memory, execution_mode="a", context_window=200_000,
            )
        assert "Distilled Knowledge" in result
        assert "knowledge content here" in result

    def test_t4_omits_dk(self, tmp_path, data_dir):
        memory = _make_mock_memory(tmp_path, data_dir)
        memory.collect_distilled_knowledge_separated.return_value = (
            [],
            [
                {
                    "name": "test_knowledge",
                    "content": "knowledge content here",
                    "confidence": 0.8,
                    "path": "/tmp/knowledge/test_knowledge.md",
                },
            ],
        )
        with patch("core.prompt.builder.load_prompt", return_value="section"):
            result = build_system_prompt(
                memory, execution_mode="a", context_window=8_000,
            )
        assert "knowledge content here" not in result

    def test_t3_omits_dk(self, tmp_path, data_dir):
        memory = _make_mock_memory(tmp_path, data_dir)
        memory.collect_distilled_knowledge_separated.return_value = (
            [],
            [
                {
                    "name": "test_knowledge",
                    "content": "knowledge content here",
                    "confidence": 0.8,
                    "path": "/tmp/knowledge/test_knowledge.md",
                },
            ],
        )
        with patch("core.prompt.builder.load_prompt", return_value="section"):
            result = build_system_prompt(
                memory, execution_mode="a", context_window=16_000,
            )
        assert "knowledge content here" not in result
