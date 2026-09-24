"""Tests for shortterm (session handoff) budget protection.

Verifies that shortterm content is NOT trimmed by the framework target,
only by the hard ceiling. Also checks that dropping shortterm items
emits a WARNING-level log.
"""
from __future__ import annotations

import logging

from core.prompt.assembler import PromptBudget, SectionEntry, _allocate_sections


def _by_id(allocated: list[SectionEntry], section_id: str) -> SectionEntry | None:
    return next((section for section in allocated if section.id == section_id), None)


class TestShorttermBudgetProtection:
    """shortterm must survive framework target trim but yield to ceiling."""

    def test_shortterm_survives_target_trim(self) -> None:
        """shortterm items are NOT dropped when total exceeds target but
        stays within ceiling."""
        identity_content = "identity " * 10  # ~10 tokens
        shortterm_content = "handoff " * 500  # ~500 tokens, well above target=100

        sections = [
            SectionEntry("identity", 1, "rigid", identity_content),
            SectionEntry(
                "shortterm",
                3,
                "elastic",
                shortterm_content,
                budget_group="shortterm",
            ),
        ]

        allocated = _allocate_sections(
            sections,
            PromptBudget(target=100, ceiling=5000),
        )

        shortterm = _by_id(allocated, "shortterm")
        assert shortterm is not None, "shortterm must NOT be dropped by target trim"
        assert shortterm.content == shortterm_content, (
            "shortterm content must be fully preserved when within ceiling"
        )

    def test_shortterm_dropped_at_ceiling(self) -> None:
        """shortterm items ARE dropped when total exceeds the hard ceiling."""
        identity_content = "identity " * 10
        shortterm_content = "handoff " * 500

        sections = [
            SectionEntry("identity", 1, "rigid", identity_content),
            SectionEntry(
                "shortterm",
                3,
                "elastic",
                shortterm_content,
                budget_group="shortterm",
            ),
        ]

        # ceiling=100 is too small to hold both identity + shortterm
        allocated = _allocate_sections(
            sections,
            PromptBudget(target=100, ceiling=100),
        )

        shortterm = _by_id(allocated, "shortterm")
        # Should be dropped or severely trimmed
        assert shortterm is None or len(shortterm.content) < len(shortterm_content), (
            "shortterm must be trimmed/dropped when exceeding ceiling"
        )

    def test_framework_elastic_still_trimmed_by_target(self) -> None:
        """Regular framework elastic items are still trimmed by target
        (regression guard: shortterm protection must not break framework trim)."""
        identity_content = "identity"
        framework_content = "framework " * 200
        shortterm_content = "handoff " * 50

        sections = [
            SectionEntry("identity", 1, "rigid", identity_content),
            SectionEntry(
                "optional_framework",
                3,
                "elastic",
                framework_content,
                budget_group="framework",
            ),
            SectionEntry(
                "shortterm",
                3,
                "elastic",
                shortterm_content,
                budget_group="shortterm",
            ),
        ]

        allocated = _allocate_sections(
            sections,
            PromptBudget(target=50, ceiling=5000),
        )

        framework = _by_id(allocated, "optional_framework")
        shortterm = _by_id(allocated, "shortterm")
        # framework elastic should be trimmed by target
        assert framework is None or len(framework.content) < len(framework_content), (
            "framework elastic must still be trimmed by target"
        )
        # shortterm should survive
        assert shortterm is not None, (
            "shortterm must survive even when framework elastic is trimmed"
        )


class TestShorttermDropWarning:
    """Dropping shortterm items must log at WARNING level."""

    def test_warning_log_on_shortterm_drop(self, caplog) -> None:  # type: ignore[no-untyped-def]
        """When shortterm items are dropped by ceiling, the allocation log
        must be at WARNING level."""
        identity_content = "identity " * 10
        shortterm_content = "handoff " * 500

        sections = [
            SectionEntry("identity", 1, "rigid", identity_content),
            SectionEntry(
                "shortterm",
                3,
                "elastic",
                shortterm_content,
                budget_group="shortterm",
            ),
        ]

        with caplog.at_level(logging.DEBUG, logger="animaworks.prompt_builder"):
            _allocate_sections(
                sections,
                PromptBudget(target=100, ceiling=100),
            )

        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "shortterm" in r.getMessage()
        ]
        assert warning_records, (
            "Dropping shortterm must produce a WARNING log"
        )

    def test_info_log_when_non_shortterm_dropped(self, caplog) -> None:  # type: ignore[no-untyped-def]
        """When only non-shortterm items are dropped, log stays at INFO."""
        identity_content = "identity " * 10
        framework_content = "framework " * 500

        sections = [
            SectionEntry("identity", 1, "rigid", identity_content),
            SectionEntry(
                "optional",
                3,
                "elastic",
                framework_content,
                budget_group="framework",
            ),
        ]

        with caplog.at_level(logging.DEBUG, logger="animaworks.prompt_builder"):
            _allocate_sections(
                sections,
                PromptBudget(target=100, ceiling=5000),
            )

        allocation_records = [
            r for r in caplog.records
            if "Prompt allocation:" in r.getMessage()
        ]
        if allocation_records:
            # Should be INFO, not WARNING
            for rec in allocation_records:
                assert rec.levelno == logging.INFO, (
                    "Dropping non-shortterm items should log at INFO, not WARNING"
                )

    def test_shortterm_trim_from_tail_default(self) -> None:
        """shortterm sections should trim from tail (preserve the head which
        contains the original request)."""
        head = "## original request\n\n" + "original " * 50
        tail = "## later context\n\n" + "later " * 50
        content = head + "\n\n" + tail

        sections = [
            SectionEntry("identity", 1, "rigid", "identity " * 10),
            SectionEntry(
                "shortterm",
                3,
                "elastic",
                content,
                budget_group="shortterm",
            ),
        ]

        # ceiling is tight enough to force trim but not complete removal
        allocated = _allocate_sections(
            sections,
            PromptBudget(target=50, ceiling=120),
        )

        shortterm = _by_id(allocated, "shortterm")
        if shortterm is not None:
            # head (original request) should be preserved, tail dropped
            assert "original" in shortterm.content, (
                "shortterm must preserve the head (original request)"
            )


class TestShorttermRigidSection:
    """Fix 2: builder emits shortterm as priority-2 rigid; assembler protects it."""

    def test_builder_emits_shortterm_as_rigid_priority_2(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """_build_group3 must add the shortterm section as rigid priority-2."""
        from unittest.mock import MagicMock

        from core.prompt.builder import _build_group3

        anima_dir = tmp_path / "animas" / "sakura"
        (anima_dir / "state").mkdir(parents=True)
        memory = MagicMock()
        memory.anima_dir = anima_dir
        memory.read_current_state.return_value = ""
        memory.read_resolutions.return_value = []

        entries = _build_group3(
            anima_dir,
            memory,
            1.0,
            "",
            "",
            "s",
            False,
            True,
            False,
            {"group3_header": "# 6. Current Situation"},
            {"truncated": "(earlier portion omitted)"},
            shortterm_text="## session handoff\ncontinue the deploy discussion",
        )

        shortterm = next((e for e in entries if e.id == "shortterm"), None)
        assert shortterm is not None, "shortterm section must be emitted"
        assert shortterm.kind == "rigid", "shortterm must be rigid (Fix 2)"
        assert shortterm.priority == 2, "shortterm must be priority 2"
        assert shortterm.budget_group == "shortterm"

    def test_rigid_shortterm_survives_target_and_elastic_ceiling_trim(self) -> None:
        """A rigid shortterm section is fully preserved while elastic content
        of the same priority is trimmed to fit the target."""
        identity_content = "identity " * 10
        framework_content = "framework " * 300
        shortterm_content = "handoff " * 200

        sections = [
            SectionEntry("identity", 1, "rigid", identity_content),
            SectionEntry(
                "optional_framework",
                3,
                "elastic",
                framework_content,
                budget_group="framework",
            ),
            SectionEntry(
                "shortterm",
                2,
                "rigid",
                shortterm_content,
                budget_group="shortterm",
            ),
        ]

        allocated = _allocate_sections(
            sections,
            PromptBudget(target=100, ceiling=5000),
        )

        shortterm = _by_id(allocated, "shortterm")
        framework = _by_id(allocated, "optional_framework")
        assert shortterm is not None, "rigid shortterm must never be target-trimmed"
        assert shortterm.content == shortterm_content, (
            "rigid shortterm content must be preserved verbatim within ceiling"
        )
        assert framework is None or len(framework.content) < len(framework_content), (
            "framework elastic must still yield to the target"
        )

    def test_rigid_shortterm_evicted_only_at_hard_ceiling(self) -> None:
        """A rigid shortterm section is evicted when the hard ceiling cannot
        hold it (all-or-nothing rigid eviction)."""
        identity_content = "identity " * 10
        shortterm_content = "handoff " * 500

        sections = [
            SectionEntry("identity", 1, "rigid", identity_content),
            SectionEntry(
                "shortterm",
                2,
                "rigid",
                shortterm_content,
                budget_group="shortterm",
            ),
        ]

        allocated = _allocate_sections(
            sections,
            PromptBudget(target=100, ceiling=100),
        )

        assert _by_id(allocated, "shortterm") is None, (
            "rigid shortterm must yield to the hard ceiling"
        )
        assert _by_id(allocated, "identity") is not None, (
            "priority-1 rigid content must survive"
        )

    def test_warning_log_on_rigid_shortterm_eviction(self, caplog) -> None:  # type: ignore[no-untyped-def]
        """Evicting a rigid shortterm section must log at WARNING level."""
        sections = [
            SectionEntry("identity", 1, "rigid", "identity " * 10),
            SectionEntry(
                "shortterm",
                2,
                "rigid",
                "handoff " * 500,
                budget_group="shortterm",
            ),
        ]

        with caplog.at_level(logging.DEBUG, logger="animaworks.prompt_builder"):
            _allocate_sections(
                sections,
                PromptBudget(target=100, ceiling=100),
            )

        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "shortterm" in r.getMessage()
        ]
        assert warning_records, (
            "Evicting a rigid shortterm section must produce a WARNING log"
        )
