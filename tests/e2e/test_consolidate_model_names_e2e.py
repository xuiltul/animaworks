from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0


"""E2E tests for the hardcoded model name consolidation refactoring.

Verifies that all model= parameters that previously defaulted to
``"anthropic/claude-sonnet-4-20250514"`` now default to ``""`` (empty string)
and correctly fall back to ``ConsolidationConfig().llm_model`` or
``AnimaDefaults().model`` at runtime.

Modified files under test:
  - core/memory/consolidation.py
  - core/memory/contradiction.py
  - core/memory/distillation.py
  - core/memory/forgetting.py
  - core/memory/reconsolidation.py
  - core/memory/validation.py
  - core/prompt/context.py
  - core/lifecycle.py
  - core/supervisor/manager.py
"""

import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


# ── Test 1: Config-driven model resolution end-to-end ────────────────────
# NOTE: TestConfigDrivenModelResolution was removed.
# It tested daily_consolidate() and weekly_integrate() which were deleted
# in the consolidation refactor. Consolidation is now Anima-driven
# via run_consolidation() with tool-call loops.


# ── Test 2: Config override via patching ─────────────────────────────────


class TestConfigOverrideViaPatching:
    """Patching config classes to return custom models changes behaviour."""

    # NOTE: test_patched_consolidation_config_model_is_used was removed.
    # It tested daily_consolidate() which was deleted in the consolidation refactor.

    @pytest.mark.asyncio
    async def test_patched_anima_defaults_model(self):
        """Patching AnimaDefaults() to return custom model propagates to ContextTracker."""
        from core.config.models import AnimaDefaults
        from core.prompt.context import ContextTracker

        custom_model = "openai/gpt-4o-mini"
        mock_defaults_instance = AnimaDefaults(model=custom_model)

        # AnimaDefaults is imported inside __post_init__ as
        # ``from core.config.models import AnimaDefaults``
        with patch(
            "core.config.models.AnimaDefaults",
            return_value=mock_defaults_instance,
        ):
            tracker = ContextTracker()
            assert tracker.model == custom_model


# ── Test 3: ContextTracker integration ────────────────────────────────────


class TestContextTrackerIntegration:
    """ContextTracker with default model resolves from AnimaDefaults."""

    def test_default_model_context_window(self):
        """ContextTracker() with default model picks up AnimaDefaults().model."""
        from core.config.models import AnimaDefaults
        from core.prompt.context import ContextTracker, resolve_context_window

        tracker = ContextTracker()
        expected_model = AnimaDefaults().model
        assert tracker.model == expected_model
        assert tracker.context_window == resolve_context_window(expected_model)

    def test_explicit_model_context_window(self):
        """ContextTracker(model='gpt-4o') returns the gpt-4o context window."""
        from core.prompt.context import ContextTracker, resolve_context_window

        tracker = ContextTracker(model="gpt-4o")
        assert tracker.model == "gpt-4o"
        assert tracker.context_window == resolve_context_window("gpt-4o")

    def test_different_models_yield_different_windows(self):
        """Claude and GPT models have different context windows."""
        from core.prompt.context import ContextTracker

        claude_tracker = ContextTracker(model="claude-sonnet-4-20250514")
        gpt_tracker = ContextTracker(model="gpt-4o")

        # Claude has 200K, GPT-4o has 128K
        assert claude_tracker.context_window == 200_000
        assert gpt_tracker.context_window == 128_000
        assert claude_tracker.context_window != gpt_tracker.context_window


# ── Test 4: No hardcoded model names in source ──────────────────────────


class TestNoHardcodedModelDefaults:
    """Verify that refactoring removed all hardcoded model name defaults."""

    # The 9 modified files
    MODIFIED_FILES = [
        "core/lifecycle.py",
        "core/memory/consolidation.py",
        "core/memory/contradiction.py",
        "core/memory/distillation.py",
        "core/memory/forgetting.py",
        "core/memory/reconsolidation.py",
        "core/memory/validation.py",
        "core/prompt/context.py",
        "core/supervisor/manager.py",
    ]

    @staticmethod
    def _project_root() -> Path:
        """Return the project root directory."""
        # tests/e2e/test_*.py -> project root is two levels up
        return Path(__file__).resolve().parent.parent.parent

    def test_no_hardcoded_model_in_function_defaults(self):
        """No function signature should have anthropic/claude-sonnet-4-* as default."""
        root = self._project_root()

        # Pattern: model: str = "anthropic/claude-sonnet-4-*"
        hardcoded_pattern = re.compile(
            r'model:\s*str\s*=\s*["\']anthropic/claude-sonnet-4'
        )

        violations: list[str] = []
        for relpath in self.MODIFIED_FILES:
            filepath = root / relpath
            if not filepath.exists():
                continue
            content = filepath.read_text(encoding="utf-8")
            for i, line in enumerate(content.splitlines(), 1):
                if hardcoded_pattern.search(line):
                    violations.append(f"{relpath}:{i}: {line.strip()}")

        assert not violations, (
            "Found hardcoded model defaults in function signatures:\n"
            + "\n".join(violations)
        )

    def test_no_hardcoded_model_in_local_assignments(self):
        """No local variable should be assigned the old hardcoded model string."""
        root = self._project_root()

        # Pattern: model = "anthropic/claude-sonnet-4-*" as a local assignment
        # Exclude ConsolidationConfig class definition (llm_model field is expected)
        assignment_pattern = re.compile(
            r'^\s+model\s*=\s*["\']anthropic/claude-sonnet-4'
        )

        violations: list[str] = []
        for relpath in self.MODIFIED_FILES:
            filepath = root / relpath
            if not filepath.exists():
                continue
            content = filepath.read_text(encoding="utf-8")
            for i, line in enumerate(content.splitlines(), 1):
                if assignment_pattern.search(line):
                    # Allow the ConsolidationConfig class field definition
                    if "llm_model" in line:
                        continue
                    violations.append(f"{relpath}:{i}: {line.strip()}")

        assert not violations, (
            "Found hardcoded model in local variable assignments:\n"
            + "\n".join(violations)
        )

    def test_empty_string_default_pattern_present(self):
        """Verify that the empty-string default pattern (model: str = '') is used."""
        root = self._project_root()

        empty_default_pattern = re.compile(r'model:\s*str\s*=\s*""')
        files_with_empty_default: list[str] = []

        for relpath in self.MODIFIED_FILES:
            filepath = root / relpath
            if not filepath.exists():
                continue
            content = filepath.read_text(encoding="utf-8")
            if empty_default_pattern.search(content):
                files_with_empty_default.append(relpath)

        # At minimum, consolidation.py and context.py should have this pattern
        assert len(files_with_empty_default) >= 2, (
            f"Expected at least 2 files with model: str = '' default, "
            f"found {len(files_with_empty_default)}: {files_with_empty_default}"
        )

    def test_fallback_pattern_present(self):
        """Verify that 'if not model:' + ConsolidationConfig fallback is used."""
        root = self._project_root()

        # Files that should have the fallback pattern (all except context.py)
        # NOTE: consolidation.py removed — its model-accepting methods were deleted
        # in the consolidation refactor.
        expected_fallback_files = [
            "core/memory/contradiction.py",
            "core/memory/distillation.py",
            "core/memory/forgetting.py",
            "core/memory/reconsolidation.py",
            "core/memory/validation.py",
        ]

        fallback_pattern = re.compile(
            r"if not model:.*?ConsolidationConfig\(\)\.llm_model",
            re.DOTALL,
        )

        missing: list[str] = []
        for relpath in expected_fallback_files:
            filepath = root / relpath
            if not filepath.exists():
                continue
            content = filepath.read_text(encoding="utf-8")
            if not fallback_pattern.search(content):
                missing.append(relpath)

        assert not missing, (
            "Expected 'if not model: ... ConsolidationConfig().llm_model' in:\n"
            + "\n".join(missing)
        )


# ── Test 5: Lifecycle and Supervisor model resolution ────────────────────
# NOTE: TestLifecycleModelResolution was removed.
# It tested that lifecycle/supervisor handlers pass model= to
# ConsolidationEngine.daily_consolidate() and weekly_integrate(),
# which were deleted in the consolidation refactor. Lifecycle now calls
# anima.run_consolidation() instead; supervisor still uses
# ConsolidationEngine but without the old one-shot LLM methods.
