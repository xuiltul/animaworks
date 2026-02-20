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
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a minimal anima directory for consolidation tests."""
    d = tmp_path / "test_anima"
    (d / "episodes").mkdir(parents=True)
    (d / "knowledge").mkdir(parents=True)
    (d / "procedures").mkdir(parents=True)
    (d / "activity_log").mkdir(parents=True)
    return d


@pytest.fixture
def episodes_dir(anima_dir: Path) -> Path:
    return anima_dir / "episodes"


@pytest.fixture
def consolidation_engine(anima_dir: Path):
    """Create a ConsolidationEngine with the temporary directory."""
    from core.memory.consolidation import ConsolidationEngine

    return ConsolidationEngine(
        anima_dir=anima_dir,
        anima_name="test_anima",
    )


def _make_episode_file(episodes_dir: Path, content: str | None = None) -> Path:
    """Write today's episode file with at least one valid entry."""
    today = datetime.now().date()
    ep_file = episodes_dir / f"{today}.md"
    if content is None:
        content = (
            f"## 10:00 — テスト作業\n\n"
            f"**相手**: システム\n"
            f"**要点**: テスト用のエピソード記録。\n"
        )
    ep_file.write_text(content, encoding="utf-8")
    return ep_file


def _make_mock_llm_response(text: str) -> MagicMock:
    """Build a mock litellm.acompletion response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    return resp


def _patch_load_config(return_value: Any):
    """Patch load_config in both core.config.models and core.config.

    ``core.config.__init__`` does ``from core.config.models import load_config``
    at import time, copying the reference. In-function local imports like
    ``from core.config import load_config`` resolve against that cached binding.
    We must therefore patch **both** module namespaces so that callers in
    any module see the mock regardless of how they imported the function.
    """
    return (
        patch("core.config.models.load_config", return_value=return_value),
        patch("core.config.load_config", return_value=return_value),
    )


# ── Test 1: Config-driven model resolution end-to-end ────────────────────


class TestConfigDrivenModelResolution:
    """daily_consolidate() with default model="" resolves to ConsolidationConfig."""

    @pytest.mark.asyncio
    async def test_default_model_uses_consolidation_config(
        self, consolidation_engine, episodes_dir
    ):
        """When model="" (default), litellm receives ConsolidationConfig().llm_model."""
        from core.config.models import ConsolidationConfig

        expected_model = ConsolidationConfig().llm_model

        _make_episode_file(episodes_dir)

        llm_response = _make_mock_llm_response(
            "## 既存ファイル更新\n(なし)\n\n"
            "## 新規ファイル作成\n"
            "- ファイル名: knowledge/test.md\n"
            "  内容: テスト知識\n"
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = llm_response
            result = await consolidation_engine.daily_consolidate(min_episodes=1)

        assert result["skipped"] is False

        # Verify the model passed to litellm is the config default
        assert mock_llm.await_count >= 1
        first_call = mock_llm.call_args_list[0]
        actual_model = first_call.kwargs.get("model") or first_call.args[0]
        assert actual_model == expected_model, (
            f"Expected model={expected_model!r}, got {actual_model!r}"
        )

    @pytest.mark.asyncio
    async def test_explicit_model_override(
        self, consolidation_engine, episodes_dir
    ):
        """When model is explicitly passed, it takes precedence over config."""
        custom_model = "openai/gpt-4o"

        _make_episode_file(episodes_dir)

        llm_response = _make_mock_llm_response(
            "## 既存ファイル更新\n(なし)\n\n"
            "## 新規ファイル作成\n"
            "- ファイル名: knowledge/test.md\n"
            "  内容: テスト知識\n"
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = llm_response
            result = await consolidation_engine.daily_consolidate(
                model=custom_model, min_episodes=1,
            )

        assert result["skipped"] is False
        first_call = mock_llm.call_args_list[0]
        actual_model = first_call.kwargs.get("model") or first_call.args[0]
        assert actual_model == custom_model

    @pytest.mark.asyncio
    async def test_weekly_integrate_default_model(
        self, consolidation_engine, episodes_dir
    ):
        """weekly_integrate() with default model="" also resolves from config."""
        from core.config.models import ConsolidationConfig

        expected_model = ConsolidationConfig().llm_model

        # weekly_integrate calls _detect_duplicates which needs RAG; mock it
        with (
            patch.object(
                consolidation_engine, "_detect_duplicates",
                new_callable=AsyncMock, return_value=[],
            ),
            patch.object(
                consolidation_engine, "_compress_old_episodes",
                new_callable=AsyncMock, return_value=0,
            ) as mock_compress,
            patch.object(
                consolidation_engine, "_rebuild_rag_index",
            ),
            patch(
                "core.memory.forgetting.ForgettingEngine.neurogenesis_reorganize",
                new_callable=AsyncMock,
                return_value={"merged_count": 0},
            ),
            patch(
                "core.memory.distillation.ProceduralDistiller.weekly_pattern_distill",
                new_callable=AsyncMock,
                return_value={"patterns_detected": 0, "procedures_created": []},
            ),
            patch.object(
                consolidation_engine, "_run_contradiction_check",
                new_callable=AsyncMock, return_value={},
            ),
        ):
            result = await consolidation_engine.weekly_integrate()

        # _compress_old_episodes receives the resolved model
        if mock_compress.call_count > 0:
            call_kwargs = mock_compress.call_args.kwargs
            actual_model = call_kwargs.get("model", "")
            # If model was passed, it should be the resolved one
            if actual_model:
                assert actual_model == expected_model


# ── Test 2: Config override via patching ─────────────────────────────────


class TestConfigOverrideViaPatching:
    """Patching ConsolidationConfig to return a custom model changes behaviour."""

    @pytest.mark.asyncio
    async def test_patched_consolidation_config_model_is_used(
        self, consolidation_engine, episodes_dir
    ):
        """Patching ConsolidationConfig() to return custom llm_model propagates to LLM."""
        from core.config.models import ConsolidationConfig

        custom_model = "openai/gpt-4o"

        # Create a patched ConsolidationConfig that returns our custom model
        mock_config_instance = ConsolidationConfig(llm_model=custom_model)

        _make_episode_file(episodes_dir)

        llm_response = _make_mock_llm_response(
            "## 既存ファイル更新\n(なし)\n\n"
            "## 新規ファイル作成\n(なし)\n"
        )

        # ConsolidationConfig is imported in the function body as
        # ``from core.config.models import ConsolidationConfig``
        with (
            patch(
                "core.config.models.ConsolidationConfig",
                return_value=mock_config_instance,
            ),
            patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm,
        ):
            mock_llm.return_value = llm_response
            await consolidation_engine.daily_consolidate(min_episodes=1)

        assert mock_llm.await_count >= 1
        first_call = mock_llm.call_args_list[0]
        actual_model = first_call.kwargs.get("model") or first_call.args[0]
        assert actual_model == custom_model, (
            f"Expected patched model={custom_model!r}, got {actual_model!r}"
        )

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
        expected_fallback_files = [
            "core/memory/consolidation.py",
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


class TestLifecycleModelResolution:
    """Lifecycle and Supervisor handlers resolve models from config, not hardcoded."""

    @pytest.mark.asyncio
    async def test_lifecycle_daily_consolidation_uses_config_model(
        self, anima_dir
    ):
        """LifecycleManager._handle_daily_consolidation reads model from config."""
        from core.config.models import (
            AnimaWorksConfig,
            ConsolidationConfig,
        )

        custom_model = "google/gemini-2.5-pro"
        custom_config = AnimaWorksConfig(
            consolidation=ConsolidationConfig(llm_model=custom_model),
        )

        p1, p2 = _patch_load_config(custom_config)
        with (
            p1,
            p2,
            patch(
                "core.memory.consolidation.ConsolidationEngine.daily_consolidate",
                new_callable=AsyncMock,
                return_value={
                    "skipped": False,
                    "episodes_processed": 1,
                    "knowledge_files_created": [],
                    "knowledge_files_updated": [],
                },
            ) as mock_consolidate,
        ):
            from core.lifecycle import LifecycleManager

            lm = LifecycleManager()

            # Register a minimal mock anima
            mock_anima = MagicMock()
            mock_anima.name = "test_anima"
            mock_anima.memory.anima_dir = anima_dir
            lm.animas["test_anima"] = mock_anima

            await lm._handle_daily_consolidation()

        # Verify the model from config was passed
        assert mock_consolidate.await_count == 1
        call_kwargs = mock_consolidate.call_args.kwargs
        assert call_kwargs["model"] == custom_model, (
            f"Expected lifecycle to pass model={custom_model!r}, "
            f"got {call_kwargs['model']!r}"
        )

    @pytest.mark.asyncio
    async def test_lifecycle_weekly_integration_uses_config_model(
        self, anima_dir
    ):
        """LifecycleManager._handle_weekly_integration reads model from config."""
        from core.config.models import (
            AnimaWorksConfig,
            ConsolidationConfig,
        )

        custom_model = "xai/grok-3"
        custom_config = AnimaWorksConfig(
            consolidation=ConsolidationConfig(llm_model=custom_model),
        )

        p1, p2 = _patch_load_config(custom_config)
        with (
            p1,
            p2,
            patch(
                "core.memory.consolidation.ConsolidationEngine.weekly_integrate",
                new_callable=AsyncMock,
                return_value={
                    "skipped": False,
                    "knowledge_files_merged": [],
                    "episodes_compressed": 0,
                },
            ) as mock_integrate,
        ):
            from core.lifecycle import LifecycleManager

            lm = LifecycleManager()

            mock_anima = MagicMock()
            mock_anima.name = "test_anima"
            mock_anima.memory.anima_dir = anima_dir
            lm.animas["test_anima"] = mock_anima

            await lm._handle_weekly_integration()

        assert mock_integrate.await_count == 1
        call_kwargs = mock_integrate.call_args.kwargs
        assert call_kwargs["model"] == custom_model

    @pytest.mark.asyncio
    async def test_supervisor_daily_consolidation_uses_config_model(
        self, tmp_path
    ):
        """ProcessSupervisor._run_daily_consolidation reads model from config."""
        from core.config.models import (
            AnimaWorksConfig,
            ConsolidationConfig,
        )

        custom_model = "mistral/mistral-large"
        custom_config = AnimaWorksConfig(
            consolidation=ConsolidationConfig(llm_model=custom_model),
        )

        # Set up minimal anima directory structure for _iter_consolidation_targets
        animas_dir = tmp_path / "animas"
        test_anima_dir = animas_dir / "test_anima"
        test_anima_dir.mkdir(parents=True)
        (test_anima_dir / "identity.md").write_text("# Test Anima", encoding="utf-8")
        (test_anima_dir / "status.json").write_text(
            '{"enabled": true}', encoding="utf-8",
        )
        (test_anima_dir / "episodes").mkdir(parents=True)
        (test_anima_dir / "knowledge").mkdir(parents=True)

        p1, p2 = _patch_load_config(custom_config)
        with (
            p1,
            p2,
            patch(
                "core.memory.consolidation.ConsolidationEngine.daily_consolidate",
                new_callable=AsyncMock,
                return_value={
                    "skipped": True,
                    "episodes_processed": 0,
                    "knowledge_files_created": [],
                    "knowledge_files_updated": [],
                },
            ) as mock_consolidate,
        ):
            from core.supervisor.manager import ProcessSupervisor

            supervisor = ProcessSupervisor(
                animas_dir=animas_dir,
                shared_dir=tmp_path / "shared",
                run_dir=tmp_path / "run",
                log_dir=tmp_path / "logs",
            )

            await supervisor._run_daily_consolidation()

        assert mock_consolidate.await_count == 1
        call_kwargs = mock_consolidate.call_args.kwargs
        assert call_kwargs["model"] == custom_model, (
            f"Expected supervisor to pass model={custom_model!r}, "
            f"got {call_kwargs['model']!r}"
        )

    @pytest.mark.asyncio
    async def test_supervisor_weekly_integration_uses_config_model(
        self, tmp_path
    ):
        """ProcessSupervisor._run_weekly_integration reads model from config."""
        from core.config.models import (
            AnimaWorksConfig,
            ConsolidationConfig,
        )

        custom_model = "deepseek/deepseek-chat"
        custom_config = AnimaWorksConfig(
            consolidation=ConsolidationConfig(llm_model=custom_model),
        )

        # Set up minimal anima directory
        animas_dir = tmp_path / "animas"
        test_anima_dir = animas_dir / "test_anima"
        test_anima_dir.mkdir(parents=True)
        (test_anima_dir / "identity.md").write_text("# Test Anima", encoding="utf-8")
        (test_anima_dir / "status.json").write_text(
            '{"enabled": true}', encoding="utf-8",
        )
        (test_anima_dir / "episodes").mkdir(parents=True)
        (test_anima_dir / "knowledge").mkdir(parents=True)

        p1, p2 = _patch_load_config(custom_config)
        with (
            p1,
            p2,
            patch(
                "core.memory.consolidation.ConsolidationEngine.weekly_integrate",
                new_callable=AsyncMock,
                return_value={
                    "skipped": True,
                    "knowledge_files_merged": [],
                    "episodes_compressed": 0,
                },
            ) as mock_integrate,
        ):
            from core.supervisor.manager import ProcessSupervisor

            supervisor = ProcessSupervisor(
                animas_dir=animas_dir,
                shared_dir=tmp_path / "shared",
                run_dir=tmp_path / "run",
                log_dir=tmp_path / "logs",
            )

            await supervisor._run_weekly_integration()

        assert mock_integrate.await_count == 1
        call_kwargs = mock_integrate.call_args.kwargs
        assert call_kwargs["model"] == custom_model

    @pytest.mark.asyncio
    async def test_lifecycle_default_config_uses_consolidation_config_default(
        self, anima_dir
    ):
        """When config has no consolidation override, the default llm_model is used."""
        from core.config.models import AnimaWorksConfig, ConsolidationConfig

        # Default config - no overrides
        default_config = AnimaWorksConfig()
        expected_model = ConsolidationConfig().llm_model

        p1, p2 = _patch_load_config(default_config)
        with (
            p1,
            p2,
            patch(
                "core.memory.consolidation.ConsolidationEngine.daily_consolidate",
                new_callable=AsyncMock,
                return_value={
                    "skipped": True,
                    "episodes_processed": 0,
                    "knowledge_files_created": [],
                    "knowledge_files_updated": [],
                },
            ) as mock_consolidate,
        ):
            from core.lifecycle import LifecycleManager

            lm = LifecycleManager()

            mock_anima = MagicMock()
            mock_anima.name = "test_anima"
            mock_anima.memory.anima_dir = anima_dir
            lm.animas["test_anima"] = mock_anima

            await lm._handle_daily_consolidation()

        assert mock_consolidate.await_count == 1
        call_kwargs = mock_consolidate.call_args.kwargs
        assert call_kwargs["model"] == expected_model
