from __future__ import annotations

"""E2E tests for the custom exception hierarchy.

Verifies that the exception hierarchy is properly integrated across the
entire codebase — imports work, re-exports are compatible, and no silent
exception passes remain in core/.
"""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


# ── Hierarchy availability ───────────────────────────────────

class TestExceptionHierarchyAvailability:
    """Verify core.exceptions is importable and complete."""

    def test_import_all_exceptions(self):
        """All 23 exception classes are importable from core.exceptions."""
        from core.exceptions import (
            AnimaWorksError,
            ExecutionError, LLMAPIError, LLMTimeoutError, StreamDisconnectedError,
            ToolError, ToolConfigError, ToolExecutionError, ToolNotFoundError,
            MemoryIOError, MemoryReadError, MemoryWriteError, MemoryCorruptedError,
            ProcessError, AnimaNotFoundError, AnimaNotRunningError, IPCConnectionError,
            ConfigError, ConfigNotFoundError, ConfigValidationError,
            MessagingError, RecipientNotFoundError, DeliveryError,
        )
        # All should be classes
        assert isinstance(AnimaWorksError, type)
        assert isinstance(DeliveryError, type)

    def test_base_catches_all_leaf_exceptions(self):
        """except AnimaWorksError catches every domain exception."""
        from core.exceptions import (
            AnimaWorksError,
            LLMAPIError, LLMTimeoutError, StreamDisconnectedError,
            ToolConfigError, ToolExecutionError, ToolNotFoundError,
            MemoryReadError, MemoryWriteError, MemoryCorruptedError,
            AnimaNotFoundError, AnimaNotRunningError, IPCConnectionError,
            ConfigNotFoundError, ConfigValidationError,
            RecipientNotFoundError, DeliveryError,
        )
        leaves = [
            LLMAPIError, LLMTimeoutError, StreamDisconnectedError,
            ToolConfigError, ToolExecutionError, ToolNotFoundError,
            MemoryReadError, MemoryWriteError, MemoryCorruptedError,
            AnimaNotFoundError, AnimaNotRunningError, IPCConnectionError,
            ConfigNotFoundError, ConfigValidationError,
            RecipientNotFoundError, DeliveryError,
        ]
        for exc_cls in leaves:
            try:
                raise exc_cls(f"test {exc_cls.__name__}")
            except AnimaWorksError:
                pass  # Expected
            except Exception:
                pytest.fail(f"{exc_cls.__name__} not caught by AnimaWorksError")


# ── Re-export compatibility ──────────────────────────────────

class TestReExportCompatibility:
    """Verify backward-compatible re-exports from original modules."""

    def test_stream_disconnected_error_re_export(self):
        """StreamDisconnectedError importable from core.execution.base."""
        from core.execution.base import StreamDisconnectedError as FromBase
        from core.exceptions import StreamDisconnectedError as FromExceptions
        assert FromBase is FromExceptions

    def test_stream_disconnected_error_partial_text(self):
        """StreamDisconnectedError preserves partial_text attribute."""
        from core.execution.base import StreamDisconnectedError
        exc = StreamDisconnectedError("disconnected", partial_text="partial response")
        assert exc.partial_text == "partial response"
        assert str(exc) == "disconnected"

    def test_tool_config_error_re_export(self):
        """ToolConfigError importable from core.tools._base."""
        from core.tools._base import ToolConfigError as FromBase
        from core.exceptions import ToolConfigError as FromExceptions
        assert FromBase is FromExceptions


# ── Integration: exception imports in core modules ───────────

class TestCoreModuleImports:
    """Verify that core modules have imported domain exceptions."""

    @pytest.mark.parametrize("module_path", [
        "core.execution.agent_sdk",
        "core.execution.litellm_loop",
        "core.execution.assisted",
        "core.supervisor.manager",
        "core.supervisor.runner",
        "core.supervisor.ipc",
        "core.tooling.handler",
        "core.tooling.dispatch",
        "core.anima",
        "core.agent",
        "core.messenger",
        "core.lifecycle",
        "core.outbound",
        "core.background",
    ])
    def test_module_imports_successfully(self, module_path):
        """Core module imports without error after exception hierarchy changes."""
        mod = importlib.import_module(module_path)
        assert mod is not None


# ── Zero silent passes verification ──────────────────────────

class TestNoSilentPasses:
    """Verify no silent 'except Exception: pass' remains in core/."""

    def test_no_except_exception_pass_in_core(self):
        """Scan core/ for multiline 'except Exception:\\n    pass' — must find zero."""
        core_dir = Path(__file__).resolve().parents[2] / "core"
        assert core_dir.is_dir(), f"core/ not found at {core_dir}"

        # Use multiline grep (-Pz) to match except/pass across lines
        result = subprocess.run(
            ["grep", "-Przn", r"except\s+Exception\s*:\s*\n\s+pass\b", str(core_dir)],
            capture_output=True, text=True,
        )
        matches = result.stdout.strip()
        assert matches == "", (
            f"Found silent 'except Exception: pass' in core/:\n{matches}"
        )
