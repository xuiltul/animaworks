# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ActivityLogger consolidation refactoring.

Verifies that:
- DigitalAnima and ToolHandler initialise a shared ``self._activity``
  instance instead of creating inline ``ActivityLogger()`` on every call.
- No inline ``ActivityLogger(...)`` instantiations remain outside ``__init__``.
- Write-path activity log calls are NOT wrapped in ``try/except``.
- Read-path methods (``_load_heartbeat_history``, ``_load_recent_reflections``)
  retain their ``try/except`` guard (they have recovery / fallback logic).
"""

from __future__ import annotations

import inspect
from pathlib import Path


# ── Source inspection: DigitalAnima ──────────────────────────


class TestDigitalAnimaActivityConsolidation:
    """Source-level checks on DigitalAnima activity logger consolidation."""

    def test_init_assigns_self_activity(self):
        """DigitalAnima.__init__ assigns self._activity = ActivityLogger(anima_dir)."""
        from core.anima import DigitalAnima

        source = inspect.getsource(DigitalAnima.__init__)
        assert "self._activity = ActivityLogger(anima_dir)" in source

    def test_no_inline_activity_logger_outside_init(self):
        """core/anima.py has no inline ActivityLogger(...) outside __init__."""
        source_path = Path("core/anima.py")
        source = source_path.read_text(encoding="utf-8")
        lines = source.splitlines()

        occurrences: list[tuple[int, str]] = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if "ActivityLogger(" in stripped and "self._activity" not in stripped:
                # Skip import lines
                if "import" in stripped:
                    continue
                occurrences.append((i, stripped))

        assert occurrences == [], (
            f"Inline ActivityLogger instantiation(s) found outside __init__: "
            f"{occurrences}"
        )

    def test_self_activity_is_activity_logger_type(self):
        """DigitalAnima.__init__ imports and uses ActivityLogger for self._activity."""
        from core.anima import DigitalAnima

        source = inspect.getsource(DigitalAnima.__init__)
        assert "ActivityLogger" in source

    def test_write_call_in_process_message_not_wrapped_in_try_except(self):
        """Activity log write in process_message is NOT wrapped in try/except.

        The ``self._activity.log("message_received", ...)`` call should appear
        outside any ``try:`` block that is specifically guarding it.
        """
        from core.anima import DigitalAnima

        source = inspect.getsource(DigitalAnima.process_message)
        lines = source.splitlines()

        # Find the activity log "message_received" call
        for i, line in enumerate(lines):
            if 'self._activity.log("message_received"' in line:
                # Walk backwards to check: the closest enclosing try should be
                # the lock acquisition try, NOT a dedicated activity-log guard.
                # A dedicated guard would have "try:" immediately preceding
                # (within 1-2 lines) with no other logic between.
                found_try_guard = False
                for j in range(i - 1, max(i - 3, -1), -1):
                    stripped = lines[j].strip()
                    if stripped == "try:":
                        # Check if this try is ONLY guarding the activity log
                        # (i.e. the except is "except: pass" or "except Exception: pass")
                        # Look ahead from the activity line for the matching except
                        for k in range(i + 1, min(i + 4, len(lines))):
                            exc_line = lines[k].strip()
                            if exc_line.startswith("except") and "pass" in exc_line:
                                found_try_guard = True
                                break
                            if exc_line.startswith("except"):
                                # Has recovery logic, check if next line is just pass
                                if k + 1 < len(lines) and lines[k + 1].strip() == "pass":
                                    found_try_guard = True
                                break
                        break
                assert not found_try_guard, (
                    "Activity log 'message_received' write call is still "
                    "wrapped in a dedicated try/except: pass guard"
                )
                break

    def test_load_heartbeat_history_retains_try_except(self):
        """_load_heartbeat_history still has try/except wrapping for recovery."""
        from core.anima import DigitalAnima

        source = inspect.getsource(DigitalAnima._load_heartbeat_history)
        assert "try:" in source
        assert "except Exception" in source
        assert 'return ""' in source

    def test_load_recent_reflections_retains_try_except(self):
        """_load_recent_reflections still has try/except wrapping for recovery."""
        from core.anima import DigitalAnima

        source = inspect.getsource(DigitalAnima._load_recent_reflections)
        assert "try:" in source
        assert "except Exception" in source
        assert 'return ""' in source


# ── Source inspection: ToolHandler ───────────────────────────


class TestToolHandlerActivityConsolidation:
    """Source-level checks on ToolHandler activity logger consolidation."""

    def test_init_assigns_self_activity(self):
        """ToolHandler.__init__ assigns self._activity = ActivityLogger(self._anima_dir)."""
        from core.tooling.handler import ToolHandler

        source = inspect.getsource(ToolHandler.__init__)
        assert "self._activity = ActivityLogger(self._anima_dir)" in source

    def test_no_inline_activity_logger_outside_init(self):
        """core/tooling/handler.py has no inline ActivityLogger(...) outside __init__."""
        source_path = Path("core/tooling/handler.py")
        source = source_path.read_text(encoding="utf-8")
        lines = source.splitlines()

        occurrences: list[tuple[int, str]] = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if "ActivityLogger(" in stripped and "self._activity" not in stripped:
                # Skip import lines
                if "import" in stripped:
                    continue
                occurrences.append((i, stripped))

        assert occurrences == [], (
            f"Inline ActivityLogger instantiation(s) found outside __init__: "
            f"{occurrences}"
        )

    def test_self_activity_is_activity_logger_type(self):
        """ToolHandler.__init__ imports and uses ActivityLogger for self._activity."""
        from core.tooling.handler import ToolHandler

        source = inspect.getsource(ToolHandler.__init__)
        assert "ActivityLogger" in source


# ── Cross-file count verification ────────────────────────────


class TestActivityLoggerUsageCount:
    """Verify that self._activity is used consistently across the codebase."""

    def test_anima_uses_self_activity_for_all_log_calls(self):
        """All activity log calls in core/anima.py use self._activity."""
        source_path = Path("core/anima.py")
        source = source_path.read_text(encoding="utf-8")

        # Count self._activity.log calls
        activity_calls = source.count("self._activity.log(")
        assert activity_calls > 0, "Expected self._activity.log() calls in core/anima.py"

        # Count self._activity.recent calls
        recent_calls = source.count("self._activity.recent(")
        assert recent_calls > 0, "Expected self._activity.recent() calls in core/anima.py"

    def test_handler_uses_self_activity_for_all_log_calls(self):
        """All activity log calls in core/tooling/handler.py use self._activity."""
        source_path = Path("core/tooling/handler.py")
        source = source_path.read_text(encoding="utf-8")

        # Count self._activity.log calls
        activity_calls = source.count("self._activity.log(")
        assert activity_calls > 0, (
            "Expected self._activity.log() calls in core/tooling/handler.py"
        )
