"""Unit tests for core.exceptions — unified exception hierarchy."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from core.exceptions import (
    AnimaWorksError,
    ExecutionError, LLMAPIError, LLMTimeoutError, StreamDisconnectedError,
    ToolError, ToolConfigError, ToolExecutionError, ToolNotFoundError,
    MemoryIOError, MemoryReadError, MemoryWriteError, MemoryCorruptedError,
    ProcessError, AnimaNotFoundError, AnimaNotRunningError, IPCConnectionError,
    ConfigError, ConfigNotFoundError, ConfigValidationError,
    MessagingError, RecipientNotFoundError, DeliveryError,
)

# ── Helpers ──────────────────────────────────────────────────

ALL_EXCEPTIONS = [
    AnimaWorksError,
    ExecutionError, LLMAPIError, LLMTimeoutError, StreamDisconnectedError,
    ToolError, ToolConfigError, ToolExecutionError, ToolNotFoundError,
    MemoryIOError, MemoryReadError, MemoryWriteError, MemoryCorruptedError,
    ProcessError, AnimaNotFoundError, AnimaNotRunningError, IPCConnectionError,
    ConfigError, ConfigNotFoundError, ConfigValidationError,
    MessagingError, RecipientNotFoundError, DeliveryError,
]

LEAF_EXCEPTIONS = [
    LLMAPIError, LLMTimeoutError, StreamDisconnectedError,
    ToolConfigError, ToolExecutionError, ToolNotFoundError,
    MemoryReadError, MemoryWriteError, MemoryCorruptedError,
    AnimaNotFoundError, AnimaNotRunningError, IPCConnectionError,
    ConfigNotFoundError, ConfigValidationError,
    RecipientNotFoundError, DeliveryError,
]

FAMILY_MAP: dict[type[AnimaWorksError], list[type[AnimaWorksError]]] = {
    ExecutionError: [LLMAPIError, LLMTimeoutError, StreamDisconnectedError],
    ToolError: [ToolConfigError, ToolExecutionError, ToolNotFoundError],
    MemoryIOError: [MemoryReadError, MemoryWriteError, MemoryCorruptedError],
    ProcessError: [AnimaNotFoundError, AnimaNotRunningError, IPCConnectionError],
    ConfigError: [ConfigNotFoundError, ConfigValidationError],
    MessagingError: [RecipientNotFoundError, DeliveryError],
}


# ── 1. All inherit from base ────────────────────────────────


class TestAllInheritFromBase:
    @pytest.mark.parametrize("exc_cls", ALL_EXCEPTIONS, ids=lambda c: c.__name__)
    def test_all_inherit_from_base(self, exc_cls: type) -> None:
        assert issubclass(exc_cls, AnimaWorksError)
        assert issubclass(exc_cls, Exception)


# ── 2. Family hierarchy ─────────────────────────────────────


class TestFamilyHierarchy:
    @pytest.mark.parametrize(
        "parent, children",
        list(FAMILY_MAP.items()),
        ids=lambda x: x.__name__ if isinstance(x, type) else None,
    )
    def test_family_hierarchy(
        self,
        parent: type[AnimaWorksError],
        children: list[type[AnimaWorksError]],
    ) -> None:
        for child in children:
            assert issubclass(child, parent), (
                f"{child.__name__} should be a subclass of {parent.__name__}"
            )
            assert issubclass(child, AnimaWorksError)


# ── 3. Catch-all with base ──────────────────────────────────


class TestCatchAllWithBase:
    @pytest.mark.parametrize("exc_cls", LEAF_EXCEPTIONS, ids=lambda c: c.__name__)
    def test_catch_all_with_base(self, exc_cls: type[AnimaWorksError]) -> None:
        with pytest.raises(AnimaWorksError):
            raise exc_cls("test message")


# ── 4. Family catch ─────────────────────────────────────────


class TestFamilyCatch:
    @pytest.mark.parametrize(
        "parent, children",
        list(FAMILY_MAP.items()),
        ids=lambda x: x.__name__ if isinstance(x, type) else None,
    )
    def test_family_catch(
        self,
        parent: type[AnimaWorksError],
        children: list[type[AnimaWorksError]],
    ) -> None:
        for child in children:
            with pytest.raises(parent):
                raise child(f"testing {child.__name__}")


# ── 5. Family isolation ─────────────────────────────────────


class TestFamilyIsolation:
    def test_tool_error_not_caught_by_execution_error(self) -> None:
        """ToolError should NOT be caught by except ExecutionError."""
        with pytest.raises(ToolError):
            try:
                raise ToolError("tool problem")
            except ExecutionError:
                pytest.fail("ExecutionError should not catch ToolError")

    def test_memory_error_not_caught_by_tool_error(self) -> None:
        """MemoryIOError should NOT be caught by except ToolError."""
        with pytest.raises(MemoryIOError):
            try:
                raise MemoryIOError("memory problem")
            except ToolError:
                pytest.fail("ToolError should not catch MemoryIOError")

    def test_config_error_not_caught_by_process_error(self) -> None:
        """ConfigError should NOT be caught by except ProcessError."""
        with pytest.raises(ConfigError):
            try:
                raise ConfigError("config problem")
            except ProcessError:
                pytest.fail("ProcessError should not catch ConfigError")

    def test_messaging_error_not_caught_by_config_error(self) -> None:
        """MessagingError should NOT be caught by except ConfigError."""
        with pytest.raises(MessagingError):
            try:
                raise MessagingError("messaging problem")
            except ConfigError:
                pytest.fail("ConfigError should not catch MessagingError")


# ── 6. StreamDisconnectedError partial_text ──────────────────


class TestStreamDisconnectedPartialText:
    def test_default_partial_text(self) -> None:
        exc = StreamDisconnectedError("boom")
        assert exc.partial_text == ""

    def test_custom_partial_text(self) -> None:
        exc = StreamDisconnectedError("boom", partial_text="hello")
        assert exc.partial_text == "hello"

    def test_str_returns_message(self) -> None:
        exc = StreamDisconnectedError("boom", partial_text="hello")
        assert str(exc) == "boom"

    def test_default_message(self) -> None:
        exc = StreamDisconnectedError()
        assert str(exc) == "Stream disconnected"
        assert exc.partial_text == ""


# ── 7. Re-export from execution.base ────────────────────────


class TestReExportExecutionBase:
    def test_re_export_execution_base(self) -> None:
        from core.execution.base import StreamDisconnectedError as ReExported

        assert ReExported is StreamDisconnectedError


# ── 8. Re-export from tools._base ───────────────────────────


class TestReExportToolsBase:
    def test_re_export_tools_base(self) -> None:
        from core.tools._base import ToolConfigError as ReExported

        assert ReExported is ToolConfigError


# ── 9. str representation ───────────────────────────────────


class TestStrRepresentation:
    @pytest.mark.parametrize("exc_cls", ALL_EXCEPTIONS, ids=lambda c: c.__name__)
    def test_str_representation(self, exc_cls: type[AnimaWorksError]) -> None:
        msg = f"test error from {exc_cls.__name__}"
        exc = exc_cls(msg)
        assert str(exc) == msg
        assert msg in repr(exc)
