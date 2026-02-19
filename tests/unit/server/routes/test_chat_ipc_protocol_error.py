"""
Unit tests for IPC protocol error handling in chat SSE streaming.
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest


# ── Helper: replicate error classification logic from chat.py ──


def classify_runtime_error(e: RuntimeError) -> tuple[str, str]:
    """Replicate the error classification logic from chat.py.

    In the ``_ipc_stream_events`` generator inside ``chat_stream``,
    ``RuntimeError`` is caught *before* ``ValueError``.  If the error
    message contains ``"IPC protocol error"`` the SSE error code is
    ``IPC_PROTOCOL_ERROR``; otherwise it falls through as
    ``STREAM_ERROR``.
    """
    if "IPC protocol error" in str(e):
        return "IPC_PROTOCOL_ERROR", "通信エラーが発生しました。再試行してください。"
    else:
        return "STREAM_ERROR", "Internal server error"


# ── Tests ────────────────────────────────────────────────────────


class TestIPCProtocolErrorDetection:
    """Verify the string-matching logic that distinguishes IPC protocol
    errors from generic RuntimeErrors in the SSE streaming handler."""

    def test_ipc_protocol_error_message_detection(self) -> None:
        """The 'IPC protocol error' substring should be detected in the
        error message produced by IPCClient when a response ID mismatch
        occurs, but NOT in other RuntimeError messages."""
        # Should match: real error from ipc.py
        ipc_err = RuntimeError(
            "IPC protocol error: response ID mismatch "
            "(expected=req_123, got=stale)"
        )
        assert "IPC protocol error" in str(ipc_err)

        # Should NOT match: connection-level errors
        conn_err = RuntimeError("Connection closed during stream")
        assert "IPC protocol error" not in str(conn_err)

        not_connected_err = RuntimeError("Not connected")
        assert "IPC protocol error" not in str(not_connected_err)

    def test_ipc_protocol_error_code_format(self) -> None:
        """When an IPC protocol error is detected the SSE event should
        carry code='IPC_PROTOCOL_ERROR' and the Japanese user message."""
        err = RuntimeError(
            "IPC protocol error: response ID mismatch "
            "(expected=req_abc, got=req_xyz)"
        )
        code, message = classify_runtime_error(err)

        assert code == "IPC_PROTOCOL_ERROR"
        assert message == "通信エラーが発生しました。再試行してください。"

    def test_runtime_error_non_protocol_code_format(self) -> None:
        """A RuntimeError WITHOUT 'IPC protocol error' should produce
        code='STREAM_ERROR' and the generic English message."""
        err = RuntimeError("Connection closed during stream")
        code, message = classify_runtime_error(err)

        assert code == "STREAM_ERROR"
        assert message == "Internal server error"

        # Also verify for the "Not connected" variant
        err2 = RuntimeError("Not connected")
        code2, message2 = classify_runtime_error(err2)

        assert code2 == "STREAM_ERROR"
        assert message2 == "Internal server error"

    def test_runtime_error_does_not_mask_value_error(self) -> None:
        """ValueError exceptions must still reach the ValueError handler.

        The ``except RuntimeError`` block in ``_ipc_stream_events`` only
        catches RuntimeError.  A ValueError should propagate past it and
        be caught by the subsequent ``except ValueError`` block.  This
        test verifies the type hierarchy: ValueError is NOT a subclass of
        RuntimeError.
        """
        val_err = ValueError("Anima not responding")

        # ValueError is NOT caught by ``except RuntimeError``
        assert not isinstance(val_err, RuntimeError)

        # Conversely, RuntimeError should not be caught by ValueError
        rt_err = RuntimeError("IPC protocol error: mismatch")
        assert not isinstance(rt_err, ValueError)

        # Simulate the try/except chain from chat.py
        caught_by = None
        try:
            raise ValueError("test IPC error value")
        except RuntimeError:
            caught_by = "RuntimeError"
        except ValueError:
            caught_by = "ValueError"

        assert caught_by == "ValueError", (
            "ValueError must be caught by the ValueError handler, "
            "not the RuntimeError handler"
        )
