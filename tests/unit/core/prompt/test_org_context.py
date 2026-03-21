"""Unit tests for core.prompt.org_context helpers."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


def test_is_mcp_mode_includes_d():
    from core.prompt.org_context import _is_mcp_mode

    assert _is_mcp_mode("d") is True
