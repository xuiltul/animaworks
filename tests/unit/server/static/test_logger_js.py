# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for server/static/shared/logger.js — _getSessionId fallback.

Validates that the logger's session ID generation handles both secure
(HTTPS/localhost) and insecure (HTTP + LAN IP) contexts by checking
the JavaScript source for the required fallback pattern.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Path to the logger.js file under test
_LOGGER_JS = Path(__file__).resolve().parents[4] / "server" / "static" / "shared" / "logger.js"


class TestGetSessionIdFallback:
    """Verify _getSessionId() has crypto.randomUUID fallback."""

    @pytest.fixture(autouse=True)
    def _load_source(self) -> None:
        assert _LOGGER_JS.exists(), f"logger.js not found at {_LOGGER_JS}"
        self.source = _LOGGER_JS.read_text(encoding="utf-8")

    def test_checks_randomuuid_availability(self) -> None:
        """Must feature-detect crypto.randomUUID before calling it."""
        assert "typeof crypto.randomUUID === 'function'" in self.source, (
            "_getSessionId must check typeof crypto.randomUUID before use"
        )

    def test_has_getRandomValues_fallback(self) -> None:
        """Must fall back to crypto.getRandomValues for insecure contexts."""
        assert "crypto.getRandomValues" in self.source, (
            "_getSessionId must use crypto.getRandomValues as fallback"
        )

    def test_no_bare_randomuuid_call(self) -> None:
        """Must NOT call crypto.randomUUID() without a guard.

        The only occurrence of crypto.randomUUID should be inside the
        typeof check or the guarded branch, never as a bare call.
        """
        lines = self.source.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip the typeof check line
            if "typeof crypto.randomUUID" in stripped:
                continue
            # Skip comment lines
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            # If randomUUID appears, it should be guarded (inside an if block)
            if "crypto.randomUUID()" in stripped:
                # Verify it's indented (inside a conditional block)
                assert line.startswith("      ") or line.startswith("\t\t"), (
                    f"Line {i}: crypto.randomUUID() appears to be called "
                    f"without a typeof guard: {stripped}"
                )

    def test_session_id_length_consistent(self) -> None:
        """Both branches should produce a 12-character session ID.

        - Secure branch: crypto.randomUUID().slice(0, 12)
        - Fallback branch: Uint8Array(6) → 6 bytes → 12 hex chars
        """
        assert ".slice(0, 12)" in self.source, (
            "Secure branch must slice UUID to 12 characters"
        )
        assert "Uint8Array(6)" in self.source, (
            "Fallback branch must use 6 bytes (= 12 hex chars)"
        )

    def test_sessionStorage_persistence(self) -> None:
        """Session ID must be persisted to sessionStorage."""
        assert "sessionStorage.setItem" in self.source
        assert "sessionStorage.getItem" in self.source

    def test_typeof_crypto_check(self) -> None:
        """Must check that crypto object exists before accessing methods."""
        assert "typeof crypto !== 'undefined'" in self.source, (
            "Must guard against environments where crypto is undefined"
        )
