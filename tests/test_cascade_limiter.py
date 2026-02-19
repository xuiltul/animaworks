from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tests for ConversationDepthLimiter.

Covers:
- Pair key normalization (symmetric)
- Windowed expiration
- Blocking at max_depth
- current_depth reporting
"""

import time
from unittest.mock import patch

import pytest

from core.cascade_limiter import ConversationDepthLimiter


class TestPairKeyNormalization:
    """Pair key is always sorted, so (a, b) == (b, a)."""

    def test_pair_key_symmetric(self):
        limiter = ConversationDepthLimiter()
        assert limiter._pair_key("alice", "bob") == limiter._pair_key("bob", "alice")

    def test_pair_key_same(self):
        limiter = ConversationDepthLimiter()
        assert limiter._pair_key("alice", "bob") == ("alice", "bob")


class TestCheckAndRecord:
    """Test check_and_record allows up to max_depth, then blocks."""

    def test_allows_up_to_max_depth(self):
        limiter = ConversationDepthLimiter(max_depth=3)
        assert limiter.check_and_record("alice", "bob") is True
        assert limiter.check_and_record("bob", "alice") is True
        assert limiter.check_and_record("alice", "bob") is True
        # 4th should be blocked
        assert limiter.check_and_record("bob", "alice") is False

    def test_different_pairs_independent(self):
        limiter = ConversationDepthLimiter(max_depth=2)
        assert limiter.check_and_record("alice", "bob") is True
        assert limiter.check_and_record("alice", "bob") is True
        assert limiter.check_and_record("alice", "bob") is False  # blocked

        # Different pair should still be allowed
        assert limiter.check_and_record("alice", "charlie") is True
        assert limiter.check_and_record("alice", "charlie") is True

    def test_symmetric_counting(self):
        """A -> B and B -> A count towards the same depth."""
        limiter = ConversationDepthLimiter(max_depth=2)
        assert limiter.check_and_record("alice", "bob") is True   # depth=1
        assert limiter.check_and_record("bob", "alice") is True   # depth=2
        assert limiter.check_and_record("alice", "bob") is False  # blocked


class TestWindowExpiration:
    """Entries expire after the window, freeing up depth."""

    def test_expired_entries_evicted(self):
        limiter = ConversationDepthLimiter(window_s=10, max_depth=2)
        assert limiter.check_and_record("alice", "bob") is True
        assert limiter.check_and_record("alice", "bob") is True
        assert limiter.check_and_record("alice", "bob") is False  # blocked

        # Simulate time passing beyond window
        with patch("time.monotonic", return_value=time.monotonic() + 11):
            assert limiter.check_and_record("alice", "bob") is True  # allowed again


class TestCurrentDepth:
    """Test current_depth reporting."""

    def test_zero_when_empty(self):
        limiter = ConversationDepthLimiter()
        assert limiter.current_depth("alice", "bob") == 0

    def test_increments_with_exchanges(self):
        limiter = ConversationDepthLimiter(max_depth=10)
        limiter.check_and_record("alice", "bob")
        assert limiter.current_depth("alice", "bob") == 1
        limiter.check_and_record("bob", "alice")
        assert limiter.current_depth("alice", "bob") == 2

    def test_symmetric(self):
        limiter = ConversationDepthLimiter(max_depth=10)
        limiter.check_and_record("alice", "bob")
        assert limiter.current_depth("bob", "alice") == 1


class TestModuleSingleton:
    """Test the module-level singleton is shared."""

    def test_singleton_exists(self):
        from core.cascade_limiter import depth_limiter
        assert isinstance(depth_limiter, ConversationDepthLimiter)
