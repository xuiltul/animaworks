from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for sender_name path traversal prevention in PrimingEngine.

Covers:
- _channel_a_sender_profile rejects sender_name containing ../
- _channel_a_sender_profile works for valid sender names
"""

import tempfile
from pathlib import Path

import pytest

from core.memory.priming import PrimingEngine


@pytest.fixture
def temp_dirs():
    """Create temporary anima and shared directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        anima_dir = base / "animas" / "test_anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "knowledge").mkdir()

        shared_dir = base / "shared"
        users_dir = shared_dir / "users"
        users_dir.mkdir(parents=True)

        # Create a valid user profile
        valid_user = users_dir / "alice"
        valid_user.mkdir()
        (valid_user / "index.md").write_text("# Alice\nValid profile.", encoding="utf-8")

        # Create a file outside users/ that traversal might reach
        (shared_dir / "secret.md").write_text("SECRET DATA", encoding="utf-8")

        yield anima_dir, shared_dir


@pytest.mark.asyncio
async def test_sender_traversal_returns_empty(temp_dirs):
    """sender_name with ../ should return empty string, not file contents."""
    anima_dir, shared_dir = temp_dirs
    engine = PrimingEngine(anima_dir, shared_dir=shared_dir)

    from unittest.mock import patch
    with patch("core.paths.get_shared_dir", return_value=shared_dir):
        result = await engine._channel_a_sender_profile("../secret")

    assert result == ""


@pytest.mark.asyncio
async def test_sender_deep_traversal_returns_empty(temp_dirs):
    """Deep traversal in sender_name should return empty."""
    anima_dir, shared_dir = temp_dirs
    engine = PrimingEngine(anima_dir, shared_dir=shared_dir)

    from unittest.mock import patch
    with patch("core.paths.get_shared_dir", return_value=shared_dir):
        result = await engine._channel_a_sender_profile("../../etc/passwd")

    assert result == ""


@pytest.mark.asyncio
async def test_valid_sender_returns_profile(temp_dirs):
    """Valid sender_name should return profile content normally."""
    anima_dir, shared_dir = temp_dirs
    engine = PrimingEngine(anima_dir, shared_dir=shared_dir)

    from unittest.mock import patch
    with patch("core.paths.get_shared_dir", return_value=shared_dir):
        result = await engine._channel_a_sender_profile("alice")

    assert "Alice" in result
    assert "Valid profile" in result


@pytest.mark.asyncio
async def test_nonexistent_sender_returns_empty(temp_dirs):
    """Non-existent sender_name should return empty string."""
    anima_dir, shared_dir = temp_dirs
    engine = PrimingEngine(anima_dir, shared_dir=shared_dir)

    from unittest.mock import patch
    with patch("core.paths.get_shared_dir", return_value=shared_dir):
        result = await engine._channel_a_sender_profile("nonexistent_user")

    assert result == ""
