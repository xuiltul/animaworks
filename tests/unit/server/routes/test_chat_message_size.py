"""Unit tests for chat message size validation."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from server.routes.chat import MAX_CHAT_MESSAGE_SIZE


def test_max_chat_message_size_is_10mb():
    """MAX_CHAT_MESSAGE_SIZE should be 10MB."""
    assert MAX_CHAT_MESSAGE_SIZE == 10 * 1024 * 1024
