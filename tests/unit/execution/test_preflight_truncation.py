"""Unit tests for preflight truncation guard relaxation."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import types


class TestPreflightTruncation:
    """Test that the relaxed truncation guard works correctly."""

    def test_truncation_preserves_minimum_chars(self):
        """Truncation should never reduce system content below 2000 chars."""
        from core.execution._litellm_context import ContextMixin

        mixin = ContextMixin()
        mixin._resolve_cw = lambda: 8000
        mixin._model_config = types.SimpleNamespace(model="test-model")

        sys_content = "x" * 5000
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": "hello"},
        ]
        tools = []

        mock_litellm = types.SimpleNamespace()
        mock_litellm.token_counter = lambda **kw: 10000  # exceeds 8000

        llm_kwargs = {"max_tokens": 4096}

        mixin._preflight_clamp(llm_kwargs, messages, tools, mock_litellm)

        # System content should still have at least 2000 chars after truncation
        assert len(messages[0]["content"]) >= 2000
