# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E: Verify background tool name consistency across modules.

Ensures that _DEFAULT_ELIGIBLE_TOOLS, BackgroundTaskConfig.eligible_tools,
and _BG_POOL_TOOLS all use actual tool schema names (not category names)
that match the schemas returned by image_gen.get_tool_schemas().
"""
from __future__ import annotations

import pytest

from core.background import _DEFAULT_ELIGIBLE_TOOLS
from core.config.models import BackgroundTaskConfig
from core.execution.litellm_loop import LiteLLMExecutor


class TestBgToolNameConsistency:
    """Cross-module consistency: tool names must match actual schemas."""

    @pytest.fixture
    def image_gen_schema_names(self) -> set[str]:
        """Load actual tool schema names from image_gen module."""
        from core.tools.image_gen import get_tool_schemas
        return {s["name"] for s in get_tool_schemas()}

    def test_default_eligible_contains_all_image_gen_schemas(
        self, image_gen_schema_names: set[str],
    ):
        """_DEFAULT_ELIGIBLE_TOOLS must include every image_gen schema name."""
        for name in image_gen_schema_names:
            assert name in _DEFAULT_ELIGIBLE_TOOLS, (
                f"image_gen schema '{name}' missing from _DEFAULT_ELIGIBLE_TOOLS"
            )

    def test_bg_pool_tools_contains_all_image_gen_schemas(
        self, image_gen_schema_names: set[str],
    ):
        """_BG_POOL_TOOLS must include every image_gen schema name."""
        for name in image_gen_schema_names:
            assert name in LiteLLMExecutor._BG_POOL_TOOLS, (
                f"image_gen schema '{name}' missing from _BG_POOL_TOOLS"
            )

    def test_config_eligible_contains_all_image_gen_schemas(
        self, image_gen_schema_names: set[str],
    ):
        """BackgroundTaskConfig default eligible_tools must include all image_gen schemas."""
        config = BackgroundTaskConfig()
        for name in image_gen_schema_names:
            assert name in config.eligible_tools, (
                f"image_gen schema '{name}' missing from BackgroundTaskConfig.eligible_tools"
            )

    def test_no_category_names_in_defaults(self):
        """Old category name 'image_generation' must not appear in any tool name set."""
        assert "image_generation" not in _DEFAULT_ELIGIBLE_TOOLS
        assert "image_generation" not in LiteLLMExecutor._BG_POOL_TOOLS
        config = BackgroundTaskConfig()
        assert "image_generation" not in config.eligible_tools
