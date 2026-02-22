from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Tests for consolidation sanitization utilities.

After the consolidation refactor, the LLM one-shot pipeline methods
(daily_consolidate, weekly_integrate, _summarize_episodes, etc.) were removed.
This file retains tests for the still-present _sanitize_llm_output utility.
"""

from pathlib import Path

import pytest


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def temp_anima_dir(tmp_path: Path) -> Path:
    """Create a temporary anima directory structure."""
    anima_dir = tmp_path / "test_anima"
    (anima_dir / "episodes").mkdir(parents=True)
    (anima_dir / "knowledge").mkdir(parents=True)
    return anima_dir


@pytest.fixture
def consolidation_engine(temp_anima_dir: Path):
    """Create a ConsolidationEngine instance."""
    from core.memory.consolidation import ConsolidationEngine

    return ConsolidationEngine(
        anima_dir=temp_anima_dir,
        anima_name="test_anima",
    )


# ── Sanitize Tests ───────────────────────────────────────────


class TestSanitizeLLMOutput:
    """Test _sanitize_llm_output static method."""

    def test_strip_markdown_fence(self, consolidation_engine: object) -> None:
        """Markdown code fences are removed."""
        text = "```markdown\n# Title\n\nContent.\n```"
        result = consolidation_engine._sanitize_llm_output(text)
        assert "```" not in result
        assert "# Title" in result
        assert "Content." in result

    def test_strip_md_fence(self, consolidation_engine: object) -> None:
        """Short ``md`` fence variant is also removed."""
        text = "```md\n# Title\n\nContent.\n```"
        result = consolidation_engine._sanitize_llm_output(text)
        assert "```" not in result
        assert "# Title" in result

    def test_no_fence(self, consolidation_engine: object) -> None:
        """Content without fences is returned unchanged (modulo strip)."""
        text = "## Heading\n\nBody text."
        result = consolidation_engine._sanitize_llm_output(text)
        assert result == text

    def test_bare_fence(self, consolidation_engine: object) -> None:
        """Plain ``` fences are also handled."""
        text = "```\n## Output\n\nData.\n```"
        result = consolidation_engine._sanitize_llm_output(text)
        assert "```" not in result
        assert "## Output" in result

    def test_preserves_internal_content(self, consolidation_engine: object) -> None:
        """Only wrapper fences are stripped; mid-content dashes are kept."""
        text = "```markdown\n## Title\n\nSome content.\n```"
        result = consolidation_engine._sanitize_llm_output(text)
        assert "## Title" in result
        assert "Some content." in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
