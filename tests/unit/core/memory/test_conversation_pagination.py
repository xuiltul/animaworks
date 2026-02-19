# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for conversation history pagination.

Validates the offset/limit pagination logic used by the
``/api/animas/{name}/conversation/full`` endpoint.  The slicing
algorithm is:

    total = len(state.turns)
    end   = max(0, total - offset)
    start = max(0, end - limit)
    paginated = state.turns[start:end]

This returns the *latest* turns first (offset=0), and paging
backward retrieves progressively older turns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.memory.conversation import ConversationMemory, ConversationTurn
from core.schemas import ModelConfig


@pytest.fixture
def conv_mem(tmp_path):
    """Create a ConversationMemory with test data."""
    anima_dir = tmp_path / "animas" / "test-anima"
    anima_dir.mkdir(parents=True)
    (anima_dir / "state").mkdir()
    (anima_dir / "transcripts").mkdir()

    model_config = ModelConfig(model="claude-sonnet-4-20250514")
    return ConversationMemory(anima_dir, model_config)


def _paginate(turns, limit, offset):
    """Apply the same pagination logic used in memory_routes.py."""
    total = len(turns)
    end = max(0, total - offset)
    start = max(0, end - limit)
    return turns[start:end]


class TestConversationPagination:
    """Test offset/limit pagination of conversation turns."""

    def _populate_turns(self, conv_mem, count):
        """Add *count* turns to conversation memory and persist."""
        state = conv_mem.load()
        for i in range(count):
            state.turns.append(
                ConversationTurn(
                    role="human" if i % 2 == 0 else "assistant",
                    content=f"Message {i}",
                    timestamp=f"2026-02-17T10:00:{i:02d}",
                    token_estimate=10,
                )
            )
        conv_mem.save()
        return state

    def test_load_latest_page(self, conv_mem):
        """offset=0 returns the latest turns."""
        self._populate_turns(conv_mem, 50)
        state = conv_mem.load()

        paginated = _paginate(state.turns, limit=20, offset=0)

        assert len(paginated) == 20
        assert paginated[0].content == "Message 30"
        assert paginated[-1].content == "Message 49"

    def test_load_second_page(self, conv_mem):
        """offset=20 returns the next older batch."""
        self._populate_turns(conv_mem, 50)
        state = conv_mem.load()

        paginated = _paginate(state.turns, limit=20, offset=20)

        assert len(paginated) == 20
        assert paginated[0].content == "Message 10"
        assert paginated[-1].content == "Message 29"

    def test_load_last_page_partial(self, conv_mem):
        """Last page may have fewer items than limit."""
        self._populate_turns(conv_mem, 50)
        state = conv_mem.load()

        paginated = _paginate(state.turns, limit=20, offset=40)

        assert len(paginated) == 10
        assert paginated[0].content == "Message 0"
        assert paginated[-1].content == "Message 9"

    def test_offset_beyond_total_returns_empty(self, conv_mem):
        """Offset == total returns empty list."""
        self._populate_turns(conv_mem, 10)
        state = conv_mem.load()

        paginated = _paginate(state.turns, limit=20, offset=10)

        assert len(paginated) == 0

    def test_offset_exceeds_total_returns_empty(self, conv_mem):
        """Offset > total returns empty list."""
        self._populate_turns(conv_mem, 10)
        state = conv_mem.load()

        paginated = _paginate(state.turns, limit=20, offset=100)

        assert len(paginated) == 0

    def test_small_conversation_fits_in_one_page(self, conv_mem):
        """When total < limit, all turns are returned."""
        self._populate_turns(conv_mem, 5)
        state = conv_mem.load()

        paginated = _paginate(state.turns, limit=20, offset=0)

        assert len(paginated) == 5
        assert paginated[0].content == "Message 0"
        assert paginated[-1].content == "Message 4"

    def test_empty_conversation(self, conv_mem):
        """Empty conversation returns empty list."""
        state = conv_mem.load()

        paginated = _paginate(state.turns, limit=20, offset=0)

        assert len(paginated) == 0

    def test_continuity_across_pages(self, conv_mem):
        """Pages are continuous: page2 last message is adjacent to page1 first."""
        self._populate_turns(conv_mem, 100)
        state = conv_mem.load()

        # Page 1: latest 30
        page1 = _paginate(state.turns, limit=30, offset=0)

        # Page 2: next 30
        page2 = _paginate(state.turns, limit=30, offset=30)

        # page2 last message should be adjacent to page1 first message
        assert (
            int(page2[-1].content.split()[-1]) + 1
            == int(page1[0].content.split()[-1])
        )

    def test_limit_one(self, conv_mem):
        """limit=1 returns exactly one turn per page."""
        self._populate_turns(conv_mem, 5)
        state = conv_mem.load()

        page0 = _paginate(state.turns, limit=1, offset=0)
        page1 = _paginate(state.turns, limit=1, offset=1)
        page4 = _paginate(state.turns, limit=1, offset=4)

        assert len(page0) == 1
        assert page0[0].content == "Message 4"
        assert len(page1) == 1
        assert page1[0].content == "Message 3"
        assert len(page4) == 1
        assert page4[0].content == "Message 0"

    def test_full_traversal_collects_all_turns(self, conv_mem):
        """Iterating through all pages collects every turn exactly once."""
        self._populate_turns(conv_mem, 37)
        state = conv_mem.load()

        all_contents = []
        offset = 0
        limit = 10
        while True:
            page = _paginate(state.turns, limit=limit, offset=offset)
            if not page:
                break
            all_contents = [t.content for t in page] + all_contents
            offset += limit

        assert len(all_contents) == 37
        for i, content in enumerate(all_contents):
            assert content == f"Message {i}"
