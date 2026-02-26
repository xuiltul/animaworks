# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Japanese-aware sentence splitting for streaming TTS."""

from __future__ import annotations

import re

# ── Split pattern ──────────────────────────────────────────────

_SPLIT_PATTERN = re.compile(r"(?<=[。！？\!\?\n])")


# ── Batch splitter ──────────────────────────────────────────────


def split_sentences(text: str) -> list[str]:
    """Split text into sentences at Japanese/English punctuation.

    Args:
        text: Input text.

    Returns:
        List of non-empty sentence strings.
    """
    parts = _SPLIT_PATTERN.split(text)
    return [p.strip() for p in parts if p.strip()]


# ── Streaming splitter ──────────────────────────────────────────


class StreamingSentenceSplitter:
    """Accumulate text_delta chunks and yield complete sentences."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        """Feed text chunk, return list of complete sentences (may be empty).

        Args:
            text: Incoming text delta.

        Returns:
            List of complete sentences extracted so far.
        """
        self._buffer += text
        sentences: list[str] = []
        while True:
            match = _SPLIT_PATTERN.search(self._buffer)
            if match:
                sentence = self._buffer[: match.end()].strip()
                self._buffer = self._buffer[match.end() :]
                if sentence:
                    sentences.append(sentence)
            else:
                break
        return sentences

    def flush(self) -> str | None:
        """Flush remaining buffer as final sentence.

        Returns:
            Remaining text as sentence, or None if buffer empty.
        """
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining or None
