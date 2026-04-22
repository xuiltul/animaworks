# Copyright 2026 AnimaWorks
# Licensed under the Apache License, Version 2.0
from __future__ import annotations


def __getattr__(name: str):  # noqa: N807
    """Lazy-import FactExtractor so the package is importable before extractor.py exists."""
    if name == "FactExtractor":
        from core.memory.extraction.extractor import FactExtractor

        return FactExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FactExtractor"]
