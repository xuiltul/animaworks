from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract ``@name`` mentions from Board text."""

import re
from collections.abc import Iterable

_MENTION_RE = re.compile(r"@([a-z0-9_-]+)", re.IGNORECASE | re.ASCII)


def extract_mentions(text: str, known_names: Iterable[str]) -> set[str]:
    """Return known names mentioned in *text*, preserving their canonical spelling."""
    canonical_names = {name.lower(): name for name in known_names}
    mentions: set[str] = set()

    for match in _MENTION_RE.finditer(text):
        token = match.group(1).lower()
        if token == "all":
            mentions.add("all")
            continue

        name = canonical_names.get(token)
        if name is None:
            name = canonical_names.get(token.rstrip("-_"))
        if name is not None:
            mentions.add(name)

    return mentions
