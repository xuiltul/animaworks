# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings (legacy route modules)."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "anima.status_json_invalid": {
        "ja": "Anima '{name}' の status.json が壊れているか読み取れないため、操作を中止しました。",
        "en": "Anima '{name}' status.json is invalid or unreadable; the operation was aborted.",
    },
    "anima.bootstrap_complete_requires_identity": {
        "ja": "Anima '{name}' の人格（IDENTITY.md / AGENTS.md）がまだ書かれていないため、完了扱いにできません。「やり直す」を選んでください。",
        "en": "Anima '{name}' has no IDENTITY.md / AGENTS.md yet, so setup cannot be marked complete. Choose retry instead.",
    },
}
