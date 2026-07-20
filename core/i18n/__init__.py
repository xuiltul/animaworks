# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Lightweight i18n support for runtime strings."""

from __future__ import annotations

import logging

from core.i18n.strings import _merge_strings

logger = logging.getLogger(__name__)

_STRINGS: dict[str, dict[str, str]] = _merge_strings()
_STRINGS.update(
    {
        "interactive.expired": {
            "ja": "この承認リクエストは期限切れです",
            "en": "This approval request has expired",
        },
        "interactive.not_found": {
            "ja": "この承認リクエストが見つかりません（登録に失敗した可能性があります。送信元のanimaに再送を依頼してください）",
            "en": "This approval request was not found (it may have failed to register; ask the sending anima to resend)",
        },
        "interactive.already_resolved": {
            "ja": "既に回答済みです",
            "en": "Already resolved",
        },
        "interactive.unauthorized": {
            "ja": "この操作の権限がありません",
            "en": "You are not authorized for this action",
        },
        "interactive.resolved_by": {
            "ja": "{actor} が {decision} しました",
            "en": "{actor} chose {decision}",
        },
        "interactive.comment_modal_title": {
            "ja": "コメント",
            "en": "Comment",
        },
        "interactive.comment_modal_submit": {
            "ja": "送信",
            "en": "Submit",
        },
        "interactive.comment_modal_label": {
            "ja": "コメント",
            "en": "Comment",
        },
        "interactive.comment_submitted": {
            "ja": "コメントを送信しました",
            "en": "Comment submitted.",
        },
        "interactive.error": {
            "ja": "エラーが発生しました",
            "en": "An error occurred.",
        },
        "interactive.fallback_header": {
            "ja": "回答方法:",
            "en": "How to respond:",
        },
        "interactive.fallback_instruction": {
            "ja": "番号を返信してください。",
            "en": "Reply with a number.",
        },
        "interactive.fallback_url_or": {
            "ja": "または: {url}",
            "en": "Or: {url}",
        },
    }
)


class _SafeFormatDict(dict):
    """Dict that returns ``{key}`` for missing keys during format_map."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def t(key: str, locale: str | None = None, **kwargs: object) -> str:
    """Get localized string with optional format args.

    Args:
        key: Dot-separated key (e.g. "handler.not_subordinate").
        locale: Override locale. If None, uses config.locale.
        **kwargs: Values to substitute into {placeholder} in the template.

    Returns:
        Localized string. Falls back to en, then ja, then key if not found.
    """
    from core.paths import _get_locale

    loc = locale or _get_locale()
    if not isinstance(loc, str) or loc not in ("ja", "en", "zh", "ko"):
        loc = "ja"
    entry = _STRINGS.get(key, {})
    template = entry.get(loc) or entry.get("en") or entry.get("ja", key)
    if kwargs:
        return template.format_map(_SafeFormatDict({k: str(v) for k, v in kwargs.items()}))
    return template


__all__ = ["t", "_STRINGS", "_SafeFormatDict"]
