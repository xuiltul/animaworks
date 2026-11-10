# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings for vault and 1Password operations."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "vault.operator_only": {
        "ja": "{command} はオペレーター専用です（ANIMAWORKS_ANIMA_DIR が設定されています）",
        "en": "{command} is operator-only (ANIMAWORKS_ANIMA_DIR is set)",
        "ko": "{command}은(는) 운영자 전용입니다 (ANIMAWORKS_ANIMA_DIR가 설정되어 있습니다)",
    },
    "vault.managed_store": {
        "ja": "{name} は 1Password 管理です。1Password で変更してから `animaworks vault sync` を実行してください",
        "en": "{name} is managed by 1Password. Change it in 1Password, then run `animaworks vault sync`",
        "ko": "{name}는 1Password에서 관리됩니다. 1Password에서 변경한 뒤 `animaworks vault sync`를 실행하세요",
    },
    "vault.managed_delete": {
        "ja": "{name} は 1Password 管理です。1Password 側で削除してから `animaworks vault sync` を実行してください",
        "en": "{name} is managed by 1Password. Delete it in 1Password, then run `animaworks vault sync`",
        "ko": "{name}는 1Password에서 관리됩니다. 1Password에서 삭제한 뒤 `animaworks vault sync`를 실행하세요",
    },
    "vault.non_tty_pending": {
        "ja": "変更がありますが標準入力が TTY ではないため書き込みません。適用するには --yes を付けてください",
        "en": "Changes are pending, but stdin is not a TTY so nothing was written; pass --yes to apply them",
        "ko": "변경 사항이 있지만 표준 입력이 TTY가 아니어서 쓰지 않았습니다. 적용하려면 --yes를 지정하세요",
    },
    "vault.sync_confirm": {
        "ja": "{count} 件の変更を shared に適用しますか？ [y/N] ",
        "en": "Apply {count} change(s) to shared? [y/N] ",
        "ko": "shared에 {count}건의 변경을 적용할까요? [y/N] ",
    },
    "vault.sync_cancelled": {
        "ja": "変更は適用しませんでした",
        "en": "No changes were applied",
        "ko": "변경 사항을 적용하지 않았습니다",
    },
    "vault.op_not_found": {
        "ja": "1Password CLI が見つかりません: {command}",
        "en": "1Password CLI not found: {command}",
        "ko": "1Password CLI를 찾을 수 없습니다: {command}",
    },
    "vault.rollback_missing": {
        "ja": "ロールバックできる previous がありません: {name}",
        "en": "No previous value is available to roll back: {name}",
        "ko": "롤백할 previous 값이 없습니다: {name}",
    },
    "vault.sources_empty": {
        "ja": "登録された 1Password 参照はありません",
        "en": "No 1Password references are registered",
        "ko": "등록된 1Password 참조가 없습니다",
    },
}
