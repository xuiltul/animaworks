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
    "vault.write_forbidden": {
        "ja": "秘密を書き換えるには、ログインした owner セッションとパスワードの再入力が必要です",
        "en": "Sign in as the owner and re-enter the password to change vault values",
        "ko": "비밀 값을 변경하려면 owner 세션으로 로그인하고 암호를 다시 입력해야 합니다",
    },
    "vault.write_setup": {
        "ja": "セットアップが完了するまで秘密を書き換えることはできません",
        "en": "Vault values cannot be changed until setup is complete",
        "ko": "설정이 완료될 때까지 비밀 값을 변경할 수 없습니다",
    },
    "vault.read_forbidden": {
        "ja": "秘密の概要を表示する権限がありません",
        "en": "You are not allowed to view the vault overview",
        "ko": "비밀 개요를 볼 권한이 없습니다",
    },
    "vault.op_forbidden": {
        "ja": "1Password の読み取りには local_trust またはログインした owner セッションが必要です",
        "en": "Reading 1Password requires local_trust or a signed-in owner session",
        "ko": "1Password를 읽으려면 local_trust 또는 로그인한 owner 세션이 필요합니다",
    },
    "vault.op_error": {
        "ja": "1Password の読み取りに失敗しました",
        "en": "Failed to read from 1Password",
        "ko": "1Password 읽기에 실패했습니다",
    },
    "vault.invalid_request": {
        "ja": "リクエストの形式または値が正しくありません",
        "en": "The request format or values are invalid",
        "ko": "요청 형식 또는 값이 올바르지 않습니다",
    },
    "vault.migrate_server_running": {
        "ja": "サーバーを停止するか、設定画面の秘密タブから移してください",
        "en": "Stop the server or migrate credentials from the Secrets tab in Settings",
        "ko": "서버를 중지하거나 설정의 비밀 탭에서 이전하세요",
    },
    "vault.migrate_failed": {
        "ja": "設定の秘密を移行できませんでした",
        "en": "Could not migrate config credentials",
        "ko": "설정 비밀 값을 이전하지 못했습니다",
    },
    "vault.managed_config": {
        "ja": "{name} は 1Password 管理です。1Password で変更してから `animaworks vault sync` を実行してください",
        "en": "{name} is managed by 1Password. Change it in 1Password, then run `animaworks vault sync`",
        "ko": "{name}는 1Password에서 관리됩니다. 1Password에서 변경한 뒤 `animaworks vault sync`를 실행하세요",
    },
}
