# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "pending_executor.dep_result_header": {
        "ja": "## 先行タスク [{dep_id}] の結果",
        "en": "## Preceding task [{dep_id}] result",
    },
    "pending_executor.none_value": {
        "ja": "(なし)",
        "en": "(none)",
    },
    "pending_executor.task_completed": {
        "ja": "(タスク完了)",
        "en": "(task completed)",
    },
    "pending_executor.task_exec_end": {
        "ja": "タスク完了: {title} — {result}",
        "en": "Task completed: {title} — {result}",
    },
    "pending_executor.task_exec_start": {
        "ja": "タスク実行開始: {title}",
        "en": "Task execution started: {title}",
    },
    "pending_executor.model_override": {
        "ja": "タスクのモデル上書き: {requested} で実行（解決: {resolved}）",
        "en": "Task model override: running with {requested} (resolved: {resolved})",
        "ko": "태스크 모델 오버라이드: {requested}로 실행 (해결: {resolved})",
    },
    "pending_executor.task_fail_notify": {
        "ja": (
            "[タスク失敗通知]\nタスクID: {task_id}\nタスク: {title}\nエラー: {error}\n"
            "必要なら再委譲を判断してください。"
        ),
        "en": (
            "[Task Failure]\nTask ID: {task_id}\nTask: {title}\nError: {error}\n"
            "Decide whether to re-delegate if needed."
        ),
        "ko": (
            "[작업 실패 알림]\n작업 ID: {task_id}\n작업: {title}\n오류: {error}\n필요한 경우 재위임 여부를 판단하세요."
        ),
    },
    "pending_executor.workspace_not_specified": {
        "ja": "(指定なし)",
        "en": "(not specified)",
    },
    "supervisor.zombie_reaped": {
        "ja": "zombie reaper: {count}個の子プロセスを回収しました",
        "en": "zombie reaper: reaped {count} child process(es)",
    },
}
