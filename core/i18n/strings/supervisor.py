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
    "pending_executor.task_cancelled": {
        "ja": "タスクはキャンセルされました",
        "en": "Task was cancelled",
        "ko": "작업이 취소되었습니다",
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
            "実行済みの操作・成果を確認し、継続、条件待ち、取り消しを判断してください。元の入力は保存されています。"
        ),
        "en": (
            "[Task Failure]\nTask ID: {task_id}\nTask: {title}\nError: {error}\n"
            "Check existing effects and artifacts, then decide whether to resume, wait for a condition, or cancel. Original input is retained."
        ),
        "ko": (
            "[작업 실패 알림]\n작업 ID: {task_id}\n작업: {title}\n오류: {error}\n실행된 조치와 결과를 확인하고 재개, 조건 대기, 취소를 판단하세요. 원래 입력은 보존됩니다."
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
