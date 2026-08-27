# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "pending_executor.continuation_checkpoint": {
        "ja": (
            "## 前回実行のcheckpoint（自動継続 {count}回目）\n"
            "このタスクは前回セッションを正常完了と判定できなかったため自動継続されている。\n"
            "終了種別: {stop_kind}\n"
            "以下は前回セッションの末尾出力と実行済みツールの記録である。実施済みの操作を\n"
            "重複実行せず、残作業を完了させ、完了時は必ず "
            '`update_task(status="done", result="...")` を呼ぶこと。\n'
            "外部要因（権限不足・依存待ち・環境障害など）で進められない場合は、同じ操作を"
            '繰り返さず `update_task(status="blocked", summary="障害内容")` を宣言し、可能な限り '
            "`unblock_check` を添えて"
            "停止すること。\n"
            "### 前回の出力（末尾）\n{output}\n"
            "### 実行済みツール（最大30件）\n{records}"
        ),
        "en": (
            "## Previous execution checkpoint (automatic continuation {count})\n"
            "The previous session could not be considered complete, so this task was continued automatically.\n"
            "Stop kind: {stop_kind}\n"
            "The previous session's final output and executed tools follow. Do not repeat completed operations; "
            'finish the remaining work and call `update_task(status="done", result="...")` when done.\n'
            "If an external factor (missing permission, dependency, environment failure) prevents progress, "
            'do not repeat the same operations; declare `update_task(status="blocked", summary="<blocker>")`, '
            "including `unblock_check` whenever possible, and stop.\n"
            "### Previous output (tail)\n{output}\n"
            "### Executed tools (up to 30)\n{records}"
        ),
        "ko": (
            "## 이전 실행 checkpoint(자동 계속 {count}회차)\n"
            "이전 세션을 정상 완료로 판단할 수 없어 이 작업이 자동으로 계속되었습니다.\n"
            "종료 유형: {stop_kind}\n"
            "이전 세션의 마지막 출력과 실행한 도구 기록입니다. 완료한 작업을 반복하지 말고 남은 작업을 끝낸 뒤 "
            '`update_task(status="done", result="...")`를 호출하세요.\n'
            "외부 요인(권한 부족, 의존성 대기, 환경 장애 등)으로 진행할 수 없으면 같은 작업을 반복하지 말고 "
            '`update_task(status="blocked", summary="<장애 내용>")`을 선언하고 가능하면 `unblock_check`를 첨부한 뒤 중지하세요.\n'
            "### 이전 출력(마지막 부분)\n{output}\n"
            "### 실행한 도구(최대 30개)\n{records}"
        ),
    },
    "pending_executor.crash_checkpoint": {
        "ja": (
            "## 前回実行のcheckpoint（自動継続 {count}回目）\n"
            "前回はプロセス異常終了のため自動継続されている。実施済みの操作を確認し、残作業を完了させること。"
        ),
        "en": (
            "## Previous execution checkpoint (automatic continuation {count})\n"
            "The previous process terminated unexpectedly. Review completed operations and finish the remaining work."
        ),
        "ko": (
            "## 이전 실행 checkpoint(자동 계속 {count}회차)\n"
            "이전 프로세스가 비정상 종료되었습니다. 완료한 작업을 확인하고 남은 작업을 끝내세요."
        ),
    },
    "pending_executor.declaration_probe": {
        "ja": (
            "タスク {task_id} のセッションが完了宣言なしで終了しました。新しい作業は一切せず、"
            "現状に応じて今すぐ update_task を1回だけ呼んでください。完遂済みなら "
            'status="done" と result、外部要因で進められないなら status="blocked" と summary（可能な限り '
            "unblock_check も指定）、"
            'バックグラウンド処理を待っているだけなら status="in_progress" と '
            'summary="[待機] <何を待っているか>" を指定してください。'
        ),
        "en": (
            "Task {task_id} ended without a completion declaration. Do no new work. Call update_task exactly once "
            'now: use status="done" with result if complete, status="blocked" with summary and, whenever possible, '
            "unblock_check if an external blocker "
            'prevents progress, or status="in_progress" with summary="[waiting] <what you are waiting for>" if only '
            "waiting for background work."
        ),
        "ko": (
            "작업 {task_id} 세션이 완료 선언 없이 종료되었습니다. 새로운 작업은 하지 말고 지금 update_task를 "
            '정확히 한 번 호출하세요. 완료했다면 status="done"과 result, 외부 요인으로 진행할 수 없다면 '
            'status="blocked"와 summary(가능하면 unblock_check도 지정), 백그라운드 작업을 기다리는 중이라면 status="in_progress"와 '
            'summary="[대기] <기다리는 대상>"을 지정하세요.'
        ),
    },
    "pending_executor.recovered_checkpoint": {
        "ja": (
            "## プロセス異常終了から回収したcheckpoint\n"
            "### 前回の出力（末尾）\n{output}\n"
            "### 実行済みツール（最大30件）\n{records}"
        ),
        "en": (
            "## Checkpoint recovered after an unexpected process termination\n"
            "### Previous output (tail)\n{output}\n"
            "### Executed tools (up to 30)\n{records}"
        ),
        "ko": (
            "## 프로세스 비정상 종료에서 복구한 checkpoint\n"
            "### 이전 출력(마지막 부분)\n{output}\n"
            "### 실행한 도구(최대 30개)\n{records}"
        ),
    },
    "pending_executor.dep_result_header": {
        "ja": "## 先行タスク [{dep_id}] の結果",
        "en": "## Preceding task [{dep_id}] result",
    },
    "pending_executor.machine_directive": {
        "ja": (
            "🔴 MUST: machineツール使用が指定されたタスクです。\n5ステップ以上の重い処理は animaworks-tool machine run で外部エージェントに必ず委託し、\n出力を検証の上、不十分なら再度machineで修正してください。"
        ),
        "en": (
            "🔴 MUST: This task specifies the use of the machine tool.\nDelegate heavy work (5+ steps) to an external agent via animaworks-tool machine run.\nVerify the output and re-run machine if the result is insufficient."
        ),
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
    "blocked_recovery.reprobe_instruction": {
        "ja": (
            "blockerが解消済みか確認し、解消なら続行、未解消ならblocked宣言し直すこと。"
            "可能ならunblock_checkを添えること。"
        ),
        "en": (
            "Check whether the blocker is resolved: continue if it is, otherwise declare blocked again. "
            "Attach an unblock_check whenever possible."
        ),
        "ko": (
            "blocker가 해소되었는지 확인하고, 해소되었으면 계속 진행하고 아니면 다시 blocked를 선언하세요. "
            "가능하면 unblock_check를 함께 제시하세요."
        ),
    },
    "blocked_recovery.stale_check_instruction": {
        "ja": (
            "unblock_check が長時間失敗し続けている。まず自分で立て直すこと: "
            "(1) checkが陳腐化していないか（当時のexact SHAを固定している等）を現状と照合する "
            "(2) タスク自体が現行headに追い越されて無意味なら done/cancelled に落とす "
            "(3) 継続が必要なら、現状で本当に成立しうる unblock_check を付け直して blocked を宣言し直す。"
            "同じcheckをそのまま付け直さないこと。"
        ),
        "en": (
            "The unblock_check has been failing for a long time. Fix it yourself first: "
            "(1) compare the check against reality (does it pin an exact SHA that has moved on?), "
            "(2) close it as done/cancelled if the current head made the task obsolete, "
            "(3) if it must continue, re-declare blocked with a check that can actually pass now. "
            "Do not re-attach the same check unchanged."
        ),
        "ko": (
            "unblock_check가 오랫동안 계속 실패하고 있습니다. 먼저 스스로 정리하세요: "
            "(1) check가 낡지 않았는지(당시의 exact SHA 고정 등) 현재 상태와 대조하고, "
            "(2) 현재 head에 밀려 무의미해졌다면 done/cancelled로 종료하고, "
            "(3) 계속해야 한다면 지금 실제로 통과할 수 있는 unblock_check로 다시 blocked를 선언하세요. "
            "같은 check를 그대로 다시 붙이지 마세요."
        ),
    },
    "blocked_recovery.manual_intervention_instruction": {
        "ja": (
            "タスク {task_id}（{anima_name}）は unblock_check を持たないため自動再開されない。"
            "人手で確認し、完了なら done/cancelled に、継続なら unblock_check を付けて "
            "blocked を宣言し直すこと。\n\n{original_instruction}"
        ),
        "en": (
            "Task {task_id} ({anima_name}) has no unblock_check, so it will not resume automatically. "
            "Check it by hand: move it to done/cancelled if finished, or re-declare blocked with an "
            "unblock_check attached if it continues.\n\n{original_instruction}"
        ),
        "ko": (
            "작업 {task_id}({anima_name})은 unblock_check가 없어 자동으로 재개되지 않습니다. "
            "직접 확인하여 완료되었으면 done/cancelled로 옮기고, 계속한다면 unblock_check를 붙여 "
            "blocked를 다시 선언하세요.\n\n{original_instruction}"
        ),
    },
    "pending_executor.descriptor_recovery_suffix": {
        "ja": "descriptor消失からの自動復旧。タスクの実態を確認して続行すること。",
        "en": "Automatic recovery from a lost descriptor. Verify the task's actual state before continuing.",
        "ko": "descriptor 소실로부터의 자동 복구입니다. 작업의 실제 상태를 확인한 뒤 계속하세요.",
    },
    "supervisor.zombie_reaped": {
        "ja": "zombie reaper: {count}個の子プロセスを回収しました",
        "en": "zombie reaper: reaped {count} child process(es)",
    },
}
