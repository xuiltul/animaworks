# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "cascade.activity_read_failed": {
        "ja": "GlobalOutboundLimitExceeded: アクティビティログ読み取り失敗のため送信をブロックしました",
        "en": ("GlobalOutboundLimitExceeded: Sending blocked because the activity log could not be read"),
    },
    "cascade.daily_limit": {
        "ja": (
            "GlobalOutboundLimitExceeded: 24時間あたりの送信上限（{max_per_day}通）に到達しています（現在{daily_count}通/24h）。 このターンではsend_messageを使わず、送信内容をcurrent_state.mdに記録して次のセッションで送信してください。"
        ),
        "en": (
            "GlobalOutboundLimitExceeded: Daily send limit ({max_per_day} messages) reached ({daily_count} msgs/24h). Do not use send_message this turn. Record the message content in current_state.md and send it in the next session."
        ),
    },
    "cascade.hourly_limit": {
        "ja": (
            "GlobalOutboundLimitExceeded: 1時間あたりの送信上限（{max_per_hour}通）に到達しています（現在{hourly_count}通/1h, {daily_count}通/24h）。{reset_at} このターンではsend_messageを使わず、送信内容をcurrent_state.mdに記録して次のセッションで送信してください。"
        ),
        "en": (
            "GlobalOutboundLimitExceeded: Hourly send limit ({max_per_hour} messages) reached ({hourly_count} msgs/1h, {daily_count} msgs/24h).{reset_at} Do not use send_message this turn. Record the message content in current_state.md and send it in the next session."
        ),
    },
    "cascade.hourly_reset_at": {
        "ja": " 次の送信可能時刻（目安）: {reset_time}",
        "en": " Estimated next send time: {reset_time}",
    },
    "heartbeat.history_plan_entry": {
        "ja": "- {ts}: [計画] {plan}",
        "en": "- {ts}: [Plan] {plan}",
    },
    "memory.company_vision_summary_reference": {
        "ja": "（全文: {path}）",
        "en": "(Full vision: {path})",
        "ko": "(전체 비전: {path})",
    },
    "heartbeat.current_state_cleanup_required": {
        "ja": (
            "【要整理】current_state.md が {current_chars} 文字あり、整理目安 {cleanup_chars} 文字を超えています（機械トリム上限: {max_chars} 文字）。"
            "本題に入る前に current_state.md を自分で整理してください: "
            "(1) 完了・解決済み・期限切れの項目は削除する（経緯はepisodesに自動記録済みなので消してよい）、"
            "(2) 継続中の案件は1件1行まで圧縮する（見出し＋要点数行ではなく1行にする）、"
            "(3) cronの実行記録・同期カーソル・定期チェックの「差分なし」報告は current_state.md ではなく "
            "state/ 配下の専用ファイル（例: state/<タスク名>-cursor.md）に移す。"
            "整理後は {target_chars} 文字以内を目安とする。"
            "放置すると上限超過時にシステムが古い方から機械的に切り捨てるため、重要な項目が失われる恐れがあります。"
        ),
        "en": (
            "[Cleanup required] Your current_state.md is {current_chars} chars, above the {cleanup_chars}-char cleanup threshold "
            "(hard-trim limit: {max_chars} chars). "
            "Before the main task, reorganize current_state.md yourself: "
            "(1) delete completed, resolved, or expired items (their history is already auto-recorded in episodes), "
            "(2) compress each ongoing item to one line, not a heading plus several key lines, "
            "(3) move cron run records, sync cursors, and no-diff check reports out of current_state.md "
            "into dedicated files under state/ (e.g. state/<task>-cursor.md). "
            "Aim for {target_chars} chars or less after cleanup. "
            "If left as is, the system will mechanically drop the oldest content once the limit is exceeded, "
            "and important items may be lost."
        ),
        "ko": (
            "[정리 필요] current_state.md가 {current_chars}자이며 정리 기준 {cleanup_chars}자를 초과했습니다 "
            "(기계적 정리 상한: {max_chars}자). "
            "주 작업 전에 current_state.md를 직접 정리하세요: "
            "(1) 완료·해결·만료된 항목은 삭제하세요(이력은 episodes에 자동 기록되어 있습니다). "
            "(2) 진행 중인 각 건은 제목과 여러 핵심 줄이 아니라 한 건당 한 줄로 압축하세요. "
            "(3) cron 실행 기록, 동기화 커서, 변경 없음 보고는 current_state.md가 아닌 state/ 아래 전용 파일 "
            "(예: state/<task>-cursor.md)로 옮기세요. 정리 후 {target_chars}자 이내를 목표로 하세요. "
            "방치하면 상한 초과 시 시스템이 오래된 내용부터 기계적으로 잘라 중요한 항목이 사라질 수 있습니다."
        ),
    },
    "heartbeat.heartbeat_md_cleanup_required": {
        "ja": (
            "【要圧縮】heartbeat.md が {current_kb}KB あり、上限 {max_kb}KB を超えています。"
            "このファイルは毎回の heartbeat プロンプトに丸ごと読み込まれるため、肥大すると毎回の巡回が重くなり、"
            "恒常の手順が経緯の中に埋もれます。本題に入る前に heartbeat.md を自分で書き直してください: "
            "(1) 特定の PR 番号・日付に紐づく経緯や事例、終わった案件のゲート・観測項目は削除する"
            "（経緯は episodes と knowledge に残っている）、"
            "(2) 同じルールの重複は 1 か所にまとめる、"
            "(3) 毎回の巡回で使う判断と手順の要点だけを残す。詳しいコマンドや手順は procedures/ 配下のファイルに移し、"
            "heartbeat.md にはその参照1行を残す、"
            "(4) 「## 活動時間」「## 通知ルール」セクションは変えない。"
            "{archive_notice}"
            '書き直し後は {target_kb}KB 以内を目安とし、write_memory_file(path="heartbeat.md", mode="overwrite") で保存する。'
        ),
        "en": (
            "[Compaction required] Your heartbeat.md is {current_kb}KB, over the {max_kb}KB limit. "
            "This file is loaded in full into every heartbeat prompt, so bloat makes every run heavier "
            "and buries the recurring steps under case history. Before the main task, rewrite heartbeat.md yourself: "
            "(1) delete history and examples tied to specific PR numbers or dates, and gates/observation items "
            "for finished cases (their history remains in episodes and knowledge), "
            "(2) merge duplicated rules into one place, "
            "(3) keep only the decision and procedure essentials used on every run. Move detailed commands and procedures "
            "to files under procedures/ and leave a one-line pointer in heartbeat.md, "
            "(4) leave the '## 活動時間' and '## 通知ルール' sections unchanged. "
            "{archive_notice} "
            'Aim for {target_kb}KB or less and save with write_memory_file(path="heartbeat.md", mode="overwrite").'
        ),
        "ko": (
            "[압축 필요] heartbeat.md가 {current_kb}KB이며 상한 {max_kb}KB를 초과했습니다. "
            "이 파일은 매 heartbeat 프롬프트에 전체 포함되므로 비대해지면 매번 실행이 무거워지고 반복 절차가 이력에 묻힙니다. "
            "주 작업 전에 heartbeat.md를 직접 다시 작성하세요: "
            "(1) 특정 PR 번호·날짜에 연결된 이력·사례와 완료된 건의 게이트·관찰 항목은 삭제하세요 "
            "(이력은 episodes와 knowledge에 남아 있습니다). (2) 중복 규칙은 한곳으로 합치세요. "
            "(3) 매번 순회에서 사용하는 판단과 절차의 핵심만 남기세요. 자세한 명령과 절차는 procedures/ 아래 파일로 옮기고 "
            "heartbeat.md에는 해당 참조를 한 줄 남기세요. "
            "(4) '## 活動時間' 및 '## 通知ルール' 섹션은 변경하지 마세요. {archive_notice} "
            '정리 후 {target_kb}KB 이내를 목표로 하고 write_memory_file(path="heartbeat.md", mode="overwrite")로 저장하세요.'
        ),
    },
    "heartbeat.heartbeat_md_archive_notice": {
        "ja": "書き換え前の版は {path} に退避済みです。",
        "en": "The pre-edit version has been archived at {path}.",
        "ko": "수정 전 버전은 {path}에 보관했습니다.",
    },
    "scheduler.cron_fallback_description": {
        "ja": "cron.mdの「{task_name}」の指示に従って処理してください。",
        "en": "Follow the instructions for '{task_name}' in cron.md.",
    },
    "governor.supervisor_notify": {
        "ja": "[Governor] {anima} をクォータ超過により一時停止しました。理由: {reason}",
        "en": "[Governor] {anima} has been suspended due to quota limit. Reason: {reason}",
    },
    "governor.human_notify": {
        "ja": "Governor: {anima} がクォータ超過で停止されました。理由: {reason}",
        "en": "Governor: {anima} suspended due to quota. Reason: {reason}",
    },
    "governor.human_notify_subject": {
        "ja": "Governor アラート",
        "en": "Governor Alert",
    },
}
