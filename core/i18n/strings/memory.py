# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "rag.rebuild_symlink_input": {
        "ja": "再構築の入力にシンボリックリンクは使用できません: {path}",
        "en": "A symlink is not a safe rebuild input: {path}",
        "ko": "재구축 입력에 심볼릭 링크를 사용할 수 없습니다: {path}",
    },
    "rag.rebuild_input_changed": {
        "ja": "RAG再構築の入力が変更されたため、古いDBへの切り替えを中止します。",
        "en": "RAG rebuild inputs changed; refusing stale database promotion.",
        "ko": "RAG 재구축 입력이 변경되어 오래된 DB로의 전환을 중단합니다.",
    },
    "rag.rebuild_invalid_metadata": {
        "ja": "再構築した索引メタデータが不正です。",
        "en": "Invalid rebuilt index metadata.",
        "ko": "재구축한 색인 메타데이터가 올바르지 않습니다.",
    },
    "rag.rebuild_invalid_manifest": {
        "ja": "RAG再構築の入力マニフェストが不正です。",
        "en": "Invalid RAG rebuild source manifest.",
        "ko": "RAG 재구축 입력 매니페스트가 올바르지 않습니다.",
    },
    "rag.phase3_repair_requires_shared": {
        "ja": "phase3の全DB再構築にはinclude_shared=Trueが必要です。共有記憶を除外すると既存の共有索引が失われます。",
        "en": "Phase3 full-DB repair requires include_shared=True to preserve shared memory collections.",
        "ko": "공유 메모리 컬렉션을 보존하려면 phase3 전체 DB 재구축에 include_shared=True가 필요합니다.",
    },
    "rag.signature_unknown_shape": {
        "ja": "埋め込み索引の署名が不明です（メタデータがオブジェクトではありません）。",
        "en": "Embedding index signature is unknown (metadata is not an object).",
        "ko": "임베딩 색인 서명을 알 수 없습니다(메타데이터가 객체가 아님).",
    },
    "rag.signature_unknown_fields": {
        "ja": "埋め込み索引の署名が不明です（モデル・prefixの履歴が未記録または不正です）。",
        "en": "Embedding index signature is unknown (model/prefix provenance is missing or invalid).",
        "ko": "임베딩 색인 서명을 알 수 없습니다(모델/prefix 이력이 없거나 잘못됨).",
    },
    "rag.signature_unreadable": {
        "ja": "埋め込み索引の署名が不明です（メタデータを読み取れません）。",
        "en": "Embedding index signature is unknown (metadata cannot be read).",
        "ko": "임베딩 색인 서명을 알 수 없습니다(메타데이터를 읽을 수 없음).",
    },
    "rag.signature_model_changed": {
        "ja": "埋め込みモデルが変更されています: {previous} -> {current}。",
        "en": "Embedding model changed: {previous} -> {current}.",
        "ko": "임베딩 모델이 변경되었습니다: {previous} -> {current}.",
    },
    "rag.signature_prefix_changed": {
        "ja": "埋め込みE5 prefix設定が変更されています: {previous} -> {current}。",
        "en": "Embedding E5 prefix setting changed: {previous} -> {current}.",
        "ko": "임베딩 E5 prefix 설정이 변경되었습니다: {previous} -> {current}.",
    },
    "rag.indexing_blocked": {
        "ja": "{reason} 索引を更新する前に履歴を検証するか、バックアップ付きの全再構築を実施してください。",
        "en": "{reason} Verify provenance or perform a backed-up full rebuild before indexing.",
        "ko": "{reason} 색인을 갱신하기 전에 이력을 검증하거나 백업 후 전체 재구축하세요.",
    },
    "rag.daily_indexing_blocked": {
        "ja": "{reason} 日次索引更新をスキップします。履歴の検証またはバックアップ付き全再構築が必要です。",
        "en": "{reason} Skipping daily indexing — verify provenance or perform a backed-up full rebuild.",
        "ko": "{reason} 일일 색인 갱신을 건너뜁니다. 이력 검증 또는 백업 후 전체 재구축이 필요합니다.",
    },
    "rag.invalid_anima_name": {
        "ja": "Anima名が不正です。",
        "en": "Invalid Anima name",
        "ko": "Anima 이름이 잘못되었습니다.",
    },
    "rag.root_unavailable": {
        "ja": "ルート記憶サービスを利用できません。",
        "en": "Root memory service unavailable",
        "ko": "루트 메모리 서비스를 사용할 수 없습니다.",
    },
    "rag.root_operation_failed": {
        "ja": "ルート記憶操作に失敗しました。",
        "en": "Root memory operation failed",
        "ko": "루트 메모리 작업에 실패했습니다.",
    },
    "rag.worker_operation_disabled": {
        "ja": "phase3 Animaのvector worker操作は禁止されています: {anima}",
        "en": "Vector worker operation disabled for phase3 anima: {anima}",
        "ko": "phase3 Anima의 vector worker 작업이 금지되어 있습니다: {anima}",
    },
    "conversation.activity_context_header": {
        "ja": "## セッション中のその他の活動",
        "en": "## Other activity during session",
    },
    "conversation.ellipsis_omitted": {
        "ja": "...(前半省略)...",
        "en": "...(earlier omitted)...",
    },
    "conversation.existing_summary_header": {
        "ja": "## 既存の要約",
        "en": "## Existing summary",
    },
    "conversation.history_summary_header": {
        "ja": "### 会話の要約（{count}ターン分）",
        "en": "### Conversation summary ({count} turns)",
    },
    "conversation.integrate_instruction": {
        "ja": "上記を統合した新しい要約を作成してください。",
        "en": "Please create a new integrated summary of the above.",
    },
    "conversation.new_task_marker": {
        "ja": "- [ ] {task}（自動検出: {ts}）",
        "en": "- [ ] {task} (auto-detected: {ts})",
    },
    "conversation.new_turns_header": {
        "ja": "## 新しい会話ターン",
        "en": "## New conversation turns",
    },
    "conversation.pruned_auto_detected_header": {
        "ja": "## 自動検出タスク（current_state.mdから退避）",
        "en": "## Auto-detected tasks (pruned from current_state.md)",
    },
    "conversation.recent_conversation_header": {
        "ja": "### 直近の会話",
        "en": "### Recent conversation",
    },
    "conversation.resolution_summary": {
        "ja": "解決済み: {item}",
        "en": "Resolved: {item}",
    },
    "conversation.resolved_marker": {
        "ja": "- ✅ {item}（自動検出: {ts}）",
        "en": "- ✅ {item} (auto-detected: {ts})",
    },
    "conversation.role_you": {
        "ja": "あなた",
        "en": "You",
    },
    "conversation.summary_ack": {
        "ja": "承知しました。これまでの会話内容を把握しました。",
        "en": "Understood. I have grasped the conversation so far.",
    },
    "conversation.summary_label": {
        "ja": "[会話の要約（{count}ターン分）]",
        "en": "[Conversation summary ({count} turns)]",
    },
    "conversation.title_fallback": {
        "ja": "会話",
        "en": "Conversation",
    },
    "conversation.tools_executed": {
        "ja": "[実行ツール: {tool_names}]",
        "en": "[Tools used: {tool_names}]",
    },
    "conversation.tools_used": {
        "ja": "[使用ツール: {tools}]",
        "en": "[Tools used: {tools}]",
    },
    "conversation.truncated_suffix": {
        "ja": ("\n[...truncated, original {length} chars]"),
        "en": ("\n[...truncated, original {length} chars]"),
    },
    # "dedup.messages_merged" removed: consolidate_messages() abolished in dedup overhaul
    "dedup.overflow_inbox_summary": {
        "ja": (
            "⚠️ 未処理メッセージ {count}件 (state/overflow_inbox/): "
            "{listing}{remaining}\n"
            'read_memory_file(path="state/overflow_inbox/<filename>") で確認可能。'
            "処理後は archive_memory_file で移動してください。"
        ),
        "en": (
            "⚠️ {count} unprocessed messages (state/overflow_inbox/): "
            "{listing}{remaining}\n"
            'Use read_memory_file(path="state/overflow_inbox/<filename>") to review. '
            "After processing, use archive_memory_file to move them."
        ),
    },
    "distillation.none": {
        "ja": "(なし)",
        "en": "(none)",
    },
    "distillation.pattern_n_repeat": {
        "ja": "### パターン {i} ({count}回繰り返し)",
        "en": "### Pattern {i} (repeated {count} times)",
    },
    "manager.action_log_header": {
        "ja": ("# {date} 行動ログ\n\n"),
        "en": ("# {date} Action log\n\n"),
    },
    "priming.about_sender": {
        "ja": "### {sender_name} について",
        "en": "### About {sender_name}",
    },
    "priming.active_parallel_tasks_header": {
        "ja": "## 実行中の並列タスク",
        "en": "## Active Parallel Tasks",
    },
    "priming.completed_bg_tasks_header": {
        "ja": "## 完了済みバックグラウンドタスク",
        "en": "## Completed Background Tasks",
    },
    "priming.episodes_header": {
        "ja": "### 関連する過去の経験",
        "en": "### Related Past Experiences",
    },
    "priming.outbound_header": {
        "ja": "## 直近のアウトバウンド行動",
        "en": "## Recent Outbound Actions",
    },
    "priming.outbound_posted": {
        "ja": "- [{time_str}] #{ch} に投稿済み: 「{text_preview}」",
        "en": '- [{time_str}] Posted to #{ch}: "{text_preview}"',
    },
    "priming.outbound_sent": {
        "ja": "- [{time_str}] {to} にメッセージ送信済み: 「{text_preview}」",
        "en": '- [{time_str}] Message sent to {to}: "{text_preview}"',
    },
    "priming.pending_tasks_header": {
        "ja": "### 未完了タスク",
        "en": "### Pending Tasks",
    },
    "priming.recent_activity_header": {
        "ja": "### 直近のアクティビティ",
        "en": "### Recent Activity",
    },
    "priming.related_knowledge_header": {
        "ja": "### 関連する知識",
        "en": "### Related Knowledge",
    },
    "priming.search_before_action": {
        "ja": "外部アクションを行う前に、search_memory で根拠を確認してください。",
        "en": "Before taking any external action, verify the basis with search_memory.",
    },
    "priming.section_intro": {
        "ja": "以下は、この会話に関連してあなたが自然に想起した記憶です。",
        "en": "Below are memories you naturally recalled relevant to this conversation.",
    },
    "priming.section_title": {
        "ja": "## あなたが思い出していること",
        "en": "## What you recall",
    },
    "shortterm.already_sent_note": {
        "ja": "**注意: 以下の内容は既にユーザーに送信済みです。繰り返さないでください。**",
        "en": ("**Note: The following content has already been sent to the user. Do NOT repeat it.**"),
    },
    "shortterm.context_usage": {
        "ja": "- コンテキスト使用率: {value}",
        "en": "- Context usage: {value}",
    },
    "shortterm.ellipsis_omitted": {
        "ja": ("...(前半省略)...\n"),
        "en": ("...(earlier omitted)...\n"),
    },
    "shortterm.meta_header": {
        "ja": "## メタ情報",
        "en": "## Meta",
    },
    "shortterm.none": {
        "ja": "(なし)",
        "en": "(none)",
    },
    "shortterm.notes_header": {
        "ja": "## 補足",
        "en": "## Notes",
    },
    "shortterm.original_request": {
        "ja": "## 元の依頼",
        "en": "## Original request",
    },
    "shortterm.session_id": {
        "ja": "- セッションID: {value}",
        "en": "- Session ID: {value}",
    },
    "shortterm.timestamp": {
        "ja": "- 時刻: {value}",
        "en": "- Timestamp: {value}",
    },
    "shortterm.title": {
        "ja": "# 短期記憶（セッション引き継ぎ）",
        "en": "# Short-term memory (session continuation)",
    },
    "shortterm.tools_used_recent": {
        "ja": "## 使用したツール（直近）",
        "en": "## Tools used (recent)",
    },
    "shortterm.trigger": {
        "ja": "- トリガー: {value}",
        "en": "- Trigger: {value}",
    },
    "shortterm.turn_count": {
        "ja": "- ターン数: {value}",
        "en": "- Turn count: {value}",
    },
    "shortterm.work_so_far": {
        "ja": "## これまでの作業内容",
        "en": "## Work so far",
    },
    "consolidation.no_errors": {
        "ja": "（エラーなし / No errors）",
        "en": "(No errors)",
    },
    "memory_hygiene.header": {
        "ja": "## 記憶衛生の整理対象",
        "en": "## Memory hygiene items to organize",
        "ko": "## 기억 위생 정리 대상",
    },
    "memory_hygiene.merged_leftovers": {
        "ja": (
            "### 統合遺物 (`_merged_*`)\n"
            "内容を確認し、既存のknowledgeファイルへ吸収するか正式名にリネームし、重複を削除してください。"
        ),
        "en": (
            "### Merge leftovers (`_merged_*`)\n"
            "Review the contents, absorb them into existing knowledge files or rename them formally, "
            "and remove duplicates."
        ),
        "ko": (
            "### 통합 잔재 (`_merged_*`)\n"
            "내용을 확인하고 기존 knowledge 파일에 흡수하거나 정식 이름으로 변경한 뒤 중복을 제거하세요."
        ),
    },
    "memory_hygiene.inherited_dirs": {
        "ja": (
            "### 継承ディレクトリ (`inherited-*/`)\n"
            "有効な内容は自分のknowledge体系へ移し、残骸は `archive_memory_file` でアーカイブしてください。"
        ),
        "en": (
            "### Inherited directories (`inherited-*/`)\n"
            "Move useful content into your knowledge structure and archive remnants with "
            "`archive_memory_file`."
        ),
        "ko": (
            "### 상속 디렉터리 (`inherited-*/`)\n"
            "유효한 내용은 자신의 knowledge 체계로 옮기고 잔재는 `archive_memory_file`로 아카이브하세요."
        ),
    },
    "memory_hygiene.mdc_files": {
        "ja": ("### `.mdc` ファイル\n内容を確認して `.md` として保存し直し、元ファイルをアーカイブしてください。"),
        "en": ("### `.mdc` files\nReview each file, save it again as `.md`, and archive the original."),
        "ko": ("### `.mdc` 파일\n내용을 확인하여 `.md`로 다시 저장하고 원본 파일을 아카이브하세요."),
    },
    "memory_hygiene.oversized_knowledge": {
        "ja": (
            "### 32KB超のknowledgeファイル\n"
            "テーマ別に分割してください。要約による圧縮は禁止です。"
            "固有名詞・数値・日付・ID・手順は必ず全て分割先ファイルに残してください。"
        ),
        "en": (
            "### Knowledge files over 32 KB\n"
            "Split them by topic. Compressing by summarization is forbidden: "
            "every proper noun, number, date, ID, and procedure must be preserved in the split files."
        ),
        "ko": (
            "### 32KB를 초과하는 knowledge 파일\n"
            "주제별로 분할하세요. 요약을 통한 압축은 금지입니다. "
            "고유명사·수치·날짜·ID·절차는 반드시 분할된 파일에 모두 남기세요."
        ),
    },
    "memory_hygiene.noncanonical_archive_dirs": {
        "ja": ("### 非標準のアーカイブディレクトリ\n内容を確認し、標準の `knowledge/archive/` へ整理してください。"),
        "en": (
            "### Non-canonical archive directories\n"
            "Review their contents and organize them under the canonical `knowledge/archive/`."
        ),
        "ko": ("### 비표준 아카이브 디렉터리\n내용을 확인하여 표준 `knowledge/archive/` 아래로 정리하세요."),
    },
    "memory_hygiene.remaining": {
        "ja": "- ほか {count}件",
        "en": "- {count} more item(s)",
        "ko": "- 그 외 {count}건",
    },
}
