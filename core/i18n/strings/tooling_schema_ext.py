# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings (schema.* part 2)."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "schema.skill.context": {
        "ja": "スキルに渡す補足コンテキスト（任意）",
        "en": "Supplementary context to pass to the skill (optional)",
    },
    "schema.skill.desc": {
        "ja": "スキル・手順書を発動する。skill_nameで指定したスキルの全文を返す。",
        "en": ("Invoke a skill or procedure. Returns the full text of the skill specified by skill_name."),
    },
    "schema.skill.skill_name": {
        "ja": "発動するスキル名（個人スキル、共通スキル、手順書）",
        "en": "Skill name to invoke (personal skill, common skill, or procedure)",
    },
    "schema.task_tracker.desc": {
        "ja": (
            "delegate_task で委譲したタスクの進捗を追跡する。自分のタスクキューから delegated ステータスのエントリを取得し、部下側の最新ステータスと突き合わせて返す。"
        ),
        "en": (
            "Track progress of tasks delegated via delegate_task. Retrieves delegated-status entries from your task queue and cross-references them with the subordinate's latest status."
        ),
    },
    "schema.task_tracker.status": {
        "ja": "フィルタ（all: 全件, active: 進行中, completed: 完了済み）。デフォルト: active",
        "en": ("Filter (all: all tasks, active: in-progress, completed: finished). Default: active"),
    },
    "schema.text_format.args_label": {
        "ja": "引数",
        "en": "Args",
    },
    "schema.text_format.example": {
        "ja": '{"tool": "ツール名", "arguments": {"引数名": "値"}}',
        "en": '{"tool": "tool_name", "arguments": {"arg_name": "value"}}',
    },
    "schema.text_format.fewshot1_prompt": {
        "ja": "ユーザー: docker ps して",
        "en": "User: run docker ps",
    },
    "schema.text_format.fewshot2_prompt": {
        "ja": "ユーザー: 今のメモリ使用量を教えて",
        "en": "User: show current memory usage",
    },
    "schema.text_format.fewshot_header": {
        "ja": "### 使用例",
        "en": "### Examples",
    },
    "schema.text_format.header": {
        "ja": "## 利用可能なツール",
        "en": "## Available Tools",
    },
    "schema.text_format.instruction": {
        "ja": "外部情報の取得やコマンド実行が必要な場合は、**必ず**以下の形式で ```json コードブロックを出力してツールを呼び出してください:",
        "en": (
            "When you need external information or command execution, you **MUST** output a ```json code block to invoke a tool:"
        ),
    },
    "schema.text_format.required_label": {
        "ja": "(必須)",
        "en": "(required)",
    },
    "schema.text_format.rule_no_empty_promise": {
        "ja": "「調べます」「確認します」とだけ言って終わらないでください。調べるならツールを呼び出してください。",
        "en": 'Do NOT just say "I\'ll check" without actually calling a tool.',
    },
    "schema.text_format.rule_no_fabricate": {
        "ja": "**重要**: コマンド出力・ファイル内容・プロセス情報などを推測や想像で生成してはいけません。必ずツールで取得してください。",
        "en": (
            "**Important**: NEVER fabricate command output, file contents, or system information. Always use a tool to retrieve real data."
        ),
    },
    "schema.text_format.rule_one_call": {
        "ja": "1回のメッセージでツール呼び出しは1つだけにしてください。",
        "en": "Only one tool call per message.",
    },
    "schema.text_format.rule_plain_text": {
        "ja": "ツールを使う必要がなければ、普通にテキストで返答してください。",
        "en": "If you don't need to use a tool, respond with plain text.",
    },
    "schema.text_format.rule_wait": {
        "ja": "ツールの実行結果は次のメッセージで提供されます。結果を待ってから回答してください。",
        "en": ("Tool results will be provided in the next message. Wait for results before answering."),
    },
    "schema.text_format.tools_header": {
        "ja": "### ツール一覧",
        "en": "### Tool List",
    },
    "schema.update_task.desc": {
        "ja": "タスクのステータスを更新する。完了時は status='done'、中断時は status='cancelled' に設定する。",
        "en": ("Update a task's status. Set status='done' on completion, status='cancelled' on abort."),
    },
    "schema.update_task.status": {
        "ja": "新しいステータス",
        "en": "New status",
    },
    "schema.update_task.summary": {
        "ja": "更新後の要約（任意）",
        "en": "Updated summary (optional)",
    },
    "schema.update_task.task_id": {
        "ja": "タスクID（backlog_task時に返されたID）",
        "en": "Task ID (the ID returned by backlog_task)",
    },
    "schema.vault_get.desc": {
        "ja": (
            "暗号化されたクレデンシャルvaultから値を取得する。APIキー、パスワード、トークンなどの秘密情報を安全に保管・取得できる。sectionとkeyを指定して値を取得する。"
        ),
        "en": (
            "Retrieve a value from the encrypted credential vault. Securely stores and retrieves secrets such as API keys, passwords, and tokens. Specify section and key to get a value."
        ),
    },
    "schema.vault_get.key": {
        "ja": "キー名（例: 'api_key', 'master_password'）",
        "en": "Key name (e.g. 'api_key', 'master_password')",
    },
    "schema.vault_get.section": {
        "ja": "セクション名（例: 'shared', 'bitwarden', 'bank'）",
        "en": "Section name (e.g. 'shared', 'bitwarden', 'bank')",
    },
    "schema.vault_list.desc": {
        "ja": "暗号化されたクレデンシャルvaultのセクション・キー一覧を表示する。値は表示されない（セクション名とキー名のみ）。",
        "en": (
            "List sections and keys in the encrypted credential vault. Values are not shown (section and key names only)."
        ),
    },
    "schema.vault_list.section": {
        "ja": "セクション名（省略時は全セクション一覧）",
        "en": "Section name (omit to list all sections)",
    },
    "schema.vault_store.desc": {
        "ja": "暗号化されたクレデンシャルvaultに値を保存する。APIキー、パスワード、トークンなどの秘密情報を暗号化して保管する。",
        "en": (
            "Store a value in the encrypted credential vault. Encrypts and stores secrets such as API keys, passwords, and tokens."
        ),
    },
    "schema.vault_store.key": {
        "ja": "キー名（例: 'api_key', 'master_password'）",
        "en": "Key name (e.g. 'api_key', 'master_password')",
    },
    "schema.vault_store.section": {
        "ja": "セクション名（例: 'shared', 'bitwarden', 'bank'）",
        "en": "Section name (e.g. 'shared', 'bitwarden', 'bank')",
    },
    "schema.vault_store.value": {
        "ja": "保存する値（暗号化されて保存される）",
        "en": "Value to store (will be encrypted)",
    },
    "schema.todo_write.desc": {
        "ja": (
            "セッション内タスクチェックリストを作成・更新する。3ステップ以上の複雑なタスクで使用すること。"
            "merge=falseで全リスト置換、merge=trueで既存リストにマージ（idで照合）。"
            "in_progressは同時に1つのみ許可。最大20アイテム。"
        ),
        "en": (
            "Create or update a structured task checklist for the current session. "
            "Use when a task requires 3 or more distinct steps. "
            "merge=false replaces the entire list; merge=true merges by id. "
            "Only one task may be in_progress at a time. Maximum 20 items."
        ),
    },
    "schema.todo_write.todos": {
        "ja": "タスクアイテムの配列",
        "en": "Array of todo items",
    },
    "schema.todo_write.id": {
        "ja": "タスクの一意識別子",
        "en": "Unique identifier for the todo item",
    },
    "schema.todo_write.content": {
        "ja": "タスクの内容・説明",
        "en": "Description/content of the todo item",
    },
    "schema.todo_write.status": {
        "ja": "タスクの状態 (pending=未着手, in_progress=実行中, completed=完了)",
        "en": "Task status (pending=not started, in_progress=working on, completed=done)",
    },
    "schema.todo_write.merge": {
        "ja": "trueで既存リストにマージ（idで照合し、変更のあるフィールドのみ更新）。falseで全置換。デフォルトfalse",
        "en": (
            "If true, merge into existing todos by id (update only changed fields). "
            "If false, replace the entire list. Default false."
        ),
    },
}
