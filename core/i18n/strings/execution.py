# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "agent.omitted_rest": {
        "ja": ("\n\n（以降省略）"),
        "en": ("\n\n(omitted)"),
    },
    "agent.priming_tier_light_header": {
        "ja": ("## あなたが思い出していること\n\n### {sender_name} について\n\n"),
        "en": ("## What you recall\n\n### About {sender_name}\n\n"),
    },
    "agent.recent_dialogue_consider": {
        "ja": "進行中のタスクや指示がある場合、この内容を考慮してください。",
        "en": "Consider this content if there are ongoing tasks or instructions.",
    },
    "agent.recent_dialogue_header": {
        "ja": "## 直近の対話履歴",
        "en": "## Recent dialogue history",
    },
    "agent.recent_dialogue_intro": {
        "ja": "以下はユーザーとの直近の対話です。",
        "en": "Below is your recent dialogue with the user.",
    },
    "agent.stream_retry_exhausted": {
        "ja": "ストリームが{retry_count}回切断されました。最大リトライ回数に達しました。",
        "en": "Stream disconnected {retry_count} time(s). Max retries reached.",
    },
    "assisted.intent_reprompt": {
        "ja": (
            'ツールを使う意図があるようですが、実際にツールが呼び出されていません。必要な操作を以下の形式で出力してください:\n\n```json\n{"tool": "ツール名", "arguments": {"引数名": "値"}}\n```'
        ),
        "en": (
            'You indicated intent to use a tool but did not actually call one. Please output the tool call in the following format:\n\n```json\n{"tool": "tool_name", "arguments": {"arg_name": "value"}}\n```'
        ),
    },
    "assisted.output_truncated": {
        "ja": "... [出力切り捨て: 元のサイズ {size}バイト]",
        "en": "... [Output truncated: original size {size} bytes]",
    },
    "assisted.tool_exec_error": {
        "ja": "ツール実行エラー: {error}",
        "en": "Tool execution error: {error}",
    },
    "assisted.tool_result_header": {
        "ja": "ツール実行結果:",
        "en": "Tool execution result:",
    },
    "assisted.unknown_tool": {
        "ja": "エラー: 不明なツール '{tool_name}' です。利用可能なツール: {available}",
        "en": "Error: Unknown tool '{tool_name}'. Available tools: {available}",
    },
    "litellm_context.compact_system": {
        "ja": "以下のAIアシスタントと人間の作業会話を簡潔に要約してください。主要な発見、決定事項、ツール結果、未完了の項目をすべて保持してください。要約のみを出力してください。",
        "en": "Summarize the following work conversation between an AI assistant and a human concisely. Preserve all key findings, decisions, tool results, and pending items. Output only the summary.",
    },
    "litellm_context.compact_summary_prefix": {
        "ja": "[前回の作業要約]",
        "en": "[Previous work summary]",
    },
    "cursor_agent.not_installed": {
        "ja": "cursor-agent CLIが見つかりません。`curl https://cursor.com/install -fsS | bash` でインストールし、`agent login` でログインしてください。",
        "en": "cursor-agent CLI not found. Install with `curl https://cursor.com/install -fsS | bash` and run `agent login`.",
    },
    "cursor_agent.not_authenticated": {
        "ja": "cursor-agentが未認証です。`agent login` を実行してCursorアカウントにログインしてください。",
        "en": "cursor-agent is not authenticated. Run `agent login` to sign in to your Cursor account.",
    },
    "cursor_agent.session_resume_failed": {
        "ja": "cursor-agentセッションの復元に失敗しました（chatId={chat_id}）。新規セッションで再試行します。",
        "en": "Failed to resume cursor-agent session (chatId={chat_id}). Retrying with a fresh session.",
    },
    "cursor_agent.session_rotated": {
        "ja": "cursor-agentセッションをローテーションしました（ターン{turn_count}、最大{max_turns}）。新規セッションを開始します。",
        "en": "Rotated cursor-agent session (turn {turn_count}, max {max_turns}). Starting fresh session.",
    },
    "cursor_agent.timeout": {
        "ja": "[cursor-agent タイムアウト: {timeout}秒以内に完了しませんでした]",
        "en": "[cursor-agent timeout: did not complete within {timeout} seconds]",
    },
    "gemini_cli.not_installed": {
        "ja": "Gemini CLIが見つかりません。`npm install -g @google/gemini-cli` でインストールし、`gemini auth login` でログインしてください。",
        "en": "Gemini CLI not found. Install with `npm install -g @google/gemini-cli` and run `gemini auth login`.",
    },
    "gemini_cli.not_authenticated": {
        "ja": "Gemini CLIが未認証です。`gemini auth login` を実行してGoogleアカウントにログインしてください。",
        "en": "Gemini CLI is not authenticated. Run `gemini auth login` to sign in to your Google account.",
    },
    "gemini_cli.timeout": {
        "ja": "[Gemini CLI タイムアウト: {timeout}秒以内に完了しませんでした]",
        "en": "[Gemini CLI timeout: did not complete within {timeout} seconds]",
    },
    "grok_cli.not_installed": {
        "ja": "Grok CLIが見つかりません。`curl -fsSL https://x.ai/cli/install.sh | bash` でインストールし、`grok login` でログインしてください。",
        "en": "Grok CLI not found. Install with `curl -fsSL https://x.ai/cli/install.sh | bash` and run `grok login`.",
        "ko": "Grok CLI가 없습니다. `curl -fsSL https://x.ai/cli/install.sh | bash`로 설치하고 `grok login`을 실행하세요.",
    },
    "grok_cli.not_authenticated": {
        "ja": "Grok CLIが未認証です。`grok login` を実行してxAIアカウントにログインしてください。",
        "en": "Grok CLI is not authenticated. Run `grok login` to sign in to your xAI account.",
        "ko": "Grok CLI가 인증되지 않았습니다. `grok login`을 실행하여 xAI 계정에 로그인하세요.",
    },
    "grok_cli.timeout": {
        "ja": "[Grok CLI タイムアウト: {timeout}秒間進捗がなかったため打ち切りました]",
        "en": "[Grok CLI timeout: no progress for {timeout} seconds]",
        "ko": "[Grok CLI 시간 초과: {timeout}초 동안 진행이 없어 중단했습니다]",
    },
    "sdk_hooks.task_no_subtask": {
        "ja": (
            "BLOCKED: TaskExecセッション内でAgent/Task/submit_tasksサブタスクは起動できません（再帰防止）。"
            "自分で直接Bash/Read/Grep等のツールを使って作業するか、"
            '重い処理は `animaworks-tool machine run "指示" -d ワークスペースパス` で外部エージェントに委託してください。'
        ),
        "en": (
            "BLOCKED: Cannot spawn Agent/Task/submit_tasks subtasks from a TaskExec session (recursion prevention). "
            "Use Bash/Read/Grep and other tools directly, or delegate heavy work via "
            '`animaworks-tool machine run "instruction" -d workspace_path`.'
        ),
    },
    "sdk_hooks.submit_tasks_unavailable": {
        "ja": (
            "BLOCKED: submit_tasksは通常チャット/heartbeat/cron/TaskExecでは利用できません。"
            "この場ではRead/Bash/Grep/Edit等で直接作業してください。"
            "バックグラウンド実行が必要な場合は、明示的なバックグラウンド実行ワークフローから起動してください。"
        ),
        "en": (
            "BLOCKED: submit_tasks is not available in normal chat/heartbeat/cron/TaskExec sessions. "
            "Do the work directly with Read/Bash/Grep/Edit here. "
            "Use an explicit background execution workflow when background task submission is required."
        ),
    },
    "sdk_hooks.agent_task_blocked": {
        "ja": (
            "BLOCKED: Agent/Taskツールは無効です（サブエージェント起動は禁止）。"
            "以下の方法で作業してください:\n"
            "• **直接実行**: Read, Bash, Grep, Edit 等のツールで自分で作業する（推奨）\n"
            '• **重い処理**: `animaworks-tool machine run "指示" -d ワークスペースパス` で外部エージェントに委託する\n'
            "• **部下に委譲**: `delegate_task` で部下にタスクを委任する（部下が実行する）\n"
            "Agent/Taskの代わりに上記を使ってください。"
        ),
        "en": (
            "BLOCKED: Agent/Task tools are disabled (sub-agent spawning is not allowed). "
            "Use one of these methods instead:\n"
            "• **Direct execution**: Use Read, Bash, Grep, Edit, etc. to do the work yourself (recommended)\n"
            '• **Heavy work**: Use `animaworks-tool machine run "instruction" -d workspace_path` for an external agent\n'
            "• **Delegation**: Use `delegate_task` to delegate to a subordinate (they execute it)\n"
            "Use these instead of Agent/Task."
        ),
    },
    "action_rule.deny_reason": {
        "ja": "アクションルールにより一時停止: 実行前に以下のルールを確認してください",
        "en": "Paused by action rule: please review the following rule before proceeding",
    },
    "action_rule.system_message": {
        "ja": (
            "<action-rule>\n{rule_content}\n</action-rule>\n\n"
            "上記ルールを確認した上で、必要な事前アクションを実行してから再試行してください。"
        ),
        "en": (
            "<action-rule>\n{rule_content}\n</action-rule>\n\n"
            "Please review the rule above, take any required pre-actions, then retry."
        ),
    },
}
