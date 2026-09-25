# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Domain-specific i18n strings."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "agent.context_cannot_fit_safely": {
        "ja": "必須の指示・権限・タスク文脈を保持すると入力上限を超えるため実行を停止しました（推定 {estimated} / 上限 {limit} トークン）。入力を短くするか、より大きなコンテキストのモデルを設定してください。",
        "en": "Execution stopped because required instructions, permissions, and task context cannot fit safely ({estimated} estimated tokens / {limit} limit). Shorten the input or configure a model with a larger context window.",
        "ko": "필수 지시, 권한, 태스크 맥락을 보존하면 입력 한도를 초과하여 실행을 중지했습니다(추정 {estimated} / 한도 {limit} 토큰). 입력을 줄이거나 컨텍스트가 더 큰 모델을 설정하세요.",
    },
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
    "task.compaction_summary_instructions": {
        "ja": (
            "次の見出しで要約してください。\n"
            "## タスクの目的と完了条件\n"
            "## 完了した手順（結果・根拠となるID/コミット/URL/数値を正確に）\n"
            "## 進行中の手順と次にやること\n"
            "## 判明した事実・制約・失敗した試行とその理由\n"
            "## 読んだファイル\n"
            "## 変更したファイル\n"
            "## 未解決の疑問\n"
            "固有名詞・ID・パス・コマンドは省略せず原文のまま残してください。"
        ),
        "en": (
            "Summarize using these headings:\n"
            "## Task purpose and completion criteria\n"
            "## Completed steps (preserve results and exact evidence IDs/commits/URLs/numbers)\n"
            "## In-progress steps and next actions\n"
            "## Discovered facts, constraints, and failed attempts with reasons\n"
            "## Files read\n"
            "## Files changed\n"
            "## Unresolved questions\n"
            "Do not omit or alter proper nouns, IDs, paths, or commands; preserve them verbatim."
        ),
        "ko": (
            "다음 제목으로 요약하세요:\n"
            "## 작업 목적 및 완료 조건\n"
            "## 완료한 단계(결과와 근거 ID/커밋/URL/수치를 정확히 보존)\n"
            "## 진행 중인 단계와 다음 작업\n"
            "## 확인된 사실, 제약, 실패한 시도와 그 이유\n"
            "## 읽은 파일\n"
            "## 변경한 파일\n"
            "## 해결되지 않은 질문\n"
            "고유명사, ID, 경로, 명령은 생략하거나 바꾸지 말고 원문 그대로 보존하세요."
        ),
    },
    "task.compacted_activity_summary": {
        "ja": "タスク実行中の文脈を圧縮しました",
        "en": "Compacted task execution context",
        "ko": "작업 실행 문맥을 압축했습니다",
    },
    "task.compacted_after_activity_summary": {
        "ja": "文脈圧縮後に同一タスクセッションを再開しました",
        "en": "Resumed the same task session after context compaction",
        "ko": "문맥 압축 후 동일한 작업 세션을 재개했습니다",
    },
    "task.compaction_continue_prompt": {
        "ja": (
            "文脈を圧縮しました。以下が元のタスク指示です。圧縮前の続きから作業を再開し、"
            "完了済みの手順は繰り返さないでください。\n\n"
            "## 元のタスク指示\n{original_prompt}"
        ),
        "en": (
            "The context has been compacted. The original task instructions are below. Resume from where "
            "you left off before compaction, and do not repeat steps that are already complete.\n\n"
            "## Original task instructions\n{original_prompt}"
        ),
        "ko": (
            "문맥을 압축했습니다. 아래는 원래 작업 지시입니다. 압축 전 진행하던 지점부터 다시 시작하고, "
            "이미 완료한 단계는 반복하지 마세요.\n\n"
            "## 원래 작업 지시\n{original_prompt}"
        ),
    },
    "assisted.tool_exec_error": {
        "ja": "ツール実行エラー: {error}",
        "en": "Tool execution error: {error}",
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
        "ja": "cursor-agentセッションをローテーションしました（ターン{turn_count}）。新規セッションを開始します。",
        "en": "Rotated cursor-agent session (turn {turn_count}). Starting fresh session.",
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
            "自分で直接Bash/Read/Grep等のツールを使って作業してください。"
        ),
        "en": (
            "BLOCKED: Cannot spawn Agent/Task/submit_tasks subtasks from a TaskExec session (recursion prevention). "
            "Use Bash/Read/Grep and other tools directly."
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
            "• **部下に委譲**: `delegate_task` で部下にタスクを委任する（部下が実行する）\n"
            "Agent/Taskの代わりに上記を使ってください。"
        ),
        "en": (
            "BLOCKED: Agent/Task tools are disabled (sub-agent spawning is not allowed). "
            "Use one of these methods instead:\n"
            "• **Direct execution**: Use Read, Bash, Grep, Edit, etc. to do the work yourself (recommended)\n"
            "• **Delegation**: Use `delegate_task` to delegate to a subordinate (they execute it)\n"
            "Use these instead of Agent/Task."
        ),
    },
    "sdk_hooks.submit_tasks_success": {
        "ja": (
            "成功: submit_tasks でタスク {task_ids} をキューに投入した。"
            "これらは自分の TaskExecutor が後で実行するため、このターンではそれ以上手を出さない。"
        ),
        "en": (
            "Success: tasks {task_ids} are queued in your TaskExecutor for later execution. "
            "Do not touch them further in this turn."
        ),
        "ko": (
            "성공: submit_tasks로 작업 {task_ids}을(를) 큐에 넣었습니다. "
            "이는 나중에 자신의 TaskExecutor가 실행하므로 이번 턴에서는 더 이상 다루지 마세요."
        ),
    },
    "action_rule.attached": {
        "ja": "この操作に関係する行動ルール。内容に反していたら、いま是正すること。",
        "en": "Action rules related to this call. If the action conflicted with them, correct it now.",
        "ko": "이 작업과 관련된 행동 규칙입니다. 내용에 위배된다면 지금 바로 잡으십시오.",
    },
    "executor.unavailable_no_configured_fallback": {
        "ja": "実行方式 {mode}（{model}）を利用できません。利用可能な fallback_models を設定してください。",
        "en": "Execution mode {mode} ({model}) is unavailable. Configure an available fallback_models route.",
        "ko": "실행 모드 {mode} ({model})를 사용할 수 없습니다. 사용 가능한 fallback_models 경로를 설정하세요.",
    },
    "executor.codex_unavailable_no_openai_cred": {
        "ja": (
            "Codex SDK が利用できず、フォールバック先 openai/* の認証情報も無いため実行不能。"
            "openai-codex パッケージの復元が必要"
        ),
        "en": (
            "Codex SDK is unavailable and no credentials for the openai/* fallback; cannot execute. "
            "Restore the openai-codex package."
        ),
    },
}
