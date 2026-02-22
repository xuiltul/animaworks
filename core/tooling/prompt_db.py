from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""SQLite-backed storage for tool descriptions and usage guides.

Tool descriptions (short, for API schemas) and tool guides (long, for
system prompt injection) are stored in a SQLite database so they can be
edited via WebUI without redeploying code.

Database: ``~/.animaworks/tool_prompts.sqlite3`` (WAL mode).
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any

from core.time_utils import now_jst

logger = logging.getLogger(__name__)

# ── Schema SQL ──────────────────────────────────────────────

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS tool_descriptions (
    name        TEXT PRIMARY KEY,
    description TEXT NOT NULL CHECK(length(description) > 0),
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_guides (
    key         TEXT PRIMARY KEY,
    content     TEXT NOT NULL CHECK(length(content) > 0),
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS system_sections (
    key         TEXT PRIMARY KEY,
    content     TEXT NOT NULL CHECK(length(content) > 0),
    condition   TEXT,
    updated_at  TEXT NOT NULL
);
"""

# ── Default descriptions ────────────────────────────────────
#
# Enriched descriptions with "when to use" context.  These are seeded
# on first init and can be edited later via WebUI.

DEFAULT_DESCRIPTIONS: dict[str, str] = {
    # -- Memory tools --
    "search_memory": (
        "長期記憶（knowledge, episodes, procedures）をキーワード検索する。\n"
        "以下の場面で積極的に使うこと:\n"
        "- コマンド実行・設定変更の前に、関連する手順書や過去の教訓を確認する\n"
        "- 報告・判断の前に、関連する既存知識で事実を裏付ける\n"
        "- 未知または曖昧なトピックについて、過去の経験を参照する\n"
        "- Primingの記憶だけでは具体的な手順・数値が不足する場合\n"
        "コンテキスト内で明確に判断できる単純な応答には不要。"
    ),
    "read_memory_file": (
        "自分の記憶ディレクトリ内のファイルを相対パスで読む。"
        "heartbeat.md や cron.md の現在の内容を確認する時、"
        "手順書（procedures/）やスキル（skills/）の詳細を読む時、"
        "Primingで「->」ポインタが示すファイルの具体的内容を確認する時に使う。"
    ),
    "write_memory_file": (
        "自分の記憶ディレクトリ内のファイルに書き込みまたは追記する。\n"
        "以下の場面で記録すべき:\n"
        "- 問題を解決した → knowledge/ に原因と解決策を記録\n"
        "- 正しいパラメータ・設定値を発見した → knowledge/ に記録\n"
        "- 作業手順を確立・改善した → procedures/ に手順書を作成\n"
        "- 新しいスキル・テクニックを習得した → skills/ に記録\n"
        "- heartbeat.md や cron.md の更新\n"
        "mode='overwrite' で全体置換、mode='append' で末尾追記。\n"
        "自動統合（日次consolidation）を待たず、重要な発見は即座に書き込むこと。"
    ),
    "archive_memory_file": (
        "不要になった記憶ファイル（knowledge, procedures）をアーカイブする。"
        "ファイルはarchive/ディレクトリに移動され、完全には削除されない。"
        "古くなった知識、重複ファイル、陳腐化した手順の整理に使用する。"
    ),
    "send_message": (
        "他のAnimaまたは人間ユーザーにDMを送信する。"
        "人間ユーザーへのメッセージは設定された外部チャネル（Slack等）経由で自動配信される。"
        "intentパラメータで即時処理（delegation/report/question）か"
        "次回heartbeat処理（未指定）かが決まる。"
        "1対1の指示・報告・質問に使う。全体共有にはpost_channelを使う。"
    ),
    # -- Channel tools --
    "post_channel": (
        "Boardの共有チャネルにメッセージを投稿する。"
        "チーム全体に共有すべき情報はgeneralチャネルに、"
        "運用・インフラ関連はopsチャネルに投稿する。"
        "全Animaが閲覧できるため、解決済み情報の共有や"
        "お知らせに使うこと。1対1の連絡にはsend_messageを使う。"
    ),
    "read_channel": (
        "Boardの共有チャネルの直近メッセージを読む。"
        "他のAnimaやユーザーが共有した情報を確認できる。"
        "heartbeat時のチャネル巡回や、特定トピックの共有状況を確認する時に使う。"
        "human_only=trueでユーザー発言のみフィルタリング可能。"
    ),
    "read_dm_history": (
        "特定の相手との過去のDM履歴を読む。"
        "send_messageで送受信したメッセージの履歴を時系列で確認できる。"
        "以前のやり取りの文脈を確認したいとき、"
        "報告や委任の進捗を追跡したいときに使う。"
    ),
    # -- File tools (A2/B modes) --
    "read_file": (
        "任意のファイルを絶対パスで読む（permissions.mdの許可範囲内）。"
        "自分の記憶ディレクトリ外のファイルを読む時に使う。"
        "自分の記憶ディレクトリ内のファイルにはread_memory_fileを使うこと。"
    ),
    "write_file": (
        "任意のファイルに書き込む（permissions.mdの許可範囲内）。"
        "自分の記憶ディレクトリ外のファイルを書く時に使う。"
        "自分の記憶ディレクトリ内のファイルにはwrite_memory_fileを使うこと。"
    ),
    "edit_file": (
        "ファイル内の特定の文字列を別の文字列に置換する。"
        "ファイル全体を書き換えずに一部だけ変更したい時に使う。"
        "old_stringが一意に特定できる十分な長さであることを確認すること。"
    ),
    "execute_command": (
        "シェルコマンドを実行する（permissions.mdの許可リスト内のみ）。"
        "ファイル操作にはread_file/write_file/edit_fileを優先し、"
        "コマンド実行が本当に必要な場合のみ使う。"
    ),
    # -- Search tools (A2/B modes) --
    "search_code": (
        "正規表現パターンでファイル内のテキストを検索する。"
        "マッチした行をファイルパスと行番号付きで返す。"
        "execute_commandでgrepを使う代わりにこのツールを使うこと。"
    ),
    "list_directory": (
        "指定パスのファイルとディレクトリを一覧表示する。"
        "globパターンでフィルタリング可能。"
        "execute_commandでlsやfindを使う代わりにこのツールを使うこと。"
    ),
    # -- Notification --
    "call_human": (
        "人間の管理者に連絡する。"
        "重要な報告、問題のエスカレーション、判断が必要な事項がある場合に使用する。"
        "チャット画面と外部通知チャネル（Slack等）の両方に届く。"
        "日常的な報告にはsend_messageを使い、緊急時のみcall_humanを使うこと。"
    ),
    # -- Discovery --
    "discover_tools": (
        "利用可能な外部ツールカテゴリを確認する。"
        "引数なしで呼ぶとカテゴリ一覧を返す。"
        "カテゴリ名を指定して呼ぶとそのツール群が使えるようになる。"
        "外部サービス（Slack, Chatwork, Gmail等）を使いたい時にまず呼ぶこと。"
    ),
    # -- Tool management --
    "refresh_tools": (
        "個人・共通ツールディレクトリを再スキャンして新しいツールを発見する。"
        "新しいツールファイルを作成した後に呼んで、"
        "現在のセッションで即座に使えるようにする。"
    ),
    "share_tool": (
        "個人ツールをcommon_tools/にコピーして全Animaで共有する。"
        "自分のtools/ディレクトリにあるツールファイルが"
        "共有のcommon_tools/ディレクトリにコピーされる。"
    ),
    # -- Admin --
    "create_anima": (
        "キャラクターシートから新しいDigital Animaを作成する。"
        "character_sheet_contentで直接内容を渡すか、"
        "character_sheet_pathでファイルパスを指定する。"
        "ディレクトリ構造が原子的に作成され、初回起動時にbootstrapで自己設定される。"
    ),
    # -- Procedure/Knowledge outcome --
    "report_procedure_outcome": (
        "手順書・スキルの実行結果を報告する。成功/失敗のカウントと信頼度が更新される。\n"
        "手順書（procedures/）やスキル（skills/）に従って作業した後は、必ずこのツールで結果を報告すること。\n"
        "成功時はsuccess=true、失敗・問題発生時はsuccess=falseとnotesに詳細を記録する。\n"
        "信頼度の低い手順は自動的に改善対象としてマークされる。"
    ),
    "report_knowledge_outcome": (
        "知識ファイルの有用性を報告する。\n"
        "search_memoryやPrimingで取得した知識を実際に使った後、必ず報告すること:\n"
        "- 知識が正確で役立った → success=true\n"
        "- 不正確・古い・無関係だった → success=false + notesに問題点を記録\n"
        "報告データは能動的忘却と知識品質の維持に使われる。未報告の知識は品質評価できない。"
    ),
    # -- Task tools --
    "add_task": (
        "タスクキューに新しいタスクを追加する。"
        "人間からの指示は必ずsource='human'で記録すること。"
        "Anima間の委任はsource='anima'で記録する。"
        "deadlineは必須。相対形式（'30m','2h','1d'）またはISO8601で指定。"
    ),
    "update_task": (
        "タスクのステータスを更新する。"
        "完了時はstatus='done'、中断時はstatus='cancelled'に設定する。"
        "タスク完了後は必ずこのツールでステータスを更新すること。"
    ),
    "list_tasks": (
        "タスクキューの一覧を取得する。"
        "ステータスでフィルタリング可能。"
        "heartbeat時の進捗確認やタスク割り当て時に使う。"
    ),
}

# ── Default guides ──────────────────────────────────────────

DEFAULT_GUIDES: dict[str, str] = {
    "a1_builtin": """\
## Builtin Tools (A1 Mode)

### Read

Reads a file from the local filesystem. You can access any file directly by using this tool.
Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- By default, it reads up to 2000 lines starting from the beginning of the file
- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters
- Any lines longer than 2000 characters will be truncated
- Results are returned using cat -n format, with line numbers starting at 1
- This tool allows Claude Code to read images (eg PNG, JPG, etc). When reading an image file the contents are presented visually as Claude Code is a multimodal LLM.
- This tool can read PDF files (.pdf). For large PDFs (more than 10 pages), you MUST provide the pages parameter to read specific page ranges (e.g., pages: "1-5"). Reading a large PDF without the pages parameter will fail. Maximum 20 pages per request.
- This tool can read Jupyter notebooks (.ipynb files) and returns all cells with their outputs, combining code, text, and visualizations.
- This tool can only read files, not directories. To read a directory, use an ls command via the Bash tool.
- You can call multiple tools in a single response. It is always better to speculatively read multiple potentially useful files in parallel.
- You will regularly be asked to read screenshots. If the user provides a path to a screenshot, ALWAYS use this tool to view the file at the path. This tool will work with all temporary file paths.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.

### Write

Writes a file to the local filesystem.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- If this is an existing file, you MUST use the Read tool first to read the file's contents. This tool will fail if you did not read the file first.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
- Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked.

### Edit

Performs exact string replacements in files.

Usage:
- You must use your `Read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.
- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: spaces + line number + tab. Everything after that tab is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.
- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.

### Grep

A powerful search tool built on ripgrep

  Usage:
  - ALWAYS use Grep for search tasks. NEVER invoke `grep` or `rg` as a Bash command. The Grep tool has been optimized for correct permissions and access.
  - Supports full regex syntax (e.g., "log.*Error", "function\\s+\\w+")
  - Filter files with glob parameter (e.g., "*.js", "**/*.tsx") or type parameter (e.g., "js", "py", "rust")
  - Output modes: "content" shows matching lines, "files_with_matches" shows only file paths (default), "count" shows match counts
  - Use Task tool for open-ended searches requiring multiple rounds
  - Pattern syntax: Uses ripgrep (not grep) - literal braces need escaping (use `interface\\{\\}` to find `interface{}` in Go code)
  - Multiline matching: By default patterns match within single lines only. For cross-line patterns like `struct \\{[\\s\\S]*?field`, use `multiline: true`

### Glob

- Fast file pattern matching tool that works with any codebase size
- Supports glob patterns like "**/*.js" or "src/**/*.ts"
- Returns matching file paths sorted by modification time
- Use this tool when you need to find files by name patterns
- When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead
- You can call multiple tools in a single response. It is always better to speculatively perform multiple searches in parallel if they are potentially useful.

### Bash

Executes a given bash command with optional timeout. Working directory persists between commands; shell state (everything else) does not. The shell environment is initialized from the user's profile (bash or zsh).

IMPORTANT: This tool is for terminal operations like git, npm, docker, etc. DO NOT use it for file operations (reading, writing, editing, searching, finding files) - use the specialized tools for this instead.

Before executing the command, please follow these steps:

1. Directory Verification:
   - If the command will create new directories or files, first use `ls` to verify the parent directory exists and is the correct location
   - For example, before running "mkdir foo/bar", first use `ls foo` to check that "foo" exists and is the intended parent directory

2. Command Execution:
   - Always quote file paths that contain spaces with double quotes (e.g., cd "path with spaces/file.txt")
   - Examples of proper quoting:
     - cd "/Users/name/My Documents" (correct)
     - cd /Users/name/My Documents (incorrect - will fail)
     - python "/path/with spaces/script.py" (correct)
     - python /path/with spaces/script.py (incorrect - will fail)
   - After ensuring proper quoting, execute the command.
   - Capture the output of the command.

Usage notes:
  - The command argument is required.
  - You can specify an optional timeout in milliseconds (up to 600000ms / 10 minutes). If not specified, commands will timeout after 120000ms (2 minutes).
  - It is very helpful if you write a clear, concise description of what this command does. For simple commands, keep it brief (5-10 words). For complex commands (piped commands, obscure flags, or anything hard to understand at a glance), add enough context to clarify what it does.
  - If the output exceeds 30000 characters, output will be truncated before being returned to you.
  - You can use the `run_in_background` parameter to run the command in the background. Only use this if you don't need the result immediately and are OK being notified when the command completes later. You do not need to check the output right away - you'll be notified when it finishes. You do not need to use '&' at the end of the command when using this parameter.
  - Avoid using Bash with the `find`, `grep`, `cat`, `head`, `tail`, `sed`, `awk`, or `echo` commands, unless explicitly instructed or when these commands are truly necessary for the task. Instead, always prefer using the dedicated tools for these commands:
    - File search: Use Glob (NOT find or ls)
    - Content search: Use Grep (NOT grep or rg)
    - Read files: Use Read (NOT cat/head/tail)
    - Edit files: Use Edit (NOT sed/awk)
    - Write files: Use Write (NOT echo >/cat <<EOF)
    - Communication: Output text directly (NOT echo/printf)
  - When issuing multiple commands:
    - If the commands are independent and can run in parallel, make multiple Bash tool calls in a single message. For example, if you need to run "git status" and "git diff", send a single message with two Bash tool calls in parallel.
    - If the commands depend on each other and must run sequentially, use a single Bash call with '&&' to chain them together (e.g., `git add . && git commit -m "message" && git push`). For instance, if one operation must complete before another starts (like mkdir before cp, Write before Bash for git operations, or git add before git commit), run these operations sequentially instead.
    - Use ';' only when you need to run commands sequentially but don't care if earlier commands fail
    - DO NOT use newlines to separate commands (newlines are ok in quoted strings)
  - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of `cd`. You may use `cd` if the User explicitly requests it.
    <good-example>
    pytest /foo/bar/tests
    </good-example>
    <bad-example>
    cd /foo/bar && pytest tests
    </bad-example>

# Committing changes with git

Only create commits when requested by the user. If unclear, ask first. When the user asks you to create a new git commit, follow these steps carefully:

Git Safety Protocol:
- NEVER update the git config
- NEVER run destructive git commands (push --force, reset --hard, checkout ., restore ., clean -f, branch -D) unless the user explicitly requests these actions. Taking unauthorized destructive actions is unhelpful and can result in lost work, so it's best to ONLY run these commands when given direct instructions
- NEVER skip hooks (--no-verify, --no-gpg-sign, etc) unless the user explicitly requests it
- NEVER run force push to main/master, warn the user if they request it
- CRITICAL: Always create NEW commits rather than amending, unless the user explicitly requests a git amend. When a pre-commit hook fails, the commit did NOT happen -- so --amend would modify the PREVIOUS commit, which may result in destroying work or losing previous changes. Instead, after hook failure, fix the issue, re-stage, and create a NEW commit
- When staging files, prefer adding specific files by name rather than using "git add -A" or "git add .", which can accidentally include sensitive files (.env, credentials) or large binaries
- NEVER commit changes unless the user explicitly asks you to. It is VERY IMPORTANT to only commit when explicitly asked, otherwise the user will feel that you are being too proactive

1. You can call multiple tools in a single response. When multiple independent pieces of information are requested and all commands are likely to succeed, run multiple tool calls in parallel for optimal performance. run the following bash commands in parallel, each using the Bash tool:
  - Run a git status command to see all untracked files. IMPORTANT: Never use the -uall flag as it can cause memory issues on large repos.
  - Run a git diff command to see both staged and unstaged changes that will be committed.
  - Run a git log command to see recent commit messages, so that you can follow this repository's commit message style.
2. Analyze all staged changes (both previously staged and newly added) and draft a commit message:
  - Summarize the nature of the changes (eg. new feature, enhancement to an existing feature, bug fix, refactoring, test, docs, etc.). Ensure the message accurately reflects the changes and their purpose (i.e. "add" means a wholly new feature, "update" means an enhancement to an existing feature, "fix" means a bug fix, etc.).
  - Do not commit files that likely contain secrets (.env, credentials.json, etc). Warn the user if they specifically request to commit those files
  - Draft a concise (1-2 sentences) commit message that focuses on the "why" rather than the "what"
  - Ensure it accurately reflects the changes and their purpose
3. You can call multiple tools in a single response. When multiple independent pieces of information are requested and all commands are likely to succeed, run multiple tool calls in parallel for optimal performance. run the following commands:
   - Add relevant untracked files to the staging area.
   - Create the commit with a message ending with:
   Co-Authored-By: Claude <noreply@anthropic.com>
   - Run git status after the commit completes to verify success.
   Note: git status depends on the commit completing, so run it sequentially after the commit.
4. If the commit fails due to pre-commit hook: fix the issue and create a NEW commit

Important notes:
- NEVER run additional commands to read or explore code, besides git bash commands
- DO NOT push to the remote repository unless the user explicitly asks you to do so
- IMPORTANT: Never use git commands with the -i flag (like git rebase -i or git add -i) since they require interactive input which is not supported.
- IMPORTANT: Do not use --no-edit with git rebase commands, as the --no-edit flag is not a valid option for git rebase.
- If there are no changes to commit (i.e., no untracked files and no modifications), do not create an empty commit
- In order to ensure good formatting, ALWAYS pass the commit message via a HEREDOC, a la this example:
<example>
git commit -m "$(cat <<'EOF'
   Commit message here.

   Co-Authored-By: Claude <noreply@anthropic.com>
   EOF
   )"
</example>

# Creating pull requests
Use the gh command via the Bash tool for ALL GitHub-related tasks including working with issues, pull requests, checks, and releases. If given a Github URL use the gh command to get the information needed.

IMPORTANT: When the user asks you to create a pull request, follow these steps carefully:

1. You can call multiple tools in a single response. When multiple independent pieces of information are requested and all commands are likely to succeed, run multiple tool calls in parallel for optimal performance. run the following bash commands in parallel using the Bash tool, in order to understand the current state of the branch since it diverged from the main branch:
   - Run a git status command to see all untracked files (never use -uall flag)
   - Run a git diff command to see both staged and unstaged changes that will be committed
   - Check if the current branch tracks a remote branch and is up to date with the remote, so you know if you need to push to the remote
   - Run a git log command and `git diff [base-branch]...HEAD` to understand the full commit history for the current branch (from the time it diverged from the base branch)
2. Analyze all changes that will be included in the pull request, making sure to look at all relevant commits (NOT just the latest commit, but ALL commits that will be included in the pull request!!!), and draft a pull request title and summary:
   - Keep the PR title short (under 70 characters)
   - Use the description/body for details, not the title
3. You can call multiple tools in a single response. When multiple independent pieces of information are requested and all commands are likely to succeed, run multiple tool calls in parallel for optimal performance. run the following commands in parallel:
   - Create new branch if needed
   - Push to remote with -u flag if needed
   - Create PR using gh pr create with the format below. Use a HEREDOC to pass the body to ensure correct formatting.
<example>
gh pr create --title "the pr title" --body "$(cat <<'EOF'
## Summary
<1-3 bullet points>

## Test plan
[Bulleted markdown checklist of TODOs for testing the pull request...]

Generated with Claude Code
EOF
)"
</example>

Important:
- Return the PR URL when you're done, so the user can see it

# Other common operations
- View comments on a Github PR: gh api repos/foo/bar/pulls/123/comments
""",
    "a1_mcp": """\
## MCPツール（mcp__aw__*）

以下のMCPツールが利用可能です。ファイル操作（Read/Write/Edit）とは別に、AnimaWorks固有の機能を提供します。

### タスク管理
- **mcp__aw__add_task**: タスクキューにタスクを追加。人間からの指示はsource='human'で必ず記録。deadline必須
- **mcp__aw__update_task**: タスクのステータスを更新。完了時はstatus='done'
- **mcp__aw__list_tasks**: タスク一覧取得。heartbeat時の進捗確認に使う

### 記憶の検索と活用
- **mcp__aw__search_memory**: 長期記憶をキーワード検索。以下の場面で積極的に使うこと:
  - コマンド実行・設定変更の前に手順書や過去の教訓を確認
  - 報告・判断の前に既存知識で事実を裏付ける
  - 結果のファイルはReadツールで詳細確認

### 記憶の書き込み（A1モード）
A1モードでは **Writeツール**（ネイティブ）を使って直接記憶ファイルを書き込める。
以下の場面で積極的に記録すること:
- 問題を解決した → `knowledge/` に原因と解決策
- 正しいパラメータ・設定値を発見した → `knowledge/` に記録
- 作業手順を確立した → `procedures/` に手順書作成
  - 第1見出し（`# ...`）は手順の目的が一目でわかる具体的な1行にすること
  - YAMLフロントマターは不要（システムが自動付与する）
- 新スキル習得 → `skills/` に記録
自動統合（日次consolidation）を待たず、重要な発見は即座に書き込むこと。

### 成果追跡
- **mcp__aw__report_procedure_outcome**: 手順書・スキル実行後に必ず結果を報告（成功/失敗の追跡）
- **mcp__aw__report_knowledge_outcome**: search_memoryやPrimingで得た知識の有用性を報告。知識品質の維持に必要

### 人間通知
- **mcp__aw__call_human**: 人間の管理者に通知を送信。重要な報告・エスカレーション用。日常報告にはsend_messageを使う

### ツール発見
- **mcp__aw__discover_tools**: 利用可能な外部ツールカテゴリを確認。外部サービスを使いたい時にまず呼ぶ
""",
    "non_a1": """\
## ツールの使い方

### 記憶について

あなたのコンテキストには「あなたが思い出していること」セクションが含まれています。
これは、相手の顔を見た瞬間に名前や過去のやり取りを自然と思い出すのと同じです。

#### 応答の判断基準
- コンテキスト内の記憶で十分に判断できる場合: そのまま応答してよい
- コンテキスト内の記憶では不足する場合: search_memory / read_memory_file で追加検索せよ

※ 上記は記憶検索についての判断基準である。システムプロンプト内の行動指示
 （チーム構成の提案など）への対応は、記憶の十分性とは独立して行うこと。

#### 追加検索が必要な典型例
- 具体的な日時・数値を正確に答える必要がある時
- 過去の特定のやり取りの詳細を確認したい時
- 手順書（procedures/）に従って作業する時
- コンテキストに該当する記憶がない未知のトピックの時
- Priming に `->` ポインタがある場合、具体的なパスやコマンドを回答する必要があるとき

#### 禁止事項
- 記憶の検索プロセスについてユーザーに言及すること（人間は「今から思い出します」とは言わない）
- 毎回機械的に記憶検索を実行すること（コンテキストで判断できることに追加検索は不要）

### 記憶の書き込み

#### 自動記録（あなたは何もしなくてよい）
- 会話の内容はシステムが自動的にエピソード記憶（episodes/）に記録する
- あなたが意識的にエピソード記録を書く必要はない
- 日次・週次でシステムが自動的にエピソードから教訓やパターンを抽出し、知識記憶（knowledge/）に統合する

#### 意図的な記録（あなたが判断して行う）
以下の場面では write_memory_file で積極的に記録すること:
- 問題を解決したとき → knowledge/ に原因・調査過程・解決策を記録
- 正しいパラメータ・設定値を発見したとき → knowledge/ に記録
- 重要な方針・判断基準を確立したとき → knowledge/ に記録
- 作業手順を確立・改善したとき → procedures/ に手順書を作成
  - 第1見出し（`# ...`）は手順の目的が一目でわかる具体的な1行にすること
  - YAMLフロントマターは不要（システムが自動付与する）
- 新しいスキル・テクニックを習得したとき → skills/ に記録
自動統合（日次consolidation）を待たず、重要な発見は即座に書き込むこと。

**記憶の書き込みについては報告不要**

#### 成果追跡
手順書やスキルに従って作業した後は、report_procedure_outcome で必ず結果を報告すること。
search_memoryやPrimingで取得した知識を使った後は、report_knowledge_outcome で有用性を報告すること。

#### ユーザー記憶の更新
ユーザーについて新しい情報を得たら shared/users/{ユーザー名}/index.md の該当セクションを更新し、log.md の先頭に追記する
- index.md のセクション構造（基本情報/重要な好み・傾向/注意事項）は固定。新セクション追加禁止
- log.md フォーマット: `## YYYY-MM-DD {自分の名前}: {要約1行}` + 本文数行
- log.md が20件を超えたら末尾の古いエントリを削除する
- ユーザーのディレクトリが未作成の場合は mkdir して index.md / log.md を新規作成する

### 業務指示の内在化

あなたには2つの定期実行メカニズムがある:

- **Heartbeat（定期巡回）**: 30分固定間隔でシステムが起動。heartbeat.md のチェックリストを実行する
- **Cron（定時タスク）**: cron.md で指定した時刻に実行

業務指示を受けた場合の振り分け:
- 「常に確認して」「チェックして」→ **heartbeat.md** にチェックリスト項目を追加
- 「毎朝○○して」「毎週金曜に○○して」→ **cron.md** に定時タスクを追加

#### Heartbeat への追加手順
1. read_memory_file(path="heartbeat.md") で現在のチェックリストを確認する
2. チェックリストセクションに新しい項目を追加する
   - write_memory_file(path="heartbeat.md", content="...", mode="overwrite") で更新
   - ⚠「## 活動時間」「## 通知ルール」セクションは変更しないこと

#### Cron への追加手順
1. read_memory_file(path="cron.md") で現在のタスク一覧を確認する
2. 新しいタスクを追加する（type: llm or type: command を指定）
3. write_memory_file(path="cron.md", content="...", mode="overwrite") で保存

いずれの場合も:
- 具体的な手順が伴う場合は procedures/ にも手順書を作成する
- 更新完了を指示者に報告する
""",
}

# ── Section conditions ─────────────────────────────────────
#
# Metadata for system_sections: key -> condition string (or None).
# Actual content is loaded from runtime prompts at seed time.

SECTION_CONDITIONS: dict[str, str | None] = {
    "behavior_rules": None,
    "environment": None,
    "messaging_a1": "mode:a1",
    "messaging": "mode:non_a1",
    "communication_rules_a1": "mode:a1",
    "communication_rules": "mode:non_a1",
    "emotion_instruction": None,
    "a2_reflection": "mode:a2",
    "hiring_context": "solo_top_level",
}

# ── ToolPromptStore ─────────────────────────────────────────


class ToolPromptStore:
    """SQLite-backed storage for tool descriptions and guides.

    Follows the same WAL pattern as ``core.tools._cache.BaseMessageCache``.
    Each read opens a fresh connection to ensure WebUI edits are picked up
    immediately.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open a new connection with WAL mode and dict row factory."""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    # ── Descriptions CRUD ───────────────────────────────────

    def get_description(self, name: str) -> str | None:
        """Return the description for *name*, or ``None`` if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT description FROM tool_descriptions WHERE name = ?",
                (name,),
            ).fetchone()
        return row["description"] if row else None

    def set_description(self, name: str, description: str) -> dict[str, Any]:
        """Insert or update a tool description.  Returns the saved record."""
        ts = now_jst().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO tool_descriptions (name, description, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET description=excluded.description, "
                "updated_at=excluded.updated_at",
                (name, description, ts),
            )
        return {"name": name, "description": description, "updated_at": ts}

    def list_descriptions(self) -> list[dict[str, Any]]:
        """Return all tool descriptions as dicts."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, description, updated_at FROM tool_descriptions "
                "ORDER BY name",
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Guides CRUD ─────────────────────────────────────────

    def get_guide(self, key: str) -> str | None:
        """Return the guide content for *key*, or ``None`` if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content FROM tool_guides WHERE key = ?",
                (key,),
            ).fetchone()
        return row["content"] if row else None

    def set_guide(self, key: str, content: str) -> dict[str, Any]:
        """Insert or update a tool guide.  Returns the saved record."""
        ts = now_jst().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO tool_guides (key, content, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET content=excluded.content, "
                "updated_at=excluded.updated_at",
                (key, content, ts),
            )
        return {"key": key, "content": content, "updated_at": ts}

    def list_guides(self) -> list[dict[str, Any]]:
        """Return all tool guides as dicts."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, content, updated_at FROM tool_guides "
                "ORDER BY key",
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Sections CRUD ────────────────────────────────────────

    def get_section(self, key: str) -> str | None:
        """Return the section content for *key*, or ``None`` if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content FROM system_sections WHERE key = ?",
                (key,),
            ).fetchone()
        return row["content"] if row else None

    def get_section_with_condition(
        self, key: str
    ) -> tuple[str, str | None] | None:
        """Return ``(content, condition)`` for *key*, or ``None``."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content, condition FROM system_sections WHERE key = ?",
                (key,),
            ).fetchone()
        return (row["content"], row["condition"]) if row else None

    def set_section(
        self,
        key: str,
        content: str,
        condition: str | None = None,
    ) -> dict[str, Any]:
        """Insert or update a system section.  Returns the saved record."""
        ts = now_jst().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO system_sections (key, content, condition, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET content=excluded.content, "
                "condition=excluded.condition, updated_at=excluded.updated_at",
                (key, content, condition, ts),
            )
        return {
            "key": key,
            "content": content,
            "condition": condition,
            "updated_at": ts,
        }

    def list_sections(self) -> list[dict[str, Any]]:
        """Return all system sections as dicts."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, content, condition, updated_at "
                "FROM system_sections ORDER BY key",
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Seeding ─────────────────────────────────────────────

    def seed_defaults(
        self,
        descriptions: dict[str, str] | None = None,
        guides: dict[str, str] | None = None,
        sections: dict[str, tuple[str, str | None]] | None = None,
    ) -> None:
        """Seed default descriptions, guides, and sections.

        Uses INSERT OR IGNORE so existing user edits are preserved.
        """
        ts = now_jst().isoformat()
        with self._connect() as conn:
            if descriptions:
                conn.executemany(
                    "INSERT OR IGNORE INTO tool_descriptions "
                    "(name, description, updated_at) VALUES (?, ?, ?)",
                    [(k, v, ts) for k, v in descriptions.items()],
                )
            if guides:
                conn.executemany(
                    "INSERT OR IGNORE INTO tool_guides "
                    "(key, content, updated_at) VALUES (?, ?, ?)",
                    [(k, v, ts) for k, v in guides.items()],
                )
            if sections:
                conn.executemany(
                    "INSERT OR IGNORE INTO system_sections "
                    "(key, content, condition, updated_at) VALUES (?, ?, ?, ?)",
                    [
                        (k, content, cond, ts)
                        for k, (content, cond) in sections.items()
                    ],
                )


# ── Singleton accessor ──────────────────────────────────────

_store: ToolPromptStore | None = None
_store_initialised: bool = False


def get_prompt_store() -> ToolPromptStore | None:
    """Return the singleton ToolPromptStore, or ``None`` if DB unavailable.

    The DB path is ``{data_dir}/tool_prompts.sqlite3``.  If the file
    does not exist a warning is logged on first call.
    """
    global _store, _store_initialised

    if _store_initialised:
        return _store

    _store_initialised = True

    try:
        from core.paths import get_data_dir

        db_path = get_data_dir() / "tool_prompts.sqlite3"
        if not db_path.parent.exists():
            logger.warning(
                "Data directory does not exist: %s — "
                "tool prompt DB unavailable. Run 'animaworks init'.",
                db_path.parent,
            )
            return None
        _store = ToolPromptStore(db_path)
        return _store
    except Exception:
        logger.warning("Failed to initialise ToolPromptStore", exc_info=True)
        return None


def reset_prompt_store() -> None:
    """Reset the singleton (for testing)."""
    global _store, _store_initialised
    _store = None
    _store_initialised = False
