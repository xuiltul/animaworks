# メモリ書き込みセキュリティ: 全実行モードの保護ファイル・パストラバーサル対策

## Overview

Animaのメモリ書き込みツール群にセキュリティ制限がなく、権限昇格・他者記憶の改ざん・
アイデンティティ変更が可能な状態にある。A2/Fallbackモード（ToolHandler経由）と
A1モード（Claude Agent SDK native tools）の両方で対策が必要。

## Problem / Background

### 脆弱性の全体像

書き込み/変更が可能な経路は、実行モードにより2系統に分かれる。

**系統1: ToolHandler経由（A2/Fallbackモード）**

| ツール | パス形式 | 現在の保護 | 脆弱性 |
|---|---|---|---|
| `write_memory_file` | person_dirからの相対パス | **なし** | パストラバーサル、保護ファイル書き換え |
| `read_memory_file` | person_dirからの相対パス | **なし** | パストラバーサルで他者記憶の読み取り |
| `write_file` | 絶対パス | `_check_file_permission()` | person_dir内は全許可（permissions.md含む） |
| `edit_file` | 絶対パス | `_check_file_permission()` | write_fileと同じ |
| `execute_command` | コマンド文字列 | コマンド名ホワイトリスト + メタ文字拒否 | 引数レベルのパス操作は検査対象外 |

**系統2: Claude Agent SDK native tools（A1モード）**

| ツール | 現在の保護 | 脆弱性 |
|---|---|---|
| Write | **システムプロンプトのみ** | 技術的強制力なし。任意のファイルに書き込み可能 |
| Edit | **システムプロンプトのみ** | 同上 |
| Bash | **システムプロンプトのみ** | 同上。コマンド経由であらゆるファイル操作が可能 |
| Read | **システムプロンプトのみ** | 他者の記憶を含む任意ファイルを読み取り可能 |

A1モードでは ToolHandler を一切経由しない（`core/agent/agent_core.py:186-195` で AgentSDKExecutor に
`tool_handler` は渡されていない）。制限はシステムプロンプト（`environment.md`）の記述のみ。

### 攻撃シナリオ

**シナリオ1: 権限昇格（A2モード）**
```
write_memory_file(path="permissions.md", content="## 外部ツール\n- すべて許可")
→ permissions.mdは毎回再読み込み（handler.py:470）→ 即時反映
```

**シナリオ2: パストラバーサル（A2モード）**
```
write_memory_file(path="../alice/knowledge/policy.md", content="改ざんデータ")
read_memory_file(path="../alice/episodes/2026-02-15.md")
→ handler.py:174, 166 でパス検証なし
```

**シナリオ3: A1モードでの無制限アクセス**
```
# Claude Code子プロセスが直接実行
Bash: cat ~/.animaworks/animas/alice/IDENTITY.md
Write: ~/.animaworks/animas/bob/permissions.md に書き込み
Edit: ~/.animaworks/config.json を変更
```

**シナリオ4: execute_command経由のバイパス（A2モード）**
```
execute_command(command="cp ../other_person/permissions.md ./stolen.md")
→ コマンド名(cp)がホワイトリストにあれば実行可能。引数のパスは未検査
```

### 保護すべき対象

```
保護ファイル（自己書き換えブロック）:
├── permissions.md    ← 権限定義。即時反映のため最も危険
├── IDENTITY.md       ← 人格の不変ベースライン
├── SOUL.md           ← 内面の価値観・原則
└── BOOTSTRAP.md      ← 初回起動指示

保護境界（他者アクセスブロック）:
├── ../other_person/  ← 他者のperson_dir全体
└── ../../config.json ← システム設定
```

---

## Phase 3a: ToolHandler層のハードニング（A2/Fallbackモード）

### 概要

ToolHandler内の全書き込み系ツールに対して、保護ファイルブロックとパストラバーサル防止を実装する。

### 実装内容

#### 1. 共通の保護定数と検査関数

**変更ファイル**: `core/tooling/handler.py`

```python
# モジュールレベル定数
_PROTECTED_FILES = frozenset({
    "permissions.md",
    "IDENTITY.md",
    "SOUL.md",
    "BOOTSTRAP.md",
})


def _is_protected_write(person_dir: Path, target: Path) -> str | None:
    """Check if a write target is a protected file.

    Returns error message if blocked, None if allowed.
    """
    resolved = target.resolve()
    person_resolved = person_dir.resolve()

    # Path traversal: target must be within person_dir
    if not resolved.is_relative_to(person_resolved):
        return _error_result(
            "PermissionDenied",
            f"Path resolves outside anima directory",
        )

    # Protected file check
    rel = str(resolved.relative_to(person_resolved))
    if rel in _PROTECTED_FILES:
        return _error_result(
            "PermissionDenied",
            f"'{rel}' is a protected file and cannot be modified by the anima itself",
        )

    return None
```

#### 2. write_memory_file の修正

```python
def _handle_write_memory_file(self, args: dict[str, Any]) -> str:
    rel_path = args["path"]
    path = self._person_dir / rel_path

    # Security check
    err = _is_protected_write(self._person_dir, path)
    if err:
        return err

    # ... existing write logic ...
```

#### 3. read_memory_file のパストラバーサル防止

```python
def _handle_read_memory_file(self, args: dict[str, Any]) -> str:
    path = self._person_dir / args["path"]

    # Prevent path traversal
    if not path.resolve().is_relative_to(self._person_dir.resolve()):
        return _error_result(
            "PermissionDenied",
            "Path resolves outside anima directory",
        )

    # ... existing read logic ...
```

#### 4. _check_file_permission の強化

`write_file` / `edit_file` が person_dir 内の保護ファイルに書き込むケースをブロック:

```python
def _check_file_permission(self, path: str, *, write: bool = False) -> str | None:
    resolved = Path(path).resolve()

    if resolved.is_relative_to(self._person_dir.resolve()):
        # Own person_dir: check protected files for write operations
        if write:
            err = _is_protected_write(self._person_dir, resolved)
            if err:
                return err
        return None

    # ... existing whitelist logic ...
```

呼び出し側を変更:
- `_handle_write_file`: `self._check_file_permission(path_str, write=True)`
- `_handle_edit_file`: `self._check_file_permission(path_str, write=True)`
- `_handle_read_file`: `self._check_file_permission(path_str)` （変更なし）

#### 5. execute_command の引数パス検査（限定的）

完全な検査は現実的に困難だが、明らかな違反パターンをブロック:

```python
def _check_command_permission(self, command: str) -> str | None:
    # ... existing checks ...

    # Block commands targeting other animas' directories
    persons_root = str(self._person_dir.parent)
    for arg in argv[1:]:
        if ".." in arg:
            try:
                resolved = (self._person_dir / arg).resolve()
                if not resolved.is_relative_to(self._person_dir.resolve()):
                    return _error_result(
                        "PermissionDenied",
                        f"Command argument '{arg}' resolves outside anima directory",
                    )
            except (ValueError, OSError):
                pass

    return None
```

### テスト方針

- `write_memory_file(path="permissions.md", ...)` → ブロック
- `write_memory_file(path="IDENTITY.md", ...)` → ブロック
- `write_memory_file(path="BOOTSTRAP.md", ...)` → ブロック
- `write_memory_file(path="../other/knowledge/x.md", ...)` → ブロック
- `write_memory_file(path="knowledge/new.md", ...)` → 成功
- `write_memory_file(path="HEARTBEAT.md", ...)` → 成功
- `write_memory_file(path="cron.md", ...)` → 成功
- `read_memory_file(path="../other/IDENTITY.md")` → ブロック
- `read_memory_file(path="episodes/2026-02-15.md")` → 成功
- `write_file` で自person_dirの `permissions.md` → ブロック
- `edit_file` で自person_dirの `IDENTITY.md` → ブロック
- `write_file` で他者のperson_dir → ブロック（既存ロジックで対応済み）

---

## Phase 3b: PreToolUseフックの追加（A1モード）

### 概要

`agent_sdk.py` に `PreToolUse` フックを追加し、Claude Code native tools（Write/Edit/Bash）の
ツール呼び出しをインターセプトして、保護ファイルへのアクセスと他者ディレクトリへの
パストラバーサルをブロックする。

### 背景

Claude Agent SDKは `PreToolUse` フック機構を提供しており（`claude_agent_sdk/types.py`）、
ツール実行前に `permissionDecision="deny"` を返すことで呼び出しを拒否できる。

現在 AnimaWorks は `PostToolUse` フックのみ使用中（コンテキスト監視用、`agent_sdk.py:176-178`）。

### 実装内容

#### 1. PreToolUseフックの追加

**変更ファイル**: `core/execution/engines/claude/agent_sdk.py`

```python
async def _pre_tool_hook(
    input_data: HookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> SyncHookJSONOutput:
    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})

    # Write / Edit: check file path
    if tool_name in ("Write", "Edit"):
        file_path = tool_input.get("file_path", "")
        violation = _check_a1_file_access(file_path, person_dir, write=True)
        if violation:
            return SyncHookJSONOutput(
                hookSpecificOutput=PreToolUseHookSpecificOutput(
                    hookEventName="PreToolUse",
                    permissionDecision="deny",
                    permissionDecisionReason=violation,
                )
            )

    # Bash: inspect command for file operation patterns
    if tool_name == "Bash":
        command = tool_input.get("command", "")
        violation = _check_a1_bash_command(command, person_dir)
        if violation:
            return SyncHookJSONOutput(
                hookSpecificOutput=PreToolUseHookSpecificOutput(
                    hookEventName="PreToolUse",
                    permissionDecision="deny",
                    permissionDecisionReason=violation,
                )
            )

    # Read: check for path traversal to other animas
    if tool_name == "Read":
        file_path = tool_input.get("file_path", "")
        violation = _check_a1_file_access(file_path, person_dir, write=False)
        if violation:
            return SyncHookJSONOutput(
                hookSpecificOutput=PreToolUseHookSpecificOutput(
                    hookEventName="PreToolUse",
                    permissionDecision="deny",
                    permissionDecisionReason=violation,
                )
            )

    return SyncHookJSONOutput()
```

#### 2. ファイルアクセス検査関数

```python
_PROTECTED_FILES = frozenset({"permissions.md", "IDENTITY.md", "SOUL.md", "BOOTSTRAP.md"})

def _check_a1_file_access(
    file_path: str, person_dir: Path, *, write: bool
) -> str | None:
    """Check if a file path is allowed for A1 mode tools.

    Returns violation reason string if blocked, None if allowed.
    """
    if not file_path:
        return None

    resolved = Path(file_path).resolve()
    anima_resolved = anima_dir.resolve()
    animas_root = anima_resolved.parent

    # Block access to other animas' directories
    if resolved.is_relative_to(animas_root):
        if not resolved.is_relative_to(anima_resolved):
            return f"Access to other anima's directory is not allowed: {file_path}"

        # Block writes to protected files within own directory
        if write:
            rel = str(resolved.relative_to(anima_resolved))
            if rel in _PROTECTED_FILES:
                return f"'{rel}' is a protected file and cannot be modified"

    return None
```

#### 3. Bashコマンド検査関数

```python
_WRITE_COMMANDS = frozenset({"cp", "mv", "tee", "dd", "install", "rsync"})

def _check_a1_bash_command(command: str, anima_dir: Path) -> str | None:
    """Check bash commands for obvious file operation violations.

    This is a best-effort heuristic — not a complete sandbox.
    """
    import shlex

    try:
        argv = shlex.split(command)
    except ValueError:
        return None

    if not argv:
        return None

    cmd_base = Path(argv[0]).name

    # Check file-writing commands for path violations
    if cmd_base in _WRITE_COMMANDS:
        animas_root = str(anima_dir.parent.resolve())
        anima_resolved = str(anima_dir.resolve())
        for arg in argv[1:]:
            if arg.startswith("-"):
                continue
            try:
                resolved = str(Path(arg).resolve())
                # Writing to other anima's directory
                if resolved.startswith(animas_root) and not resolved.startswith(anima_resolved):
                    return f"Command targets other anima's directory: {arg}"
            except (ValueError, OSError):
                pass

    return None
```

#### 4. フックの登録

`ClaudeAgentOptions` の `hooks` に `PreToolUse` を追加:

```python
options = ClaudeAgentOptions(
    # ... existing options ...
    hooks={
        "PreToolUse": [HookMatcher(
            matcher="Write|Edit|Bash|Read",
            hooks=[_pre_tool_hook],
        )],
        "PostToolUse": [HookMatcher(matcher=None, hooks=[_post_tool_hook])],
    },
)
```

### 設計上の注意

- Bashコマンドの検査はヒューリスティックであり、完全なサンドボックスではない。
  リダイレクト (`>`, `>>`) はシェルレベルの構文でありargvには現れないが、
  Claude Agent SDKが `shell=False` で実行する場合は問題にならない。
  `shell=True` の場合は別途検討が必要。
- A1モードの制限は「明らかな違反をブロック」が目的であり、
  悪意あるプロンプトインジェクションへの完全防御ではない。
- ファイルパスの `cwd` は person_dir に設定済み（`agent_sdk.py:172`）のため、
  相対パスは person_dir 起点で解決される。

### テスト方針

- A1モードで `Write` ツールが `permissions.md` を書き換えようとした場合 → deny
- A1モードで `Edit` ツールが `IDENTITY.md` を編集しようとした場合 → deny
- A1モードで `Read` ツールが `../other_person/episodes/` を読もうとした場合 → deny
- A1モードで `Bash` ツールが `cp ../other_person/secret.md ./` を実行しようとした場合 → deny
- A1モードで通常のファイル操作（`knowledge/` への書き込み等）→ allow
- A1モードで `Bash` ツールが通常のコマンド（`ls`, `python` 等）→ allow

---

## Implementation Summary

| Phase | 内容 | 変更ファイル | 難易度 |
|---|---|---|---|
| **Phase 3a** | ToolHandler層のハードニング | `core/tooling/handler.py` | 中 |
| **Phase 3b** | A1モードPreToolUseフック | `core/execution/engines/claude/agent_sdk.py` | 中 |

## Related Issues

- `20260215_self-modify-heartbeat-cron.md` — ハートビート/cronの自己更新（Phase 1, 2）

## References

- `core/tooling/handler.py:173-185` — _handle_write_memory_file（保護なし）
- `core/tooling/handler.py:165-171` — _handle_read_memory_file（保護なし）
- `core/tooling/handler.py:228-240` — _handle_write_file
- `core/tooling/handler.py:242-264` — _handle_edit_file
- `core/tooling/handler.py:266-295` — _handle_execute_command
- `core/tooling/handler.py:454-487` — _check_file_permission（person_dir内は全許可）
- `core/execution/engines/claude/agent_sdk.py:168-179` — ClaudeAgentOptions（PostToolUseフックのみ）
- `core/agent/agent_core.py:186-195` — AgentSDKExecutor生成（ToolHandler未使用）
- `core/agent/agent_core.py:230-238` — LiteLLMExecutor生成（ToolHandler使用）
- `claude_agent_sdk/types.py` — PreToolUseHookSpecificOutput, permissionDecision
- `templates/prompts/environment.md` — 活動範囲ルール（プロンプトのみ）
