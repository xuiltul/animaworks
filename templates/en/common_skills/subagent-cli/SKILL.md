---
name: subagent-cli
description: >-
  Skill for running external AI agent CLIs (codex exec, cursor-agent -p) as subagents
  via Bash in non-interactive mode. Provides execution procedure, options, and output
  handling when delegating complex coding tasks, code review, or multi-file changes.
  Applies when Bash permission is available in Mode S/C/D/G/A/B. In Mode C/D/G (codex/* / cursor/* / gemini/*), the
  framework runs each engine directly, so manually invoking the matching CLI is usually unnecessary.
  "subagent", "codex", "cursor-agent", "write code", "implement",
  "code review", "refactoring"
---

# subagent-cli

Run external AI agent CLIs as subprocesses via Bash to delegate complex coding tasks.
Use as a "power tool" to extend execution capability while keeping your identity, judgment, and memory.

## Relationship with Framework Execution Modes

This skill applies **only when the Bash tool is available**.

| Mode | Implementation | Bash | Skill Applicability |
|------|----------------|------|---------------------|
| **Mode S** | `agent_sdk.py` (Claude Agent SDK) | Available by default | Applies. Read/Write/Edit/Bash/Grep/Glob/WebFetch/WebSearch available |
| **Mode C** | `codex_sdk.py` (Codex SDK) | Depends on Codex CLI toolset | **codex exec not needed** — Framework runs Codex directly. cursor-agent / claude -p can be invoked via Bash (when Bash is available) |
| **Mode D** | Cursor Agent (cursor-agent subprocess) | Depends on Cursor CLI toolset | **cursor-agent -p not needed** — Framework runs cursor-agent directly. MCP integration. Tool access similar to Mode S but via the cursor-agent binary. codex exec / claude -p can be invoked via Bash (when Bash is available) |
| **Mode G** | Gemini CLI (gemini subprocess) | Depends on Gemini CLI toolset | **Manual gemini invocation not needed** — Framework runs it directly. MCP integration, stream-json output. Other CLIs can be invoked via Bash (when Bash is available) |
| **Mode A/B** | LiteLLM + tool_use / 1-shot | Only when permitted in permissions.json | Applies if Bash is permitted |

**Important**: For Mode C (`codex/*`), Mode D (`cursor/*`), and Mode G (`gemini/*`), the framework runs each engine directly. You do not need to call `codex exec` (Mode C), `cursor-agent -p` (Mode D), or the Gemini CLI (Mode G) from Bash yourself. Refer to the relevant sections only when you explicitly want a different CLI (cursor-agent, claude -p, codex exec, etc.).

## Tool Selection Priority

**Choose by cost efficiency.**

| Priority | Tool | Cost | Best For |
|----------|------|------|----------|
| 1 | `codex exec` | Lowest (Codex) | Code generation, editing, review |
| 2 | `cursor-agent -p` | Low (Cursor) | Code generation, editing, multi-file |
| 3 | `claude -p` | High (Claude API) | Last resort. Only when the above two fail |

**Rule: Try codex exec first. Fall back to cursor-agent → claude only on failure or unsuitable tasks.**

## When to Use

- Multi-file code changes
- Test creation or modification
- Code review
- Refactoring
- Bug investigation and implementation
- New feature implementation

## When NOT to Use

- Small edit in a single file (do it yourself)
- Memory read/write (use your tools)
- External API calls (use dedicated tools)
- Search or research only (web_search or Read is enough)

---

## 1. codex exec (Recommended)

**Applicability**: Mode S or Mode A/B (with Bash permission). Not needed in Mode C — the framework runs Codex directly. In Mode D/G, the framework runs those engines; skip this section unless you intentionally use codex as an alternative.

### Basic Syntax

```bash
codex exec --full-auto -C /path/to/workspace "prompt"
```

Specify the project path for working directory `-C`. For the main project, `$ANIMAWORKS_PROJECT_DIR` may be available (set in Mode S Bash execution environment).

### Key Options

| Option | Description |
|--------|-------------|
| `--full-auto` | Auto-approve + sandbox (workspace-write) |
| `-C /path` | Working directory (required) |
| `-m model` | Model (e.g., `o4-mini`, `o3`) |
| `--sandbox workspace-write` | Workspace write permission (included in full-auto) |
| `--json` | JSONL output |
| `-o file` | Write final message to file |
| `--ephemeral` | Do not save session file |

### Examples

#### Code Generation

```bash
codex exec --full-auto --ephemeral -C /home/main/dev/myproject \
  "Implement Markdown parser in src/utils/parser.py. Do not break existing tests."
```

#### Code Review

```bash
codex exec --full-auto --ephemeral -C /home/main/dev/myproject \
  review
```

#### Test Creation

```bash
codex exec --full-auto --ephemeral -C /home/main/dev/myproject \
  "Create unit tests for src/utils/parser.py in tests/test_parser.py."
```

#### Save Result to File

```bash
codex exec --full-auto --ephemeral -C /home/main/dev/myproject \
  -o /tmp/codex_result.txt \
  "Analyze this project's architecture and suggest improvements."
```

---

## 2. cursor-agent -p (Alternative)

**Applicability**: Mode S or Mode A/B (with Bash permission). Also applies in Mode C/G when Bash is available. In Mode D, the framework runs cursor-agent — manual `cursor-agent -p` is usually unnecessary.

### Basic Syntax

```bash
cursor-agent -p --trust --force --workspace /path/to/workspace "prompt"
```

### Key Options

| Option | Description |
|--------|-------------|
| `-p` / `--print` | Non-interactive mode (required) |
| `--trust` | Auto-trust workspace |
| `--force` | Auto-approve commands |
| `--workspace /path` | Working directory (required) |
| `--model model` | Model (e.g., `sonnet-4`, `gpt-5`) |
| `--output-format text\|json` | Output format |
| `--mode plan\|ask` | Read-only mode (for investigation) |

### Examples

#### Code Generation

```bash
cursor-agent -p --trust --force \
  --workspace /home/main/dev/myproject \
  "Add POST /users endpoint to src/api/routes.py. Include validation."
```

#### Read-Only Investigation

```bash
cursor-agent -p --trust --mode ask \
  --workspace /home/main/dev/myproject \
  "Are there security issues in this auth flow?"
```

#### Save Result to File

```bash
cursor-agent -p --trust --force \
  --workspace /home/main/dev/myproject \
  --output-format text \
  "Find modules with low test coverage and improve them" > /tmp/cursor_result.txt
```

---

## 3. claude -p (Fallback)

**Applicability**: Mode S or Mode A/B (with Bash permission). Also applies in Mode C/D/G when Bash is available.

Use only when codex/cursor-agent cannot handle the task. API cost is high.

### Basic Syntax

```bash
claude -p --dangerously-skip-permissions --output-format text "prompt"
```

### Key Options

| Option | Description |
|--------|-------------|
| `-p` / `--print` | Non-interactive mode (required) |
| `--dangerously-skip-permissions` | Skip permission check |
| `--model model` | Model (e.g., `sonnet`, `haiku`) |
| `--allowedTools "tools"` | Restrict allowed tools (e.g., `"Read Edit Bash(git:*)"`) |
| `--output-format text\|json` | Output format |
| `--max-budget-usd N` | Cost cap (USD) |
| `--no-session-persistence` | Do not save session |

### Example

```bash
claude -p --dangerously-skip-permissions --no-session-persistence \
  --model haiku --max-budget-usd 0.5 \
  --output-format text \
  "Improve error handling in src/core/parser.py"
```

---

## Writing Prompts

Subagents do not have AnimaWorks context. Write clear, self-contained prompts.

### Good Prompt

```
Implement a Python module with these requirements:

File: src/utils/validator.py

Requirements:
- Pydantic v2 BaseModel-based validator
- email, username, password fields
- Password: 8+ chars, alphanumeric
- Raise custom exception on validation error

Constraints:
- from __future__ import annotations at top
- Google-style docstring
- Do not break existing tests
```

### Bad Prompt

```
Fix the validation somehow
```

→ No context, "somehow" is vague.

---

## Handling Output

### Capture stdout

```bash
RESULT=$(codex exec --full-auto --ephemeral -C /path "prompt" 2>/dev/null)
echo "$RESULT"
```

### Via File (Recommended for codex)

```bash
codex exec --full-auto --ephemeral -C /path \
  -o /tmp/result.txt "prompt"
# Read result
cat /tmp/result.txt
```

### Success/Failure from Exit Code

```bash
codex exec --full-auto --ephemeral -C /path "prompt"
if [ $? -eq 0 ]; then
  echo "Success"
else
  echo "Failed — fallback to cursor-agent"
  cursor-agent -p --trust --force --workspace /path "same prompt"
fi
```

---

## Background Execution (Important)

Subagent runs can take **5–20+ minutes**.
Foreground execution blocks the session, so **always run in the background**.

### Basic Pattern: nohup + Result File

```bash
nohup codex exec --full-auto --ephemeral -C /path/to/workspace \
  -o /tmp/codex_result.txt \
  "prompt" > /tmp/codex_stdout.log 2>&1 &
echo "PID: $!"
```

For cursor-agent:

```bash
nohup cursor-agent -p --trust --force \
  --workspace /path/to/workspace \
  "prompt" > /tmp/cursor_result.txt 2>&1 &
echo "PID: $!"
```

### Completion Check

```bash
# Check if process is still running
ps -p <PID> > /dev/null 2>&1 && echo "Running" || echo "Done"

# Read result (after completion)
cat /tmp/codex_result.txt
# or
cat /tmp/cursor_result.txt
```

### Timeout

Use `timeout` to avoid runaway runs:

```bash
nohup timeout 30m codex exec --full-auto --ephemeral -C /path \
  -o /tmp/codex_result.txt \
  "prompt" > /tmp/codex_stdout.log 2>&1 &
```

- Recommended timeout: **30 min** (`30m`)
- Small tasks: **10 min** (`10m`)
- Large refactors: **60 min** (`60m`)

### Continue Other Work While Running

After background run, you may proceed with other tasks without waiting.
Periodically check process status; when done, read the result and record in episodes/.

---

## Safety Guidelines

1. **Always specify working directory** — Otherwise runs in current directory
2. **Do not include secrets in prompts** — API keys, passwords, etc.
3. **codex runs in sandbox with `--full-auto`** — Writes outside workspace are restricted
4. **Check changes with git diff after execution** — Verify no unintended changes
5. **Use `--ephemeral`** — Prevents session file accumulation

---

## Fallback Strategy

```
1. Try codex exec
   ↓ failure or poor quality
2. Retry with cursor-agent -p
   ↓ failure or poor quality
3. Final attempt with claude -p (with --max-budget-usd)
   ↓ still failure
4. Try yourself or report to supervisor
```

## Notes

- Subagents cannot access AnimaWorks memory or tools. They are "coding hands" only
- Record execution results in your episodes/ and accumulate patterns in knowledge/
- Runs take 5–20+ minutes. Always run in background and set timeout
- Work in git-tracked repositories (easier tracking and rollback)
- In Mode S, `ANIMAWORKS_ANIMA_DIR` and `ANIMAWORKS_PROJECT_DIR` are set as environment variables when Bash runs
