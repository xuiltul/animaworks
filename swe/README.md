# SWE-bench Multi-Agent Evaluation

AnimaWorks の組織構造（Architect → Investigator → Reviewer）で SWE-bench Verified を解く評価基盤。

## Quick Start

### 1. Docker で実行

```bash
cd swe/

# 環境変数を設定
export ANTHROPIC_API_KEY=sk-...
export OPENAI_API_KEY=sk-...

# テスト問題で動作確認
docker compose run swe --setup-team --test

# SWE-bench Verified 5インスタンス実行
docker compose run swe --setup-team --run --instances 5
```

### 2. ローカルで実行

```bash
# チーム作成
python3 swe/team_setup.py setup

# テスト問題
python3 swe/runner.py --setup-team --test --port 18502

# SWE-bench Verified
python3 swe/runner.py --setup-team --run --instances 5 --port 18502
```

### 3. サーバーが既に起動中の場合

```bash
python3 swe/runner.py --no-server --test --port 18502
```

### 4. CI auto-fix loop v0

Failed GitHub Actions runs can be repaired by a small unattended loop:

```bash
# Detect the latest failed CI run for the current branch/HEAD, fetch failed logs,
# ask swe-architect to edit the checkout, run local gates, ask swe-reviewer, then commit.
python3 -m swe.ci_autofix --mode ci --repo xuiltul/animaworks --branch my-branch --max-iter 3
```

The default local quality gate is the v0 fast gate from Issue #205:

```bash
ruff check core/ cli/ server/
ruff format --check core/ cli/ server/
pytest tests/unit -m "not live and not azure and not ollama"
```

Use `--push` only when GitHub Actions will actually run for the branch, for example an open PR targeting
`main` or a workflow that accepts that branch. This repository's CI currently runs on `main` pushes and PRs to
`main`, so arbitrary branch pushes do not create CI runs by themselves.

For deterministic local repair without GitHub or LLM calls, provide a fixer and gate commands:

```bash
python3 -m swe.ci_autofix \
  --mode local \
  --fix-command "python3 /path/to/fixer.py" \
  --review-command "python3 /path/to/reviewer.py" \
  --gate-command "python3 -m pytest tests/unit/swe/test_ci_autofix.py -q"
```

After three failed attempts, the loop runs `animaworks-tool call_human ... --priority high`. If notification is
not configured, it writes a fallback summary to `swe/results/ci_autofix_escalation_*.md`.

## チーム構成

| Agent | Model | Role | Mode |
|-------|-------|------|------|
| swe-architect | claude-sonnet-4-6 | 分析・実装 | S (Agent SDK) |
| swe-investigator | openai/qwen3.5-35b-a3b | 調査・セカンドオピニオン | A (LiteLLM) |
| swe-reviewer | codex/gpt-5.4 | パッチレビュー | C (Codex CLI) |

## 処理フロー

```
Runner
  ├─ Phase 1: Architect が問題を分析（Mode S: Bash/Read/Grep で探索）
  ├─ Phase 2: Investigator がセカンドオピニオン（テキストベース）
  ├─ Phase 3: Architect がフィードバックを踏まえて実装
  ├─ Phase 4: Reviewer がパッチをレビュー
  └─ git diff → predictions.jsonl
```

## 結果

`swe/results/` に出力:
- `test_result.json` — テスト問題の結果
- `predictions.jsonl` — SWE-bench 予測（ハーネス入力形式）

## 設定

`swe/configs/team.json` でチーム構成をカスタマイズ可能。
