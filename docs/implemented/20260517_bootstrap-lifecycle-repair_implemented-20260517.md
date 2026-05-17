# Bootstrap Lifecycle Repair — 初期設定失敗を完了扱いにせず自己修復する

## Overview

Anima の初回セットアップを `bootstrap.md` の有無だけで判定する実装を廃止し、明示的な bootstrap lifecycle state と修復 CLI を導入する。blank Anima はユーザーとの対話で「どんな存在にしたいですか？」から始め、character sheet がある Anima だけ background bootstrap を自動実行する。

## Problem / Background

### Current State

- 起動直後、`needs_bootstrap` が true なら background bootstrap が自動実行される。`core/supervisor/manager.py:336`
- `needs_bootstrap` は `bootstrap.md` の存在だけで判定される。`core/anima.py:368`
- `run_bootstrap()` 後に `bootstrap.md` が残っていると、内容検証なしで `bootstrap.md.auto_resolved` に退避される。`core/_anima_messaging.py:242`
- blank Anima の bootstrap template は、identity が未定義の場合にユーザーへ「私をどんな存在にしたいですか？」と聞くよう指示している。`templates/ja/bootstrap.md:21`
- UI は `bootstrapping` / `failed` / `max_retries` の表示を持つが、runtime 側の lifecycle 判定は `bootstrap.md` に依存している。`server/static/pages/chat/chat-renderer.js:77`

### Root Cause

1. **完了判定がファイル有無だけ** — `core/anima.py:368`  
   `identity.md` / `injection.md` の未定義状態や bootstrap の実成果を検証していない。

2. **interactive bootstrap と background bootstrap が分離されていない** — `core/supervisor/manager.py:344`, `templates/ja/bootstrap.md:21`  
   ユーザー入力が必要な blank Anima でも、サーバ起動直後に background bootstrap が走る。

3. **安全弁が失敗を成功に見せる** — `core/_anima_messaging.py:242`  
   `bootstrap.md.auto_resolved` はループ防止にはなるが、未定義 identity のまま通常稼働へ進ませる。

4. **修復手順が手作業に散らばる** — `cli/parser.py:321`, `cli/commands/anima_mgmt.py:314`  
   会話履歴、prompt logs、episodes、shortterm、vectordb、thread id、retry count をまとめて安全に掃除する CLI がない。

### Impact

| Component | Impact | Description |
|-----------|--------|-------------|
| `core/_anima_messaging.py` | Direct | bootstrap 失敗時に `.auto_resolved` へ逃げ、未完成状態を隠す |
| `core/supervisor/manager.py` | Direct | blank Anima にも background bootstrap を自動投入する |
| `core/anima.py` | Direct | `needs_bootstrap` が `bootstrap.md` の有無だけで、状態を表現できない |
| `server/routes/animas.py` | Indirect | UI に `pending_user_input` / `needs_repair` を返せない |
| `cli/commands/anima_mgmt.py` | Indirect | runtime の clean reset / retry repair が手作業になる |

## Decided Approach / 確定方針

### Design Decision

確定: `state/bootstrap_state.json` を導入し、bootstrap を `pending_user_input`, `running`, `completed`, `failed`, `needs_repair` の明示状態で管理する。blank Anima は background bootstrap を起動せず `pending_user_input` として通常チャットへ開放し、bootstrap template に従ってユーザーとの対話で初期設定を進める。character sheet が存在する場合だけ background bootstrap を許可する。bootstrap 完了時は validator を通し、未定義 identity / injection や `.auto_resolved` を成功扱いにしない。

### Rejected Alternatives

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Bootstrap template だけ修正 | 実装量が少ない | `needs_bootstrap` と `.auto_resolved` の欠陥が残り、未完成状態を検出できない | **Rejected**: 状態管理の根本原因を解決しない |
| Retry 回数だけ増やす | 一時的な API 失敗には効く | ユーザー入力待ちの blank Anima は何度実行しても完了しない | **Rejected**: interactive と background の分離がない |
| 常に background bootstrap を継続 | 自動化に寄せられる | ユーザーが初期人格を決める flow と競合し、会話に割り込む | **Rejected**: blank 初期作成の UX を壊す |
| 手順書で手動復旧 | 実装不要 | 複数ファイル掃除が必要で、再事故とデータ消し過ぎのリスクが高い | **Rejected**: self-repair 要件を満たさない |
| **State + validator + repair CLI** | 状態が明示され、UI/CLI/runner が同じ真実を見られる | 実装範囲は広い | **Adopted**: 今回の事故原因をすべて閉じられる |

### Key Decisions from Discussion

1. **`bootstrap.md` の有無だけで完了判定しない** — 理由: 今回の事故の直接原因であり、未定義 identity を見逃すため。
2. **未定義 identity かつ `character_sheet.md` なしは `pending_user_input` にする** — 理由: ユーザーに「どんな存在にしたいですか？」と聞く必要があるため。
3. **`character_sheet.md` ありだけ background bootstrap を自動実行する** — 理由: ユーザー入力なしで初期設定できる十分な入力があるため。
4. **`bootstrap.md.auto_resolved` は成功ではなく `needs_repair` にする** — 理由: ループ防止と完了判定を混同しないため。
5. **`repair-bootstrap --fresh` は model / credential / execution_mode を保持する** — 理由: API 設定まで消すと復旧負荷が高くなるため。
6. **修復時は archive を作る** — 理由: 会話履歴や logs を消す操作が含まれるため、誤操作時に戻せる状態を残す。

### Changes by Module

| Module | Change Type | Description |
|--------|------------|-------------|
| `core/bootstrap_state.py` | New | bootstrap state 読み書き、状態推定、validator、repair helper を実装する |
| `core/anima.py` | Modify | `needs_bootstrap` を state-aware にし、`bootstrap_state` / `needs_background_bootstrap` を追加する |
| `core/_anima_messaging.py` | Modify | `.auto_resolved` 自動成功扱いを廃止し、bootstrap run 後に validator / finalizer を呼ぶ |
| `core/supervisor/manager.py` | Modify | 起動直後は `needs_background_bootstrap` が true の場合だけ `_run_bootstrap()` を投入する |
| `core/supervisor/runner.py` | Modify | `get_status` に `bootstrap_state`, `needs_user_input`, `needs_repair`, `needs_background_bootstrap` を含める |
| `server/routes/animas.py` | Modify | `/api/animas` の応答に bootstrap state を含める |
| `server/static/pages/chat/*` | Modify | `pending_user_input` は通常チャット可能、`needs_repair` はエラー表示と repair guidance を出す |
| `cli/parser.py` | Modify | `animaworks anima repair-bootstrap` subcommand を追加する |
| `cli/commands/anima_mgmt.py` | Modify | `cmd_anima_repair_bootstrap` を追加する |
| `templates/ja/bootstrap.md`, `templates/en/bootstrap.md` | Modify | interactive と character-sheet bootstrap の前提を明記する |
| `tests/unit/test_bootstrap.py` | Modify | validator / state transition tests を追加する |
| `tests/e2e/test_bootstrap_flow.py` | Modify | blank と character-sheet の起動挙動を分けて検証する |

### State Schema

Target file: `state/bootstrap_state.json`

```json
{
  "version": 1,
  "state": "pending_user_input",
  "mode": "interactive",
  "reason": "identity_undefined_without_character_sheet",
  "started_at": null,
  "completed_at": null,
  "updated_at": "2026-05-17T00:00:00+09:00",
  "last_error": "",
  "validation_errors": [],
  "retry_count": 0
}
```

Allowed `state` values:

| State | Meaning | Background bootstrap |
|-------|---------|----------------------|
| `pending_user_input` | blank Anima がユーザー入力待ち | 起動しない |
| `running` | background bootstrap 実行中 | 実行中 |
| `completed` | validator 通過済み | 起動しない |
| `failed` | 実行例外または timeout | retry policy に従う |
| `needs_repair` | 部分初期化、`.auto_resolved`、未定義 identity など | 起動しない |

### Validator Rules

`validate_bootstrap(anima_dir)` は次を検証する。

| Rule | Failure state |
|------|---------------|
| `identity.md` が存在し、`未定義` / `undefined` を含まない | `needs_repair` |
| `injection.md` が存在し、`未定義` / `undefined` を含まない | `needs_repair` |
| `bootstrap.md.auto_resolved` / `bootstrap.md.failed` が存在しない | `needs_repair` |
| `character_sheet.md` が処理済みで残っていない | `needs_repair` |
| bootstrap 起点の pending task が残っていない | `needs_repair` |

Safe finalization:

- identity / injection が定義済みで、`bootstrap.md` だけが残っている場合は、`state/bootstrap_archive/bootstrap-<timestamp>.md` に archive してから `bootstrap.md` を削除し、`completed` にする。
- identity / injection が未定義の場合は `bootstrap.md` を残し、`needs_repair` にする。

### CLI Contract

Command:

```bash
animaworks anima repair-bootstrap <name> --status
animaworks anima repair-bootstrap <name> --retry
animaworks anima repair-bootstrap <name> --fresh
```

Behavior:

| Option | Behavior |
|--------|----------|
| `--status` | validator を実行し、state, reason, validation_errors, suggested action を表示する。ファイル変更なし |
| `--retry` | `.auto_resolved` / `.failed` を `bootstrap.md` に戻し、retry count / shortterm / Codex thread id / incomplete journals を掃除する。model 設定は保持 |
| `--fresh` | Anima ディレクトリを archive し、blank template から再作成する。`status.json` の `model`, `credential`, `execution_mode`, `background_model`, `background_credential` は保持する。会話履歴、activity log、prompt log、episodes、token_usage、shortterm、vectordb、task queue、repair artifacts は新規状態にする |

`--retry` と `--fresh` は同時指定不可。server が running の場合は対象 Anima を API で disable/stop し、失敗したら操作を中止して明示エラーを出す。

## Edge Cases

| Case | Handling |
|------|----------|
| blank Anima, `identity.md` 未定義, `character_sheet.md` なし | `pending_user_input`; background bootstrap を起動しない; chat は通常通り通す |
| blank Anima にユーザーが最初の chat を送る | `bootstrap.md` が system prompt に含まれ、「どんな存在にしたいですか？」から interactive bootstrap が進む |
| `character_sheet.md` あり | `needs_background_bootstrap=true`; background bootstrap を起動する |
| LLM が files を更新したが `bootstrap.md` を消し忘れた | validator が identity/injection 定義済みなら archive 後に削除し `completed` |
| LLM が `submit_tasks` へ bootstrap 作業を逃がした | pending task が残っている間は `completed` にしない |
| `.auto_resolved` が既に存在する既存 runtime | `needs_repair`; `repair-bootstrap --status` が原因として表示する |
| `--fresh` 実行時に API key が config にある | API key は表示しない。Anima の `status.json` 参照名だけ保持する |
| server 停止中に `--fresh` | offline mode で実行し、server API は呼ばない |
| server 稼働中に `--fresh` | 対象 Anima を stop/disable してから実行する。停止できない場合はファイル操作を行わない |

## Implementation Plan

### Phase 1: Bootstrap State Core

| # | Task | Target |
|---|------|--------|
| 1-1 | `BootstrapState` dataclass と JSON read/write を実装 | `core/bootstrap_state.py` |
| 1-2 | `derive_bootstrap_state(anima_dir)` を実装し、既存 runtime も状態推定できるようにする | `core/bootstrap_state.py` |
| 1-3 | `validate_bootstrap(anima_dir)` と safe finalizer を実装 | `core/bootstrap_state.py` |
| 1-4 | unit tests を追加 | `tests/unit/test_bootstrap.py` |

**Completion condition**: `.auto_resolved`, 未定義 identity, character sheet 有無、bootstrap.md 残存を state と validation errors に正しく変換できる。

### Phase 2: Runtime Integration

| # | Task | Target |
|---|------|--------|
| 2-1 | `DigitalAnima.bootstrap_state`, `needs_background_bootstrap`, state-aware `needs_bootstrap` を追加 | `core/anima.py` |
| 2-2 | `run_bootstrap()` の `.auto_resolved` rename を削除し、validator / finalizer に置換 | `core/_anima_messaging.py` |
| 2-3 | supervisor 起動時の判定を `needs_background_bootstrap` に変更 | `core/supervisor/manager.py` |
| 2-4 | `get_status` に bootstrap state fields を追加 | `core/supervisor/runner.py` |

**Completion condition**: blank Anima は起動直後に background bootstrap されず、character-sheet Anima だけ background bootstrap される。

### Phase 3: CLI Repair

| # | Task | Target |
|---|------|--------|
| 3-1 | `repair-bootstrap` parser を追加 | `cli/parser.py` |
| 3-2 | `cmd_anima_repair_bootstrap` を実装 | `cli/commands/anima_mgmt.py` |
| 3-3 | archive / retry / fresh helper を実装 | `cli/commands/anima_mgmt.py` または `core/bootstrap_state.py` |
| 3-4 | CLI tests を追加 | `tests/unit` |

**Completion condition**: `--status`, `--retry`, `--fresh` が仕様通り動き、危険操作は archive 後に実行される。

### Phase 4: UI / API Surface

| # | Task | Target |
|---|------|--------|
| 4-1 | `/api/animas` に bootstrap state fields を追加 | `server/routes/animas.py` |
| 4-2 | chat page で `pending_user_input` は通常チャット、`needs_repair` はエラー表示にする | `server/static/pages/chat/*` |
| 4-3 | i18n 文言を追加 | `server/static/i18n/*.json` |

**Completion condition**: UI が `pending_user_input` を「ユーザー入力待ち」として扱い、`needs_repair` を「修復が必要」として表示する。

### Phase 5: E2E Tests

| # | Task | Target |
|---|------|--------|
| 5-1 | blank Anima 起動で background bootstrap が走らない test | `tests/e2e/test_bootstrap_flow.py` |
| 5-2 | character-sheet Anima 起動で background bootstrap が走る test | `tests/e2e/test_bootstrap_flow.py` |
| 5-3 | incomplete bootstrap が `needs_repair` になる test | `tests/e2e/test_bootstrap_reconciliation_e2e.py` |

**Completion condition**: lifecycle 分岐が E2E で再現され、今回の事故パターンが regression test になる。

## Scope

### In Scope

- bootstrap state file の導入
- background bootstrap の起動条件変更
- bootstrap validator / safe finalizer
- `repair-bootstrap` CLI
- API / UI status surface の追加
- unit / e2e / CLI tests

### Out of Scope

- avatar realistic asset 生成の FAL/Ollama 問題修正 — bootstrap lifecycle とは別問題
- UI の大規模 redesign — 状態表示とチャット可否だけ変更する
- 全既存 Anima の自動一括修復 — `repair-bootstrap <name>` で個別対応する
- Azure/Codex provider 設定変更 — 既に別 Issue / 実装で扱ったため

## Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| 既存 tests が `needs_bootstrap == bootstrap.md exists` を前提にしている | unit/e2e の失敗 | state-aware helper を導入し、旧挙動が必要な箇所は `bootstrap_file_exists` に分離する |
| `--fresh` が必要な設定まで消す | runtime 復旧不能 | `status.json` の model / credential / execution_mode / background settings を保存して再適用する |
| background bootstrap が必要な Anima まで止まる | character sheet 作成 flow の regression | `character_sheet.md` 有無を `needs_background_bootstrap` の必須条件にする test を追加 |
| pending task 判定が広すぎる | completed にならない | task id prefix を `bootstrap-` に限定して validation する |
| UI が `pending_user_input` を error と表示する | 初期 UX 悪化 | API field と i18n を追加し、chat をブロックしない test を追加 |

## Acceptance Criteria

- [ ] blank `animaworks anima create --name midori` 後、server 起動直後に background bootstrap が走らず `bootstrap_state.state == "pending_user_input"` になる
- [ ] `pending_user_input` の Anima へ chat を送ると、bootstrap prompt を含む通常会話として処理される
- [ ] `character_sheet.md` がある Anima は server 起動後に background bootstrap が走る
- [ ] bootstrap run 後、`identity.md` または `injection.md` が未定義なら `completed` にならず `needs_repair` になる
- [ ] `bootstrap.md.auto_resolved` または `bootstrap.md.failed` が存在する Anima は `needs_repair` と診断される
- [ ] identity / injection が定義済みで `bootstrap.md` だけ残っている場合、safe finalizer が archive 後に `bootstrap.md` を削除し `completed` にする
- [ ] `animaworks anima repair-bootstrap midori --status` が state, reason, validation errors, suggested action を表示し、ファイルを書き換えない
- [ ] `animaworks anima repair-bootstrap midori --retry` が `.auto_resolved` / `.failed` を `bootstrap.md` に戻し、retry count / shortterm / Codex thread id / incomplete journals を掃除する
- [ ] `animaworks anima repair-bootstrap midori --fresh` が runtime 履歴を archive し、blank Anima を再作成し、model / credential / execution_mode を保持する
- [ ] server running 中の `--fresh` は対象 Anima の停止に失敗した場合、ファイル操作を行わずエラー終了する
- [ ] unit, e2e, CLI tests が追加され、今回の事故パターンが regression として固定される

## References

- `core/_anima_messaging.py:201` — background bootstrap 実行本体
- `core/_anima_messaging.py:242` — `bootstrap.md.auto_resolved` への現行退避処理
- `core/supervisor/manager.py:336` — 起動直後の background bootstrap 自動投入
- `core/supervisor/manager.py:364` — bootstrap retry / max retry 管理
- `core/anima.py:368` — `needs_bootstrap` の現行判定
- `core/anima_factory.py:708` — bootstrap file 作成条件
- `core/supervisor/runner.py:790` — Anima status IPC 応答
- `server/routes/animas.py:84` — `/api/animas` response assembly
- `templates/ja/bootstrap.md:21` — 未定義 identity でユーザー入力を求める現行指示
- `cli/parser.py:321` — `anima` subcommand parser
- `cli/commands/anima_mgmt.py:314` — destructive Anima operation の archive pattern
- `tests/unit/test_bootstrap.py:130` — bootstrap unit tests の既存配置
