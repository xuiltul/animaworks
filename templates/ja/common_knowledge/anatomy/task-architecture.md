# タスクアーキテクチャ — 3層モデル

AnimaWorks のタスク管理は3つの層で構成される。
上位ほどシステムが厳格に管理し、下位ほど Anima の自由裁量に委ねられる。

## 3層の概要

```
┌─────────────────────────────────────────────────┐
│  Layer 1: 実行キュー（Execution Queue）            │  ← 最も厳格。機械的に処理
│  state/pending/*.json                            │
├─────────────────────────────────────────────────┤
│  Layer 2: タスクレジストリ（Task Registry）         │  ← 構造化。ツール経由で管理
│  state/task_queue.jsonl                          │
├─────────────────────────────────────────────────┤
│  Layer 3: Anima ノート（Working Notes）            │  ← 自由形式。自己管理
│  state/current_task.md / state/pending.md        │
└─────────────────────────────────────────────────┘
```

## Layer 1: 実行キュー（state/pending/*.json）

メッセージキュー（SQS / RabbitMQ）に相当する。

| 特性 | 説明 |
|------|------|
| フォーマット | JSON（スキーマ固定） |
| ライフサイクル | 投入 → 消費 → 削除（一時的） |
| 管理主体 | システム（PendingTaskExecutor が自動消費） |
| 書き込み元 | `submit_tasks`, `delegate_task`, SDK Task/Agent tool |
| 読み取り元 | PendingTaskExecutor（3秒ポーリング） |

タスクの完全な記述（description, acceptance_criteria, constraints, depends_on, workspace 等）を含む。
PendingTaskExecutor が検出すると `processing/` に移動して実行し、完了後に削除する。
失敗時は `failed/` に移動する。

タスクに `workspace` フィールドがある場合、レジストリで解決した絶対パスが `working_directory` として TaskExec のプロンプトに注入される。解決順序: タスクの `workspace` → `status.json` の `default_workspace` → なし。

Anima はこの層を直接操作しない。ツール経由で間接的に書き込む。

## Layer 2: タスクレジストリ（state/task_queue.jsonl）

イシュートラッカー（Jira / GitHub Issues）に相当する。

| 特性 | 説明 |
|------|------|
| フォーマット | append-only JSONL（TaskEntry スキーマ） |
| ライフサイクル | 登録 → ステータス遷移 → compact でアーカイブ（永続的） |
| 管理主体 | Anima（ツール経由） + システム（Priming 注入、compact） |
| 書き込み | `submit_tasks`, `update_task`, `delegate_task` |
| 読み取り | `format_for_priming`, Heartbeat compact（一覧は CLI: animaworks-tool task list） |

タスクの要約情報（task_id, summary, status, deadline, assignee）を保持する。
Priming の Channel E で pending / in_progress タスクがシステムプロンプトに注入される。
「何をやるべきか」の公式記録であり、人間からのタスク（source=human）は必ずここに登録する。

## Layer 3: Anima ノート（state/current_task.md, state/pending.md）

付箋メモ・個人ノートに相当する。

| 特性 | 説明 |
|------|------|
| フォーマット | Markdown（自由形式） |
| ライフサイクル | Anima が自由に作成・更新・削除 |
| 管理主体 | Anima（完全な裁量） |
| 書き込み | Anima が直接ファイル操作 |
| 読み取り | Anima 自身、Priming（current_task.md のみ） |

`current_task.md` は「今まさにやっていること」、`pending.md` は「やるべきことの自分用メモ」。
Layer 2 と内容が重複してもよい。Layer 3 は Anima の思考の場であり、自分の言葉で整理する場所。

## 層間の関係

### データの流れ

```
人間の指示 ─┬─► submit_tasks ─────────────────► Layer 2 (task_queue.jsonl)
            └─► Anima が current_task.md に記録 ► Layer 3

submit_tasks ─┬─► state/pending/*.json ──────► Layer 1 (実行キュー)
            └─► task_queue.jsonl に登録 ────► Layer 2 (タスクレジストリ)

delegate_task ─┬─► 部下の state/pending/ ──► Layer 1
               ├─► 部下の task_queue.jsonl ► Layer 2
               └─► 自分の task_queue.jsonl ► Layer 2 (status=delegated)

PendingTaskExecutor ─┬─► 完了 → task_queue を done に更新
                     └─► 失敗 → task_queue を failed に更新
```

### 同期ルール

| イベント | Layer 1 | Layer 2 | Layer 3 |
|---------|---------|---------|---------|
| submit_tasks 投入 | JSON 作成 | pending で登録 | — |
| delegate_task 投入 | JSON 作成（部下） | 両者に登録 | — |
| TaskExec 完了 | JSON 削除 | done に更新 | — |
| TaskExec 失敗 | failed/ に移動 | failed に更新 | — |
| Anima が着手 | — | in_progress に更新 | current_task.md 更新 |
| Anima が完了 | — | done に更新 | idle に戻す |
| Heartbeat 後 | — | compact 実行 | — |

### 各層が「知らなくてよい」こと

- **Layer 1** は Layer 2/3 の存在を知らない（PendingTaskExecutor は JSON を消費するだけ）
- **Layer 3** は Layer 1/2 の存在を知らなくてよい（Anima の自由メモ）
- **Layer 2** が Layer 1 と Layer 3 を橋渡しする中心的な追跡レイヤー

## 設計原則

1. **全てのタスクは Layer 2 に登録される**: submit_tasks, delegate_task いずれの経路でも task_queue.jsonl にエントリが存在する
2. **Layer 1 は一時的**: 実行キューのファイルは消費されたら消える。永続的な記録は Layer 2 が担う
3. **Layer 2 がSSoT**: タスクの「公式な状態」は task_queue.jsonl のステータスで判定する
4. **Layer 3 は自由**: Anima のワーキングメモリであり、システムは制約を課さない
5. **PendingTaskExecutor は Layer 2 を更新する**: 完了・失敗時に task_queue.jsonl のステータスを同期する
