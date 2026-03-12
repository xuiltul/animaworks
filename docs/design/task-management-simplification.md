# タスク管理の簡素化提案

## 1. 現状の問題

### 1.1 構造が複雑すぎる

現在のタスク管理は3層で構成されている：

```
Layer 1: state/pending/*.json     ← TaskExec 自動実行キュー
Layer 2: state/task_queue.jsonl   ← append-only 追跡ログ（SSOT）
Layer 3: state/current_task.md    ← エージェント個人の作業メモ
```

さらに `shared/task-board.md` が Layer 2 から自動生成され、Slack に同期される。
エージェントごとに独立した task_queue.jsonl を持ち、委任時は両者に記録される。

### 1.2 登録パスが2つあり、挙動が異なる

| | `submit_tasks` | `backlog_task` |
|---|---|---|
| Layer 1 (pending JSON) | 作成する → TaskExec 自動実行 | **作成しない** |
| Layer 2 (task_queue.jsonl) | 追記する | 追記する |
| 完了時の状態更新 | TaskExec が自動で done/failed | エージェントが手動で update_task |

`submit_tasks` はライフサイクルが完結するが、`backlog_task` は状態更新がエージェントの自主行動に依存する。

### 1.3 blocked/stale タスクが放置される

- エージェントが `update_task(status="blocked")` でタスクをブロックした後、ブロッカーが解消されても自動的に状態遷移しない
- Priming で STALE/OVERDUE マーカーは表示されるが、それを見たエージェントが何をすべきかのガイダンスがない
- 人間がブロッカーを解決して Slack で報告しても、タスクステータスに反映されない

### 1.4 エージェントごとに分散した SSOT

- 各エージェントが独立した `task_queue.jsonl` を持つ
- 組織全体のタスク状況を把握するには全エージェントの JSONL を集約する必要がある
- 委任タスクは委任者と実行者の両方に記録され、重複排除ロジックが必要

### 1.5 人間がタスク状態に介入しにくい

- task_queue.jsonl は append-only JSONL で、人間が直接編集するものではない
- 人間が「このタスクやっておいた」と報告しても、エージェントが `update_task` を呼ぶまで反映されない
- 結果として人間はタスク管理システムの外で作業し、システム内の状態が実態と乖離する

## 2. 設計思想

### 2.1 参考: vibe-kanban のアプローチ

[vibe-kanban](https://github.com/BloopAI/vibe-kanban) は AI エージェントの複数タスク管理を以下のようにシンプルに実現している：

- 1つのカンバンボード（To Do / In Progress / In Review / Done / Cancelled）
- 全エージェント・人間が同じボードを見る
- タスクにはタイトル + 作業内容がそのまま書いてある
- ステータスはカラムを移動するだけ

### 2.2 目指す姿

**「1つの共有タスクボードを全員が見て、ステータスを移動するだけ」**

- 人間もエージェントも同じファイルを読み書きする
- 構造化データではなく、Markdown で管理する（AI が自然に読み書きできる）
- CEO エージェントがタスクの優先度管理・振り分け・リマインドを行う
- 人間が「これやっといたよ」→ CEO が「これですね、Done にします」で完結
- Slack にはこのボードの内容を同期するだけ

### 2.3 現状の良い部分は残す

- **TaskExec による自動実行**: 詳細仕様がある複雑なタスクは自動実行できる仕組みを維持
- **Priming によるコンテキスト注入**: エージェントが自分のタスクを認識できる仕組みを維持
- **監査証跡**: git 履歴で代替（append-only JSONL は不要）

## 3. 提案するアーキテクチャ

### 3.1 ファイル構成

```
shared/
├── task-board.md          ← SSOT（全員共通のカンバンボード）
└── tasks/                 ← 自動実行が必要なタスクの詳細仕様（オプション）
    ├── T-005.md
    └── T-008.md
```

**1層 + オプショナルな詳細ファイル**。2層構造は廃止。

### 3.2 task-board.md のフォーマット

```markdown
# タスクボード

最終更新: 2026-03-12 20:00 by kaede

## 🔴 Blocked
| ID | タスク | 担当 | ブロッカー | 期限 |
|----|--------|------|-----------|------|
| T-003 | GA4 Secret登録 | kai | 石井さん対応待ち | 3/8 |

## 📋 To Do
| ID | タスク | 担当 | 優先度 | 期限 |
|----|--------|------|--------|------|
| T-005 | SEO記事第5弾 | sena | P2 | 3/15 |

## 🟡 In Progress
| ID | タスク | 担当 | 期限 |
|----|--------|------|------|
| T-004 | リブランディング PR | kai | 3/13 |

## ✅ Done（直近7日）
| ID | タスク | 担当 | 完了日 |
|----|--------|------|--------|
| T-001 | 丸福塗料デモ | kai | 3/10 |
```

### 3.3 運用フロー

#### タスク追加

```
kaede: task-board.md の To Do にタスク行を追加
       → 詳細が必要なら tasks/T-XXX.md を作成
       → 担当者に send_message で通知
```

#### タスク実行

```
エージェント（heartbeat で task-board.md を確認）:
  → 自分が担当の To Do/In Progress を確認
  → 作業開始時: To Do → In Progress に移動
  → 完了時: In Progress → Done に移動
```

#### ブロック・解除

```
エージェント: In Progress → Blocked に移動 + ブロッカー列に理由を記載
人間: 「ブロッカー解決したよ」と Slack で報告
kaede: 「T-003 ですね。Done にしておきます」→ task-board.md を更新
```

#### TaskExec 自動実行（オプション）

```
tasks/T-005.md が存在する場合:
  → TaskExec がここから詳細仕様を読んで自動実行
  → 完了したら task-board.md の該当行を Done に移動
tasks/ がないシンプルなタスク:
  → エージェントが直接対応 → 手動で Done に移動
```

### 3.4 Priming への注入

task-board.md をパースし、担当名でフィルタしてシステムプロンプトに注入する：

```
# あなたの現在のタスク（task-board.md より）

🔴 BLOCKED [T-003] GA4 Secret登録 — ブロッカー: 石井さん対応待ち（期限: 3/8 ⚠️ OVERDUE）
🟡 IN_PROGRESS [T-004] リブランディング PR（期限: 3/13）
```

### 3.5 Slack 同期

task-board.md の内容を変換して Slack ピン留めメッセージを更新する（現在の taskboard_generator と同等）。

### 3.6 監査証跡

task-board.md は git 管理されているため、`git log -p shared/task-board.md` で全変更履歴を追跡可能。append-only JSONL による監査証跡は不要。

## 4. 移行パス

### Phase 1: task-board.md を SSOT に昇格

- task-board.md を直接編集する `edit_taskboard` ツールを追加
- Priming の読み取り元を task_queue.jsonl → task-board.md に変更
- backlog_task / update_task を task-board.md の行操作にリダイレクト

### Phase 2: task_queue.jsonl を廃止

- TaskExec の入力を pending JSON → tasks/*.md に変更
- task_queue.jsonl への書き込みを停止
- 既存の JSONL データは git 履歴として保存

### Phase 3: ツール統合

- backlog_task + update_task + submit_tasks を統合
- 新ツール: `add_task`, `move_task`, `exec_task`

## 5. 比較表

| 観点 | 現状（3層構造） | 提案（1層 + オプション詳細） |
|------|----------------|---------------------------|
| SSOT | task_queue.jsonl（エージェント別） | task-board.md（共有1ファイル） |
| ファイル数 | エージェント数 × 3 + 共有ボード | 1 + タスク詳細ファイル数 |
| 人間の介入 | JSONL は直接編集不可 | MD を直接編集可能 |
| blocked 解除 | 再評価メカニズムなし | ボードを見るだけで状態が分かる |
| 自動実行 | pending JSON 必須 | tasks/*.md があれば起動 |
| 監査証跡 | append-only JSONL | git diff |
| Priming | JSONL パース → フォーマット | MD パース → フィルタ |
| 委任の追跡 | relay_chain + 重複排除 | 担当列を変更するだけ |
| 学習コスト | 高（3層 + 2パス + 複数ツール） | 低（1ファイル + ステータス移動） |
