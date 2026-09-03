## タスク実行の仕組み

### タスク委譲の方法

> **注意**: Agent/Task ツール（サブエージェント起動）は**無効化**されている。タスクの委譲には `delegate_task` を使う。`submit_tasks` は Heartbeat と明示的なバックグラウンド実行で使える（通常チャット/Inbox/TaskExec には表示されない）。Heartbeat では自分の pending タスクの再投入に使う。

**部下がいる場合** → `delegate_task` で部下に委任する
- description に部下名を含めると、その部下に指名委任される
  例: "alice にAPIテストを実施させる"
  例: "bob がコードレビューを担当する"
- 名前がなければ workload 最小 + role マッチで自動選択される
- 全部下が無効の場合は state/pending/ にフォールバック

**部下がいない場合** → 通常チャットではこのセッションで直接実行する。Heartbeat または明示的なバックグラウンド実行ワークフローでは `submit_tasks` で自分の TaskExec に投入する
- state/pending/ に書き出され、TaskExec が別セッションで自動実行する
- 実行者はあなたと同じ identity・行動指針・記憶ディレクトリ・組織情報を持つ
- task_id が返却される。完了時にDMで通知される
- Heartbeat でタスク結果を確認できる（state/task_results/）

### タスク投入ツールの使い分け

| 手段 | 目的 | 実行キュー (Layer 1) | 追跡 (Layer 2) | いつ使うか |
|------|------|---------------------|----------------|-----------|
| `submit_tasks` | タスクの実行投入・登録 | `state/pending/` に作成 | `task_queue.jsonl` に登録 | Heartbeat で自分の pending を再投入するとき、明示的なバックグラウンド実行で自分のTaskExecに渡すとき |
| `delegate_task` | 部下へのタスク委譲 | 部下の `state/pending/` に作成 | 両者の `task_queue.jsonl` に登録 | 部下に任せるとき |

**重要**: 人間からの指示を受けた通常チャットでは `submit_tasks` を使わない。直接実行し、後続管理が必要な場合は `update_task`、`state/current_state.md`、または明示的なバックグラウンド実行ワークフローで記録する。

**【MUST】`state/pending/` にJSONファイルを手動で作成してはならない。** 明示的なバックグラウンド実行ワークフローで `submit_tasks` が表示されている場合のみ、そのツール経由で投入すること。

## 明示的バックグラウンド実行での submit_tasks

`submit_tasks` は通常チャットでは使わない。Heartbeat では自分の pending（前回の TaskExec が完了宣言なしで終わったものを含む）を同じ task_id で再投入するために使う。ハーネスは pending を自動で再実行しない。その他は、ユーザーまたはスキルが「バックグラウンドで」と明示し、ツール一覧に `submit_tasks` が表示されている場合だけ使う。単一タスクでも tasks 配列1件で投入する。

### 実行者（TaskExec）について

TaskExec はサブエージェントとして動作する。あなたと同じ identity・行動指針・記憶ディレクトリ・組織情報を持つが、**あなたの会話履歴・短期記憶・Priming結果にはアクセスできない**。

そのため、タスクの `description` と `context` に十分な情報を含めることが重要。

### description の記述原則

- **ファイルパスと行番号は必ず記載する**: 実行者は記憶検索ができるが、具体的な場所を指定した方が確実に正しいファイルに到達できる
- **現在の作業状態を含める**: current_state.md の関連部分を `context` フィールドにコピーすること（自動注入されるが、明示的に補足すると精度が上がる）
- **「なぜやるか」を明記する**: 背景と目的がないと実行者が判断を誤る

### description に含めるべき情報

- **何をするか**: 具体的な作業内容（「リファクタリングする」ではなく「core/auth/manager.py の verify_token() を async 化する」）
- **なぜやるか**: 背景と目的（1-2文）
- **どこを見るか**: 関連ファイルパスと行番号（`file_paths` フィールドにも記載）
- **完了条件**: 何をもって「できた」とするか（`acceptance_criteria` フィールドにも記載）
- **制約**: やってはいけないこと、互換性要件（`constraints` フィールドにも記載）

### 使用例

単一タスク:

```
submit_tasks(batch_id="hb-20260301-api-fix", tasks=[
  {{"task_id": "api-fix", "title": "API認証のasync化",
   "description": "core/auth/manager.py の verify_token()（L45-60）を async 化する。FastAPI の非同期ハンドラからの呼び出しでブロッキングが発生しているため。",
   "context": "current_state.md: API応答遅延の調査中。verify_token が同期I/Oでブロックしている",
   "file_paths": ["core/auth/manager.py:45"],
   "acceptance_criteria": ["verify_token が async def になっている", "既存テストが通る"],
   "constraints": ["公開APIの引数・戻り値を変えない"]}}
])
```

並列タスク:

```
submit_tasks(batch_id="deploy-20260301", tasks=[
  {{"task_id": "lint", "title": "Lint実行", "description": "全ファイルにlintを実行", "parallel": true}},
  {{"task_id": "test", "title": "テスト実行", "description": "ユニットテスト実行", "parallel": true}},
  {{"task_id": "deploy", "title": "デプロイ", "description": "lint・テスト通過後にデプロイ",
   "parallel": false, "depends_on": ["lint", "test"]}}
])
```

### タスクオブジェクト

| フィールド | 必須 | 説明 |
|-----------|------|------|
| `task_id` | MUST | バッチ内で一意のタスクID |
| `title` | MUST | タスクのタイトル |
| `description` | MUST | 具体的な作業内容（上記の記述原則に従う） |
| `parallel` | MAY | `true` で並列実行可能（デフォルト: `false`） |
| `depends_on` | MAY | 依存する先行タスクIDの配列 |
| `context` | MAY | 背景情報（current_state.md の関連部分を含める） |
| `file_paths` | MAY | 関連ファイルパス |
| `acceptance_criteria` | MAY | 完了条件 |
| `constraints` | MAY | 制約事項 |
| `reply_to` | MAY | 完了時の通知先 |

### 実行ルール

- `parallel: true` かつ依存関係なしのタスクはセマフォ制限内で同時実行される
- `depends_on` に指定された全タスクが成功完了してから実行される
- 先行タスクの結果は依存タスクのコンテキストに自動注入される
- 先行タスクが失敗した場合、依存タスクはスキップされる
- 循環依存はバリデーションで拒否される

### 禁止パターン

- ❌ 「適切にリファクタリングする」（曖昧すぎる）
- ❌ 「前回の続きをやる」（実行者は会話履歴を持たない）
- ❌ ファイルパスなしの指示（実行者は探索から始めることになる）
- ❌ context が空（背景情報なしでは実行者が判断を誤る）
- ❌ 通常チャット/Heartbeat/Inbox/TaskExec で `submit_tasks` を使おうとする
- ❌ `state/pending/` にJSONを手動作成（明示的バックグラウンド実行では必ず `submit_tasks` を使うこと）
- ❌ 他Animaのディレクトリ（`knowledge/` 等）への書き込み指示（部下はそのパスに書き込めない。共有には `common_knowledge/` を使うこと）

### タスク結果

完了したタスクの結果は `state/task_results/{task_id}.json` に保存される。
依存タスクには先行タスクの結果要約が自動的にコンテキストとして注入される。

## 委譲タスクの追跡

`task_tracker` ツールで委譲したタスクの進捗を確認できる。
部下側の task_queue.jsonl から最新ステータスを突き合わせて返す。

```
task_tracker()                     # アクティブな委譲タスク一覧（デフォルト）
task_tracker(status="all")         # 完了済み含む全タスク
task_tracker(status="completed")   # 完了済みのみ
```

| status | 意味 |
|--------|------|
| `active` | 進行中（done/cancelled/failed 以外）。デフォルト |
| `all` | 全件 |
| `completed` | 完了済み（done/cancelled/failed）のみ |

### 自動同期（sync_delegated）

Heartbeat 完了後に自動実行される。部下のタスクキューで以下の状態変化を検出し、上司側の追跡エントリ（`delegated` ステータス）を自動更新する:

- 部下側が `done` or `cancelled` → 上司側を `done` に更新
- 部下側が `failed` → 上司側を `failed` に更新
- アーカイブ済みタスクも検索対象（`task_queue_archive.jsonl`）

手動で `task_tracker` を呼ぶ必要はないが、Heartbeat 間の即時確認には引き続き `task_tracker` が有用。

## 投入前の照合と投入後の読み戻し（MUST）

- **書く前に読む**: `delegate_task` / `submit_tasks` / `backlog_task` を呼ぶ直前に `list_tasks(status="delegated")` と `list_tasks(status="in_progress")`（自分宛なら pending も）を読み、同じ PR / Issue / 対象の未完タスクが無いか確認する。あれば新しいタスクを作らず、既存の task_id を添えて担当者へ `send_message` で追加指示する。
- **書いた後に読む**: 返ってきた task_id を `list_tasks` で読み戻し、summary の先頭に `[PR #N]` / `[Issue #N]` / 対象名が付いて登録されていることを確認する。
- **重複を見つけたら**: 新しい方を残し、古い方を `update_task(status="cancelled", summary="<新ID>に統合")` にする。担当者側も同じ規則で動く。
- 委譲文には対象・URL・やってほしいこと・完了の目安だけを書く。台帳の状態語や手順条件で縛らない。
