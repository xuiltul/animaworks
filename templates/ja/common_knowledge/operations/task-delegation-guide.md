## タスク実行の仕組み

### Task tool の自動ルーティング（Sモード）

Task tool を使うと、フレームワークがあなたの組織構成に基づいて自動的にルーティングする。

**部下がいる場合** → 部下に即時委任される
- description に部下名を含めると、その部下に指名委任される
  例: "alice にAPIテストを実施させる"
  例: "bob がコードレビューを担当する"
- 名前がなければ workload 最小 + role マッチで自動選択される
- 全部下が無効の場合は state/pending/ にフォールバック

**部下がいない場合** → バックグラウンドタスクとして投入される
- state/pending/ に書き出され、TaskExec が別セッションで自動実行する
- 実行者はあなたと同じ identity・行動指針・記憶ディレクトリ・組織情報を持つ
- task_id が返却される。完了時にDMで通知される
- Heartbeat でタスク結果を確認できる（state/task_results/）

### タスク投入ツールの使い分け

| 手段 | 目的 | 実行キュー (Layer 1) | 追跡 (Layer 2) | いつ使うか |
|------|------|---------------------|----------------|-----------|
| `submit_tasks` | タスクの実行投入・登録 | `state/pending/` に作成 | `task_queue.jsonl` に登録 | 実行が必要なタスク、人間指示の記録、手動着手予定のタスク |
| `delegate_task` | 部下へのタスク委譲 | 部下の `state/pending/` に作成 | 両者の `task_queue.jsonl` に登録 | 部下に任せるとき |
| Task tool (Sモード) | 自動ルーティング委任 | 自動選択先に作成 | 登録される | Chat パスでの簡易委任 |

**重要**: 人間からの指示は `submit_tasks` で記録が MUST。単一タスクでも `submit_tasks`（tasks 配列 1 件）を使う。

Heartbeat・Inbox など Task tool を使わないパスでは `submit_tasks` / `delegate_task` を使う。

**【MUST】`state/pending/` にJSONファイルを手動で作成してはならない。** 必ず `submit_tasks` ツール経由で投入すること。`submit_tasks` は実行キューとタスクレジストリの両方に同時登録するため、追跡漏れを防げる。

## submit_tasks によるタスク投入

`submit_tasks` は実行が必要なタスクを投入する唯一の手段（部下委譲を除く）。
単一タスクでも `submit_tasks`（tasks配列1件）を使う。

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
- ❌ `state/pending/` にJSONを手動作成（必ず `submit_tasks` を使うこと）

### タスク結果

完了したタスクの結果は `state/task_results/{task_id}.json` に保存される。
依存タスクには先行タスクの結果要約が自動的にコンテキストとして注入される。
