## タスク実行の仕組み

### Task tool の自動ルーティング（Sモード）

Task tool を使うと、フレームワークがあなたの組織構成に基づいて自動的にルーティングする。

**部下がいる場合** → 部下に即時委任される
- description に部下名を含めると、その部下に指名委任される
  例: "alice にAPIテストを実施させる"
  例: "bob がコードレビューを担当する"
- 名前がなければ workload 最小 + role マッチで自動選択される
- 全部下が無効の場合は state/pending/ にフォールバック

**部下がいない場合** → サブエージェントとして即時並列実行される
- サブエージェントは自分と同じ identity・ツール・MCP を持つ
- 会話履歴・短期記憶にはアクセスできない

### delegate_task との使い分け

| 手段 | 用途 | 部下選択 |
|------|------|---------|
| Task tool | 簡易委任（自動ルーティング） | description に名前を含める |
| delegate_task | 明示的委任（期限・詳細指示あり） | name パラメータで直接指定 |

Heartbeat・Inbox など Task tool を使わないパスからの委任は delegate_task を使う。

### state/pending/ への書き出し（後で自分で実行する）

タスクを state/pending/ に書き出す際は、以下の原則に従ってください。
実行者（TaskExec）はサブエージェントとして動作します。あなたと同じidentity・行動指針・記憶ディレクトリ・組織情報を持っていますが、**あなたの会話履歴・短期記憶・Priming結果にはアクセスできません**。

### コンテキスト注入の原則
- **ファイルパスと行番号は必ず記載する**: 実行者は記憶検索ができるが、具体的な場所を指定した方が確実に正しいファイルに到達できる
- **現在の作業状態を含める**: current_task.md の関連部分を `context` フィールドにコピーすること（自動注入されるが、明示的に補足すると精度が上がる）
- **「なぜやるか」を明記する**: 背景と目的がないと実行者が判断を誤る

### 必須記載事項
- **何をするか**: 具体的な作業内容（「リファクタリングする」ではなく「core/auth/manager.py の verify_token() を async 化する」）
- **なぜやるか**: 背景と目的（1-2文）
- **どこを見るか**: 関連ファイルパスと行番号
- **完了条件**: 何をもって「できた」とするか
- **制約**: やってはいけないこと、互換性要件

### タスクファイル形式
state/pending/ ディレクトリに以下の形式のJSONファイルを作成してください:

```json
{{
    "task_type": "llm",
    "task_id": "YYYYMMDD-短い説明",
    "title": "タスクのタイトル",
    "submitted_by": "自分のAnima名",
    "submitted_at": "ISO8601形式の現在時刻",
    "description": "具体的な作業内容",
    "context": "背景情報（current_task.mdの関連部分を含む）",
    "acceptance_criteria": ["完了条件1", "完了条件2"],
    "constraints": ["制約1"],
    "file_paths": ["path/to/file.py:行番号"],
    "reply_to": "送信者名またはnull",
    "priority": 1
}}
```

### 禁止パターン
- ❌ 「適切にリファクタリングする」（曖昧）
- ❌ 「前回の続きをやる」（実行者は会話履歴を持たない）
- ❌ ファイルパスなしの指示（実行者は探索から始めることになる）
- ❌ contextフィールドが空（背景情報なしでは実行者が判断を誤る）

## 並列タスク実行（plan_tasks）

複数のタスクを依存関係付きで一括投入し、独立したタスクを並列実行できる。

### plan_tasks ツール

```
plan_tasks(batch_id="deploy-20260301", tasks=[
  {{"task_id": "lint", "title": "Lint実行", "description": "全ファイルにlintを実行", "parallel": true}},
  {{"task_id": "test", "title": "テスト実行", "description": "ユニットテスト実行", "parallel": true}},
  {{"task_id": "deploy", "title": "デプロイ", "description": "lint・テスト通過後にデプロイ",
   "parallel": false, "depends_on": ["lint", "test"]}}
])
```

| パラメータ | 必須 | 説明 |
|-----------|------|------|
| `batch_id` | MUST | バッチの一意識別子 |
| `tasks` | MUST | タスク配列（下記参照） |

#### タスクオブジェクト

| フィールド | 必須 | 説明 |
|-----------|------|------|
| `task_id` | MUST | バッチ内で一意のタスクID |
| `title` | MUST | タスクのタイトル |
| `description` | MUST | 具体的な作業内容 |
| `parallel` | MAY | `true` で並列実行可能（デフォルト: `false`） |
| `depends_on` | MAY | 依存する先行タスクIDの配列 |
| `context` | MAY | 背景情報 |
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

### 使い分け

- 単一タスク → 従来の `state/pending/` への直接書き出し
- 複数の独立タスク → `plan_tasks` で `parallel: true` を指定
- 依存関係のあるタスク群 → `plan_tasks` で `depends_on` を指定

### タスク結果

完了したタスクの結果は `state/task_results/{task_id}.json` に保存される。
依存タスクには先行タスクの結果要約が自動的にコンテキストとして注入される。
