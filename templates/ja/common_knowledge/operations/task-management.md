# タスク管理の方法

Digital Anima がタスクを受け取り、追跡し、完了させるための運用リファレンス。
タスクの進め方に迷った場合に検索・参照すること。

## タスク管理の基本構造

タスクの状態は `state/` ディレクトリ内のファイルとタスクキューで管理する。

| リソース | 役割 |
|---------|------|
| `state/current_task.md` | 今取り組んでいるタスク（1つ） |
| `state/pending.md` | 手動メモ用のバックログ（自由形式） |
| `state/pending/` ディレクトリ | LLM タスク（JSON 形式）。Heartbeat・submit_tasks・Task tool・Agent tool が書き出す。TaskExec パスが自動取得・実行する |
| `state/task_queue.jsonl` | 永続タスクキュー（append-only JSONL）。人間やAnimaからの依頼を追跡する |

`state/current_task.md` は常に最新の状態を保たなければならない（MUST）。
タスク状態が変わるたびに更新すること。

### 3パス実行モデル

AnimaWorks ではタスクが3つの独立パスで処理される:

| パス | トリガー | 役割 | 実行範囲 |
|------|---------|------|---------|
| **Inbox** | DM受信 | Anima間メッセージの処理・返信 | 即時、軽量な応答のみ |
| **Heartbeat** | 定期巡回 | 状況確認・計画立案（Observe → Plan → Reflect） | 確認・判断のみ。実行は `pending/` に書き出す |
| **TaskExec** | `pending/` にタスク出現 | LLMタスクの実行 | フル実行（ツール使用含む） |

Heartbeat は **実行しない**。実行が必要なタスクを発見したら、部下がいれば `delegate_task` で委任するか、`submit_tasks` でタスク投入して TaskExec パスに委譲する。

なお、Sモード（Claude Agent SDK）の Chat パスでは **Task tool**（および Agent tool）を使うと自動ルーティングが行われる:
- 部下がいる場合 → workload 最小かつ role マッチする部下に即時委譲される（delegate_task と同様のフロー）
- 部下がいない場合、または委譲失敗時 → `state/pending/` に書き出され、TaskExec パスが実行する

### タスクキュー（backlog_task / update_task / list_tasks）

永続タスクキューは `state/task_queue.jsonl` に append-only JSONL 形式で記録される。
`backlog_task` でタスクを登録し、`update_task` でステータスを更新、`list_tasks` で一覧取得する。
キューに登録されたタスクはシステムプロンプトの Priming セクションに要約表示される。

#### backlog_task

```
backlog_task(source="human", original_instruction="月次売上レポートを作成し、aoiに提出してください", assignee="自分自身の名前", summary="月次レポート作成", deadline="1d")
```

| パラメータ | 必須 | 説明 |
|-----------|------|------|
| `source` | MUST | `human`（人間からの依頼）/ `anima`（Animaからの委譲） |
| `original_instruction` | MUST | 元の指示文（委任時は原文引用を含める。最大10,000文字） |
| `assignee` | MUST | 担当者名（自分自身または委任先のAnima名） |
| `summary` | MUST | タスクの1行要約（省略時または空の場合は original_instruction の先頭100文字が使用される） |
| `deadline` | MUST | 期限。相対形式 `30m` / `2h` / `1d` または ISO8601 |
| `relay_chain` | MAY | 委任経路（例: `["aoi", "taro"]`） |

- 人間からの指示を受けたら、必ず `backlog_task` で `source="human"` を指定して記録する（MUST）
- `source: human` のタスクは最優先で処理する（MUST）
- キューのタスクは Heartbeat で確認され、着手時に `update_task` で `in_progress` に更新する

#### update_task

タスクのステータスを更新する。完了時は `done`、中断時は `cancelled`、失敗時は `failed` に設定する。

```
update_task(task_id="abc123def456", status="in_progress")
update_task(task_id="abc123def456", status="done", summary="レポート作成完了")
```

| パラメータ | 必須 | 説明 |
|-----------|------|------|
| `task_id` | MUST | タスクID（backlog_task 時に返されたID） |
| `status` | MUST | `pending` / `in_progress` / `done` / `cancelled` / `blocked` / `failed` |
| `summary` | MAY | 更新後の要約 |

#### list_tasks

タスクキューの一覧を取得する。ステータスでフィルタリング可能。

```
list_tasks()                     # 全件
list_tasks(status="pending")     # 未着手のみ
list_tasks(status="in_progress") # 進行中のみ
list_tasks(status="done")        # 完了のみ
list_tasks(status="failed")      # 失敗のみ
```

#### タスクキューの状態とマーカー

| 状態 | 意味 |
|------|------|
| `pending` | 未着手 |
| `in_progress` | 作業中 |
| `done` | 完了 |
| `cancelled` | 取り消し |
| `blocked` | ブロック中 |
| `failed` | 失敗（TaskExec 等で実行に失敗した場合） |
| `delegated` | 委譲済み（delegate_task で部下に委譲した追跡用） |

Priming 表示では、人間由来タスク（source=human）に 🔴 HIGH マーカー、30分以上更新されていないタスクに ⚠️ STALE、期限超過タスクに 🔴 OVERDUE マーカーが付く。

## current_task.md の使い方

`current_task.md` は「今まさに取り組んでいるタスク」を記録するファイル。
1つのタスクのみを記載する（MUST）。複数の並行タスクがあっても、最優先の1つだけをここに書く。

### フォーマット

```markdown
status: in-progress
task: Slack連携機能のテスト
assigned_by: aoi
started: 2026-02-15 10:00
context: |
  aoiからの指示: Slack APIの接続テストを行い、
  #generalチャンネルへの投稿が正常に動作するか確認する。
  テスト完了後に結果を報告すること。
blockers: なし
```

### フィールド説明

| フィールド | 必須 | 説明 |
|-----------|------|------|
| `status` | MUST | タスクの状態（後述の状態遷移を参照） |
| `task` | MUST | タスクの簡潔な説明（1行） |
| `assigned_by` | SHOULD | 誰から受けたタスクか。自発タスクなら `self` |
| `started` | SHOULD | 着手日時 |
| `context` | SHOULD | タスクの詳細・背景情報 |
| `blockers` | SHOULD | ブロッカーがあれば記載。なければ `なし` |

### アイドル状態

タスクがない場合は以下のように記載する:

```markdown
status: idle
```

`idle` は正常な状態であり、次のタスクが来るまで待機していることを意味する。
ハートビートで確認した際に `idle` であれば、特にアクションは不要（`HEARTBEAT_OK`）。

## pending.md の使い方

`pending.md` は「まだ着手していないが、やるべきタスク」の一覧を管理するバックログ。
優先度順に並べる（SHOULD）。

### フォーマット

```markdown
# Pending Tasks

## [HIGH] Gmail通知テンプレートの作成
assigned_by: aoi
received: 2026-02-15 11:00
deadline: 2026-02-16 EOD
notes: |
  新規クライアント向けの自動通知メール。
  テンプレート案を3パターン作成して aoi に提出。

## [MEDIUM] ナレッジベースの整理
assigned_by: self
received: 2026-02-14 09:00
notes: |
  knowledge/ 配下のファイルにタグ付けを行い、
  検索効率を改善する。

## [LOW] 週次レポートフォーマットの改善案
assigned_by: aoi
received: 2026-02-13 15:00
notes: |
  現在のレポート形式を見直し、改善提案をまとめる。
  急ぎではない。
```

### 優先度ラベル

| ラベル | 意味 | 目安 |
|--------|------|------|
| `[URGENT]` | 緊急 | 他の全タスクに優先して即座に着手（MUST） |
| `[HIGH]` | 高 | 当日中に着手すべき |
| `[MEDIUM]` | 中 | 今週中に着手すべき |
| `[LOW]` | 低 | 余裕がある時に着手 |

優先度が未指定のタスクは `[MEDIUM]` として扱う（SHOULD）。

## タスク状態遷移

タスクは以下の状態を遷移する。状態変更時は必ず current_task.md を更新すること（MUST）。

```
received → in-progress → completed
                      ↘ blocked → in-progress（再開）
                                ↘ cancelled
```

### 各状態の定義

| 状態 | 意味 | current_task.md に記載 |
|------|------|----------------------|
| `received` | タスクを受け取ったが未着手 | pending.md に記載 |
| `in-progress` | 現在作業中 | current_task.md に記載 |
| `completed` | 完了 | idle に戻す + episodes/ にログ |
| `blocked` | ブロッカーにより中断 | current_task.md に blockers を記載 |
| `cancelled` | 取り消し | idle に戻す + episodes/ にログ |

### 状態遷移の手順

**received → in-progress（着手）**:
1. pending.md から該当タスクを削除する
2. current_task.md に `status: in-progress` で記載する
3. episodes/ に「タスク着手」をログする（SHOULD）

**in-progress → completed（完了）**:
1. current_task.md を `status: idle` に戻す
2. episodes/ に「タスク完了」と結果の要約をログする（MUST）
3. タスクの依頼者に結果を報告する（assigned_by が他者の場合 MUST）
4. pending.md に次のタスクがあれば、最優先のものを current_task.md に移す

**in-progress → blocked（ブロック）**:
1. current_task.md の `status` を `blocked` に変更する
2. `blockers` フィールドに具体的なブロック理由を記載する（MUST）
3. ブロック解消のアクションを取る（後述のブロック対応フロー参照）
4. pending.md の次優先タスクがあれば、並行して着手を検討する（MAY）

## 複数タスクの優先度管理

複数のタスクが同時に存在する場合の判断基準:

1. **URGENT は最優先**: `[URGENT]` タスクが来たら、現在のタスクを中断してでも着手する（MUST）
2. **上司からのタスクを優先**: supervisor からの指示は同レベルの他タスクより優先する（SHOULD）
3. **締め切り順**: deadline が近いものから着手する（SHOULD）
4. **先入れ先出し**: 同優先度・同締め切りなら受信順に処理する（MAY）

### タスク中断時の手順

優先度の高いタスクが割り込んだ場合:

1. current_task.md の現在タスクの進捗をメモする（MUST）
2. 現在タスクを pending.md に戻す（状態と進捗メモ付き）
3. 新しいタスクを current_task.md に記載する

pending.md に戻す際のフォーマット例:

```markdown
## [HIGH] Slack連携機能のテスト（中断中）
assigned_by: aoi
received: 2026-02-15 10:00
progress: |
  API接続テストは完了。チャンネル投稿テストの途中で中断。
  残作業: #general への投稿テスト、エラーハンドリング確認
```

## ブロックされたタスクの対応フロー

タスクがブロックされた場合、以下の手順で対応する。

### ステップ1: ブロック原因の特定と記録

current_task.md の `blockers` に具体的な原因を記載する（MUST）。

```markdown
status: blocked
task: AWS S3バケット設定
blockers: |
  AWS クレデンシャルが未設定。
  config.json に aws credential が存在しない。
  aoi に設定依頼が必要。
```

### ステップ2: 解消アクション

ブロック原因に応じたアクションを取る:

| 原因 | アクション |
|------|-----------|
| 情報不足 | 依頼者に質問メッセージを送る（SHOULD） |
| 権限不足 | supervisor に権限追加を依頼する（SHOULD） |
| 外部依存 | 待ちであることを依頼者に報告する（SHOULD） |
| 技術的問題 | knowledge/ や procedures/ を検索し、解決策を探す。見つからなければ報告する |

### ステップ3: 別タスクへの切り替え

ブロック解消に時間がかかる場合、pending.md の次のタスクに着手してよい（MAY）。
ブロックされたタスクは pending.md に移し、ブロック理由を残す。

```markdown
## [HIGH] AWS S3バケット設定（ブロック中）
assigned_by: aoi
received: 2026-02-15 10:00
blocked_reason: AWS クレデンシャル未設定。aoi に依頼済み（2026-02-15 11:00）
```

### ステップ4: ブロック解消後の再開

ブロックが解消されたら（例: メッセージで通知を受けたら）:
1. pending.md から該当タスクを取り出す
2. 優先度を再評価し、current_task.md に移すか判断する
3. 着手する場合は `status: in-progress` に変更し、作業を再開する

## タスクファイルのテンプレート

### current_task.md — アイドル状態

```markdown
status: idle
```

### current_task.md — 作業中

```markdown
status: in-progress
task: {タスク名}
assigned_by: {依頼者名 or self}
started: {YYYY-MM-DD HH:MM}
context: |
  {タスクの詳細・背景情報}
blockers: なし
```

### current_task.md — ブロック中

```markdown
status: blocked
task: {タスク名}
assigned_by: {依頼者名 or self}
started: {YYYY-MM-DD HH:MM}
context: |
  {タスクの詳細・背景情報}
blockers: |
  {ブロック理由の具体的な説明}
  {解消に向けて取ったアクション}
```

### pending.md — バックログ

```markdown
# Pending Tasks

## [{優先度}] {タスク名}
assigned_by: {依頼者名}
received: {YYYY-MM-DD HH:MM}
deadline: {YYYY-MM-DD or なし}
notes: |
  {タスクの詳細}
```

## episodes/ へのタスクログ記録

タスクの着手・完了・ブロック等の状態変化は episodes/ に記録する（SHOULD）。
ファイル名は `YYYY-MM-DD.md`（日別ログ）。

```markdown
## 10:00 タスク着手: Slack連携テスト

aoi からの指示を受け、Slack API の接続テストを開始。
permissions.md で slack: yes を確認済み。

## 11:30 タスク完了: Slack連携テスト

Slack API 接続テスト完了。#general への投稿テストも成功。
結果を aoi に報告済み。

[IMPORTANT] Slack API のレート制限: 1分間に最大1メッセージの制限あり。
バースト送信時は間隔を空ける必要がある。
```

重要な学びには `[IMPORTANT]` タグを付ける（SHOULD）。後のハートビートや記憶統合で優先的に抽出される。

## 並列タスク実行（submit_tasks）

`submit_tasks` ツールを使うと、複数のタスクを依存関係付きで一括投入し、並列実行できる。
TaskExec がDAG（有向非巡回グラフ）として依存関係を解決し、独立タスクを同時実行する。

### 使い方

```
submit_tasks(batch_id="build-20260301", tasks=[
  {{"task_id": "compile", "title": "コンパイル", "description": "ソースをビルド", "parallel": true}},
  {{"task_id": "lint", "title": "Lint", "description": "静的解析", "parallel": true}},
  {{"task_id": "package", "title": "パッケージ", "description": "ビルド成果物をパッケージ化",
   "depends_on": ["compile", "lint"]}}
])
```

| パラメータ | 必須 | 説明 |
|-----------|------|------|
| `batch_id` | MUST | バッチの一意識別子 |
| `tasks[].task_id` | MUST | バッチ内で一意のタスクID |
| `tasks[].title` | MUST | タスクタイトル |
| `tasks[].description` | MUST | 作業内容 |
| `tasks[].parallel` | MAY | `true` で並列実行可能（デフォルト: `false`） |
| `tasks[].depends_on` | MAY | 先行タスクIDの配列 |
| `tasks[].acceptance_criteria` | MAY | 完了条件の配列 |
| `tasks[].constraints` | MAY | 制約の配列 |
| `tasks[].file_paths` | MAY | 関連ファイルパスの配列 |

### 実行の仕組み

1. `submit_tasks` がバリデーション（ID一意性、依存先存在、循環検出）を行う
2. タスクファイルが `state/pending/` に `batch_id` 付きで書き出される
3. submit_tasks 実行後、TaskExec（PendingTaskExecutor）は即座にタスクを検出する（wake によりポーリングを待たない）
4. TaskExec がバッチを検出し、トポロジカルソートで実行順を決定する
5. 依存なしの `parallel: true` タスクはセマフォ上限内で同時実行される
6. 先行タスクの結果は依存タスクのコンテキストに自動注入される
7. 先行タスクが失敗した場合、依存タスクはスキップされる
8. タスクは書き出しから24時間以内に実行されないとスキップされる（TTL）

### 並列実行の上限

同時実行数は `config.json` の `background_task.max_parallel_llm_tasks`（デフォルト: 3、1〜10）で制御される。

### タスク結果の保存

完了タスクの結果要約は `state/task_results/{task_id}.md` に保存される（最大2,000文字）。
依存タスクはこの結果をコンテキストとして自動的に受け取る。先行タスクが失敗した場合、依存タスクはスキップされ `FAILED: {理由}` が記録される。
各タスク完了時、submit_tasks を実行した Anima に完了通知が DM で送られる。

### submit_tasks の使い分け

| シナリオ | 方法 |
|---------|------|
| 単一タスク | `submit_tasks`（tasks配列1件で投入） |
| 複数独立タスク | `submit_tasks` で `parallel: true` |
| 依存関係付きタスク群 | `submit_tasks` で `depends_on` 指定 |
| 部下への委譲 | `delegate_task`（別メカニズム） |

**注意**: `state/pending/` にJSONを手動で書き出してはならない。必ず `submit_tasks` ツール経由で投入すること。`submit_tasks` は Layer 1（実行キュー）と Layer 2（タスクレジストリ）の両方に同時登録するため、タスクの追跡漏れを防げる。

## タスク委譲（delegate_task / Task tool）

部下を持つ Anima（スーパーバイザー）は `delegate_task` ツールでタスクを部下に委譲できる。
Sモードの Chat パスでは Task tool（および Agent tool）でも委譲可能。Task tool は部下を指名するパラメータを持たず、workload 最小かつ role マッチで自動選択される。

### delegate_task の動作

1. 部下のタスクキューにタスクが追加される（source="anima"）
2. 部下の `state/pending/` にタスクJSONが書き込まれ、即時実行される
3. 部下に DM が自動送信される
4. 自分のキューに追跡エントリが作成される（status="delegated"）

### 使い方

```
delegate_task(name="dave", instruction="API テストを実施して結果を報告してください", deadline="2d", summary="API テスト")
```

| パラメータ | 必須 | 説明 |
|-----------|------|------|
| `name` | MUST | 委譲先の直属部下のAnima名 |
| `instruction` | MUST | タスクの指示内容 |
| `deadline` | MUST | 期限。相対形式 `30m` / `2h` / `1d` または ISO8601 |
| `summary` | MAY | タスクの1行要約（省略時は instruction の先頭100文字） |

### 委譲タスクの追跡

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

### 委譲を受けた側の対応

1. DM で委譲メッセージを受信する
2. タスクキューに自動的にタスクが登録される
3. 内容を確認し、不明点があれば委譲元に質問する（SHOULD）
4. 完了したら委譲元に結果を報告する（MUST）
