# タスク管理の方法

Digital Anima がタスクを受け取り、追跡し、完了させるための運用リファレンス。
タスクの進め方に迷った場合に検索・参照すること。

## タスク管理の基本構造

タスクの状態は `state/` ディレクトリ内のファイルとタスクキューで管理する。

| リソース | 役割 |
|---------|------|
| `state/current_state.md` | ワーキングメモリ（今の状態・観察・計画・ブロッカー） |
| `state/pending/` ディレクトリ | LLM タスク（JSON 形式）。Heartbeat・submit_tasks・Task tool・Agent tool が書き出す。TaskExec パスが自動取得・実行する |
| `state/task_queue.jsonl` | 永続タスクキュー（append-only JSONL）。人間やAnimaからの依頼を追跡する |
| `state/task_results/` ディレクトリ | TaskExec の完了結果を保存（`{task_id}.md`、最大2000文字）。依存タスクに自動注入。7日TTL |

`state/current_state.md` は常に最新の状態を保たなければならない（MUST）。
タスク状態が変わるたびに更新すること。

### 3パス実行モデル

AnimaWorks ではタスクが3つの独立パスで処理される:

| パス | トリガー | 役割 | 実行範囲 |
|------|---------|------|---------|
| **Inbox** | DM受信 | Anima間メッセージの処理・返信 | 即時、軽量な応答のみ |
| **Heartbeat** | 定期巡回 | 状況確認・計画立案（Observe → Plan → Reflect） | 確認・判断のみ。実行は `pending/` に書き出す |
| **TaskExec** | `pending/` にタスク出現 | LLMタスクの実行 | フル実行（ツール使用含む） |

Heartbeat は **実行しない**。実行が必要なタスクを発見したら、部下がいれば `delegate_task` で委任するか、`submit_tasks` でタスク投入して TaskExec パスに委譲する。

なお、MCP 統合モード（S/C/D/G: Claude Agent SDK・Codex CLI・Cursor Agent・Gemini CLI）の Chat パスでは **Task tool**（および Agent tool）を使うと自動ルーティングが行われる:
- 部下がいる場合 → workload 最小かつ role マッチする部下に即時委譲される（delegate_task と同様のフロー）
- 部下がいない場合、または委譲失敗時 → `state/pending/` に書き出され、TaskExec パスが実行する

### タスクキュー（submit_tasks / update_task / 一覧はCLI）

永続タスクキューは `state/task_queue.jsonl` に append-only JSONL 形式で記録される。
`submit_tasks` でタスクを登録し、`update_task` でステータスを更新する。一覧取得は CLI の `animaworks-tool task list` を使用する。
キューに登録されたタスクはシステムプロンプトの Priming セクションに要約表示される。

#### submit_tasks（タスク登録 — 自分自身が実行する）

> **重要**: `submit_tasks` で投入したタスクは**あなた自身の TaskExec** が実行します（部下には送られません）。部下にタスクを委任する場合は `delegate_task` を使ってください。

タスクの作成・登録には `submit_tasks` を使用する。単一タスクの場合は tasks 配列に1件だけ指定する。

```
submit_tasks(batch_id="human-20260313", tasks=[
  {"task_id": "t1", "title": "月次レポート作成", "description": "月次売上レポートを作成し、aoiに提出してください", "parallel": true}
])
```

| パラメータ | 必須 | 説明 |
|-----------|------|------|
| `batch_id` | MUST | バッチの一意識別子 |
| `tasks[].task_id` | MUST | バッチ内で一意のタスクID |
| `tasks[].title` | MUST | タスクタイトル（1行要約） |
| `tasks[].description` | MUST | 元の指示文（委任時は原文引用を含める） |
| `tasks[].parallel` | MAY | `true` で並列実行可能（単一タスクでは `true` 推奨） |
| `tasks[].depends_on` | MAY | 先行タスクIDの配列 |
| `tasks[].workspace` | MAY | 作業ディレクトリ。ワークスペースエイリアス（例: `aischreiber`）を指定すると TaskExec がそのディレクトリで実行する。省略時は Anima のデフォルト |

- 人間からの指示を受けたら、必ず `submit_tasks` でタスクキューに登録する（MUST）
- 人間由来タスク（source=human 相当）は最優先で処理する（MUST）
- キューのタスクは Heartbeat で確認され、着手時に `update_task` で `in_progress` に更新する

#### update_task

タスクのステータスを更新する。完了時は `done`、中断時は `cancelled`、失敗時は `failed` に設定する。

```
update_task(task_id="abc123def456", status="in_progress")
update_task(task_id="abc123def456", status="done", summary="レポート作成完了")
```

| パラメータ | 必須 | 説明 |
|-----------|------|------|
| `task_id` | MUST | タスクID（submit_tasks 時に返されたID） |
| `status` | MUST | `pending` / `in_progress` / `done` / `cancelled` / `blocked` / `failed` |
| `summary` | MAY | 更新後の要約 |

#### タスク一覧の取得（CLI）

タスクキューの一覧は `animaworks-tool task list` で取得する。ステータスでフィルタリング可能。

```
Bash: animaworks-tool task list                    # 全件
Bash: animaworks-tool task list --status pending   # 未着手のみ
Bash: animaworks-tool task list --status in_progress
Bash: animaworks-tool task list --status done
Bash: animaworks-tool task list --status failed
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

## current_state.md の使い方

`current_state.md` はワーキングメモリとして、今の状態・観察・計画・ブロッカーを記録するファイル。
タスク一覧ではない。タスクの追跡は `task_queue.jsonl` で行う。

- **サイズ制限**: 3000文字。Heartbeat 時に自動クリーンアップされる
- **アイドル状態**: タスクがない場合は `status: idle` を記載する

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

## タスク状態遷移

タスクは `task_queue.jsonl` で追跡し、`current_state.md` は作業中のワーキングコンテキストを記録する。

```
submit_tasks で登録 → update_task(status="in_progress") → 作業 → update_task(status="done")
                                                                ↘ blocked → 報告 → 別タスクへ
```

### 状態遷移の手順

**着手**:
1. `task_queue.jsonl` からタスクを選び、`update_task(task_id="...", status="in_progress")` で更新する
2. `current_state.md` に `status: in-progress` で作業コンテキストを記載する

**完了**:
1. `update_task(task_id="...", status="done", summary="...")` でタスクを完了にする
2. `current_state.md` を `status: idle` に戻す
3. タスクの依頼者に結果を報告する（assigned_by が他者の場合 MUST）

**ブロック**:
1. `current_state.md` の `status` を `blocked` にし、`blockers` に具体的な理由を記載する（MUST）
2. ブロック解消のアクションを取る（後述のブロック対応フロー参照）
3. ブロック解消に時間がかかる場合、`task_queue.jsonl` の別タスクに着手してよい（MAY）

## 複数タスクの優先度管理

複数のタスクが `task_queue.jsonl` に存在する場合の判断基準:

1. **人間由来タスクを最優先**: source=human 相当のタスクは最優先で処理する（MUST）
2. **上司からのタスクを優先**: supervisor からの指示は同レベルの他タスクより優先する（SHOULD）
3. **締め切り順**: deadline が近いものから着手する（SHOULD）
4. **先入れ先出し**: 同優先度・同締め切りなら受信順に処理する（MAY）

### タスク中断時の手順

優先度の高いタスクが割り込んだ場合:

1. 現在タスクを `update_task(status="pending")` でキューに戻す
2. `current_state.md` の進捗をメモしてから、新しいタスクのコンテキストに切り替える

## ブロックされたタスクの対応フロー

タスクがブロックされた場合、以下の手順で対応する。

### ステップ1: ブロック原因の特定と記録

current_state.md の `blockers` に具体的な原因を記載する（MUST）。

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

ブロック解消に時間がかかる場合、`task_queue.jsonl` の次のタスクに着手してよい（MAY）。
ブロックされたタスクは `update_task(status="blocked")` で記録し、解消後に再着手する。

## タスクファイルのテンプレート

### current_state.md — アイドル状態

```markdown
status: idle
```

### current_state.md — 作業中

```markdown
status: in-progress
task: {タスク名}
assigned_by: {依頼者名 or self}
started: {YYYY-MM-DD HH:MM}
context: |
  {タスクの詳細・背景情報}
blockers: なし
```

### current_state.md — ブロック中

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

## episodes/ へのタスクログ記録

タスクの着手・完了・ブロック等の状態変化は episodes/ に記録する（SHOULD）。
ファイル名は `YYYY-MM-DD.md`（日別ログ）。

```markdown
## 10:00 タスク着手: Slack連携テスト

aoi からの指示を受け、Slack API の接続テストを開始。
permissions.json で slack: yes を確認済み。

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
| `tasks[].workspace` | MAY | 作業ディレクトリ。ワークスペースエイリアスを指定すると TaskExec がそのディレクトリで実行する |
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

完了タスクの結果要約は `state/task_results/{task_id}.md` に保存される（最大2,000文字、7日TTL）。
依存タスクはこの結果をコンテキストとして自動的に受け取る。先行タスクが失敗した場合、依存タスクはスキップされ `FAILED: {理由}` が記録される。
各タスク完了時、submit_tasks を実行した Anima に完了通知が DM で送られる。

### submit_tasks と delegate_task の使い分け

| シナリオ | 方法 | 実行者 |
|---------|------|--------|
| 自分でバックグラウンド実行したいタスク | `submit_tasks` | **自分自身** |
| 複数独立タスクを並列で自分が実行 | `submit_tasks` で `parallel: true` | **自分自身** |
| 依存関係付きタスク群を自分が実行 | `submit_tasks` で `depends_on` 指定 | **自分自身** |
| **部下に作業を任せたい** | **`delegate_task`** | **部下** |

**注意**: `state/pending/` にJSONを手動で書き出してはならない。必ず `submit_tasks` ツール経由で投入すること。`submit_tasks` は Layer 1（実行キュー）と Layer 2（タスクレジストリ）の両方に同時登録するため、タスクの追跡漏れを防げる。

## タスク委譲（delegate_task / Task tool） — 部下が実行する

> **重要**: `delegate_task` は**部下の TaskExec** がタスクを実行します（あなた自身は実行しません）。自分でバックグラウンド実行したい場合は `submit_tasks` を使ってください。

部下を持つ Anima（スーパーバイザー）は `delegate_task` ツールでタスクを部下に委譲できる。
MCP 統合モード（S/C/D/G）の Chat パスでは Task tool（および Agent tool）でも委譲可能。Task tool は部下を指名するパラメータを持たず、workload 最小かつ role マッチで自動選択される。

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
| `name` | MUST | 委譲先の直属部下Anima名 |
| `instruction` | MUST | タスクの指示内容 |
| `deadline` | MUST | 期限。相対形式 `30m` / `2h` / `1d` または ISO8601 |
| `summary` | MAY | タスクの1行要約（省略時は instruction の先頭100文字） |
| `workspace` | MAY | 作業ディレクトリ。ワークスペースエイリアスを指定すると委譲先がそのディレクトリで作業する |

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
