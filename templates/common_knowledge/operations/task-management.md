# タスク管理の方法

Digital Anima がタスクを受け取り、追跡し、完了させるための運用リファレンス。
タスクの進め方に迷った場合に検索・参照すること。

## タスク管理の基本構造

タスクの状態は `state/` ディレクトリ内の2つのファイルで管理する。

| ファイル | 役割 |
|---------|------|
| `state/current_task.md` | 今取り組んでいるタスク（1つ） |
| `state/pending.md` | 待機中・未着手のタスク一覧（バックログ） |

この2つのファイルは常に最新の状態を保たなければならない（MUST）。
タスク状態が変わるたびに、対応するファイルを更新すること。

## current_task.md の使い方

`current_task.md` は「今まさに取り組んでいるタスク」を記録するファイル。
1つのタスクのみを記載する（MUST）。複数の並行タスクがあっても、最優先の1つだけをここに書く。

### フォーマット

```markdown
status: in-progress
task: Slack連携機能のテスト
assigned_by: hinata
started: 2026-02-15 10:00
context: |
  hinataからの指示: Slack APIの接続テストを行い、
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
assigned_by: hinata
received: 2026-02-15 11:00
deadline: 2026-02-16 EOD
notes: |
  新規クライアント向けの自動通知メール。
  テンプレート案を3パターン作成して hinata に提出。

## [MEDIUM] ナレッジベースの整理
assigned_by: self
received: 2026-02-14 09:00
notes: |
  knowledge/ 配下のファイルにタグ付けを行い、
  検索効率を改善する。

## [LOW] 週次レポートフォーマットの改善案
assigned_by: hinata
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
assigned_by: hinata
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
  hinata に設定依頼が必要。
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
assigned_by: hinata
received: 2026-02-15 10:00
blocked_reason: AWS クレデンシャル未設定。hinata に依頼済み（2026-02-15 11:00）
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

hinata からの指示を受け、Slack API の接続テストを開始。
permissions.md で slack: yes を確認済み。

## 11:30 タスク完了: Slack連携テスト

Slack API 接続テスト完了。#general への投稿テストも成功。
結果を hinata に報告済み。

[IMPORTANT] Slack API のレート制限: 1分間に最大1メッセージの制限あり。
バースト送信時は間隔を空ける必要がある。
```

重要な学びには `[IMPORTANT]` タグを付ける（SHOULD）。後のハートビートや記憶統合で優先的に抽出される。
