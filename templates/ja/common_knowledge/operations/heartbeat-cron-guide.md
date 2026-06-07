# 定期実行の設定と運用

ハートビート（定期巡回）と Cron（定時タスク）の設定方法・運用ガイド。
定期実行の挙動を変更したい場合や、新しい定時タスクを追加したい場合に参照すること。

## ハートビートとは

ハートビートは、Digital Anima が定期的に自動起動し、状況を確認・計画する仕組み。
人間が定期的に受信箱を確認し、進行中の仕事を見直すのと同じ行動を自動化したもの。

### 重要: ハートビートは「確認と計画」のみ

ハートビートの役割は **Observe（観察）→ Plan（計画）→ Reflect（振り返り）** の3フェーズに限定される。

- MUST: ハートビート内では状況確認・計画立案・振り返りのみを行う
- MUST NOT: ハートビート内で長時間の実行タスク（コーディング、大量のツール呼び出し等）を行わない
- MUST: 実行が必要なタスクを発見したら、部下がいれば `delegate_task` で委任するか、`submit_tasks` でタスク投入する

書き出されたタスクは **TaskExec パス**（`PendingTaskExecutor`）がポーリングで取得・実行する。
`submit_tasks` が書く LLM タスクは `state/pending/` を監視し、最大約3秒の間隔で拾われる（同一ループが `state/background_tasks/pending/` の CLI 投入タスクも処理する）。

### ハートビートと会話の並行動作

ハートビートと人間との会話は **別ロック** で管理されるため、同時に動作できる。
ハートビート実行中でも、人間からのメッセージには即座に応答可能。

### submit_tasks によるタスク投入

ハートビートで実行すべきタスクを発見した場合、`submit_tasks` ツールでタスクを投入する:

```
submit_tasks(batch_id="hb-20260301-api-test", tasks=[
  {"task_id": "api-test", "title": "APIテスト実施",
   "description": "Slack API接続テストを実施し、全エンドポイントの結果をレポートにまとめる。完了後 aoi に報告する。"}
])
```

`submit_tasks` は Layer 1（実行キュー `state/pending/`）と Layer 2（タスクレジストリ `task_queue.jsonl`）の両方に同時登録する。
TaskExec が JSON を `processing/` へ移動したうえで LLM セッションで実行する。失敗時は `state/pending/failed/` に退避される。

**長時間 CLI ツール**（`animaworks-tool submit …`）は別経路で `state/background_tasks/pending/` に書かれ、`BackgroundTaskManager`（`core/background.py`）がバックグラウンド実行する。詳細は `operations/background-tasks.md` を参照。

**注意**: `state/pending/` に手動で JSON を置くのは非推奨。`submit_tasks` ツール経由で投入すること（バリデーションとキュー同期が省略されるため）。

単一タスクでも `submit_tasks`（tasks配列1件）を使う。
複数の独立タスクは `parallel: true` で並列実行、依存関係がある場合は `depends_on` を指定する。
詳細は task-management を参照。

### ハートビートのトリガー種別

ハートビートには2種類のトリガーがある:

| トリガー | 説明 |
|---------|------|
| 定期ハートビート | `config.json` の `heartbeat.interval_minutes` に従い、APScheduler が定期的に起動 |
| メッセージトリガー | Inbox に未読メッセージが到着した際に即座に起動（Inbox パスとして処理） |

メッセージトリガーには以下のセーフガードが組み込まれている:
- **クールダウン**: 前回のメッセージ起動完了から一定時間以内は再起動しない（`config.json` の `heartbeat.msg_heartbeat_cooldown_s`、デフォルト300秒）
- **カスケード検出**: 2者間で一定時間内に往復が閾値を超えるとループとみなし抑制する（`heartbeat.cascade_window_s` デフォルト30分、`heartbeat.cascade_threshold` デフォルト3）
- **意図フィルタ**: `intent` が `heartbeat.actionable_intents`（デフォルト `report`, `question`）に含まれるメッセージがある場合のみ即時ハートビート。それ以外（例: 軽い ack 系）は定期ハートビートまで待つ
- **往復の深さ制限**: `heartbeat.depth_window_s`（デフォルト600秒）と `heartbeat.max_depth`（デフォルト6）で、同一ペアの短期間の往復過多を抑止

## heartbeat.md の設定

`heartbeat.md` は各 Anima の設定ファイルで、活動時間・チェック項目を定義する。
ハートビートの実行間隔は `config.json` の `heartbeat.interval_minutes` で設定可能（1〜1440分、デフォルト30）。`heartbeat.md` では変更できない。
各 Anima には名前ベースの 0〜9 分オフセットが付与され、同時起動を分散する。
ファイルパス: `~/.animaworks/animas/{name}/heartbeat.md`

上司が配下 Anima の `heartbeat.md` を編集する場合は、直接ファイル操作ではなく `read_memory_file` / `write_memory_file` を使い、`../{anima_name}/heartbeat.md` のような相対パスで指定する。

### フォーマット

```markdown
# Heartbeat: {name}

## 活動時間
24時間（サーバー設定タイムゾーン）

## チェックリスト
- Inboxに未読メッセージがあるか
- 進行中タスクにブロッカーが発生していないか
- 自分の作業領域に新しいファイルが置かれていないか
- 何もなければ何もしない（HEARTBEAT_OK）

## 通知ルール
- 緊急と判断した場合のみ関係者に通知
- 同じ内容の通知は24時間以内に繰り返さない
```

### 設定フィールド

**実行間隔**:
- `config.json` の `heartbeat.interval_minutes` で設定（1〜1440分、デフォルト30）。`heartbeat.md` での変更は不可

**活動時間**（SHOULD）:
- `HH:MM - HH:MM` 形式で記載する（例: `9:00 - 22:00`）
- この時間外はハートビートが起動しない
- 未設定時のデフォルト: 24時間（全時間帯）
- タイムゾーンは `config.json` の `system.timezone` で設定可能。未設定時はシステムタイムゾーンを自動検出

**チェックリスト**（MUST）:
- ハートビート起動時にエージェントが確認する項目
- 箇条書き（`- ` 始まり）で記載する
- エージェントのプロンプトにチェックリスト内容がそのまま渡される
- カスタマイズ可能: Anima の役割に合わせて項目を追加・変更してよい

### チェックリストのカスタマイズ例

デフォルト（全 Anima 共通）:
```markdown
## チェックリスト
- Inboxに未読メッセージがあるか
- 進行中タスクにブロッカーがないか
- 何もなければ何もしない（HEARTBEAT_OK）
```

開発担当の例:
```markdown
## チェックリスト
- Inboxに未読メッセージがあるか
- 進行中タスクにブロッカーが発生していないか
- 監視対象のGitHubリポジトリに新しいIssueやPRがないか
- CI/CDの失敗アラートがないか
- 何もなければ何もしない（HEARTBEAT_OK）
```

コミュニケーション担当の例:
```markdown
## チェックリスト
- Inboxに未読メッセージがあるか
- Slackの未読メンションがないか
- 返信待ちのメールがないか
- 進行中タスクにブロッカーがないか
- 何もなければ何もしない（HEARTBEAT_OK）
```

### 実行モデル（コスト最適化）

Heartbeat / Inbox / Cron は `background_model` が設定されている場合、メインモデルの代わりにそのモデルで実行される。
Chat（人間との対話）と TaskExec（実作業）はメインモデルを維持する。

設定方法: `animaworks anima set-background-model {名前} claude-sonnet-4-6`
詳細は `reference/operations/model-guide.md` の「バックグラウンドモデル」セクションを参照。

### ハートビートの内部動作

- **クラッシュ復旧**: 前回ハートビートが失敗した場合、`state/recovery_note.md` にエラー情報が保存される。次回起動時にプロンプトに注入され、復旧後にファイルは削除される。
- **振り返り記録**: ハートビート出力に `[REFLECTION]...[/REFLECTION]` ブロックがあると、activity_log に `heartbeat_reflection` として記録され、次回以降のハートビートコンテキストに含まれる。
- **部下チェック**: 部下を持つ Anima には、ハートビート・Cron のプロンプトに部下の状態確認指示が自動注入される。
- **セッション時間制限**（`config.json` の `heartbeat`）: `soft_timeout_seconds`（デフォルト300秒）経過でラップアップ用のリマインダを注入、`hard_timeout_seconds`（デフォルト600秒）でセッションを強制終了。`max_turns` を設定すると、ハートビート専用のターン上限として per-anima の `max_turns` を上書きできる。
- **アイドル時の自動コンパクト**: `heartbeat.idle_compaction_minutes`（デフォルト10分）— ストリーム終了からこの時間経過後にアイドル自動コンパクションが走る（実行エンジン側の設定）。
- **Board 投稿の間隔**: `heartbeat.channel_post_cooldown_s`（デフォルト300秒、0 で無制限）— 同一 Anima の `post_channel` 連投を抑止。

### 定期ハートビートのスケジュール方式

実効間隔（Activity Level 適用後・下限5分丸め）が **60分以下かつ 60 を割り切る** ときは、APScheduler の `CronTrigger` で分スロットに分散登録される（名前ベース 0〜9 分オフセットと組み合わせ）。

それ以外（例: 実効61分、または43分のように 60 を割り切れない間隔）は **1分ごとのポーリング**（`_heartbeat_check`）で「前回からの経過」により発火する。古い `IntervalTrigger` 由来の不具合回避のための実装。

### バックグラウンドツールと DM ログ（core/background.py）

`core/background.py` はハートビート/Cron のスケジュールそのものではなく、**長時間ツール呼び出しのバックグラウンド実行**（状態の JSON 永続化を含む）と **レガシー共有 DM ログ（`shared/dm_logs/`）のローテーション**を担う。運用の詳細・CLI 経路は `operations/background-tasks.md` も参照。

#### BackgroundTaskManager

- **保存先**: `state/background_tasks/{task_id}.json`。`task_id` は UUID の先頭 12 文字（16 進）。各ファイルに `task_id`, `anima_name`, `tool_name`, `tool_args`, `status`, `created_at`, `completed_at`, `result`, `error` が記録される。
- **状態 (`TaskStatus`)**: `pending` / `running` / `completed` / `failed`。`submit` / `submit_async` では投入直後から `running` で JSON が書かれ、完了または例外で `completed` / `failed` に更新される。
- **実行 API**: `submit(tool_name, tool_args, execute_fn)` は同期 callable を `asyncio` の `run_in_executor` でスレッドプール実行する。`submit_async` は非同期 callable をそのまま await する。`get_task`（メモリ優先、無ければディスク）、`list_tasks`（ディスク上の JSON もマージ、作成時刻の新しい順）、`active_count`（メモリ上の `running` 件数）を提供する。
- **完了コールバック**: `on_complete` に渡した非同期関数は、タスク保存後に呼ばれる。コールバック内で例外が出てもタスク結果は保持され、ログに記録されるのみ（典型的には `state/background_notifications/` への書き込みと組み合わせ、**次回ハートビート**で読み取り・削除され会話コンテキストへ取り込まれる経路）。
- **候補判定 `is_eligible(tool_name)`**: マップにキーが存在すればバックグラウンド対象。キーは次の両方を受け付ける。(1) **スキーマ名**（例: `generate_3d_model`）— Mode A の外部ツールディスパッチなど。(2) **`ツール名:サブコマンド`**（例: `image_gen:pipeline`）— 各ツールモジュールの `EXECUTION_PROFILE` で `background_eligible: true` のエントリが `get_eligible_tools_from_profiles()` によりこの形式で登録される（Mode S の `submit` 経路など）。
- **候補ツールマップの構築 `from_profiles()`**: 次の 3 層を dict の `update` でマージし、**後勝ち**で上書きする。(1) `_DEFAULT_ELIGIBLE_TOOLS`（コード既定）(2) 引数 `profiles`（`EXECUTION_PROFILE` 集約）(3) 引数 `config_eligible`（通常は `config.json` の `background_task.eligible_tools` から `threshold_s` を展開した `名前 → 秒`）。値はプロファイル連携用の期待秒（整数）。
- **コード既定 `_DEFAULT_ELIGIBLE_TOOLS`（秒）**: `generate_character_assets` 30、`generate_fullbody` / `generate_bustup` / `generate_icon` / `generate_chibi` 各 30、`generate_3d_model` / `generate_rigged_model` / `generate_animations` 各 30、`local_llm` 60、`run_command` 60、`machine_run` 600。
- **掃除 `cleanup_old_tasks(max_age_hours=24)`**: `status` が `completed` / `failed` で `completed_at` が **引数で指定した時間（デフォルト 24 時間）より古い** JSON を削除する。加えて `running` のまま `created_at` から **48 時間超**経過したファイルはクラッシュ孤児として削除する。戻り値は削除件数。
- **`result_retention_hours` について**: `config.json` の `background_task.result_retention_hours` はスキーマ上あるが、**`BackgroundTaskManager.cleanup_old_tasks` はこの値を読まない**（デフォルトはメソッド引数 `max_age_hours=24`）。運用で保持時間を変える場合は、呼び出し側が `max_age_hours` に合わせる想定。

#### rotate_dm_logs（システム Cron）

- **実行タイミング**: ライフサイクル（`core/lifecycle/system_crons.py` 等）のシステム Cron で **毎日 04:30**（サーバー設定タイムゾーン）。`core/supervisor/_mgr_scheduler.py` 側でも同一 ID のジョブが登録される構成。
- **対象**: `shared/dm_logs/*.jsonl`（ファイル名に `.archive.` を含むものはスキップ）。
- **動作**: `core.time_utils` のローカル現在時刻基準で、各行 JSON の `ts`（ISO 形式）をパースし、**デフォルト 7 日**より古いエントリを `{stem}.{YYYYMMDD}.archive.jsonl` に追記アーカイブしたうえで、現行ファイルから除去する。`ts` の解釈に失敗した行は **現行ファイルに残す**（データ損失防止）。
- **その他のサーバー定時ジョブ**: Anima 単位の `cron.md` とは別に、ライフサイクルがメモリ保守・RAG 等のシステム Cron を登録する（例: 日次コンソリデーション 02:00、日次インデックス 04:00）。時刻は設定タイムゾーン基準。DM ローテーションは上記 04:30。

### ハートビート設定のホットリロード

heartbeat.md をファイルシステム上で更新すると、次回のハートビート実行時に `_check_schedule_freshness()` が変更を検出し、SchedulerManager がスケジュールを自動リロードする。
サーバーの再起動は不要（MAY skip restart）。APScheduler のジョブが再登録される。

## Per-anima Heartbeat間隔設定

### status.jsonでの設定

各Animaの `status.json` に `heartbeat_interval_minutes` を設定することで、Anima個別のHeartbeat間隔を指定できます。

```json
{
  "heartbeat_interval_minutes": 60
}
```

- 設定可能範囲: 1〜1440分（1日）
- 未設定の場合: `config.json` の `heartbeat.interval_minutes`（デフォルト30分）にフォールバック
- Anima自身が `write_memory_file` で `status.json` を更新して自己調整可能

### 推奨ガイドライン

| 状況 | 推奨間隔 | 理由 |
|------|----------|------|
| アクティブな開発プロジェクト中 | 15〜30分 | 頻繁な状況把握が必要 |
| 通常業務 | 30〜60分 | デフォルト。バランスの取れた頻度 |
| 低負荷・待機状態 | 60〜120分 | コスト節約。タスクがなければ長めに |
| 長期休眠・非アクティブ | 120〜1440分 | 最小限の巡回で状況把握 |

### Activity Level との関係

グローバルな Activity Level（10%〜400%）が設定されている場合、実効間隔は以下の式で計算されます:

```
実効間隔 = ベース間隔 / (Activity Level / 100)
```

例: ベース30分、Activity Level 50% → 実効60分
例: ベース30分、Activity Level 200% → 実効15分

- 実効間隔の下限は5分（どれだけブーストしても5分未満にはならない）
- Activity Level 100%以下では max_turns も比例してスケールダウン（下限3ターン）
- Activity Level 100%以上では max_turns は変更なし（間隔のみ短縮）

### Activity Schedule（時間帯別自動切替 / ナイトモード）

Activity Level を時間帯に応じて自動的に切り替える仕組み。
夜間や休日にコストを抑えたい場合や、業務時間帯にのみ活発に動作させたい場合に使用する。

#### 仕組み

- `config.json` の `activity_schedule` に時間帯エントリを設定する
- 1分ごとに現在時刻をチェックし、該当する時間帯のレベルに Activity Level を自動変更
- Activity Level が変わると、全 Anima のハートビートが即座にリスケジュールされる

#### 設定フォーマット

各エントリは `start`（開始時刻）、`end`（終了時刻）、`level`（Activity Level %）の3フィールド:

```json
{
  "activity_schedule": [
    {"start": "09:00", "end": "22:00", "level": 100},
    {"start": "22:00", "end": "06:00", "level": 30}
  ]
}
```

- 時刻は `HH:MM` 形式（24時間表記）
- **日付跨ぎ対応**: `"22:00"` 〜 `"06:00"` のように start > end の指定が可能（深夜帯をカバー）
- `level` は 10〜400 の範囲
- 最大24エントリまで
- 空配列 `[]` でスケジュールモードを無効化（固定 Activity Level に戻る）

#### 設定方法

- **Settings UI**: ナイトモードのチェックボックス + 時間帯・レベル設定
- **API**: `PUT /api/settings/activity-schedule` に上記 JSON を送信
- **設定ファイル直接編集**: `config.json` の `activity_schedule` を編集後、サーバー再起動

#### 注意点

- Activity Level を手動で変更すると、現在の時間帯に該当するスケジュールエントリも連動して更新される
- スケジュールはサーバー起動時にも即座に適用される（起動時点の時刻で該当レベルに設定）
- どの時間帯にも該当しない場合は、最後に設定された Activity Level が維持される

## Cron タスクとは

Cron は「決められた時間に自動実行されるタスク」。ハートビートが「定期巡回」なのに対し、Cron は「定時業務」。

例:
- 毎朝9:00に業務計画を立てる
- 毎週金曜17:00に週次振り返りをする
- 毎日2:00にバックアップスクリプトを実行する

## cron.md の設定

Cron タスクは `cron.md` に Markdown + YAML 形式で定義する。
ファイルパス: `~/.animaworks/animas/{name}/cron.md`

上司が配下 Anima の `cron.md` を編集する場合は、直接ファイル操作ではなく `read_memory_file` / `write_memory_file` を使い、`../{anima_name}/cron.md` のような相対パスで指定する。

### 基本フォーマット

各タスクは `## タスク名` の見出しで始まり、本文の先頭に `schedule:` ディレクティブで標準5フィールド cron 式を記載する。

```markdown
# Cron: {name}

## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
長期記憶から昨日の進捗を確認し、今日のタスクを計画する。
理念と目標に照らして優先順位を判断する。
結果は state/current_state.md に書き出す。

## 週次振り返り
schedule: 0 17 * * 5
type: llm
今週のepisodes/を読み返し、パターンを抽出してknowledge/に統合する。
```

旧形式（`## タスク名（毎日 9:00 JST）` のように括弧内にスケジュールを書く形式）は、`animaworks migrate-cron` で新形式に変換できる。

### CronTask のスキーマ

各タスクは内部的に以下の `CronTask` モデルにパースされる:

| フィールド | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `name` | str | （必須） | タスク名。`##` 見出しから抽出される |
| `schedule` | str | （必須） | 標準5フィールド cron 式。`schedule:` ディレクティブから抽出 |
| `type` | str | `"llm"` | タスク種別: `"llm"` または `"command"` |
| `description` | str | `""` | LLM 型の指示文（type: llm で使用） |
| `command` | str \| None | `None` | Command 型の bash コマンド |
| `tool` | str \| None | `None` | Command 型の内部ツール名 |
| `args` | dict \| None | `None` | tool の引数（YAML 形式） |
| `skip_pattern` | str \| None | `None` | Command 型: stdout がこの正規表現にマッチしたら follow-up LLM をスキップ |
| `trigger_heartbeat` | bool | `True` | Command 型: `False` ならコマンド出力後の follow-up cron LLM をスキップ |

## LLM 型 Cron タスク

`type: llm` はエージェント（LLM）が判断・思考を伴って実行するタスク。
description に書かれた指示がプロンプトとしてエージェントに渡される。

### 特徴

- エージェントがツールを使い、記憶を検索し、判断を下す
- 結果は不定形（タスクごとに異なる出力）
- 実行にはモデルの API コールが必要（コストが発生する）

### 記述例

```markdown
## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
昨日の episodes/ を読み返し、今日のタスクを計画する。
優先順位は理念と目標に照らして判断する。
結果は state/current_state.md に書き出す。
task_queue.jsonl の未着手タスクも確認し、必要なら優先度を見直す。
```

description（`type:` 行の後の本文）には以下を含めるべき（SHOULD）:
- 何を確認するか（入力）
- どう判断するか（基準）
- 何を出力するか（成果物）

## Command 型 Cron タスク

`type: command` はエージェントの判断を介さず、決まったコマンドやツールを実行するタスク。
確定的な処理（バックアップ、通知送信等）に適している。

### bash コマンド型

```markdown
## バックアップ実行
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

`command:` に bash コマンドを1行で記載する。
コマンドはシェル経由で実行される。

### 内部ツール型

```markdown
## Slack朝の挨拶
schedule: 0 9 * * 1-5
type: command
tool: slack_send
args:
  channel: "#general"
  message: "おはようございます！本日もよろしくお願いします。"
```

`tool:` に内部ツール名、`args:` に YAML 形式で引数を記載する。
args は YAML のインデントブロックとしてパースされる（2スペースインデント）。

### Command 型の follow-up 制御

Command 型タスクは、コマンドが正常終了かつ stdout がある場合、その出力を LLM に渡して follow-up 分析を行う（heartbeat 相当のコンテキストで実行）。

- **`trigger_heartbeat: false`** — follow-up LLM をスキップする（出力の分析が不要な場合）
- **`skip_pattern: <正規表現>`** — stdout がこの正規表現にマッチしたら follow-up をスキップする

```markdown
## ログ取得（出力分析不要）
schedule: 0 8 * * *
type: command
trigger_heartbeat: false
command: /usr/local/bin/fetch-logs.sh

## 監視チェック（"OK" のときは分析不要）
schedule: */15 * * * *
type: command
skip_pattern: ^OK$
command: /usr/local/bin/health-check.sh
```

### LLM 型と Command 型の使い分け

| 観点 | LLM 型 | Command 型 |
|------|--------|-----------|
| 判断が必要か | はい | いいえ |
| API コスト | あり | なし |
| 出力の予測可能性 | 不定形 | 確定的 |
| 適したタスク | 計画立案、振り返り、文章作成 | バックアップ、通知送信、データ取得 |
| エラー時の対応 | エージェントが自律対処 | ログに記録のみ |

迷った場合のガイドライン:
- 「毎回同じことをするだけ」→ Command 型（SHOULD）
- 「状況に応じて判断が変わる」→ LLM 型（SHOULD）
- 「コマンド実行 + 結果の解釈」→ LLM 型で description にコマンド実行を指示

## スケジュール記法

cron.md の `schedule:` ディレクティブには **標準5フィールド cron 式** を記載する。

### 標準 cron 式（必須）

```
分 時 日 月 曜日
```

例:
- `0 9 * * *` — 毎日 9:00
- `0 9 * * 1-5` — 平日 9:00
- `*/30 9-17 * * *` — 9:00〜17:00 の30分ごと
- `0 2 1 * *` — 毎月1日 2:00
- `0 17 * * 5` — 毎週金曜 17:00

タイムゾーンは `config.json` の `system.timezone` で設定可能。未設定時はシステムタイムゾーンを自動検出。

### 日本語スケジュールからの移行

旧形式（`## タスク名（毎日 9:00 JST）`）で書かれた cron.md は、`animaworks migrate-cron` で標準 cron 式に変換できる。変換対応表:

| 日本語記法 | cron 式例 |
|-----------|----------|
| `毎日 HH:MM` | `0 9 * * *` |
| `平日 HH:MM` | `0 9 * * 1-5` |
| `毎週{曜日} HH:MM` | `0 17 * * 5`（金曜） |
| `毎月N日 HH:MM` | `0 9 1 * *` |
| `X分毎` | `*/5 * * * *` |
| `X時間毎` | `0 */2 * * *` |

`隔週`、`毎月最終日`、`第N曜日` は自動変換不可。手動で cron 式を記載する。

## cron_logs の確認方法

Cron タスクの実行結果はサーバーログに記録される。
WebSocket 経由で `anima.cron` イベントとしてブロードキャストもされる。

ログの確認方法:
- サーバーログ: `animaworks.lifecycle` ロガーの INFO レベル
- Web UI: ダッシュボードのアクティビティフィードに表示
- episodes/: LLM 型タスクの場合、エージェント自身が episodes/ にログを書く（SHOULD）

LLM 型タスクの結果は `CycleResult` として記録され、以下の情報を含む:
- `trigger`: `"cron"`
- `action`: エージェントの行動要約
- `summary`: 結果の要約テキスト
- `duration_ms`: 実行時間（ミリ秒）
- `context_usage_ratio`: コンテキスト使用率

## よくある Cron 設定例

### 基本セット（全 Anima 推奨）

```markdown
# Cron: {name}

## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
episodes/ から昨日の行動を確認し、task_queue.jsonl の未着手タスクを見直す。
今日の優先タスクを決め、state/current_state.md を更新する。

## 週次振り返り
schedule: 0 17 * * 5
type: llm
今週の episodes/ を読み返し、パターンや教訓を抽出する。
重要な知見は knowledge/ に書き出す。
繰り返し行った作業があれば procedures/ に手順化を検討する。
```

### 外部連携タスク

```markdown
## Slack日報送信
schedule: 0 18 * * 1-5
type: command
tool: slack_send
args:
  channel: "#daily-report"
  message: "本日の業務完了しました。詳細は明日の朝礼で共有します。"

## GitHub Issue 確認
schedule: 0 10 * * 1-5
type: llm
担当リポジトリの新しい Issue と PR を確認する。
重要なものがあれば supervisor に報告する。
```

### 記憶メンテナンス

```markdown
## 知識の棚卸し
schedule: 0 10 1 * *
type: llm
knowledge/ の全ファイルを確認し、古い情報や矛盾する記載を整理する。
重要度の低い知識はアーカイブを検討する。

## 手順書の更新確認
schedule: 0 10 * * 1
type: llm
procedures/ の手順書を確認し、実際の運用と乖離がないか見直す。
変更があれば手順書を更新する。
```

### コメントアウト

実行したくないタスクは HTML コメントで囲む:

```markdown
<!--
## 一時停止中のタスク
schedule: 0 15 * * *
type: llm
このタスクは一時的に停止中。
-->
```

コメント内の `## ` 見出しはパーサーに無視される。

## Cron 設定のホットリロード

cron.md を更新すると、heartbeat.md と同様にスケジュールが自動リロードされる。
Anima 自身が cron.md を書き換えた場合も即座に反映される（self-modify パターン）。
配下 Anima の `cron.md` / `heartbeat.md` / `injection.md` / `status.json` を上司が編集する場合も、write memoryツールで `../{anima_name}/cron.md` のように指定する。Read / Write / Edit / apply_patch / `Path.write_text` / シェルリダイレクト等の直接ファイル操作は使わない。

リロード時の動作:
1. 該当 Anima の既存 cron ジョブを全て削除
2. 更新後の cron.md をパースし、新しいジョブを登録
3. ログに `Schedule reloaded for '{name}'` が出力される

自分で cron.md を更新する場合の注意点:
- 見出し（`## タスク名`）の直後に `schedule:` ディレクティブを置く（MUST）
- スケジュールは標準5フィールド cron 式で記載する（MUST）
- type 行は schedule の直後に置く（SHOULD）
