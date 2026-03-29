---
name: cron-management
description: >-
  cron.mdを正しいフォーマットで読み書きするスキル。定時タスクの追加・更新・削除手順を提供する。
  Use when: cron.mdの編集、cron式の追加、LLM型・コマンド型タスクの追加・削除、定時ジョブのメンテナンスが必要なとき。
---

## フレームワーク側の実装（参照用）

### cron（定時タスク）

cron の**パース**は `core/schedule_parser.py`（`parse_cron_md` / `parse_schedule`）、**登録・実行・リロード**は `core/supervisor/scheduler_manager.py`（APScheduler、`AsyncIOScheduler(timezone=get_app_timezone())`）が担当する。

### `core/background.py`（cron とは別系統）

このモジュールは **cron のスケジューリングをしない**。長時間ツール呼び出しのバックグラウンド実行と、DM ログのローテーションを担当する。cron と混同しないこと。

**`BackgroundTaskManager`**

- **役割**: 対象ツールを `asyncio` 上で非同期実行し、状態をディスクに保存する。
- **永続化先**: `state/background_tasks/{task_id}.json`（`pending` / `running` / `completed` / `failed`、結果文字列・エラー・タイムスタンプ）。
- **公開 API（概要）**:
  - `submit(tool_name, tool_args, execute_fn)` — 同期実行関数をスレッドプールで走らせる。
  - `submit_async(...)` — 非同期 `execute_fn` 版。
  - `get_task` / `list_tasks` / `active_count` — 照会。
  - `cleanup_old_tasks(max_age_hours=24)` — 完了・失敗タスクを **24 時間**超で削除。加えて `running` のまま **48 時間**超経過した JSON（クラッシュ孤児）も削除。
- **`from_profiles(anima_dir, ..., profiles, config_eligible)`**: バックグラウンド対象ツール集合は次の **3 層マージ**（後勝ち）。
  1. コード内デフォルト `_DEFAULT_ELIGIBLE_TOOLS`（Mode A 互換の固定一覧）
  2. 各ツールモジュールの `EXECUTION_PROFILE` から `background_eligible: true` のエントリ（`core.tools._base.get_eligible_tools_from_profiles`）。キーは `"{tool}:{subcommand}"` 形式（Mode S の `submit` と整合）
  3. `config.json` 等から渡る `config_eligible`（明示上書き）
- **`is_eligible(tool_name)`**: 次のどちらの名前でも照合可能 — スキーマ名（例: `generate_3d_model`）、プロファイルキー（例: `image_gen:3d`）。
- **デフォルトで `_DEFAULT_ELIGIBLE_TOOLS` に含まれる例**（値は目安秒）: `generate_character_assets` / `generate_fullbody` / `generate_bustup` / `generate_icon` / `generate_chibi` / `generate_3d_model` / `generate_rigged_model` / `generate_animations`（各 30）、`local_llm` / `run_command`（各 60）、`machine_run`（600）。
- **完了フック**: `on_complete` に `Callable[[BackgroundTask], Awaitable[None]]` を設定可能。アプリ側でここから `state/background_notifications/` への Markdown 通知などを書く（`background.py` 自体は通知ディレクトリを触らない）。

**`rotate_dm_logs`**

- `shared/dm_logs/*.jsonl` のエントリを `max_age_days`（デフォルト **7 日**）で切り、古い行を `{stem}.{YYYYMMDD}.archive.jsonl` に追記アーカイブする。cron タスクではないが、バックグラウンド系メンテと同一ファイルに定義されているため参照用に記載。

---

## cron.mdの構造

### 全体構成

```markdown
# Cron: {自分の名前}

## タスク名1
schedule: 0 9 * * *
type: llm
タスクの説明文...

## タスク名2
schedule: */5 * * * *
type: command
command: /path/to/script.sh
```

### 絶対に守るべきルール

1. **各タスクは `## タスク名` で始める**（H2見出し。H3やH1は使わない）
2. **`schedule:` 行は必須**（`schedule:` というキーワードで始める。`###` 見出しにしない）
3. **スケジュールは標準5フィールドcron式のみ**（`09:00` や `毎週金曜 17:00` は不可）
4. **`type:` 行は必須**（`llm` または `command`）
5. **タスク間に空行を入れる**（可読性のため）

### やってはいけない書き方

```markdown
❌ ### */5 * * * *           ← H3見出しにcron式を書いてはいけない
❌ ### 09:00                 ← 自然言語の時刻表記は不可
❌ ### 毎週金曜 17:00         ← 日本語のスケジュール表記は不可
❌ cron: 0 9 * * *           ← キー名は "schedule:" であること（"cron:" ではない）
❌ interval: 5m              ← interval形式は不可
❌ schedule: 0 9 * * * *     ← 6フィールドは不可（5フィールドのみ）
```

### 正しい書き方

```markdown
✅ schedule: 0 9 * * *       ← "schedule:" + 半角スペース + 5フィールドcron式
✅ schedule: */5 * * * *
✅ schedule: 30 21 * * *
✅ schedule: 0 17 * * 5
```

---

## 5フィールドcron式リファレンス

### フィールド構成

```
schedule: 分 時 日 月 曜日
```

| フィールド | 位置 | 範囲 | 説明 |
|-----------|------|------|------|
| 分 | 1 | 0-59 | 何分に実行するか |
| 時 | 2 | 0-23 | 何時に実行するか（24時間制） |
| 日 | 3 | 1-31 | 何日に実行するか |
| 月 | 4 | 1-12 | 何月に実行するか |
| 曜日 | 5 | 0-6 | 何曜日に実行するか（0=月, 6=日） |

**注意**: 曜日は **0=月曜日, 6=日曜日**（APSchedulerの仕様。一般的なcronの0=日曜日とは異なる）

### 特殊文字

| 文字 | 意味 | 例 |
|------|------|-----|
| `*` | すべての値 | `* * * * *` = 毎分 |
| `*/n` | n間隔 | `*/5 * * * *` = 5分ごと |
| `n-m` | 範囲 | `0 9-17 * * *` = 9時〜17時の毎正時 |
| `n,m` | リスト | `0 9,12,18 * * *` = 9時・12時・18時 |
| `n-m/s` | 範囲+間隔 | `0 9-17/2 * * *` = 9時〜17時の2時間ごと |

### よく使うスケジュール例

#### 毎日系

| やりたいこと | cron式 | 解説 |
|-------------|--------|------|
| 毎朝9:00 | `0 9 * * *` | 分=0, 時=9 |
| 毎朝9:30 | `30 9 * * *` | 分=30, 時=9 |
| 毎日12:00（正午） | `0 12 * * *` | 分=0, 時=12 |
| 毎日18:00 | `0 18 * * *` | 分=0, 時=18 |
| 毎晩21:30 | `30 21 * * *` | 分=30, 時=21 |
| 毎日深夜2:00 | `0 2 * * *` | 分=0, 時=2 |

#### 間隔系

| やりたいこと | cron式 | 解説 |
|-------------|--------|------|
| 5分ごと | `*/5 * * * *` | 分=*/5（0,5,10,...,55） |
| 10分ごと | `*/10 * * * *` | 分=*/10（0,10,20,...,50） |
| 15分ごと | `*/15 * * * *` | 分=*/15（0,15,30,45） |
| 30分ごと | `*/30 * * * *` | 分=*/30（0,30） |
| 1時間ごと | `0 * * * *` | 毎時0分 |
| 2時間ごと | `0 */2 * * *` | 0時,2時,4時,...,22時の0分 |
| 業務時間中のみ5分ごと | `*/5 9-17 * * *` | 9:00〜17:55の5分間隔 |
| 業務時間中のみ1時間ごと | `0 9-17 * * *` | 9:00〜17:00の毎正時 |

#### 曜日系

| やりたいこと | cron式 | 解説 |
|-------------|--------|------|
| 平日の毎朝9:00 | `0 9 * * 0-4` | 曜日=0-4（月〜金） |
| 毎週月曜9:00 | `0 9 * * 0` | 曜日=0（月） |
| 毎週金曜17:00 | `0 17 * * 4` | 曜日=4（金） |
| 毎週金曜18:00 | `0 18 * * 4` | 曜日=4（金） |
| 平日の業務時間中30分ごと | `*/30 9-17 * * 0-4` | 平日9:00〜17:30 |
| 週末の毎朝10:00 | `0 10 * * 5,6` | 曜日=5,6（土日） |

#### 月次系

| やりたいこと | cron式 | 解説 |
|-------------|--------|------|
| 毎月1日の9:00 | `0 9 1 * *` | 日=1 |
| 毎月15日の12:00 | `0 12 15 * *` | 日=15 |
| 毎月最終営業日近く（28日） | `0 17 28 * *` | 日=28（近似） |
| 四半期初日9:00（1,4,7,10月） | `0 9 1 1,4,7,10 *` | 月=1,4,7,10 |

---

## タスクタイプ詳細

### type: llm — LLM判断タスク

思考・判断が必要なタスク。`schedule:` と `type: llm` の後に、自由記述でタスク内容を書く。

```markdown
## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
長期記憶から昨日の進捗を確認し、今日のタスクを計画する。
理念と目標に照らして優先順位を判断する。
結果は state/current_state.md に書き出す。
```

- 説明文はそのままLLMへのプロンプトとして渡される
- 具体的なアウトプット（何を書き出すか）を明記すると効果的
- 複数行OK
- 本文にフェンス付きコードブロック（\`\`\`）が含まれると、パーサーが警告ログを出す（確定コマンドなら `type: command` を検討）

### type: command — コマンド実行タスク

決定論的に実行するbashコマンドやツール呼び出し。

#### パターンA: bashコマンド

```markdown
## バックアップ実行
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

- `command:` に実行するコマンドを1行で書く
- シェルリダイレクト（`>`, `>>`, `|`）は使用可能
- 複数行のコマンドは非推奨（1行にまとめるか、スクリプトファイルにする）

#### パターンB: ツール呼び出し

```markdown
## Slack朝の通知
schedule: 0 9 * * 0-4
type: command
tool: slack_channel_post
args:
  channel_id: "C0123456789"
  text: "おはようございます！"
```

- `tool:` にツール名（`get_tool_schemas()` / `permissions.md` で許可されたスキーマ名。例: Slack 投稿は `slack_channel_post` など）
- `args:` 以降はYAMLブロック形式でインデント2スペース
- `ToolHandler.handle(tool, args)` で実行され、結果文字列が stdout 相当として扱われる

### オプション: skip_pattern

`type: command` でコマンドが**成功**（`exit_code == 0`）し、かつ標準出力が空でない場合にだけ走る**フォローアップ cron LLM**（heartbeat 同等コンテキストの分析セッション）を、stdout がこの正規表現にマッチするとき**抑制**する。

```markdown
## Chatwork未返信チェック
schedule: */5 * * * *
type: command
command: chatwork_cli.py unreplied --json
skip_pattern: "^\[\]$"
```

- `skip_pattern:` には正規表現を書く（実行時は `re.search(skip_pattern, stdout)`）
- 正規表現に `[]` 等のYAML特殊文字を含む場合は引用符（`"..."` または `'...'`）で囲む。パーサーは外側の引用符を自動除去する
- **パース時**に無効な正規表現 → 警告ログのうえ `skip_pattern` は未設定扱い（フォローアップ抑制なし）
- **実行時**に `re.search` が例外（無効パターンの残存など）→ 警告ログのうえ**抑制せず**フォローアップを実行
- 上の例では、未返信が0件（`[]`）の場合にフォローアップをスキップする

### オプション: trigger_heartbeat

コマンド成功時にフォローアップ cron LLM を走らせるかをタスク単位で制御する（`SchedulerManager._run_cron_task` の評価順に従う）。

```markdown
## Chatwork未返信チェック
schedule: */15 * * * *
type: command
command: animaworks-tool chatwork unreplied
skip_pattern: "^\[\]$"
trigger_heartbeat: false
```

- **フォローアップが検討される条件**（すべて満たすとき）: `exit_code == 0` かつ stdout（`.strip()` 後）が空でない
- **stdout の長さ**: `run_cron_command` がスケジューラに返す `stdout` は先頭 **1000 文字**に切り詰められる。フォローアップ LLM に渡るのもこのプレビュー。完全ログは `state/cron_logs/`（JSONL）側を参照
- `trigger_heartbeat: false` — 上記条件を満たしてもフォローアップ cron LLM を**実行しない**（`skip_pattern` より先に評価）
- `trigger_heartbeat: true`（デフォルト）— 条件を満たせばフォローアップを実行（次に `skip_pattern` を評価）
- `false`, `no`, `0` を指定すると抑制。それ以外は true 扱い
- フォローアップ cron LLM は heartbeat 同等のプロンプトフィルタ（バックグラウンド用コンテキスト）で動く
- `exit_code != 0` や stdout が空のときは、フォローアップは**そもそも起動しない**（`skip_pattern` / `trigger_heartbeat` は無関係）

---

## 型の使い分け判断基準

### type: command を使うべきケース
- 実行するコマンドが完全に確定している
- パラメータが固定（リージョン、クラスタ名、プロファイル等）
- 結果の判断はcron LLMセッションに任せる

### type: llm を使うべきケース
- 状況に応じて実行内容を変える必要がある
- 複数のツールを組み合わせた調査が必要
- 人間的な判断・分析が実行段階で必要

### 禁止パターン
- type: llm にコードブロック（確定コマンド）を含める
  → そのコマンドは type: command にすべき
- 「この通りに実行すること」と書いているのに type: llm
  → LLMはコマンドを正確に再現できない。type: command を使え

### Animaは「コマンドを暗記する人」ではない
type: command は人間がスクリプトを保存するのと同じ。
Animaの価値は結果を見て判断する力にある。
確定的な実行はフレームワークに任せ、
Animaには判断・分析・報告に集中させること。

---

## cron ヘルス通知（自動）

スケジューラは問題検知時に `state/background_notifications/cron_health_{タイムスタンプ}.md` を生成する。次回の heartbeat または cron 実行のコンテキストで読み取り・対応される想定。

**レイヤー1（セットアップ／`reload_schedule` 直後）** — `parse_cron_md` 結果と raw テキストを照合:

- タスクは定義されているが**有効なスケジュールが1件も登録できない**（式がすべて無効など）
- 行頭に空白がある `schedule:` 行が raw に含まれる（コードフェンス内のインデント付き行なども検出。通常は **`schedule:` を行頭（または行全体を trim して `schedule:` で始まる形）** に書く）
- raw に `schedule:` という文字列があるのに、パーサーが **1件もタスクを返さない**

**レイヤー2（3時間ごと）** — 登録済みのユーザー cron ジョブが1件以上あるのに、直近 **3時間** の activity_log に `cron_executed` が **0件** のとき警告（実行自体が動いていない可能性）

---

## cron.md操作手順

### 新規タスク追加

1. 自分の `cron.md` を読み込む
2. ファイル末尾に新しいセクションを追加する
3. **書き込み前にフォーマットを確認する**（下記チェックリスト参照）
4. ファイルを書き込む

```markdown
## 新しいタスク名
schedule: <5フィールドcron式>
type: llm|command
<説明またはcommand/tool行>
```

### 既存タスク変更

1. `cron.md` を読み込む
2. 該当セクション（`## タスク名` から次の `##` の直前まで）を特定
3. 変更する行（`schedule:`, `type:`, 説明文等）を編集
4. ファイルを書き込む

### タスク削除

1. `cron.md` を読み込む
2. 該当セクション全体（`## タスク名` から次の `##` の直前まで）を削除
3. ファイルを書き込む

### タスクの一時無効化

HTMLコメントで囲むとパーサーがスキップする:

```markdown
<!--
## 一時停止中のタスク
schedule: 0 9 * * *
type: llm
このタスクは一時的に停止中。
-->
```

---

## 書き込み前チェックリスト

cron.mdを更新する前に、以下を**必ず**確認すること:

- [ ] 各タスクが `## タスク名` で始まっているか（`###` や `#` ではない）
- [ ] `schedule:` 行があるか（`###` 見出しや自然言語ではない）
- [ ] スケジュールが5フィールドcron式か（`分 時 日 月 曜日`）
- [ ] 各フィールドの値が有効範囲内か（分: 0-59, 時: 0-23, 日: 1-31, 月: 1-12, 曜日: 0-6）
- [ ] `type:` 行があるか（`llm` または `command`）
- [ ] command型の場合、`command:` または `tool:` があるか
- [ ] tool型の場合、`args:` のインデントが正しいか（2スペース）
- [ ] タスク間に空行があるか
- [ ] `schedule:` をコードブロック内に書いていないか（ヘルス警告の原因になりうる）

### バリデーション方法

書き込み後、以下のコマンドで正しくパースされるか確認できる:

```bash
# プロジェクトルートで実行。ANIMAWORKS_ANIMA_DIR 未設定時は ~/.animaworks/animas/default を使用
python -c "
from core.schedule_parser import parse_cron_md, parse_schedule
import os
from pathlib import Path

cron_path = Path(os.environ.get('ANIMAWORKS_ANIMA_DIR', '~/.animaworks/animas/default')) / 'cron.md'
content = cron_path.expanduser().read_text()
tasks = parse_cron_md(content)
for t in tasks:
    trigger = parse_schedule(t.schedule)
    status = '✅' if trigger else '❌ パース失敗'
    print(f'{status} {t.name}: schedule=\"{t.schedule}\" type={t.type}')
"
```

すべてのタスクに ✅ が表示されれば正常。❌ が出た場合はスケジュール式を修正すること。

---

## 完全な記述例

```markdown
# Cron: example_anima

## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
長期記憶から昨日の進捗を確認し、今日のタスクを計画する。
理念と目標に照らして優先順位を判断する。
結果は state/current_state.md に書き出す。

## Chatwork未返信チェック
schedule: */5 9-18 * * 0-4
type: command
command: chatwork-cli unreplied --json > $ANIMAWORKS_ANIMA_DIR/state/chatwork_unreplied.json
skip_pattern: "^\[\]$"
trigger_heartbeat: false

## Slack朝の挨拶
schedule: 0 9 * * 0-4
type: command
tool: slack_channel_post
args:
  channel_id: "C0123456789"
  text: "おはようございます！今日もよろしくお願いします。"

## 週次振り返り
schedule: 0 17 * * 4
type: llm
今週のepisodes/を読み返し、パターンを抽出してknowledge/に統合する。
改善点があれば procedures/ に手順を追記する。

## 月次レポート
schedule: 0 10 1 * *
type: llm
先月のepisodes/とknowledge/を分析し、月次サマリーレポートを作成する。
レポートは knowledge/monthly_report_YYYY-MM.md として保存する。
```

---

## よくある間違いと修正

| 間違い | 正しい書き方 | 原因 |
|--------|------------|------|
| `### */5 * * * *` | `schedule: */5 * * * *` | H3見出しはタスク区切りに使えない |
| `### 09:00` | `schedule: 0 9 * * *` | 自然言語時刻はパース不能 |
| `### 毎週金曜 17:00` | `schedule: 0 17 * * 4` | 日本語表記はパース不能 |
| `schedule: 9:00` | `schedule: 0 9 * * *` | HH:MM形式は5フィールドcron式ではない |
| `schedule: every 5 minutes` | `schedule: */5 * * * *` | 英語表記はパース不能 |
| `schedule: 0 9 * * 7` | `schedule: 0 9 * * 6` | 曜日7は範囲外（0-6） |
| `schedule: 0 9 * * SUN` | `schedule: 0 9 * * 6` | 曜日名は使えない（数字のみ） |
| `schedule: 0 25 * * *` | `schedule: 0 23 * * *` | 時は0-23の範囲 |
| `schedule: 60 * * * *` | `schedule: 0 * * * *` | 分は0-59の範囲 |
| `skip_pattern: ^\[\]$`（引用符なし） | `skip_pattern: "^\[\]$"` | `[]` はYAMLで空リストと解釈される。引用符で囲む |

---

## 注意事項

- **ホットリロード**: `cron.md` / `heartbeat.md` を Anima の `write_memory_file` 等で更新すると、スケジュール変更コールバックで `reload_schedule` が走り即時再登録される。外部の直接編集は、次回 **heartbeat または任意の cron が発火する直前** の mtime チェック（`_check_schedule_freshness`）でも検出され、同様にリロードされる
- **リロード直後の古いジョブ**: mtime 変化を検知したタイミングでスケジューラが再構築されると、**直前の定義に紐づいた cron 発火は「stale」としてスキップ**されうる（意図的な二重実行防止）
- **タイムゾーン**: APScheduler は `get_app_timezone()` を使用。`config.json` の `system.timezone` に IANA 名（例: `Asia/Tokyo`）を設定可能。**空文字**のときは OS タイムゾーンを自動検出し、失敗時は `Asia/Tokyo` にフォールバック
- **同時実行**: **タスク名が異なれば**、同一分に複数ジョブが重なっても `asyncio.create_task` により並行しうる。同一タスク名は実行中に再入しない（`_cron_running` でスキップ）。各ジョブは `max_instances=1`
- command 型は失敗してもプロセスは止めない（`cron_executed` が記録され、`stderr` / exit_code がログに残る）。**フォローアップ cron LLM は成功かつ stdout があるときのみ**（前述）
- `type: llm` の定期実行には **バックグラウンド用モデル**（`status.json` の `background_model` 等、未設定時はメインモデル）が使われる
- command の詳細出力は `state/cron_logs/`（日次 JSONL）に蓄積され、保持日数は設定の `cron_log_retention_days`（デフォルト30日）でハウスキーピングされる
- **他のAnimaのtools/ディレクトリにアクセスする場合は、権限を事前に確認すること**
