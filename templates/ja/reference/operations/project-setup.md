# プロジェクト設定方法

AnimaWorks の設定構造と Anima 追加手順のリファレンス。
設定変更が必要な場面で検索・参照すること。

## ランタイムディレクトリの初期化

ランタイムデータは `~/.animaworks/`（または `ANIMAWORKS_DATA_DIR`）に配置される。
初回セットアップ時は `animaworks init` でテンプレートから初期化する。

### init コマンド

| コマンド | 説明 |
|---------|------|
| `animaworks init` | ランタイムディレクトリを初期化（非対話）。既に存在する場合は何もしない |
| `animaworks init --force` | 既存データにテンプレート差分をマージ。新規ファイルのみ追加、`prompts/` は上書き |
| `animaworks init --skip-anima` | インフラのみ初期化。Anima 作成をスキップ |
| `animaworks init --template NAME` | テンプレートから Anima を非対話で作成 |
| `animaworks init --from-md PATH` | MD ファイルから Anima を非対話で作成 |
| `animaworks init --blank NAME` | ブランク Anima を非対話で作成 |

**推奨フロー**: `animaworks init` 実行後、`animaworks start` でサーバーを起動し、ウェブ UI のセットアップウィザードで Anima を追加する。

### 初期化時に作成されるディレクトリ

`ensure_runtime_dir`（`core/init.py`）により以下が作成される:

- `animas/` — Anima ディレクトリ
- `shared/inbox/` — 受信メッセージキュー
- `shared/users/` — ユーザープロファイル
- `shared/channels/` — 共有チャネル（general, ops の初期ファイル）
- `shared/dm_logs/` — DM 履歴（フォールバック用）
- `tmp/attachments/` — 添付ファイル一時保存
- `common_skills/` / `common_knowledge/` — 共通スキル・知識
- `prompts/` / `company/` — プロンプト・組織テンプレート
- `tool_prompts.sqlite3` — ツールプロンプト DB
- `models.json` — モデル名→実行モードマッピング（`config_defaults/` からコピー）

起動のたびに `common_skills` と `common_knowledge` はテンプレートから増分同期され、新規エントリのみが追加される（既存ファイルは保持）。

## config.json の全体構造

AnimaWorks の統合設定ファイルは `~/.animaworks/config.json` に配置される。
全ての設定は `AnimaWorksConfig` モデルで定義されており、以下のトップレベルフィールドを持つ。

```json
{
  "version": 1,
  "setup_complete": true,
  "locale": "ja",
  "system": { "mode": "server", "log_level": "INFO" },
  "credentials": {
    "anthropic": { "api_key": "sk-ant-..." },
    "openai": { "api_key": "sk-..." }
  },
  "model_modes": {},
  "anima_defaults": { "model": "claude-sonnet-4-6", "max_tokens": 8192 },
  "animas": {
    "aoi": { "supervisor": null, "speciality": null },
    "taro": { "supervisor": "aoi", "speciality": null }
  },
  "consolidation": { "daily_enabled": true, "daily_time": "02:00" },
  "rag": { "enabled": true },
  "priming": { "dynamic_budget": true },
  "image_gen": {}
}
```

**注意**: `animas` セクションは組織レイアウト（`supervisor`, `speciality`）のみを保持する。モデル名・credential・max_turns 等のモデル設定は各 Anima の `status.json` に記録される（後述の「Anima 設定の解決」参照）。

各セクションの役割:

| セクション | 説明 |
|-----------|------|
| `version` | 設定スキーマバージョン（現在 `1`） |
| `setup_complete` | 初回セットアップ完了フラグ |
| `locale` | UI言語（`"ja"` / `"en"`） |
| `system` | サーバーモード・ログレベル |
| `credentials` | APIキー・エンドポイント（名前付き） |
| `model_modes` | モデル名→実行モードの上書きマップ |
| `anima_defaults` | 全 Anima 共通のデフォルト設定 |
| `animas` | Anima の組織レイアウト（supervisor, speciality）。モデル設定は status.json |
| `consolidation` | 記憶統合（日次/週次）の設定 |
| `rag` | RAG（埋め込みベクトル検索）の設定 |
| `priming` | プライミング（自動記憶取得）のトークン予算 |
| `image_gen` | 画像生成のスタイル設定 |

<!-- AUTO-GENERATED:START config_fields -->
### 設定項目リファレンス（自動生成）

#### Anima設定 (per-anima overrides)

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `supervisor` | `str | None` | None |  |
| `speciality` | `str | None` | None |  |
| `model` | `str | None` | None |  |

#### デフォルト値 (anima_defaults)

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `model` | `str` | `"claude-sonnet-4-6"` |  |
| `fallback_model` | `str | None` | None |  |
| `background_model` | `str | None` | None |  |
| `background_credential` | `str | None` | None |  |
| `max_tokens` | `int` | `8192` |  |
| `max_turns` | `int` | `20` |  |
| `credential` | `str` | `"anthropic"` |  |
| `context_threshold` | `float` | `0.5` |  |
| `max_chains` | `int` | `2` |  |
| `conversation_history_threshold` | `float` | `0.3` |  |
| `execution_mode` | `str | None` | None |  |
| `supervisor` | `str | None` | None |  |
| `speciality` | `str | None` | None |  |
| `thinking` | `bool | None` | None |  |
| `thinking_effort` | `str | None` | None |  |
| `llm_timeout` | `int` | `600` |  |
| `mode_s_auth` | `str | None` | None |  |
| `max_outbound_per_hour` | `int | None` | None |  |
| `max_outbound_per_day` | `int | None` | None |  |
| `max_recipients_per_run` | `int | None` | None |  |

#### AnimaWorksConfig トップレベル

| セクション | 説明 |
|-----------|------|
| `version` | 設定ファイルバージョン |
| `setup_complete` | セットアップ完了フラグ |
| `locale` | ロケール設定 |
| `system` | システム設定（モード、ログレベル） |
| `credentials` | API認証情報 |
| `model_modes` | モデル名→実行モードマッピング |
| `model_context_windows` |  |
| `model_max_tokens` |  |
| `anima_defaults` | Anima設定デフォルト値 |
| `animas` | Anima別設定オーバーライド |
| `consolidation` | 記憶統合設定 |
| `rag` | RAG（検索拡張生成）設定 |
| `priming` | プライミング（自動記憶想起）設定 |
| `image_gen` | 画像生成設定 |
| `human_notification` |  |
| `server` |  |
| `external_messaging` |  |
| `background_task` |  |
| `activity_log` |  |
| `heartbeat` |  |
| `voice` |  |
| `housekeeping` |  |
| `activity_level` |  |
| `activity_schedule` |  |
| `ui` |  |

<!-- AUTO-GENERATED:END -->

## 新しい Anima の追加方法

Anima の追加方法は3つある。いずれも `animaworks anima create` または `animaworks init` で実行する。
`animaworks anima create` は `--role` と `--supervisor` オプションがあり、推奨される。

### 方法1: テンプレートから作成

事前定義されたテンプレート（`templates/ja/anima_templates/` または `templates/en/anima_templates/` 配下）を使う方法。
テンプレートには identity.md, injection.md, permissions.json, スキル等が含まれている。

```bash
# テンプレートから作成（テンプレート名はディレクトリ名）
animaworks anima create --template <テンプレート名>

# 名前を変えて作成
animaworks anima create --template <テンプレート名> --name <anima名>
```

テンプレートが最も推奨される方法。既にキャラクター設定が整っており、bootstrap（初回起動の自己定義）をスキップできる。

### 方法2: Markdown ファイルから作成

キャラクターシート（Markdown）を用意し、それを元に Anima を生成する。

```bash
animaworks anima create --from-md /path/to/character.md [--name ken] [--role engineer] [--supervisor aoi]
```

- `--name`: Anima 名（省略時はシートから抽出）
- `--role`: ロールテンプレート（engineer, researcher, manager, writer, ops, general）。デフォルト: general
- `--supervisor`: 上司 Anima 名（キャラクターシートの「上司」を上書き）

Markdown ファイルは `character_sheet.md` として Anima ディレクトリにコピーされる。
シートの「人格」「役割・行動方針」セクションは identity.md と injection.md に反映される。
ロールテンプレートから permissions.json と specialty_prompt.md が適用される。

Markdown ファイルには以下を SHOULD で含める:
- `# Character: 名前` 形式の見出し、または基本情報テーブルの「英名」行（名前の自動抽出に使用）
- `## 基本情報` — 英名、上司、モデル等のテーブル
- `## 人格` — identity.md に反映
- `## 役割・行動方針` — injection.md に反映

### 方法3: ブランク作成

最小限のスケルトンファイルで Anima を作成する。

```bash
animaworks anima create --name aoi
```

`--name` は必須。ブランク作成では `{name}` プレースホルダが実名に置換されたスケルトンファイルが生成される。
初回起動の bootstrap で、エージェントがユーザーとの対話を通じてキャラクターを自己定義する。

### 作成後のディレクトリ構成

いずれの方法でも、以下のディレクトリとファイルが生成される:

```
~/.animaworks/animas/{name}/
├── identity.md          # 人格定義（不変ベースライン）
├── injection.md         # 役割・行動指針（可変）
├── bootstrap.md         # 初回起動指示（完了後に削除）
├── permissions.json       # ツール・コマンド権限
├── heartbeat.md         # ハートビート設定
├── cron.md              # 定時タスク設定
├── episodes/            # エピソード記憶（日別ログ）
├── knowledge/           # 意味記憶（学んだ知識）
├── procedures/          # 手続き記憶（手順書）
├── skills/              # 個人スキル
├── state/               # ワーキングメモリ
│   ├── current_state.md  # 現在のタスク
│   └── task_queue.jsonl # 永続タスクキュー（未着手タスク等）
└── shortterm/           # 短期記憶（セッション継続用）
    └── archive/
```

### Anima 名の規約

Anima 名は以下の規則に従わなければならない（MUST）:
- 小文字英数字、ハイフン(`-`)、アンダースコア(`_`)のみ使用可能
- 先頭は英字（`a-z`）であること
- アンダースコア始まりは不可（テンプレート予約）
- 例: `aoi`, `taro-dev`, `worker01`

## 実行モード（S / C / D / G / A / B）

AnimaWorks は **6** つの実行モードを持つ。モデル名から自動判定されるが、`status.json` の `execution_mode` で上書きもできる。

### Mode S (SDK): Claude Agent SDK

Claude モデル専用。Claude Code サブプロセスを使い、最もリッチなツール実行が可能。

- **対象モデル**: `claude-*`（例: `claude-sonnet-4-6`, `claude-opus-4-6`）
- **特徴**: ファイル操作、Bash 実行、記憶の自律検索を全て Claude Agent SDK 経由で行う
- **credential**: `anthropic` を使用（MUST）

### Mode C (Codex): Codex CLI

OpenAI Codex 系を Codex CLI ラッパー経由で実行する。

- **対象モデル**: `codex/*`（例: `codex/o4-mini`, `codex/gpt-4.1`）
- **特徴**: MCP ツールと AnimaWorks 外部ツールの統合パスは Mode S/D/G と同系
- **credential**: Codex / OpenAI 側の要件に従う

### Mode D (Cursor Agent): Cursor Agent CLI

Cursor の `cursor-agent` CLI を子プロセスとして実行。MCP 経由で AnimaWorks ツールを利用する。

- **対象モデル**: `cursor/*`
- **特徴**: ホストに CLI と認証が必要。失敗時は Mode A（LiteLLM）へ切り替え可能
- **credential**: Cursor / `agent login` の認証状態に依存

### Mode G (Gemini CLI): Gemini CLI

Google の `gemini` CLI を子プロセスとして実行。MCP 統合。

- **対象モデル**: `gemini/*`
- **特徴**: CLI または `GEMINI_API_KEY` が必要。フォールバック時は Mode A で `gemini/` → `google/` 等へリマップされる場合がある
- **credential**: CLI ログインまたは API キー

### Mode A (Autonomous): LiteLLM + tool_use ループ

tool_use をサポートするクラウド・ローカルモデル向け。LiteLLM でプロバイダを統一する。

- **対象モデル**: `openai/gpt-4.1`, `google/gemini-2.5-pro`, `vertex_ai/gemini-2.5-flash`, `ollama/qwen3:30b` 等
- **特徴**: LiteLLM 経由で tool_use ループを回す。ツール実行はフレームワークがディスパッチ
- **credential**: 各プロバイダに対応した credential を指定

### Mode B (Basic): Assisted（LLM は思考のみ）

tool_use 非対応モデル向け。LLM は思考のみ行い、記憶 I/O はフレームワークが代行する。

- **対象モデル**: `ollama/gemma3*`, `ollama/phi4*`, 小規模 Ollama モデル等
- **特徴**: 1ショットで応答を生成。ツール実行不可
- **credential**: 通常 `ollama` 等のローカル credential

### モード自動判定の仕組み

`~/.animaworks/models.json` で明示的にマッピングを追加できる。
未指定の場合はコード内のデフォルトパターン（fnmatch 形式）でマッチングされる。

```json
{
  "model_modes": {
    "ollama/my-custom-model": "A",
    "ollama/experimental-*": "B"
  }
}
```

判定優先順位:
1. Anima の `execution_mode` フィールド（per-anima override）
2. `~/.animaworks/models.json`（完全一致 → ワイルドカード）
3. `config.json` の `model_modes`（非推奨フォールバック）
4. コードのデフォルトパターン（完全一致 → ワイルドカード）
5. いずれにもマッチしない場合は `B`（安全側にフォールバック）

## クレデンシャル設定

API キーは `credentials` セクションで名前付きで管理する。

```json
{
  "credentials": {
    "anthropic": {
      "api_key": "sk-ant-api03-...",
      "base_url": null
    },
    "openai": {
      "api_key": "sk-...",
      "base_url": null
    },
    "ollama": {
      "api_key": "",
      "base_url": "http://localhost:11434"
    }
  }
}
```

各 Anima は `credential` フィールドでどの credential を使うか指定する。

- `api_key` — APIキー文字列。空文字の場合は環境変数からのフォールバックを試みる
- `base_url` — カスタムエンドポイント。Ollama やプロキシ利用時に設定する。`null` でデフォルト

**セキュリティ**: config.json はファイルパーミッション `0600` で保存される（MUST）。APIキーを含むため、他ユーザーからの読み取りを防ぐ。

## 権限設定（permissions.json）

各 Anima の `permissions.json` で、使えるツール・アクセス可能なパス・実行可能なコマンドを定義する。

```markdown
# Permissions: aoi

## 使えるツール
Read, Write, Edit, Bash, Grep, Glob

## 読める場所
- 自分のディレクトリ配下すべて
- /shared/ 配下

## 書ける場所
- 自分のディレクトリ配下すべて

## 実行できるコマンド
全般的なコマンド

## 実行できないコマンド
rm -rf, システム設定の変更

## 外部ツール
- image_gen: yes
- web_search: yes
- slack: no
```

権限に関するルール:
- 各 Anima は自分の `permissions.json` を起動時に読む（MUST）
- ToolHandler が権限チェックを行い、許可されていない操作をブロックする
- 外部ツール（Slack, Gmail, GitHub 等）は `外部ツール` セクションで個別に許可/拒否する
- `読める場所` / `書ける場所` は自然言語で記述し、ToolHandler が解釈する

### ブロックコマンド

`permissions.json` に `## 実行できないコマンド` セクションを記載すると、指定されたコマンドの実行がブロックされる。
システム全体のハードコードされたブロックリスト（`rm -rf /` 等の危険なコマンド）に加え、Anima 個別のブロックリストが適用される。

```markdown
## 実行できないコマンド
rm -rf, docker rm, git push --force
```

パイプライン中のコマンドも個別にチェックされる（例: `cat file | rm -rf` は `rm -rf` がブロックされる）。

## Anima 設定の解決（2層マージ）

Anima のモデル設定は **`status.json` が Single Source of Truth（SSoT）** となる。

### 設定解決の2層構造

| 優先度 | ソース | 説明 |
|--------|--------|------|
| 1（最優先） | `status.json` | 各 Anima ディレクトリに配置。モデル・実行パラメータの全設定を保持 |
| 2（フォールバック） | `anima_defaults` | `config.json` の全体デフォルト。`status.json` に未設定のフィールドに適用 |

`config.json` の `animas` セクションは **組織レイアウト**（`supervisor`, `speciality`）のみを保持する。
モデル名・credential・max_turns 等のモデル設定は `status.json` に記録される。

### status.json の構造

ファイルパス: `~/.animaworks/animas/{name}/status.json`

```json
{
  "enabled": true,
  "role": "engineer",
  "model": "claude-opus-4-6",
  "credential": "anthropic",
  "max_tokens": 16384,
  "max_turns": 200,
  "max_chains": 10,
  "context_threshold": 0.80,
  "execution_mode": null
}
```

### モデル変更

モデルを変更するには CLI コマンドを使用する:

```bash
animaworks anima set-model <anima名> <モデル名> [--credential <credential名>]

# 全 Anima のモデルを一括変更
animaworks anima set-model --all <モデル名>
```

スーパーバイザーが部下のモデルを変更する場合は `set_subordinate_model` ツールを使用する。

### 設定のリロード

`status.json` を変更した後、プロセスを再起動せずに設定を反映するには `reload` コマンドを使用する:

```bash
# 単一 Anima のリロード
animaworks anima reload <anima名>

# 全 Anima のリロード
animaworks anima reload --all
```

リロードは IPC 経由で即座に反映される（ダウンタイムなし）。実行中のセッションは旧設定で完了し、次のセッションから新設定が適用される。

**典型的な設定変更ワークフロー**:

1. `animaworks anima set-model <name> <model>` でモデルを変更
2. `animaworks anima reload <name>` で即座に反映

手動で `status.json` を編集した場合も同様に `reload` で反映できる。

### デフォルト値一覧（anima_defaults）

| フィールド | デフォルト値 | 説明 |
|-----------|-------------|------|
| `model` | `claude-sonnet-4-6` | 使用するLLMモデル |
| `max_tokens` | `8192` | 1回の応答の最大トークン数 |
| `max_turns` | `20` | 1セッションの最大ターン数 |
| `credential` | `"anthropic"` | 使用する credential 名 |
| `context_threshold` | `0.50` | コンテキスト使用率がこの閾値を超えると短期記憶を外部化 |
| `max_chains` | `2` | 自動セッション継続の最大回数 |

### 階層構造

`supervisor` フィールドのみで組織階層を定義する（`config.json` の `animas` セクションに記載）。

- `supervisor: null` — トップレベル Anima（指揮系統の最上位）
- `supervisor: "aoi"` — aoi の部下として動作

階層はメッセージングによる指示・報告で機能する。上司は部下にタスクを委任でき、部下は上司に結果を報告する。

## Anima 管理コマンド一覧

日常的な Anima の操作・管理に使う CLI コマンド。
サーバーが起動中（`animaworks start` 済み）の状態で実行する。

| コマンド | 説明 | ダウンタイム |
|---------|------|-----------|
| `animaworks anima list` | 全 Anima の一覧と状態を表示 | なし |
| `animaworks anima status [name]` | 指定 Anima（省略時は全体）のプロセス状態を表示 | なし |
| `animaworks anima reload <name>` | status.json を再読み込みしてモデル設定を即座に反映（プロセス再起動なし） | なし |
| `animaworks anima reload --all` | 全 Anima の設定を一括リロード | なし |
| `animaworks anima restart <name>` | Anima プロセスを完全に再起動（コード変更の反映時に使用） | 15-30秒 |
| `animaworks anima set-model <name> <model>` | モデルを変更（status.json を更新。反映には `reload` が必要） | なし |
| `animaworks anima set-model --all <model>` | 全 Anima のモデルを一括変更 | なし |
| `animaworks anima enable <name>` | 休止中の Anima を有効化してプロセスを起動 | — |
| `animaworks anima disable <name>` | Anima を休止（プロセス停止、status.json の enabled=false） | — |
| `animaworks anima create` | 新しい Anima を作成（`--from-md`, `--template`, `--blank`） | — |
| `animaworks anima delete <name>` | Anima を削除（デフォルトでアーカイブ保存） | — |

### サーバー管理コマンド

| コマンド | 説明 |
|---------|------|
| `animaworks start` | サーバーを起動 |
| `animaworks stop` | サーバーを停止 |
| `animaworks restart` | サーバーを完全再起動（全プロセス再生成） |
| `animaworks status` | システム全体のステータスを表示 |
| `animaworks reset` | サーバー停止後、ランタイムディレクトリを削除して再初期化（破壊的） |
| `animaworks reset --restart` | 上記の後、サーバーを再起動 |

### reload / restart / system reload の棲み分け

| コマンド | 動作 | ダウンタイム | ユースケース |
|---------|------|-----------|-------------|
| `anima reload` | IPC で ModelConfig スワップ | なし | status.json のモデル/パラメータ変更 |
| `anima restart` | プロセス kill → 再生成 | 15-30秒 | コード変更の反映、メモリリーク対策 |
| サーバー restart | 全 Anima 再起動 + 新規検出 | 15-30秒 | Anima 追加/削除の反映 |
