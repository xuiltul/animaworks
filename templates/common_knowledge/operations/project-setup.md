# プロジェクト設定方法

AnimaWorks の設定構造と Anima 追加手順のリファレンス。
設定変更が必要な場面で検索・参照すること。

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
  "anima_defaults": { "model": "claude-sonnet-4-20250514", "max_tokens": 4096 },
  "animas": {
    "hinata": { "model": "claude-sonnet-4-20250514", "supervisor": null },
    "ken": { "model": "openai/gpt-4o", "credential": "openai" }
  },
  "consolidation": { "daily_enabled": true, "daily_time": "02:00" },
  "rag": { "enabled": true },
  "priming": { "dynamic_budget": true },
  "image_gen": {}
}
```

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
| `animas` | Anima 個別の設定オーバーライド |
| `consolidation` | 記憶統合（日次/週次）の設定 |
| `rag` | RAG（埋め込みベクトル検索）の設定 |
| `priming` | プライミング（自動記憶取得）のトークン予算 |
| `image_gen` | 画像生成のスタイル設定 |

<!-- AUTO-GENERATED:START config_fields -->
### 設定項目リファレンス（自動生成）

#### Anima設定 (per-anima overrides)

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `model` | `str | None` | None |  |
| `fallback_model` | `str | None` | None |  |
| `max_tokens` | `int | None` | None |  |
| `max_turns` | `int | None` | None |  |
| `credential` | `str | None` | None |  |
| `context_threshold` | `float | None` | None |  |
| `max_chains` | `int | None` | None |  |
| `conversation_history_threshold` | `float | None` | None |  |
| `execution_mode` | `str | None` | None |  |
| `supervisor` | `str | None` | None |  |
| `speciality` | `str | None` | None |  |

#### デフォルト値 (anima_defaults)

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `model` | `str` | `"claude-sonnet-4-20250514"` |  |
| `fallback_model` | `str | None` | None |  |
| `max_tokens` | `int` | `4096` |  |
| `max_turns` | `int` | `20` |  |
| `credential` | `str` | `"anthropic"` |  |
| `context_threshold` | `float` | `0.5` |  |
| `max_chains` | `int` | `2` |  |
| `conversation_history_threshold` | `float` | `0.3` |  |
| `execution_mode` | `str | None` | None |  |
| `supervisor` | `str | None` | None |  |
| `speciality` | `str | None` | None |  |

#### AnimaWorksConfig トップレベル

| セクション | 説明 |
|-----------|------|
| `version` | 設定ファイルバージョン |
| `setup_complete` | セットアップ完了フラグ |
| `locale` | ロケール設定 |
| `system` | システム設定（モード、ログレベル） |
| `credentials` | API認証情報 |
| `model_modes` | モデル名→実行モードマッピング |
| `anima_defaults` | Anima設定デフォルト値 |
| `animas` | Anima別設定オーバーライド |
| `consolidation` | 記憶統合設定 |
| `rag` | RAG（検索拡張生成）設定 |
| `priming` | プライミング（自動記憶想起）設定 |
| `image_gen` | 画像生成設定 |

<!-- AUTO-GENERATED:END -->

## 新しい Anima の追加方法

Anima の追加方法は3つある。いずれも CLI コマンドで実行する。

### 方法1: テンプレートから作成

事前定義されたテンプレート（`templates/anima_templates/` 配下）を使う方法。
テンプレートには identity.md, injection.md, permissions.md, スキル等が含まれている。

```bash
# テンプレート一覧を確認
animaworks create-anima --template <テンプレート名>

# 名前を変えて作成
animaworks create-anima --template <テンプレート名> --name <anima名>
```

テンプレートが最も推奨される方法。既にキャラクター設定が整っており、bootstrap（初回起動の自己定義）をスキップできる。

### 方法2: Markdown ファイルから作成

キャラクターシート（Markdown）を用意し、それを元に Anima を生成する。

```bash
animaworks create-anima --from-md /path/to/character.md --name ken
```

Markdown ファイルは `character_sheet.md` として Anima ディレクトリにコピーされる。
初回起動（bootstrap）時にエージェントがその内容を読み、identity.md と injection.md を自動生成する。

Markdown ファイルには以下を SHOULD で含める:
- `# Character: 名前` 形式の見出し（名前の自動抽出に使用）
- 性格、役割、外見等のキャラクター設定

### 方法3: ブランク作成

最小限のスケルトンファイルで Anima を作成する。

```bash
animaworks create-anima --name yuki
```

ブランク作成では `{name}` プレースホルダが実名に置換されたスケルトンファイルが生成される。
初回起動の bootstrap で、エージェントがユーザーとの対話を通じてキャラクターを自己定義する。

### 作成後のディレクトリ構成

いずれの方法でも、以下のディレクトリとファイルが生成される:

```
~/.animaworks/animas/{name}/
├── identity.md          # 人格定義（不変ベースライン）
├── injection.md         # 役割・行動指針（可変）
├── bootstrap.md         # 初回起動指示（完了後に削除）
├── permissions.md       # ツール・コマンド権限
├── heartbeat.md         # ハートビート設定
├── cron.md              # 定時タスク設定
├── episodes/            # エピソード記憶（日別ログ）
├── knowledge/           # 意味記憶（学んだ知識）
├── procedures/          # 手続き記憶（手順書）
├── skills/              # 個人スキル
├── state/               # ワーキングメモリ
│   ├── current_task.md  # 現在のタスク
│   └── pending.md       # 未処理タスク一覧
└── shortterm/           # 短期記憶（セッション継続用）
    └── archive/
```

### Anima 名の規約

Anima 名は以下の規則に従わなければならない（MUST）:
- 小文字英数字、ハイフン(`-`)、アンダースコア(`_`)のみ使用可能
- 先頭は英字（`a-z`）であること
- アンダースコア始まりは不可（テンプレート予約）
- 例: `hinata`, `ken-dev`, `worker01`

## 実行モード（A1 / A2 / B）

AnimaWorks は3つの実行モードを持つ。モデル名から自動判定されるが、手動で上書きもできる。

### Mode A1: Claude Agent SDK

Claude モデル専用。Claude Code サブプロセスを使い、最もリッチなツール実行が可能。

- **対象モデル**: `claude-*`（例: `claude-sonnet-4-20250514`, `claude-opus-4-20250514`）
- **特徴**: ファイル操作、Bash 実行、記憶の自律検索を全て Claude Agent SDK 経由で行う
- **credential**: `anthropic` を使用（MUST）

### Mode A2: LiteLLM + tool_use ループ

tool_use をサポートする非 Claude モデル向け。LiteLLM でプロバイダを統一する。

- **対象モデル**: `openai/gpt-4o`, `google/gemini-pro`, `ollama/qwen3:30b` 等
- **特徴**: LiteLLM 経由で tool_use ループを回す。ツール実行はフレームワークがディスパッチ
- **credential**: 各プロバイダに対応した credential を指定

### Mode B: Assisted（LLM は思考のみ）

tool_use 非対応モデル向け。LLM は思考のみ行い、記憶 I/O はフレームワークが代行する。

- **対象モデル**: `ollama/gemma3*`, `ollama/phi4*`, 小規模 Ollama モデル等
- **特徴**: 1ショットで応答を生成。ツール実行不可
- **credential**: 通常 `ollama` 等のローカル credential

### モード自動判定の仕組み

`config.json` の `model_modes` フィールドで明示的にマッピングを追加できる。
未指定の場合はコード内のデフォルトパターン（fnmatch 形式）でマッチングされる。

```json
{
  "model_modes": {
    "ollama/my-custom-model": "A2",
    "ollama/experimental-*": "B"
  }
}
```

判定優先順位:
1. Anima の `execution_mode` フィールド（レガシー: `"autonomous"` → A2, `"assisted"` → B）
2. `config.json` の `model_modes` テーブル（完全一致 → ワイルドカード）
3. コードのデフォルトパターン（完全一致 → ワイルドカード）
4. いずれにもマッチしない場合は `B`（安全側にフォールバック）

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

## 権限設定（permissions.md）

各 Anima の `permissions.md` で、使えるツール・アクセス可能なパス・実行可能なコマンドを定義する。

```markdown
# Permissions: hinata

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
- 各 Anima は自分の `permissions.md` を起動時に読む（MUST）
- ToolHandler が権限チェックを行い、許可されていない操作をブロックする
- 外部ツール（Slack, Gmail, GitHub 等）は `外部ツール` セクションで個別に許可/拒否する
- `読める場所` / `書ける場所` は自然言語で記述し、ToolHandler が解釈する

## Anima デフォルトと個別オーバーライド

`anima_defaults` は全 Anima に適用されるベースライン設定。
`animas` セクションで Anima 名をキーとして個別に上書きできる。

### デフォルト値一覧

| フィールド | デフォルト値 | 説明 |
|-----------|-------------|------|
| `model` | `claude-sonnet-4-20250514` | 使用するLLMモデル |
| `fallback_model` | `null` | フォールバックモデル（メイン失敗時） |
| `max_tokens` | `4096` | 1回の応答の最大トークン数 |
| `max_turns` | `20` | 1セッションの最大ターン数 |
| `credential` | `"anthropic"` | 使用する credential 名 |
| `context_threshold` | `0.50` | コンテキスト使用率がこの閾値を超えると短期記憶を外部化 |
| `max_chains` | `2` | 自動セッション継続の最大回数 |
| `conversation_history_threshold` | `0.30` | 会話履歴圧縮のトリガー閾値 |
| `execution_mode` | `null` | 実行モード（`null` = モデル名から自動判定） |
| `supervisor` | `null` | 上司 Anima 名（`null` = トップレベル） |
| `speciality` | `null` | 専門領域（自由テキスト） |

### オーバーライドの仕組み

Anima 個別設定（`AnimaModelConfig`）では全フィールドが `null` 許容。
`null` のフィールドは `anima_defaults` の値が使われる。

```json
{
  "anima_defaults": {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 4096,
    "credential": "anthropic"
  },
  "animas": {
    "hinata": {
      "model": null,
      "max_tokens": 8192,
      "supervisor": null
    },
    "ken": {
      "model": "openai/gpt-4o",
      "credential": "openai",
      "supervisor": "hinata",
      "speciality": "リサーチ・情報収集"
    }
  }
}
```

上の例では:
- `hinata`: model は defaults の `claude-sonnet-4-20250514` を継承、max_tokens のみ `8192` に上書き
- `ken`: model を `openai/gpt-4o` に変更、credential も `openai` に変更、hinata を上司に設定

### 階層構造

`supervisor` フィールドのみで組織階層を定義する。

- `supervisor: null` — トップレベル Anima（指揮系統の最上位）
- `supervisor: "hinata"` — hinata の部下として動作

階層はメッセージングによる指示・報告で機能する。上司は部下にタスクを委任でき、部下は上司に結果を報告する。
