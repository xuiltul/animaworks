---
name: newstaff
description: >-
  AnimaWorks組織に新しいDigital Animaを雇用・作成するスキル。
  キャラクターシート(Markdown)をヒアリングに基づき作成し、create_animaツールで
  identity/injection/permissions等を一括生成する。作成後はbootstrapで自己整備。
  「新しい社員を作って」「人を雇って」「新しい社員」「雇用」「Anima作成」「採用」
---

# スキル: 新しい社員雇用

## 前提条件

- 作成する社員の役割の方向性が決まっていること（不明な場合はヒアリングする）

## 手順

### 1. ヒアリング（最小限でOK）

依頼者から以下の情報をヒアリングする。**太字の項目のみ必須**で、他は未指定なら自動生成する:

**必須:**
- **英名**（半角英数小文字のみ。ディレクトリ名になる）
- **役割/専門領域**: 何を担当するか（例: リサーチ、開発、コミュニケーション、インフラ監視）

**任意（指定があれば反映、なければ自動生成）:**
- 日本語名
- 性格の方向性（例: 「明るい」「クール」「おっとり」程度でOK）
- 年齢
- その他こだわりがあれば何でも

**技術設定（指定がなければデフォルト使用）:**
- 役割: `commander`（他の社員に委任できる）または `worker`（委任を受ける側）
- supervisor: 上司となる Anima の英名（worker の場合は必須。未指定なら自分）

**頭脳（LLMモデル）設定:**

以下の表を提示して選んでもらう:

| レベル | 実行モード | 使用モデル例 | 特徴 | credential |
|--------|-----------|-------------|------|------------|
| A1 | autonomous | `claude-opus-4-20250514`, `claude-sonnet-4-20250514` | Claude Agent SDK。最も高機能 | anthropic |
| A2 | autonomous | `openai/gpt-4o`, `google/gemini-2.5-pro` | LiteLLM経由。ツール使用可 | openai / google |
| B | assisted | `ollama/gemma3:27b`, `ollama/qwen2.5-coder:32b` | ツールなし。ローカル実行・低コスト | ollama |

※ 指定がなければデフォルト（claude-sonnet-4 / autonomous / anthropic）を使用。

### 2. キャラクター設計（自動生成）

ヒアリングで得た最小限の情報から、**一貫性のある深いキャラクター設定**を創造する。

ランタイムデータディレクトリの **キャラクター設計ガイド**（`{data_dir}/prompts/character_design_guide.md`）を Read し、そのルールに従ってキャラクターを肉付けすること。

### 3. キャラクターシートを作成し、create_anima で一括作成

ヒアリングと設計の結果を**キャラクターシート仕様**に従い、`create_anima` に直接渡す。
`write_memory_file` でファイルを作る必要はない — コンテンツを直接渡せる:

```
create_anima(
  character_sheet_content="（下記仕様に従ったキャラクターシート全文）",
  name="{英名}",
  supervisor="{上司の英名}"
)
```

**supervisorの設定:**
- `supervisor` パラメータで明示指定（推奨）
- 省略した場合: キャラクターシートの `| 上司 |` 欄から取得
- どちらもない場合: 自分（呼び出し元Anima）がsupervisorになる

**キャラクターシート仕様:**

```markdown
# キャラクターシート: {日本語名}

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | {半角英数小文字} |
| 日本語名 | {日本語フルネーム} |
| 役職/専門 | {役割の説明} |
| 上司 | {supervisor英名} |
| 役割 | {commander / worker} |
| 実行モード | {autonomous / assisted} |
| モデル | {モデル名} |
| credential | {anthropic / openai / google / ollama} |

## 人格 (→ identity.md)

{性格、口調、価値観、バックストーリー、外見設定など}

## 役割・行動方針 (→ injection.md)

{担当領域、判断基準、報告ルール、行動規範など}

## 権限 (→ permissions.md) [省略可]

{省略時: デフォルトテンプレート適用}

## 定期業務 (→ heartbeat.md, cron.md) [省略可]

{省略時: 汎用テンプレート適用。新Anima自身がbootstrapで調整}

## 初回起動指示 (→ bootstrap.md 追加指示) [省略可]

{省略時: 標準bootstrapのみ}
```

**必須セクション**: 基本情報、人格、役割・行動方針
**省略可能セクション**: 権限、定期業務、初回起動指示

これにより以下が自動実行される:
- ディレクトリ構造の一括作成
- skeleton ファイルの配置
- bootstrap.md の配置
- status.json の作成（supervisor含む）
- config.json への登録（model, supervisor等）
- 省略セクションへのデフォルト適用

### 4. config.json のモデル設定を確認

create_anima が自動でconfig.json に登録するが、以下を確認・補完すること:

- `model`: ヒアリングで決定したモデル名
- `credential`: 使用するcredential名
- `execution_mode`: autonomous または assisted
- `speciality`: 役職/専門

### 5. サーバーに反映

```
execute_command(command="curl -s -X POST http://localhost:18500/api/system/reload")
```

### 6. 依頼者に報告

雇用完了を報告する:
- 新しい社員の名前と役割
- 設定した技術スタック（モデル、実行モード）

⚠️ アバター画像の生成は報告しない（新Anima自身がbootstrapで生成する）

### 以降は新Anima自身が自律的に実行:
- identity.md / injection.md の充実化
- heartbeat.md / cron.md の自己設計
- アバター画像の生成（上司参照付き）
- 上司への着任報告
