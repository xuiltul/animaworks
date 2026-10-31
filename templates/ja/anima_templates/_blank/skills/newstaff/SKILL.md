---
name: newstaff
description: >-
  AnimaWorks組織に新しいDigital Animaを雇用・作成するスキル。
  キャラクターシート(Markdown)をヒアリングに基づき作成し、CLIコマンド(animaworks anima create)で
  IDENTITY/SOUL/AGENTS/permissions等を一括生成する。作成後はbootstrapで自己整備。
  「新しい社員を作って」「人を雇って」「新しい社員」「雇用」「Anima作成」「採用」「チームメンバーを増やす」「人員追加」「部下を作る」「スタッフ追加」「hire」「recruit」「team member」
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
- **性格の方向性**: 職能ではなく温度（例: 元気、ギャル、天然、熱血、お嬢、お姉さん）。クールは組織に既に2人いるなら選ばない
- **顔タイプ**: `{data_dir}/prompts/face_types.md` から1つ。既存メンバと重複しすぎないこと

**任意（指定があれば反映、なければ自動生成）:**
- 日本語名
- 年齢
- その他こだわりがあれば何でも

**技術設定（指定がなければデフォルト使用）:**
- 役割: `commander`（他の社員に委任できる）または `worker`（委任を受ける側）
- supervisor: 上司となる Anima の英名（worker の場合は必須。未指定なら自分）

**頭脳（LLMモデル）設定:**

モデルの表をユーザーに見せて選ばせない（使えないモデルが混ざるため）。ユーザーがモデルを指定した時だけ、下の表を参考に設定する。指定がなければキャラクターシートのモデル行を省略し、既定のモデルで進める:

| レベル | 実行モード | 使用モデル例 | 特徴 | credential |
|--------|-----------|-------------|------|------------|
| S | autonomous | `claude-opus-5-5`, `claude-sonnet-5-5` | Claude Agent SDK。最も高機能 | anthropic |
| A | autonomous | `openai/gpt-4.1`, `google/gemini-2.5-pro`, `vertex_ai/gemini-2.5-flash` | LiteLLM経由。ツール使用可 | openai / google / azure / vertex |
| B | assisted | `ollama/gemma3:27b`, `ollama/qwen2.5-coder:32b` | ツールなし。ローカル実行・低コスト | ollama |

※ 指定がなければモデル・実行モード・credential の行を省略する（既定値が使われる）。

### 2. キャラクター設計（自動生成）

ヒアリングで得た情報からキャラクターを肉付けする。**人格が先、職能は後。**

必ず次をこの順で Read する:
1. `{data_dir}/prompts/face_types.md`
2. `{data_dir}/prompts/character_design_guide.md`

役割からクール美人を連想しない。趣味は仕事の延長にしない。既存メンバの髪色・顔タイプ・口調とかぶらないか確認する。

### 3. キャラクターシートを作成し、CLIで一括作成

ヒアリングと設計の結果を**キャラクターシート仕様**に従いキャラクターシートをファイルに書き出し、CLIコマンドで作成する:

1. キャラクターシートをファイルに書き出す（例: `/tmp/{english_name}.md`）
2. 以下のコマンドを実行する:

```bash
animaworks anima create --from-md /tmp/{english_name}.md --name {english_name} --supervisor {supervisor_english_name}
```

**supervisorの設定:**
- `supervisor` パラメータで明示指定（推奨）
- 省略した場合: キャラクターシートの `| 上司 |` 欄から取得
- どちらもない場合: 自分（呼び出し元Anima）がsupervisorになる

**キャラクターシート仕様:**

```markdown
# キャラクターシート: {Japanese name}

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | {lowercase alphanumeric} |
| 日本語名 | {Japanese full name} |
| 役職/専門 | {Role description} |
| 上司 | {supervisor English name} |
| 役割 | {commander / worker} |
| モデル | {model name — ユーザーが指定した時だけ書く} |
| credential | {ユーザーがモデルを指定した時だけ書く} |

## 人格 (→ SOUL.md)

{Personality, speaking style, values, backstory, appearance, etc.}

## 役割・行動方針 (→ AGENTS.md)

{Responsible areas, decision criteria, reporting rules, conduct standards, etc.}

## 権限 (→ permissions.json) [省略可]

{If omitted: default template applied}

## 定期業務 (→ HEARTBEAT.md, cron.md) [省略可]

{If omitted: generic template applied. New Anima self-adjusts in bootstrap}

## 初回起動指示 (→ BOOTSTRAP.md 追加指示) [省略可]

{If omitted: standard bootstrap only}
```

**必須セクション**: 基本情報、人格、役割・行動方針
**省略可能セクション**: 権限、定期業務、初回起動指示

これにより以下が自動実行される:
- ディレクトリ構造の一括作成
- skeleton ファイルの配置
- BOOTSTRAP.md の配置
- status.json の作成（supervisor含む）
- config.json への登録（model, supervisor等）
- 省略セクションへのデフォルト適用

### 4. config.json のモデル設定を確認

`animaworks anima create` が自動でconfig.json に登録するが、以下を確認・補完すること:

- `model`: ヒアリングで決定したモデル名
- `credential`: 使用するcredential名
- `execution_mode`: autonomous または assisted
- `speciality`: 役職/専門

### 4.5. heartbeat頻度の提案・設定

新しい社員の仕事の駆動型に応じて、定期heartbeatの間隔を依頼者に提案し、合意した値を設定する:

| 駆動型 | 推奨間隔 | 例 |
|--------|---------|-----|
| 自発的な状況判断・調整が主業務 | 30分（デフォルト） | チーム調整役、進行監督 |
| メッセージ・イベント駆動が主体 | 60〜120分 | レビュー担当、開発担当 |
| cron定期業務が主体 | 120〜240分 | 監視、経理、法務 |

設定は `{data_dir}/animas/{英名}/status.json` に `"heartbeat_interval_minutes": <分>`（1〜1440）を追記する。
メッセージ起因heartbeatとcronはこの設定と無関係に動くため、間隔を延ばしても応答性は落ちない。迷ったら長めを選ぶ（空振りheartbeatはトークンの無駄になる）。

### 5. サーバーに反映

```
Bash: curl -s -X POST "${ANIMAWORKS_SERVER_URL:-http://localhost:18500}"/api/system/reload
```

### 6. 依頼者に報告

雇用完了を報告する:
- 新しい社員の名前と役割
- 設定した技術スタック（モデル、実行モード）

- 着任するとすぐ最初の小さな仕事に取りかかり、結果はあなたからまとめて伝えること

⚠️ アバター画像の生成は報告しない（新Anima自身がbootstrapで生成する）

### 以降は新Anima自身が自律的に実行:
- IDENTITY.md / SOUL.md / AGENTS.md の充実化
- HEARTBEAT.md / cron.md の自己設計
- アバター画像の生成（上司参照付き）
- 上司への着任報告
- 最初の小さな仕事（ユーザーの役に立つもの）に着手し、結果を上司へ報告
