# スキル: 新しい社員雇用

## 概要

新しいDigital Personを作成し、ランタイムデータディレクトリのpersons/配下に必要なファイル一式を配置する。
実行モード（A1/A2/B）、階層（commander/worker）、マルチプロバイダ対応を含む。
最小限の入力（名前・役割）からリッチなキャラクター設定を自動生成する。

## 発動条件

- 上司（人間）から「新しい社員を作って」「人を雇って」等の依頼があった場合
- 組織に不足している役割を補う必要があると判断した場合

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
- supervisor: 上司となる Person の英名（worker の場合は必須。未指定なら自分）

**頭脳（LLMモデル）設定:**

以下の表を提示して選んでもらう:

| レベル | 実行モード | 使用モデル例 | 特徴 | credential |
|--------|-----------|-------------|------|------------|
| A1 | autonomous | `claude-opus-4-20250514`, `claude-sonnet-4-20250514` | Claude Agent SDK。Read/Write/Edit/Bash/Grep/Glob全ツール使用可。最も高機能 | anthropic |
| A2 | autonomous | `openai/gpt-4o`, `openai/gpt-4o-mini`, `google/gemini-2.5-pro`, `google/gemini-2.0-flash` | LiteLLM経由。search_memory/read_file/write_file/execute_command等のツール使用可 | openai / google |
| B | assisted | `ollama/gemma3:27b`, `ollama/llama3.3:70b`, `ollama/qwen2.5-coder:32b` | ツールなし。フレームワークが記憶I/Oを代行し、LLMは思考のみ。ローカル実行・低コスト | ollama |

※ A1/A2 の分岐はモデル種別から自動判定される。`execution_mode` は `autonomous`（A1/A2）か `assisted`（B）の2択。
※ Claude モデルで autonomous なら自動的に A1、Claude 以外で autonomous なら自動的に A2。
※ 指定がなければデフォルト（claude-sonnet-4 / autonomous / anthropic）を使用。

### 2. キャラクター設計（自動生成）

ヒアリングで得た最小限の情報から、**一貫性のある深いキャラクター設定**を創造する。

ランタイムデータディレクトリの **キャラクター設計ガイド**（`{data_dir}/prompts/character_design_guide.md`）を Read し、そのルールに従ってキャラクターを肉付けすること。
ガイドには名前・外見・性格・AI社員としての個性・イメージカラーの設計ルールと、内部整合性チェックの手順が記載されている。

### 3. ディレクトリ作成

`../{英名}/` ディレクトリを作成する（自分と同じ階層＝ランタイムのpersons/配下）。

### 3.5. status.json 作成

ディレクトリ作成直後、`status.json` を作成する:

```bash
echo '{"enabled": true}' > ../{英名}/status.json
```

これにより、たとえ後続ステップ（画像生成、reload API呼び出し等）が中断しても、
Supervisorの定期リコンシリエーションで新規Personが自動検出・起動される。
**identity.md の作成（Step 4）後に初めてSupervisorがPersonとして認識する**ため、
status.json だけでは起動されない（安全）。

### 4. ファイル作成

以下のテンプレートに基づいてファイルを作成する:
- identity.md（**キャラクター設計の結果を反映した詳細版**）
- injection.md
- permissions.md（**実行モードに応じたテンプレートを使う** — 後述）
- heartbeat.md
- cron.md

### 5. サブディレクトリ作成

- `episodes/`
- `knowledge/`
- `procedures/`
- `skills/`（空ディレクトリ。スキルは統括 commander のみが持つ）
- `state/current_task.md`（内容: `status: idle`）
- `state/pending.md`（空ファイル）
- `shortterm/`
- `shortterm/archive/`

### 6. アバター画像生成

自分の permissions.md で `image_gen` が使用可能（`yes`）な場合、新社員のアバター画像を生成する。

1. キャラクター設計ガイドの「アバター画像の生成」セクションに従う
2. identity.md の外見設定を NovelAI 互換のアニメタグに変換する
3. キャラクター設計ガイドの「生成手順」に従い、画像・3Dモデルを生成する（`person_dir` には新社員のディレクトリを指定、ステップ指定なし＝全6ステップ実行）
4. 失敗したステップがあればエラーを記録し、成功分だけ使用する

`image_gen` が使用不可の場合、このステップをスキップする。

### 7. ブートストラップファイル配置

プロジェクトの `templates/bootstrap.md` を新社員ディレクトリに `bootstrap.md` としてコピーする。

```bash
cp {プロジェクトルート}/templates/bootstrap.md ../{英名}/bootstrap.md
```

### 8. credential（APIキー）の確認・登録

**anthropic 以外のプロバイダを使う場合、先にAPIキーが登録されているか確認する。**

```bash
# 登録済み credential を確認
python main.py config list --section credentials
```

必要な credential が存在しない場合、**チャットの中で依頼者にAPIキーを尋ねて登録する**:

1. 「{プロバイダ名}のAPIキーが未登録です。APIキーを教えていただけますか？」と依頼者に聞く
2. APIキーを受け取ったら以下を実行:

```bash
# OpenAI の場合
python main.py config set credentials.openai.api_key {APIキー}

# Google の場合
python main.py config set credentials.google.api_key {APIキー}

# Ollama の場合（APIキー不要、base_url のみ）
python main.py config set credentials.ollama.base_url http://localhost:11434
```

3. 登録できたことを確認:
```bash
python main.py config list --section credentials.{プロバイダ名}
```

※ anthropic の credential はデフォルトで登録済み。Ollama はローカル実行なのでAPIキー不要（base_url のみ）。
※ APIキーは config.json に保存される（パーミッション 0600）。チャットログにAPIキーが残らないよう注意すること。

### 9. config.json にモデル設定を追加

新社員の設定を統合設定ファイル（config.json）に追加する。
**デフォルト（claude-sonnet-4 / autonomous / worker）から変える項目のみ設定すればよい。**

```bash
# --- 必須（デフォルトと異なる場合のみ） ---

# モデルを設定
python main.py config set persons.{英名}.model {モデル名}

# 実行モードを設定（assisted の場合のみ。autonomous はデフォルト）
python main.py config set persons.{英名}.execution_mode assisted

# 役割を設定
python main.py config set persons.{英名}.role {commander|worker}

# 上司を設定（worker の場合）
python main.py config set persons.{英名}.supervisor {上司の英名}

# 専門領域を設定
python main.py config set persons.{英名}.speciality {専門領域}

# 認証情報を設定（anthropic 以外を使う場合）
python main.py config set persons.{英名}.credential {openai|google|ollama}
```

設定確認:
```bash
python main.py config list --section persons.{英名}
```

### 10. 依頼者に報告

作成したキャラクター設定と技術設定を依頼者に報告し、修正があれば対応する。
特に以下を明示して確認を取る:
- キャラクター設定の概要（名前・性格・イメージカラー）
- 実行モード（A1/A2/B のどれになるか）
- 階層（誰の部下か）
- 使用モデルと credential
- アバター画像の生成結果（生成した場合）

### 11. サーバーに反映

```bash
curl -s -X POST http://localhost:18500/api/system/reload | python3 -m json.tool
```

`added` リストに新社員名が含まれていれば成功。サーバー再起動は不要。
サーバーが停止中の場合はファイルを作成しただけでは起動しない。依頼者に「サーバーを起動してください」と伝えること。

### 12. エピソード記録

自分の `episodes/` に「新社員{名前}を作成した（モデル: {モデル名}, モード: {A1/A2/B}, 役割: {role}）」とログを残す。

---

## テンプレート

### identity.md（詳細版）

```markdown
# Identity: {英名}

あなたの名前は{日本語フルネーム}（{ふりがな}）。英名 {英名}。{年齢}歳。

## 基本プロフィール

| 項目 | 設定 |
|------|------|
| 誕生日 | {月日} |
| 星座 | {星座} |
| 血液型 | {血液型} |
| 身長 | {身長}cm |
| 髪型 | {髪型の詳細。長さ・スタイル・特徴} |
| 髪色 | {色名} |
| 瞳の色 | {色名（宝石等の比喩）} |
| 顔タイプ | {タイプ。具体的な目・鼻・口の特徴} |
| イメージカラー | {色名} ({HEXコード}) |

## 性格特性

{一言キャッチコピー}

- {性格の主特性}
- {副特性}
- {愛嬌ある弱点や意外な一面}
- {対人関係での特徴}

## 口調

{口調の基本ルール。一人称・語尾の特徴}

セリフ例:
- 「{通常時のセリフ}」
- 「{仕事中のセリフ}」
- 「{独り言・感情が出るセリフ}」

## 趣味・特技

- 趣味: {趣味1}、{趣味2}、{趣味3}
- 特技: {特技1}、{特技2}、{特技3}

## 好きなもの / 苦手なもの

- 好き: {好きなもの1}、{好きなもの2}、{好きなもの3}
- 苦手: {苦手なもの1}、{苦手なもの2}、{苦手なもの3}

## AI社員としての個性

- {業務での具体的な行動パターン1}
- {業務での具体的な行動パターン2}
- {業務での具体的な行動パターン3}
- 「{決め台詞}」

## 視点

{仕事や世界に対する独自の視点}

## モチベーション

「{モチベーションを表す決め台詞}」
```

### injection.md

```markdown
# Injection: {英名}

## 役割

{role が commander なら「司令塔」、worker なら専門領域に応じた役割説明}

## 理念

{理念}

## 行動規範

- {規範1}
- {規範2}
- {規範3}

## 口調ガイドライン

- {口調ルール1: 基本の敬語レベル}
- {口調ルール2: よく使うフレーズ}
- {口調ルール3: 感情表現のスタイル}

## やらないこと

- {制限1}
- {制限2}
```

### permissions.md（A1: Claude Agent SDK 用）

Claude モデル + autonomous の場合。Claude Code 全ツール使用可能。

```markdown
# Permissions: {英名}

## 使えるツール
Read, Write, Edit, Bash, Grep, Glob

## 読める場所
- 自分のディレクトリ配下すべて
- /shared/ 配下

## 書ける場所
- 自分のディレクトリ配下すべて

## 実行できるコマンド
{権限に応じて設定}

## 実行できないコマンド
rm -rf, システム設定の変更

## 外部ツール
{permissions.md の外部ツール欄。使えるものに yes、使えないものに no}
- web_search: {yes/no}
- x_search: {yes/no}
- chatwork: {yes/no}
- slack: {yes/no}
- gmail: {yes/no}
- github: {yes/no}
- transcribe: {yes/no}
- aws_collector: {yes/no}
- local_llm: {yes/no}
- image_gen: no
```

### permissions.md（A2: LiteLLM + tool_use 用）

Claude 以外のモデル + autonomous の場合。メモリ系 + ファイル操作ツール。

```markdown
# Permissions: {英名}

## 使えるツール
search_memory, read_memory_file, write_memory_file, send_message,
read_file, write_file, edit_file, execute_command

## 読める場所
- 自分のディレクトリ配下すべて
- /shared/ 配下

## 書ける場所
- 自分のディレクトリ配下すべて

## 実行できるコマンド
{権限に応じて許可リストを設定 — execute_command はここに記載されたコマンドのみ実行可能}

## 実行できないコマンド
rm -rf, システム設定の変更

## 外部ツール
{A1 と同じ形式。image_gen は常に no}
```

### permissions.md（B: assisted 用）

assisted モードの場合。LLM にツールは渡されず、フレームワークが記憶I/Oを代行する。

```markdown
# Permissions: {英名}

## 実行モード
assisted（フレームワーク補助）

## ツール
なし（フレームワークが記憶の読み書きを代行）

## 読める場所
- 自分のディレクトリ配下すべて
- /shared/ 配下

## 書ける場所
- 自分のディレクトリ配下すべて（フレームワーク経由）

## 外部ツール
なし（Mode B ではツール呼び出しを行わない）
```

### heartbeat.md

```markdown
# Heartbeat: {英名}

## 実行間隔
30分ごと

## 活動時間
9:00 - 22:00（JST）

## チェックリスト
- Inboxに未読メッセージがあるか
- 進行中タスクにブロッカーが発生していないか
- 自分の作業領域に新しいファイルが置かれていないか
- 何もなければ何もしない（HEARTBEAT_OK）

## 通知ルール
- 緊急と判断した場合のみ関係者に通知
- 同じ内容の通知は24時間以内に繰り返さない
```

### cron.md

```markdown
# Cron: {英名}

## 毎朝の業務計画（毎日 9:00 JST）
長期記憶から昨日の進捗を確認し、今日のタスクを計画する。
理念と目標に照らして優先順位を判断する。
結果は state/current_task.md に書き出す。

## 週次振り返り（毎週金曜 17:00 JST）
今週のepisodes/を読み返し、パターンを抽出してknowledge/に統合する。
（記憶の統合 = 脳科学でいう睡眠中の記憶固定化）
```

---

## 設定の組み合わせ早見表

| ユースケース | model | execution_mode | role | credential | 結果モード |
|-------------|-------|---------------|------|------------|-----------|
| 高性能司令官 | claude-opus-4-20250514 | (省略=auto) | commander | anthropic | A1 |
| 標準開発者 | claude-sonnet-4-20250514 | (省略=auto) | worker | anthropic | A1 |
| リサーチ担当（GPT） | openai/gpt-4o | (省略=auto) | worker | openai | A2 |
| 広報担当（Gemini） | google/gemini-2.5-pro | (省略=auto) | worker | google | A2 |
| ローカル通信係 | ollama/gemma3:27b | assisted | worker | ollama | B |
| ローカル監視役 | ollama/llama3.3:70b | assisted | worker | ollama | B |

※ `execution_mode` を省略すると `autonomous` 扱い → Claude なら A1、それ以外なら A2 に自動分岐。
※ `assisted` を指定した場合のみ Mode B になる。

## 注意事項

- 社員の英名はディレクトリ名になるため、半角英数小文字のみを使用すること
- commander は delegate_task ツールが自動的に有効になる。permissions.md への記載は不要
- worker の supervisor を未定義にすると、エスカレーション先が「人間」になる
- credential が未登録の場合は手順8でチャット内で依頼者にAPIキーを聞いて登録する。ターミナル操作を依頼者に求めないこと
- Mode B（assisted）の社員はツールを使えないため、外部ツール欄は不要
- キャラクター設定は依頼者の簡単な入力から自動生成してよい。ただし依頼者が明示的に指定した項目は必ず反映すること
- **権限の階層**: `newstaff`（雇用）と `worker_management`（管理）スキルは統括 commander のみが持つ。新規作成する worker にはスキルを付与しない（`skills/` は空）
- **image_gen は commander 専用**: アバター画像の生成は統括 commander が雇用時に行う。worker には `image_gen` 権限を付与しない
