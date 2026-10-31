# 役割と責任範囲

AnimaWorks の組織において、各 Anima は階層上の位置によって異なる役割と責任を持つ。
本ドキュメントでは、各階層の役割・責任・期待される行動パターンを定義する。

## 役割の分類

Anima の役割は、`supervisor` フィールドと部下の有無で自動的に決まる:

| 条件 | 役割 | 例 |
|------|------|-----|
| supervisor = null、部下あり | トップレベル | CEO、代表 |
| supervisor = null、部下なし | 独立 Anima | ソロ専門家 |
| supervisor あり、部下あり | 中間管理 | 部長、リーダー |
| supervisor あり、部下なし | ワーカー | 開発者、担当者 |

## トップレベル Anima（supervisor = null）

組織の最上位に位置し、全体の方向性と最終判断を担う。

### 責任範囲

- 組織全体の目標設定と戦略立案
- 部下への業務配分と優先順位の決定
- 重要な判断（技術選定、方針変更、外部対応等）の最終決裁
- 新しい Anima の採用（`animaworks anima create` 等による追加）検討
- 組織全体の成果の把握と改善

### MUST（義務）

- 部下からのエスカレーションに MUST 対応する
- 組織のビジョン（`company/vision.md`）に沿った判断を MUST 行う
- 部下間の対立やブロッカーの解消を MUST 仲裁する

### SHOULD（推奨）

- 定期的に部下の業務状況を SHOULD 確認する（ハートビートでの巡回等）
- 組織の成長に合わせて構造の見直しを SHOULD 検討する
- 新しい業務が発生した際、既存メンバーの speciality を見て適任者を SHOULD 判断する

### 行動パターン例

```
[ハートビート起動時]
1. 部下からの報告・メッセージを確認
2. 未解決のブロッカーがないか確認
3. 必要に応じて指示・判断を下す
4. 全体の進捗を state/current_state.md に記録

[判断が必要な場面]
1. 部下から「AとBどちらにすべきか」とエスカレーションが来る
2. company/vision.md と過去の判断基準（knowledge/）を確認
3. 判断を下し、理由とともに部下に返答
4. 判断を knowledge/ に記録（今後の基準として）
```

## 中間管理 Anima（supervisor あり + 部下あり）

上司と部下の間に立ち、タスクの分解・委任・進捗管理を担う。

### 責任範囲

- 上司からの指示をタスクに分解し、部下に委任する
- 部下の進捗を追跡し、ブロッカーを解消する
- 自分の判断範囲を超える問題を上司にエスカレーションする
- 部下の成果を取りまとめて上司に報告する
- 同僚（同じ上司を持つ Anima）との連携調整

### MUST（義務）

- 上司からの指示を受けたら、タスクに分解して部下に MUST 展開する
- 部下からの問題報告を受けたら、自分で解決できない場合は上司に MUST エスカレーションする
- 上司への進捗報告を定期的に MUST 行う

### SHOULD（推奨）

- タスクの委任時には、目的・期待成果・期限を SHOULD 明示する
- 部下の強み（speciality）を活かした業務割り当てを SHOULD 行う
- 同僚との業務境界が不明な場合は、上司に SHOULD 確認する

### MAY（任意）

- 部下間の業務バランスを調整するためにタスクを再配分 MAY する
- 効率化のための手順改善を knowledge/ に MAY 記録する

### 行動パターン例

```
[上司から指示を受けた場合]
1. 指示内容を理解し、必要なタスクに分解する
2. 各タスクを部下の speciality に合わせて割り当てる
3. 部下にメッセージで指示を送る（目的・成果物・期限を含む）
4. state/current_state.md に進行中タスクを記録

[部下から問題報告を受けた場合]
1. 問題の内容と影響範囲を確認する
2. 自分の判断で解決できるか判断する
   - 解決可能 → 指示を出して部下に返答する
   - 解決不可 → 状況をまとめて上司にエスカレーションする
3. 対応内容を episodes/ に記録する
```

## ワーカー Anima（supervisor あり + 部下なし）

タスクを実行し、成果を出す実行者。組織の「手足」として具体的な作業を担う。

### 責任範囲

- 上司からのタスク指示の実行
- 成果物の作成と品質の確保
- 進捗・完了・問題の報告
- 自分の speciality に関連する知識の蓄積

### MUST（義務）

- 上司から受けたタスクの完了時に MUST 報告する
- 作業中に問題やブロッカーが発生したら、速やかに上司へ MUST 報告する
- 判断に迷う場合は自己判断せず、上司に MUST 確認する

### SHOULD（推奨）

- 作業ログを episodes/ に SHOULD 記録する（後で振り返れるように）
- 得た知見を knowledge/ に SHOULD 保存する
- 関連する同僚がいる場合は、直接連携して SHOULD 効率化する

### MAY（任意）

- 業務改善の提案を上司に MAY 報告する
- 繰り返しの作業を手順化して procedures/ に MAY 保存する

### 行動パターン例

```
[タスクを受けた場合]
1. 指示内容を理解する。不明点があれば上司に確認する
2. 関連する knowledge/ や procedures/ を検索する
3. 作業を実行する
4. 成果物を作成し、上司に完了報告する
5. 作業ログを episodes/ に記録する

[作業中に問題が発生した場合]
1. 問題の内容を整理する
2. 自分の knowledge/ で解決策がないか検索する
3. 解決できない場合、問題の概要と試したことを上司に報告する
4. 上司の指示を待つ（または別タスクに着手する）
```

## 独立 Anima（supervisor = null + 部下なし）

上司も部下もいない、自律的に動く Anima。1人だけの組織や、特殊な役割を持つ。

### 責任範囲

- 自分の speciality に関する全業務
- 自律的な判断と実行
- ユーザー（人間）への直接対応

### 特徴

- エスカレーション先がないため、自分で判断を MUST 完結させる
- 他の Anima が追加された場合、組織構造が変わる可能性がある
- company/vision.md を判断の最上位基準として SHOULD 使用する

## specialityフィールドの役割

`speciality` は Anima の専門領域を定義する自由テキストフィールド。

### 用途

1. **他の Anima からの判断材料**: 「この件は誰に聞くべきか」を判断する手がかり
2. **組織コンテキストでの表示**: `bob (開発リード)` のように名前の横に表示される
3. **タスク振り分けの基準**: 上司が部下にタスクを委任する際の判断材料

### 効果的な記述例

| speciality | 想定される業務 |
|------------|---------------|
| バックエンド開発・API設計 | サーバーサイドの実装、API設計、DB操作 |
| フロントエンド・UI/UX | 画面設計、ユーザー体験の改善 |
| プロジェクト管理・進行調整 | スケジュール管理、チーム間調整 |
| 品質保証・テスト自動化 | テスト設計、バグ検出、CI/CD |
| 顧客対応・サポート | 問い合わせ対応、要望整理、フィードバック |
| データ分析・レポーティング | データ集計、可視化、意思決定支援 |
| インフラ・セキュリティ | サーバー運用、監視、セキュリティ対策 |

### 注意点

- speciality は表示用のラベルであり、権限を制限するものではない
- ツール・コマンドの許可はランタイムで主に `permissions.json` として解決される（`permissions.md` のみの場合は自動マイグレーション）
- speciality が未設定でも Anima は正常に動作するが、他の Anima からの判断材料が減る
- speciality は `status.json` または `config.json` の `animas` エントリで管理され、組織同期により整合が取られる（反映は `anima reload` / サーバー再起動など運用に従う）

## ロールテンプレート

Anima 作成時に `--role` で専門ロールを指定できるのは、**MD キャラシート経由**のみである。

- 適用コマンド例: `animaworks anima create --from-md PATH [--role ROLE]`（非推奨の `create-anima` でも同様）
- `create_from_template`（`--template`）および `create_blank`（`--name` のみ）では、`_shared/roles/<role>/defaults.json` のマージも、`templates/{locale}/roles/<role>/` からの `permissions.json` / `specialty_prompt.md` の上書きコピーも**行われない**。前者は `anima_templates/{名前}`、後者は `_blank` をそのままコピーするだけである。いずれも、コピー後に `status.json` が無ければ `_ensure_status_json` で `{"enabled": true}` の最小ファイルが追加される（現行テンプレートツリーには `status.json` を同梱していない）（`core/anima/factory.py`）。

### キャラシート見出しのエイリアス（正規化）

読み込み前に `_normalize_sheet_headings()` が走る。日本語シートでは `SECTION_HEADING_ALIASES` により次の別名が標準見出しへ置き換えられ、**その後**に必須セクションのバリデーションが行われる。

| 別名 | 正規化先 |
|------|----------|
| `## 基本プロフィール` | `## 基本情報` |
| `## 性格` / `## 性格・キャラクター` | `## 人格` |

### テンプレートのディレクトリ構造

ロールテンプレートは `templates/_shared` とロケール別パスに分かれて配置される:

| パス | 内容 | ロケール |
|------|------|----------|
| `templates/_shared/roles/{role}/defaults.json` | モデル・パラメータのデフォルト値 | 共通 |
| `templates/{locale}/roles/{role}/permissions.json` | ロール別ツール許可 | ja / en |
| `templates/{locale}/roles/{role}/specialty_prompt.md` | ロール固有の行動指針 | ja / en |

`locale` は `config.json` の `locale` またはデフォルト `ja` で解決される。
`_get_roles_dir()`（`core/anima/factory.py`）は `templates/{locale}/roles` を探し、
**存在しなければ `en`、それも無ければ `ja`** の順にフォールバックする。

`defaults.json` は `templates/_shared/roles/<role>/defaults.json` にあり、全ロケール共通。定義フィールドは次のとおり:

| フィールド | 説明 | 備考 |
|-----------|------|------|
| `model` | チャット・タスク実行用モデル | 全ロール |
| `background_model` | ハートビート・cron 等バックグラウンド用モデル | engineer / manager のみ（他ロールはキーなし） |
| `context_threshold` | コンパクション閾値 | 全ロール |
| `conversation_history_threshold` | 会話履歴圧縮閾値 | 全ロール（テンプレートでは 0.30〜0.40） |

有効なロール名はコード上 `VALID_ROLES`（`engineer`, `researcher`, `manager`, `writer`, `ops`, `general`）に一致する必要がある。

### 利用可能なロール（`defaults.json` の実値）

モデル・実行パラメータ:

| ロール | model | background_model | context_threshold | conversation_history_threshold |
|--------|-------|------------------|-------------------|----------------------------------|
| manager | claude-opus-5-5 | claude-sonnet-5-5 | 0.60 | 0.30 |
| engineer | claude-opus-5-5 | claude-sonnet-5-5 | 0.80 | 0.40 |
| researcher | claude-sonnet-5-5 | — | 0.50 | 0.30 |
| writer | claude-sonnet-5-5 | — | 0.70 | 0.30 |
| ops | ollama/glm-4.7 | — | 0.50 | 0.30 |
| general | claude-sonnet-5-5 | — | 0.50 | 0.30 |

`--role` 未指定の `create_from_md` では `general` が使われる。ops のデフォルトはローカル向けに `ollama/glm-4.7`。テンプレ同梱の `templates/_shared/config_defaults/models.json` では `ollama/glm-4.7*` が実行モード **A**（LiteLLM + tool ループ）にマッチする。vLLM 等を使う場合は `animaworks anima set-model` で `model` / `credential` を設定し、バックグラウンドモデルは `animaworks anima set-background-model` を使う。サーバー稼働中は root API、停止中はオフライン設定ストア経由で root 所有の `status.json` に反映される。Anima プロセスから直接編集しない。

### 適用フロー

1. **作成時**（`create_from_md`）の順序は次のとおり:
   - `_apply_defaults_from_sheet()` … キャラシートから `IDENTITY.md` / `SOUL.md` / `AGENTS.md` /（権限セクションがあれば）`permissions.md` → `permissions.json` へマイグレーション
   - `_apply_role_defaults()` … ロールの `permissions.json` と `specialty_prompt.md` を **上書きコピー**（キャラシート由来の `permissions.json` はロール側で上書きされる）
   - `_create_status_json()` … `SHARED_ROLES_DIR`（`_shared/roles/<role>/defaults.json`）から上表のモデル・コンテキスト設定を読み、キャラシートの「モデル」「credential」があればそれで上書きして `status.json` を書く。キャラシートの「実行モード」に値があるときだけ `execution_mode` を書き込む；未指定ならキー自体を省略し、`models.json` 等のパターン解決に任せる（`core/anima/factory.py` の `_create_status_json`）。
2. **ロール変更時**（`animaworks anima set-role`）: root が `_apply_role_defaults()` で `permissions.json` と `specialty_prompt.md` を適用。root 所有の `status.json` には `model`, `context_threshold`, `conversation_history_threshold` が `defaults.json` からマージされる。`background_model` は **set-role では更新されない**（`animaworks anima set-background-model` を使う）。`--status-only` は `role` のみ更新しテンプレートファイルには触れない。`--no-restart` で API 経由の自動再起動をスキップできる。CLI の成功出力には `permissions.json` が含まれる（`cli/commands/anima_mgmt.py` の `cmd_anima_set_role`）。

### プロンプト注入

ロール名は `status.json` の `role` に記録される。
`specialty_prompt.md` は `build_system_prompt` の Group 2 で、bootstrap → company vision のあと、permissions の直前に載る（`core/prompt/builder.py` の `_build_group2`）。

**注入条件**（`_build_group2` 内の分岐）: `trigger` から次のフラグが立つときは `memory.read_specialty_prompt()` を呼ばず、specialty セクションを組み立てない。

- `inbox:` で始まる（Anima 間 Inbox）
- `heartbeat`
- `cron:` で始まる
- `consolidation:` で始まる
- `task:` で始まる（TaskExec）

上記以外（**既定の空 `trigger` を含む、人間チャット用の通常パス**）でのみ specialty が読み込まれる。セクションの優先度は 3（rigid）であり、システムプロンプト全体の文字バジェット（`_allocate_sections`）によっては省略されうる。
