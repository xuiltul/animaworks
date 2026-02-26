---
name: animaworks-guide
description: >-
  animaworksコマンド・CLIのクイックリファレンス。
  サーバー起動停止(start/stop)、チャット送信(chat/send)、Anima管理(list/create/enable/disable/delete/restart/set-model/set-role)、
  ハートビート手動起動、RAGインデックス管理、animaworks-toolの実行方法を網羅。
  「コマンド」「使い方」「CLI」「animaworks」「起動」「停止」「再起動」「送信方法」
  「Anima作成」「ロール変更」「モデル変更」「ステータス確認」「インデックス」
---

# AnimaWorks CLI クイックリファレンス

AnimaWorks の操作は全て `animaworks` コマンドで行う。
このスキルはコマンドの書式・引数・具体例をまとめたリファレンス。

運用の考え方やルールは `common_knowledge/` を参照:
- メッセージングのルール → `communication/messaging-guide.md`
- タスク管理 → `operations/task-management.md`
- ツール体系 → `operations/tool-usage-overview.md`
- 組織構造 → `organization/structure.md`

---

## サーバー操作（基本的には使わないこと）

```bash
animaworks start              # サーバー起動
animaworks stop               # サーバー停止
animaworks restart            # サーバー再起動
animaworks status             # システム状態確認（プロセス・Anima一覧）
```

---

## Anima管理（anima サブコマンド）

### 一覧・状態確認

```bash
animaworks anima list                    # 全Anima一覧（role表示付き）
animaworks anima status                  # 全Animaのプロセス状態
animaworks anima status {名前}           # 特定Animaのプロセス状態
```

### 作成

```bash
# キャラクターシート（MD）から作成（推奨）
animaworks create-anima --from-md {ファイル} [--role {role}] [--name {名前}]

# テンプレートから作成
animaworks create-anima --template {テンプレート名} [--name {名前}]

# ブランク作成
animaworks create-anima --name {名前}
```

### 有効化・無効化・削除

```bash
animaworks anima enable {名前}           # 有効化（休養から復帰）
animaworks anima disable {名前}          # 無効化（休養）
animaworks anima delete {名前}           # 削除（ZIPアーカイブ後）
animaworks anima restart {名前}          # プロセス再起動
```

### モデル変更

```bash
animaworks anima set-model {名前} {モデル名}
animaworks anima set-model {名前} {モデル名} --credential {credential名}
animaworks anima set-model --all {モデル名}   # 全Anima一括変更
```

### ロール変更

```bash
# ロール変更（テンプレート再適用 + 自動restart）
animaworks anima set-role {名前} {role}

# status.jsonのroleフィールドのみ変更（テンプレートは触らない）
animaworks anima set-role {名前} {role} --status-only

# ファイル更新のみ・再起動しない
animaworks anima set-role {名前} {role} --no-restart
```

set-role で自動更新されるファイル:
- `status.json` — role・モデル・max_turns をロールテンプレートの標準値に更新
- `specialty_prompt.md` — ロール別専門ガイドラインに差し替え
- `permissions.md` — ロール別のツール・コマンド許可範囲に差し替え

有効なロール: `engineer`, `researcher`, `manager`, `writer`, `ops`, `general`

---

## 外部ツール実行（animaworks-tool）

Anima が外部サービス（Slack, Gmail, GitHub 等）を使う場合のコマンド。

```bash
# ヘルプ表示
animaworks-tool {ツール名} --help

# 実行
animaworks-tool {ツール名} {サブコマンド} [引数...]

# バックグラウンド実行（長時間ツール向け）
animaworks-tool submit {ツール名} {サブコマンド} [引数...]
```

### 具体例

```bash
animaworks-tool web_search query "AnimaWorks framework"
animaworks-tool slack send --channel "#general" --text "おはようございます"
animaworks-tool github issues --repo owner/repo
animaworks-tool submit image_gen pipeline "1girl, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR
```

submit の詳細 → `common_knowledge/operations/background-tasks.md`

---

## RAGインデックス管理

```bash
animaworks index rebuild                 # 全Animaのインデックス再構築
animaworks index rebuild --name {名前}   # 特定Animaのみ再構築
animaworks index status                  # インデックス状態確認
```

---

## 初期化

```bash
animaworks init                          # ランタイムディレクトリ初期化（~/.animaworks/）
animaworks init --force                  # 既存設定を上書きして再初期化
```
