# doc_freshness.py — ドキュメント鮮度チェック＆自動更新

テンプレートドキュメントの鮮度をチェックし、関連コードより古くなったドキュメントを検出・自動更新するスクリプト。

## 基本的な使い方

### レポートモード（デフォルト）

```bash
python3 scripts/doc_freshness.py
```

全マッピング済みドキュメントをチェックし、staleなものを深刻度順に表示する。

出力例:

```
=== Stale Documents: 42/52 ===

[HIGH] templates/ja/common_knowledge/operations/task-management.md
       Reason: core/background.py changed 2026-02-20 (doc: 2026-02-10, 10d stale)
[MED ] templates/ja/common_skills/cron-management/SKILL.md
       Reason: core/schedule_parser.py changed 2026-02-22 (doc: 2026-02-18, 4d stale)
```

深刻度は差分日数で判定: **HIGH**(7日+) / **MED**(3-7日) / **LOW**(1-3日)

### フィルタリング

```bash
# カテゴリで絞り込み
python3 scripts/doc_freshness.py --category common_skills
python3 scripts/doc_freshness.py --category common_knowledge

# 特定ファイルだけチェック
python3 scripts/doc_freshness.py --file common_knowledge/operations/task-management.md

# マッピング外のドキュメントも含める
python3 scripts/doc_freshness.py --all

# 基準日を手動指定（この日以降のコード変更でstale判定）
python3 scripts/doc_freshness.py --since 2026-02-15
```

`--category` の選択肢: `common_skills`, `common_knowledge`, `docs`

### JSON出力

```bash
python3 scripts/doc_freshness.py --json
python3 scripts/doc_freshness.py --json --category common_knowledge
```

他ツールとの連携用。各エントリに `doc_key`, `locale`, `severity`, `days_stale` 等が含まれる。

### 自動修正モード

```bash
# プレビュー（実際には実行しない）
python3 scripts/doc_freshness.py --fix --dry-run

# 実行（cursor-agentが必要）
python3 scripts/doc_freshness.py --fix

# モデル指定
python3 scripts/doc_freshness.py --fix --model composer-1.5

# en翻訳をスキップ（jaだけ更新）
python3 scripts/doc_freshness.py --fix --skip-en

# カテゴリ絞り込みと組み合わせ
python3 scripts/doc_freshness.py --fix --dry-run --category common_skills
```

`--fix` の動作:

1. staleなjaドキュメントごとに `cursor-agent -p --trust --force --workspace ...` で関連コードを調査・更新
2. 更新されたjaの内容を元に、対応するenドキュメントを翻訳更新
3. タイムアウト（デフォルト30分）超過時はスキップして次のファイルに進行

## フラグ一覧

| フラグ | 説明 |
|--------|------|
| `--fix` | cursor-agentで自動更新 |
| `--dry-run` | fixコマンドのプレビューのみ |
| `--model MODEL` | cursor-agentのモデル（デフォルト: `composer-1.5`） |
| `--all` | マッピング外ドキュメントも対象に含める |
| `--file PATH` | 特定ファイルだけ処理 |
| `--category CAT` | カテゴリで絞り込み |
| `--since YYYY-MM-DD` | ドキュメント更新日を手動指定 |
| `--skip-en` | en翻訳をスキップ |
| `--timeout SEC` | ファイルごとのタイムアウト秒数（デフォルト: `1800`=30分） |
| `--json` | JSON形式で出力 |

## 仕組み

### 変更検知ロジック

1. `git log -1 --format=%ct -- {doc_path}` でドキュメントの最終更新日を取得
2. マッピングされた各コードパスについて `git log -1 --format=%ct -- {code_path}` で最終変更日を取得
3. コード変更日 > ドキュメント更新日 → **stale** と判定

### DOC_SOURCE_MAP

スクリプト内に辞書として定義。キーはja側の相対パス（ロケールディレクトリ内）、値は関連ソースコードパスのリスト。

ja/enの構造差異（common_skillsのディレクトリ構造 vs フラットファイル）は `_SKILL_JA_TO_EN` マッピングで吸収。

### 対象ファイル（許可リスト方式）

- `common_knowledge/*` — Animaが参照するガイド・リファレンス
- `common_skills/*` — Animaが実行するスキル手順書
- `docs/*` — OSS公開リポジトリに同期されるドキュメント（`publish.sh` で除外されないファイル）

`docs/` カテゴリのファイルは `docs/{name}.ja.md`（日本語）と `docs/{name}.md`（英語）のペアで管理される。doc_keyは `docs/{stem}` 形式（例: `docs/vision`, `docs/memory`）。

### 対象外ファイル（自動更新すべきでないもの）

templates/ 内のそれ以外はすべて対象外:

- `prompts/` — システムプロンプトの一部。自動更新はリスク大
- `roles/` — permissions.json（設定テンプレート）+ specialty_prompt.md（プロンプトフラグメント）
- `anima_templates/_blank/*` — Anima作成時コピーされるスケルトン
- `bootstrap.md`, `company/vision.md` — ユーザー編集前提の初期テンプレート
- `common_knowledge/00_index.md` — ナビゲーション用インデックス

docs/ の非公開ディレクトリも対象外（`publish.sh` の除外パターンに基づく）:

- `docs/implemented/`, `docs/issues/`, `docs/research/`, `docs/records/`, `docs/reports/`, `docs/drafts/`, `docs/legacy/`, `docs/testing/`
- `docs/paper/` — 評価データ（コード変更追従不要）

### 前提条件

- gitリポジトリ内で実行すること
- `--fix` モードには `cursor-agent` コマンドがPATHに必要
