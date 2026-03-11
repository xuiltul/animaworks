# common_knowledge の参照経路

Anima が common_knowledge にアクセスする5つの経路と、バックグラウンドの RAG インデックス構築の仕組み。

---

## 参照経路の全体像

| # | 経路 | タイプ | Animaの意識 |
|---|------|--------|------------|
| 1 | システムプロンプトヒント | 自動 | ヒントを見て能動的にアクセス |
| 2 | Priming Channel C | 自動 | 関連知識として自動表示 |
| 3 | `search_memory` ツール | 能動 | scope指定で明示検索 |
| 4 | `read_memory_file` / `write_memory_file` | 能動 | パス指定で直接アクセス |
| 5 | Claude Code 直接ファイルI/O（Mode S） | 能動 | Read/Write等で直接アクセス |

---

## 経路1: システムプロンプトへのヒント注入

`builder.py` がシステムプロンプト構築時に、`~/.animaworks/common_knowledge/` にファイルが存在する場合、**ヒントテキスト**をGroup 4（記憶と能力）に注入する。

- **注入タイミング**: プロンプト構築時（自動）
- **内容**: common_knowledge の存在と使い方のヒント（ファイル内容は含まない）
- **除外条件**: `is_task=True`（TaskExec）の場合は省略される
- **Animaの行動**: ヒントを見て `search_memory` や `read_memory_file` で能動的にアクセス

## 経路2: Priming Channel C — RAGベクトル検索

`PrimingEngine` がメッセージのキーワードから自動的にベクトル検索を行い、個人 knowledge と共有 common_knowledge を統合してシステムプロンプトに注入する。

- **バジェット**: 700トークン（Channel C の割り当て）
- **検索対象**: `shared_common_knowledge` コレクション（ChromaDB）
- **マージ方法**: 個人 knowledge の検索結果とスコアでマージ・ソート
- **Animaの行動**: 関連する common_knowledge の断片がPrimingセクションに自動表示される

### 注意点
- 700トークンの制約があるため、全文ではなく関連断片のみ
- common_knowledge のドキュメント数が増えると個人 knowledge のチャンクが押し出されるリスクがある

## 経路3: `search_memory` ツール

Anima が `search_memory(query="...", scope="common_knowledge")` を呼ぶと、キーワード検索とベクトル検索のハイブリッドで common_knowledge を検索する。

- **キーワード検索**: `~/.animaworks/common_knowledge/` 内の .md ファイルをテキスト走査
- **ベクトル検索**: `shared_common_knowledge` コレクションを検索
- **scope指定**: `"common_knowledge"` で限定検索、`"all"`（デフォルト）でも含まれる

### 使用例
```
search_memory(query="メッセージ 送信", scope="common_knowledge")
search_memory(query="レート制限", scope="all")
```

## 経路4: `read_memory_file` / `write_memory_file`

Anima が `read_memory_file(path="common_knowledge/...")` を呼ぶと、パスプレフィックスを検出して `~/.animaworks/common_knowledge/` に解決する。

- **読み取り**: 全 Anima がアクセス可能
- **書き込み**: 全 Anima がアクセス可能（共有知識の蓄積用）
- **パストラバーサル防御**: `is_relative_to` チェックで common_knowledge 外へのアクセスを防止

### 使用例
```
read_memory_file(path="common_knowledge/00_index.md")
write_memory_file(path="common_knowledge/operations/new-guide.md", content="...")
```

## 経路5: Claude Code 直接ファイルI/O（Mode S のみ）

Mode S では Claude Code の組込みツール（Read, Write, Grep, Glob 等）で `~/.animaworks/common_knowledge/` に直接アクセスできる。

- **権限**: `handler_perms.py` で共有読み取り専用ディレクトリとして許可
- **対象モード**: Mode S（Agent SDK）のみ

---

## バックグラウンド: RAGインデックス構築

common_knowledge がベクトル検索（経路2・3）で見つかるためには、ChromaDB にインデックスされている必要がある。

### インデックスタイミング

1. **Anima起動時**: `MemoryManager` 初期化時に `_ensure_shared_knowledge_indexed()` が呼ばれ、SHA-256 ハッシュで変更を検出。変更があれば `shared_common_knowledge` コレクションに再インデックス
2. **日次04:00**: `_run_daily_indexing()` で全AnimaのベクトルDBを増分インデックス更新。common_knowledge もこのタイミングで再インデックスされる

### チャンキング戦略

`memory_type="common_knowledge"` の場合、knowledge と同じ **Markdown ヘッディング区切り** でチャンキングされる。

### コレクション名

`shared_common_knowledge`（全Anima共有の単一コレクション）

---

## reference/ との違い

| 項目 | common_knowledge | reference |
|------|-----------------|-----------|
| RAGインデックス | 対象（`shared_common_knowledge`） | **非対象** |
| `search_memory` | scope="common_knowledge" / "all" で検索可 | 検索不可 |
| Priming Channel C | 自動的に断片が表示される | 表示されない |
| `read_memory_file` | 読み書き可能 | **読み取り専用** |
| 用途 | 日常の実用ガイド・判断基準 | 詳細な技術リファレンス |
| システムプロンプト | ヒント注入あり | ヒント注入あり（別セクション） |
