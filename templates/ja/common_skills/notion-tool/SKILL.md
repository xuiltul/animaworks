---
name: notion-tool
description: >-
  Notion連携ツール。ページ・データベースの検索・取得・作成・更新をAPIで行う。
  Use when: Notionページ編集、データベース行の追加、ワークスペースの検索・取得が必要なとき。
tags: [productivity, notion, external]
---

# Notion ツール

Notion API 経由でページ・データベースの検索・取得・作成・更新を行う外部ツール。

## 呼び出し方法

**Bash**: `animaworks-tool notion <サブコマンド> [引数]` で実行

## アクション一覧

### search — ワークスペース検索
```bash
animaworks-tool notion search [検索ワード] -j
```

### get_page — ページメタデータ取得
```bash
animaworks-tool notion get-page PAGE_ID -j
```

### get_page_content — ページ本文取得
```bash
animaworks-tool notion get-page-content PAGE_ID -j
```

### get_database — データベースメタデータ取得
```bash
animaworks-tool notion get-database DATABASE_ID -j
```

### query — データベースクエリ
```bash
animaworks-tool notion query DATABASE_ID [--filter JSON] [--sorts JSON] [-n 10] -j
```
- `filter`: Notion API フィルタ JSON（任意）
- `sorts`: ソート条件の配列（任意）

### create_page — ページ作成
```bash
animaworks-tool notion create-page --parent-page-id ID --properties JSON -j
```
- `parent_page_id` または `parent_database_id` のいずれかが必須
- `children`: ページ本文ブロック配列（任意）

### update_page — ページ更新
```bash
animaworks-tool notion update-page PAGE_ID --properties JSON -j
```

### create_database — データベース作成
```bash
animaworks-tool notion create-database --parent-page-id ID --title "名前" --properties JSON -j
```

## CLI使用法

```bash
animaworks-tool notion search [検索ワード] -j
animaworks-tool notion get-page PAGE_ID -j
animaworks-tool notion get-page-content PAGE_ID -j
animaworks-tool notion get-database DATABASE_ID -j
animaworks-tool notion query DATABASE_ID [--filter JSON] [--sorts JSON] [-n 10] -j
animaworks-tool notion create-page --parent-page-id ID --properties JSON -j
animaworks-tool notion update-page PAGE_ID --properties JSON -j
animaworks-tool notion create-database --parent-page-id ID --title "名前" --properties JSON -j
```

## 注意事項

- Notion API Token は credentials に事前設定が必要
- ページ/データベースの ID はハイフン付き・なしどちらも可
- properties の構造は Notion API のスキーマに従う
