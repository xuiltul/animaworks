---
name: web-search-tool
description: >-
  Web検索ツール。Brave Search APIでインターネット上の情報を検索する。
  Use when: 最新ニュース調査、技術ドキュメント検索、事実確認、検索結果一覧の取得が必要なとき。
tags: [search, web, external]
---

# Web Search ツール

Brave Search APIを使ったWeb検索外部ツール。

## 呼び出し方法

**Bash**: `animaworks-tool web_search "検索クエリ" [オプション]` で実行

## パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| query | string | (必須) | 検索クエリ |
| count | integer | 10 | 取得件数 |
| lang | string | "ja" | 検索言語 |
| freshness | string | null | 鮮度フィルタ (pd=24h, pw=1週間, pm=1ヶ月, py=1年) |

## CLI使用法

```bash
animaworks-tool web_search "検索クエリ" [-n 10] [-l ja] [-f pd]
```

## 注意事項

- BRAVE_API_KEY の設定が必要
- 検索結果は外部ソース（untrusted）として扱われる
