---
name: x-search-tool
description: >-
  X（Twitter）検索ツール。キーワード検索と指定ユーザーのツイート取得を行う。
  Use when: X上の話題検索、特定アカウントの投稿取得、トレンド・世論の把握が必要なとき。
tags: [search, x, twitter, external]
---

# X Search ツール

X (Twitter) の検索・ツイート取得を行う外部ツール。

## 呼び出し方法

**Bash**: `animaworks-tool x_search "検索クエリ" [オプション]` または `animaworks-tool x_search --user @username` で実行

### search — キーワード検索
```bash
animaworks-tool x_search "検索クエリ" [-n 10] [--days 7]
```

### user_tweets — ユーザーのツイート取得
```bash
animaworks-tool x_search --user @username [-n 10]
```

## パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| query | string | — | 検索クエリ |
| user | string | — | ユーザー名（@付き） |
| count | integer | 10 | 取得件数 |
| days | integer | 7 | 検索対象日数 |

## CLI使用法

```bash
animaworks-tool x_search "検索クエリ" [-n 10] [--days 7]
animaworks-tool x_search --user @username [-n 10]
```

## 注意事項

- X API (Bearer Token) の設定が必要
- 検索結果は外部ソース（untrusted）として扱われる
