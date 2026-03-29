---
name: google-calendar-tool
description: >-
  Googleカレンダー連携ツール。予定の一覧取得・作成をOAuth2でCalendar API経由で行う。
  Use when: 予定の確認、新規イベント作成、スケジュール変更、カレンダー同期が必要なとき。
tags: [calendar, google, schedule, external]
---

# Google Calendar ツール

GoogleカレンダーのイベントをOAuth2で直接操作する外部ツール。

## 呼び出し方法

**Bash**: `animaworks-tool google_calendar <サブコマンド> [引数]` で実行

## アクション一覧

### list — 予定一覧取得
```bash
animaworks-tool google_calendar list [-n 20] [-d 7] [--calendar-id primary]
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| max_results | integer | 20 | 最大取得件数 |
| days | integer | 7 | 何日先まで取得するか |
| calendar_id | string | "primary" | カレンダーID |

### add — 予定追加
```bash
animaworks-tool google_calendar add "会議" --start 2026-03-04T10:00:00+09:00 --end 2026-03-04T11:00:00+09:00
```

| パラメータ | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| summary | string | Yes | イベントタイトル |
| start | string | Yes | 開始時刻（ISO8601またはYYYY-MM-DD） |
| end | string | Yes | 終了時刻（ISO8601またはYYYY-MM-DD） |
| description | string | No | 詳細説明 |
| location | string | No | 場所 |
| calendar_id | string | No | カレンダーID（デフォルト: primary） |
| attendees | array | No | 参加者メールアドレスのリスト |

## CLI使用法

```bash
animaworks-tool google_calendar list [-n 20] [-d 7] [--calendar-id primary]
animaworks-tool google_calendar add "会議" --start 2026-03-04T10:00:00+09:00 --end 2026-03-04T11:00:00+09:00
```

## 注意事項

- 初回使用時にOAuth2認証フローが必要
- credentials.json を ~/.animaworks/credentials/google_calendar/ に配置すること
- 終日イベントは start/end を YYYY-MM-DD 形式で指定
