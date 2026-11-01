---
name: google-calendar-tool
description: >-
  Google 日历集成工具。通过 OAuth2 经 Calendar API 获取日程列表和创建日程。
  Use when: 需要确认日程、创建新事件、更改日程安排、同步日历时。
tags: [calendar, google, schedule, external]
---


# Google Calendar 工具

通过 OAuth2 直接操作 Google 日历事件的外部工具。

## 调用方法

**Bash**: 使用 `animaworks-tool google_calendar <サブコマンド> [引数]` 执行

## 操作列表

### list — 获取日程列表
```bash
animaworks-tool google_calendar list [-n 20] [-d 7] [--calendar-id primary]
```

| 参数 | 类型 | 默认值 | 说明 |
|-----------|-----|-----------|------|
| max_results | integer | 20 | 最大获取条数 |
| days | integer | 7 | 获取未来多少天内的日程 |
| calendar_id | string | "primary" | 日历 ID |

### add — 添加日程
```bash
animaworks-tool google_calendar add "会議" --start 2026-03-04T10:00:00+09:00 --end 2026-03-04T11:00:00+09:00
```

| 参数 | 类型 | 必填 | 说明 |
|-----------|-----|------|------|
| summary | string | Yes | 事件标题 |
| start | string | Yes | 开始时间（ISO8601 或 YYYY-MM-DD） |
| end | string | Yes | 结束时间（ISO8601 或 YYYY-MM-DD） |
| description | string | No | 详细说明 |
| location | string | No | 地点 |
| calendar_id | string | No | 日历 ID（默认：primary） |
| attendees | array | No | 参与者电子邮件地址列表 |

## CLI 使用方法

```bash
animaworks-tool google_calendar list [-n 20] [-d 7] [--calendar-id primary]
animaworks-tool google_calendar add "会議" --start 2026-03-04T10:00:00+09:00 --end 2026-03-04T11:00:00+09:00
```

## 注意事项

- 首次使用时需要 OAuth2 认证流程
- 将 credentials.json 放置在 ~/.animaworks/credentials/google_calendar/ 中
- 全天事件请以 YYYY-MM-DD 格式指定 start/end
