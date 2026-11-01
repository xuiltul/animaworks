---
name: slack-tool
description: >-
  Slack 集成工具。进行消息发送接收、搜索、未回复确认、频道列表、表情回应。
  Use when: 需要向 Slack 发布、频道列表、线程回复、未回复确认、添加回应时。
tags: [communication, slack, external]
---


# Slack 工具

进行 Slack 消息发送接收、搜索、回应的外部工具。

## 调用方法

**Bash**: 通过 `animaworks-tool slack <サブコマンド> [引数]` 执行

## 操作列表

### send — 发送消息
```bash
animaworks-tool slack send CHANNEL MESSAGE [--thread TS]
```

### messages — 获取消息
```bash
animaworks-tool slack messages CHANNEL [-n 20]
```

### search — 搜索消息
```bash
animaworks-tool slack search KEYWORD [-c CHANNEL] [-n 50]
```

### unreplied — 确认未回复消息
```bash
animaworks-tool slack unreplied [--json]
```

### channels — 频道列表
```bash
animaworks-tool slack channels
```

### react — 表情回应
- `emoji`: Slack 的表情名称（不带冒号。例: `thumbsup`, `eyes`, `white_check_mark`）
- `message_ts`: 回应目标消息的时间戳（可从 `messages` 操作的结果中获取）
- **注意**: `react` 操作 CLI 尚未支持。通过 MCP 使用。

## CLI 使用方法

```bash
animaworks-tool slack send CHANNEL MESSAGE [--thread TS]
animaworks-tool slack messages CHANNEL [-n 20]
animaworks-tool slack search KEYWORD [-c CHANNEL] [-n 50]
animaworks-tool slack unreplied [--json]
animaworks-tool slack channels
```

## 注意事项

- Slack Bot Token 需要在 credentials 中预先配置
- 频道通过带 # 的名称或频道 ID 指定
- 回应需要 `reactions:write` 权限范围
