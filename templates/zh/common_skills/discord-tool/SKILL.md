---
name: discord-tool
description: >-
  Discord 集成工具。进行消息发送接收、搜索、服务器、频道列表、表情回应。
  Use when: 当需要在 Discord 中发送消息、列出频道、在服务器内搜索、添加表情回应、查看线程时。
tags: [communication, discord, external]
---


# Discord 工具

进行 Discord 的消息发送接收、搜索、服务器/频道列表、表情回应的外部工具。

## 调用方法

**Bash**: 通过 `animaworks-tool discord <サブコマンド> [引数]` 执行

## 操作列表

### guilds — 服务器列表
```bash
animaworks-tool discord guilds
```

### channels — 频道列表
```bash
animaworks-tool discord channels GUILD_ID
```
- `GUILD_ID`: 目标服务器（guild）的 Snowflake ID（必填）

### send — 发送消息
```bash
animaworks-tool discord send CHANNEL_ID MESSAGE [--reply-to MESSAGE_ID]
```
- `CHANNEL_ID`: 发送目标文本频道的 Snowflake ID
- `--reply-to`: 可选。回复目标消息 ID

### messages — 获取消息
```bash
animaworks-tool discord messages CHANNEL_ID [-n 20]
```

### search — 搜索消息
```bash
animaworks-tool discord search KEYWORD [-c CHANNEL_ID] [-n 50]
```

### react — 表情回应（仅通过 MCP，CLI 不支持）
- 添加表情回应。**仅通过 MCP**。CLI 中不可用。

## CLI 使用方法

```bash
animaworks-tool discord guilds
animaworks-tool discord channels GUILD_ID
animaworks-tool discord send CHANNEL_ID MESSAGE [--reply-to MESSAGE_ID]
animaworks-tool discord messages CHANNEL_ID [-n 20]
animaworks-tool discord search KEYWORD [-c CHANNEL_ID] [-n 50]
```

## 注意事项

- Discord Bot Token 需要事先在 credentials 中配置
- 频道 ID 可通过 Discord 开发者模式中右键 →「复制 ID」获取
- 消息有 2000 字符限制
- 服务器 ID、频道 ID、消息 ID 均为数字字符串（Snowflake ID）
- Bot 需要必要的权限（Send Messages、Read Message History、Add Reactions、View Channels）
