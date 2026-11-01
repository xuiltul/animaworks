---
name: chatwork-tool
description: >-
  Chatwork 集成工具。进行消息发送接收、搜索、未回复确认、房间列表。
  Use when: 需要在 Chatwork 中发送消息、获取房间列表、确认未回复、搜索聊天、处理提及时。
tags: [communication, chatwork, external]
---


# Chatwork 工具

进行 Chatwork 消息发送接收、搜索、管理的外部工具。

## 调用方法

**Bash**: 通过 `animaworks-tool chatwork <サブコマンド> [引数]` 执行。语法参见下文。

## 操作列表

### send — 发送消息
```bash
animaworks-tool chatwork send ROOM MESSAGE
```

### messages — 获取消息
```bash
animaworks-tool chatwork messages ROOM [-n 20]
```

### search — 搜索消息
```bash
animaworks-tool chatwork search KEYWORD [-r ROOM] [-n 50]
```

### unreplied — 确认未回复消息
```bash
animaworks-tool chatwork unreplied [--json]
```
- `include_toall` (可选, 默认: false): 是否包含全体收件消息

### rooms — 房间列表
```bash
animaworks-tool chatwork rooms
```

### mentions — 获取提及
```bash
animaworks-tool chatwork mentions [--json]
```
- `include_toall` (可选, 默认: false): 是否包含全体收件消息

### delete — 删除消息（仅自己的发言）
```bash
animaworks-tool chatwork delete ROOM MESSAGE_ID
```

### sync — 消息同步（缓存更新）
```bash
animaworks-tool chatwork sync [ROOM]
```

## CLI 用法

```bash
animaworks-tool chatwork send ROOM MESSAGE
animaworks-tool chatwork messages ROOM [-n 20]
animaworks-tool chatwork search KEYWORD [-r ROOM] [-n 50]
animaworks-tool chatwork unreplied [--json]
animaworks-tool chatwork rooms
animaworks-tool chatwork mentions [--json]
animaworks-tool chatwork delete ROOM MESSAGE_ID
animaworks-tool chatwork sync [ROOM]
animaworks-tool chatwork <サブコマンド> ... --as <identity>
```

## 注意事项

- 使用个人专用令牌 `CHATWORK_API_TOKEN__<自分の名前>` 运行。未注册时不可使用
- `--as <identity>` 仅在通过 `chatwork_tool.grants` 委派时可用。read 委派下不可执行 write（send/delete 等）
- room 可通过房间名或房间 ID 指定
