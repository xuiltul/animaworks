---
name: gmail-tool
description: >-
  Gmail 集成工具。通过 OAuth2 经 Gmail API 进行未读确认、正文获取、草稿创建。
  Use when: 需要确认邮件接收、读取正文、创建草稿、搜索收件箱、操作带标签邮件时。
tags: [communication, gmail, email, external]
---


# Gmail 工具

通过 OAuth2 直接操作 Gmail 邮件的外部工具。

## 调用方法

**Bash**：使用 `animaworks-tool gmail <サブコマンド> [引数]` 执行

## 操作列表

### unread — 未读邮件列表
```bash
animaworks-tool gmail unread [-n 20]
```

### read_body — 读取邮件正文
```bash
animaworks-tool gmail read MESSAGE_ID
```

### draft — 创建草稿
```bash
animaworks-tool gmail draft --to ADDR --subject SUBJ --body BODY [--thread-id TID]
```

## CLI 用法

```bash
animaworks-tool gmail unread [-n 20]
animaworks-tool gmail read MESSAGE_ID
animaworks-tool gmail draft --to ADDR --subject SUBJ --body BODY [--thread-id TID]
```

## 注意事项

- 首次使用时需要 OAuth2 认证流程
- credentials.json 和 token.json 需放置在 ~/.animaworks/ 中
