---
name: gmail-tool
description: >-
  Gmail integration tool. Check unread emails, read message body, create drafts. OAuth2 authentication.
  "gmail" "email" "unread" "draft" "inbox"
tags: [communication, gmail, email, external]
---

# Gmail Tool

External tool for Gmail operations via OAuth2 API access.

## Invocation via Bash

Use **Bash** with `animaworks-tool gmail <subcommand> [args]`:

```bash
animaworks-tool gmail unread [-n 20]
animaworks-tool gmail read MESSAGE_ID
animaworks-tool gmail draft --to ADDR --subject SUBJ --body BODY [--thread-id TID]
```

## Actions

### unread — List unread emails
```json
{"tool_name": "gmail", "action": "unread", "args": {"max_results": 20}}
```

### read_body — Read email body
```json
{"tool_name": "gmail", "action": "read_body", "args": {"message_id": "message ID"}}
```

### draft — Create draft
```json
{"tool_name": "gmail", "action": "draft", "args": {"to": "recipient@example.com", "subject": "Subject", "body": "Body text", "thread_id": "thread ID (optional)"}}
```

## CLI Usage (S/C/D/G-mode)

```bash
animaworks-tool gmail unread [-n 20]
animaworks-tool gmail read MESSAGE_ID
animaworks-tool gmail draft --to ADDR --subject SUBJ --body BODY [--thread-id TID]
```

## Notes

- OAuth2 authentication flow required on first use
- credentials.json and token.json must be in ~/.animaworks/
