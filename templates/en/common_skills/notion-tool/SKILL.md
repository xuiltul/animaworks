---
name: notion-tool
description: >-
  Notion integration tool. Search, get, create, and update pages and databases.
  "notion" "page" "database"
tags: [productivity, notion, external]
---

# Notion Tool

External tool for searching, reading, creating, and updating Notion pages and databases via the Notion API.

## Invocation via Bash

Use **Bash** with `animaworks-tool notion <subcommand> [args]`. See Actions below for syntax.

## Actions

### search — Search workspace
```json
{"tool_name": "notion", "action": "search", "args": {"query": "search term", "page_size": 10}}
```

### get_page — Get page metadata
```json
{"tool_name": "notion", "action": "get_page", "args": {"page_id": "page-id"}}
```

### get_page_content — Get page body
```json
{"tool_name": "notion", "action": "get_page_content", "args": {"page_id": "page-id"}}
```

### get_database — Get database metadata
```json
{"tool_name": "notion", "action": "get_database", "args": {"database_id": "database-id"}}
```

### query — Query database
```json
{"tool_name": "notion", "action": "query", "args": {"database_id": "database-id", "filter": {}, "sorts": [], "page_size": 10}}
```
- `filter`: Notion API filter JSON (optional)
- `sorts`: Array of sort conditions (optional)

### create_page — Create page
```json
{"tool_name": "notion", "action": "create_page", "args": {"parent_page_id": "parent-page-id", "properties": {"title": [{"text": {"content": "Title"}}]}}}
```
- Either `parent_page_id` or `parent_database_id` is required
- `children`: Array of page content blocks (optional)

### update_page — Update page
```json
{"tool_name": "notion", "action": "update_page", "args": {"page_id": "page-id", "properties": {}}}
```

### create_database — Create database
```json
{"tool_name": "notion", "action": "create_database", "args": {"parent_page_id": "parent-page-id", "title": "DB name", "properties": {}}}
```

## CLI Usage (S/C/D/G-mode)

```bash
animaworks-tool notion search [query] -j
animaworks-tool notion get-page PAGE_ID -j
animaworks-tool notion get-page-content PAGE_ID -j
animaworks-tool notion get-database DATABASE_ID -j
animaworks-tool notion query DATABASE_ID [--filter JSON] [--sorts JSON] [-n 10] -j
animaworks-tool notion create-page --parent-page-id ID --properties JSON -j
animaworks-tool notion update-page PAGE_ID --properties JSON -j
animaworks-tool notion create-database --parent-page-id ID --title "name" --properties JSON -j
```

## Notes

- Notion API Token must be configured in credentials
- Page/database IDs accept both hyphenated and non-hyphenated formats
- Property structures follow the Notion API schema
