---
name: web-search-tool
description: >-
  Web search tool. Search the internet using Brave Search API.
  "search" "web search" "look up" "brave"
tags: [search, web, external]
---

# Web Search Tool

External tool for web search via Brave Search API.

## Invocation via Bash

Use **Bash** with `animaworks-tool web_search <subcommand> [args]`:

```bash
animaworks-tool web_search "search query" [-n 10] [-l ja] [-f pd]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| query | string | (required) | Search query |
| count | integer | 10 | Number of results |
| lang | string | "ja" | Search language |
| freshness | string | null | Freshness filter (pd=24h, pw=1week, pm=1month, py=1year) |

## CLI Usage (S/C/D/G-mode)

```bash
animaworks-tool web_search "search query" [-n 10] [-l ja] [-f pd]
```

## Notes

- BRAVE_API_KEY must be configured
- Search results are treated as external (untrusted) data
