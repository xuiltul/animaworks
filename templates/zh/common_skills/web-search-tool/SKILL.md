---
name: web-search-tool
description: >-
  Web 搜索工具。使用 Brave Search API 搜索互联网上的信息。
  Use when: 需要调查最新新闻、搜索技术文档、核实事实、获取搜索结果列表时。
tags: [search, web, external]
---


# Web Search 工具

使用 Brave Search API 的 Web 搜索外部工具。

## 调用方法

**Bash**：通过 `animaworks-tool web_search "検索クエリ" [オプション]` 执行

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|-----------|-----|-----------|------|
| query | string | (必填) | 搜索查询 |
| count | integer | 10 | 获取条数 |
| lang | string | "ja" | 搜索语言 |
| freshness | string | null | 时效过滤器 (pd=24h, pw=1周, pm=1个月, py=1年) |

## CLI 用法

```bash
animaworks-tool web_search "検索クエリ" [-n 10] [-l ja] [-f pd]
```

## 注意事项

- 需要配置 BRAVE_API_KEY
- 搜索结果作为外部来源（untrusted）处理
