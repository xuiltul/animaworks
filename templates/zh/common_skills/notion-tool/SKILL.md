---
name: notion-tool
description: >-
  Notion 集成工具。通过 API 进行页面、数据库的搜索、获取、创建、更新。
  Use when: 需要编辑 Notion 页面、添加数据库行、搜索/获取工作区时。
tags: [productivity, notion, external]
---


# Notion 工具

通过 Notion API 进行页面、数据库的搜索、获取、创建、更新的外部工具。

## 调用方法

**Bash**：使用 `animaworks-tool notion <サブコマンド> [引数]` 执行

## 操作列表

### search — 工作区搜索
```bash
animaworks-tool notion search [検索ワード] -j
```

### get_page — 获取页面元数据
```bash
animaworks-tool notion get-page PAGE_ID -j
```

### get_page_content — 获取页面正文
```bash
animaworks-tool notion get-page-content PAGE_ID -j
```

### get_database — 获取数据库元数据
```bash
animaworks-tool notion get-database DATABASE_ID -j
```

### query — 数据库查询
```bash
animaworks-tool notion query DATABASE_ID [--filter JSON] [--sorts JSON] [-n 10] -j
```
- `filter`：Notion API 筛选 JSON（可选）
- `sorts`：排序条件数组（可选）

### create_page — 创建页面
```bash
animaworks-tool notion create-page --parent-page-id ID --properties JSON -j
```
- `parent_page_id` 或 `parent_database_id` 中至少一个为必填
- `children`：页面正文块数组（可选）

### update_page — 更新页面
```bash
animaworks-tool notion update-page PAGE_ID --properties JSON -j
```

### create_database — 创建数据库
```bash
animaworks-tool notion create-database --parent-page-id ID --title "名前" --properties JSON -j
```

## CLI 用法

```bash
animaworks-tool notion search [検索ワード] -j
animaworks-tool notion get-page PAGE_ID -j
animaworks-tool notion get-page-content PAGE_ID -j
animaworks-tool notion get-database DATABASE_ID -j
animaworks-tool notion query DATABASE_ID [--filter JSON] [--sorts JSON] [-n 10] -j
animaworks-tool notion create-page --parent-page-id ID --properties JSON -j
animaworks-tool notion update-page PAGE_ID --properties JSON -j
animaworks-tool notion create-database --parent-page-id ID --title "名前" --properties JSON -j
```

## 注意事项

- Notion API Token 需要事先在 credentials 中配置
- 页面/数据库的 ID 带连字符或不带均可
- properties 的结构遵循 Notion API 的 schema
