# common_knowledge 的参考路径

Anima 访问 common_knowledge 的 5 条路径，以及后台 RAG 索引构建的机制。

---

## 参考路径的整体情况

| # | 路径 | 类型 | Anima 的意识 |
|---|------|--------|------------|
| 1 | 系统提示词提示 | 自动 | 看到提示后主动访问 |
| 2 | Priming Channel C | 自动 | 作为相关知识自动显示 |
| 3 | `search_memory` 工具 | 主动 | 通过 scope 指定进行显式搜索 |
| 4 | `read_memory_file` / `write_memory_file` | 主动 | 通过路径指定直接访问 |
| 5 | Claude Code 直接文件 I/O（Mode S） | 主动 | 通过 Read/Write 等直接访问 |

---

## 路径1: 向系统提示词注入提示

`builder.py` 在构建系统提示词时，如果 `~/.animaworks/common_knowledge/` 中存在文件，则将**提示文本**注入到 Group 4（记忆与能力）中。

- **注入时机**: 构建提示词时（自动）
- **内容**: 关于 common_knowledge 的存在及使用方法的提示（不包含文件内容）
- **排除条件**: 在 `is_task=True`（TaskExec）的情况下会被省略
- **Anima 的行动**: 看到提示后，通过 `search_memory` 或 `read_memory_file` 主动访问

## 路径2: Priming Channel C / C0 — RAG 向量搜索

`PrimingEngine` 根据消息中的关键词自动进行向量搜索，将个人 knowledge 与共享 common_knowledge 整合后注入系统提示词。

- **Channel C 预算**: 1200 token
- **Channel C0 预算**: 300 token（专用于带 `[IMPORTANT]` 标签的块概要指针）
- **搜索对象**: `shared_common_knowledge` 集合（ChromaDB）
- **合并方法**: 与个人 knowledge 的搜索结果按分数合并并排序
- **信任度分离**: Channel C 的结果按 trust 级别分离（medium / untrusted）。来自外部平台的块被视为 untrusted
- **Anima 的行动**: 相关的 common_knowledge 片段会自动显示在 Priming 部分

### 注意事项
- 由于有 1200 token 的限制，因此不是全文，而是仅包含相关片段
- 当 common_knowledge 的文档数量增加时，存在个人 knowledge 的块被挤出的风险
- `[IMPORTANT]` 块会通过 Channel C0 始终注入，因此对于可靠地回忆起重要业务规则是有效的

## 路径3: `search_memory` 工具

当 Anima 调用 `search_memory(query="...", scope="common_knowledge")` 时，会通过关键词搜索与向量搜索的混合方式搜索 common_knowledge。

- **关键词搜索**: 对 `~/.animaworks/common_knowledge/` 内的 .md 文件进行文本扫描
- **向量搜索**: 搜索 `shared_common_knowledge` 集合
- **scope 指定**: `knowledge` / `episodes` / `procedures` / `common_knowledge` / `skills` / `activity_log` / `all`。通过 `"common_knowledge"` 进行限定搜索，`"all"`（默认）也会包含在内
- **`scope="all"`**: 除向量搜索的各种集合外，还会**将 activity_log 的 BM25 结果通过 RRF（Reciprocal Rank Fusion）整合**。在广泛搜索时，最近的行动日志也会进入候选

### 使用示例
```
search_memory(query="メッセージ 送信", scope="common_knowledge")
search_memory(query="レート制限", scope="all")
```

## 路径4: `read_memory_file` / `write_memory_file`

当 Anima 调用 `read_memory_file(path="common_knowledge/...")` 时，会检测路径前缀并解析到 `~/.animaworks/common_knowledge/`。

- **读取**: 所有 Anima 均可访问
- **写入**: 所有 Anima 均可访问（用于积累共享知识）
- **路径穿越防护**: 通过 `is_relative_to` 检查防止访问 common_knowledge 之外的位置

### 使用示例
```
read_memory_file(path="common_knowledge/00_index.md")
write_memory_file(path="common_knowledge/operations/new-guide.md", content="...")
```

## 路径5: Claude Code 直接文件 I/O（仅 Mode S）

在 Mode S 中，可以通过 Claude Code 的内置工具（Read、Write、Grep、Glob 等）直接访问 `~/.animaworks/common_knowledge/`。

- **权限**: 在 `handler_perms.py` 中作为共享只读目录被允许
- **适用模式**: 仅 Mode S（Agent SDK）

---

## 后台: RAG 索引构建

为了让 common_knowledge 能通过向量搜索（路径2、3）被找到，必须已在 ChromaDB 中建立索引。

### 索引时机

1. **Anima 启动时**: 在 `MemoryManager` 初始化时调用 `_ensure_shared_knowledge_indexed()`，通过 SHA-256 哈希检测变更。如有变更，则重新索引到 `shared_common_knowledge` 集合
2. **每日 04:00**: 通过 `_run_daily_indexing()` 对所有 Anima 的向量 DB 进行增量索引更新。common_knowledge 也会在此时机重新索引

### 分块策略

在 `memory_type="common_knowledge"` 的情况下，会按照与 knowledge 相同的 **Markdown 标题分隔** 进行分块。

### 集合名称

`shared_common_knowledge`（所有 Anima 共享的单一集合）

---

## 与 reference/ 的区别

| 项目 | common_knowledge | reference |
|------|-----------------|-----------|
| RAG 索引 | 对象（`shared_common_knowledge`） | **非对象** |
| `search_memory` | 可通过 `knowledge` / `episodes` / `procedures` / `common_knowledge` / `skills` / `activity_log` / `all` 搜索（`reference/` 不在对象范围内） | 不可搜索 |
| Priming Channel C | 自动显示片段 | 不显示 |
| `read_memory_file` | 可读写 | **只读** |
| 用途 | 日常实用指南、判断标准 | 详细的技术参考 |
| 系统提示词 | 有提示注入 | 有提示注入（另一部分） |
