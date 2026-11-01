# 工作区指南

Anima 工作的项目目录（工作区）的概念与用法。

工作过程中的 worktree、构建、临时文件的存放位置请参照 `operations/scratch-and-worktrees.md`。

## 什么是工作区

### 家与工作场所的概念

Anima 平时待在“自己的家”（`~/.animaworks/animas/{name}/`）里。
这里存放着 IDENTITY.md / SOUL.md、记忆、配置等 Anima 固有的数据。

进行项目工作（代码修改、调查、构建等）时，
会前往“工作场所”（工作区）进行作业。
工作区是存放项目源代码和产物的目录。

### 注册表与别名

工作区注册在组织共享的注册表（`config.json` 的 `workspaces` 部分）中。
可以通过人类赋予的短名称（别名）和路径 SHA-256 前 8 位（哈希）唯一引用。

| 形式 | 示例 | 用途 |
|------|-----|------|
| 仅别名 | `myproject` | 常规引用（冲突时建议使用带哈希的形式） |
| 完整形式 | `myproject#3af4be6e` | 零冲突的严格引用 |
| 仅哈希 | `3af4be6e` | 不知道别名时 |
| 绝对路径 | `/home/user/dev/myproject` | 直接指定（未注册到注册表也可） |

## 在工具中的使用

### submit_tasks

在各任务的 `workspace` 字段中指定别名后，
TaskExec 会将那个工作区用作工作目录。

```
submit_tasks(batch_id="build", tasks=[
  {"task_id": "t1", "title": "コンパイル", "description": "...", "workspace": "myproject", "parallel": true}
])
```

### delegate_task

在 `workspace` 字段中指定别名后，
被委派的部下会在那个工作区中作业。

```
delegate_task(name="aoi", instruction="API テストを実施して", workspace="myproject")
```

## 注册与分配

### 注册步骤

详情请参照 `common_skills/workspace-manager` 技能。
要点：

1. 在 `config.json` 的 `workspaces` 部分中添加别名和路径
2. 或者从 Python 调用 `core.org.workspace.register_workspace`
3. 目录会在注册时进行存在性检查（不存在则报错）

### 分配给部下

监督者更新部下的 `status.json` 的 `default_workspace` 字段，
分配主要工作目录。请参照 `workspace-manager` 技能。

## 常见问题

### 目录不存在

- **注册时**：尝试注册不存在的路径会报错
- **使用时**：即使已注册，如果之后目录被删除，解析时会报错
- **处理**：确认路径，并用正确的绝对路径重新注册

### 找不到别名

- **原因**：别名未注册到注册表，或拼写错误
- **处理**：在 `read_memory_file(path="config.json")` 中确认 `workspaces` 部分，并使用正确的别名

### 哈希变了

- **原因**：覆盖别名并更改路径时，哈希也会改变
- **处理**：如果使用的是完整形式（`alias#hash`），请更新为新的哈希。如果只使用别名则不受影响
