---
name: workspace-manager
description: >-
  配置工作区（工作目录）的注册、列表、删除和分配。
  Use when: 当需要将项目路径与Anima关联、管理别名、切换工作目录时。
tags: [workspace, directory, project, management]
---


# 工作区管理

管理Anima进行工作的项目目录（工作区）的技能。

## 概念

Anima平时待在“自己的家”（~/.animaworks/animas/{name}/））。
进行项目工作时，会前往“工作场所”（工作区）进行作业。

工作区注册在组织共享的注册表（config.json 的 workspaces 部分）中，通过别名#哈希进行引用。

## 别名与哈希

- 别名：由人赋予的简短名称（例：`myproject`）
- 哈希：自动赋予路径的SHA-256前8位（例：`3af4be6e`）
- 完整形式：`myproject#3af4be6e` — 冲突可能性为零
- 工具参数中可使用仅别名、完整形式、仅哈希、绝对路径中的任意一种

## 操作方法

### 注册

收到人类明确指令的顶层Anima使用 `grant_workspace_access`：

```json
{
  "alias": "finance-dashboard",
  "path": "/absolute/path/to/project",
  "make_default": true
}
```

此工具会请求 root 主机，更新组织共享注册表和 root 所有的 `permissions.json` / `status.json`。不要从 Anima 进程直接写入配置文件。

**注意**：目录不存在时会报错。
**注意**：`read_memory_file(path="config.json")` 读取自己Anima目录中的 `config.json`。不用于组织共享注册表的注册。

### 列表

组织共享注册表的列表通过 `core.org.workspace.list_workspaces()` 确认。不要使用 `read_memory_file(path="config.json")`。

### 删除

删除作为管理员操作处理。通常作业中不覆盖已有别名，而是注册新别名。

### 修改自己的默认工作区

顶层Anima在 `grant_workspace_access` 中指定 `make_default: true`。
非顶层Anima无法自行添加权限。请让人类向顶层Anima发出指令。

### 分配给部下（供监督者使用）

收到人类明确指令的顶层Anima可以指定 `target_anima` 向部下或子孙Anima授予权限：

```json
{
  "alias": "finance-dashboard",
  "path": "/absolute/path/to/project",
  "target_anima": "ritsu",
  "make_default": true
}
```

## 在工具中的使用

- **submit_tasks**：在各任务的 `workspace` 字段中指定别名
- **delegate_task**：在 `workspace` 字段中指定别名

## 注意事项

- 目录在注册时和使用时都会进行存在性确认
- 尝试注册不存在的目录会报错
- 覆盖别名时哈希也会改变，因此旧哈希引用会解析失败
- 人类无需记住哈希 — 仅用别名即可
