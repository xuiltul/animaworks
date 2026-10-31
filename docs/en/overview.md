<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/overview.md -->
<!-- i18n: source-sha256=f595d1acb48049d421a5181d012ee14ad061f4d04e7ba9d4f5697e84f26d48ea generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> Confirmed commit: b304b7dc

# Feature Overview

AnimaWorks is a runtime foundation for multiple Digital Anima to work continuously using memory, tools, and organization messages. For implementation details of each feature, refer to the corresponding chapters below and the [overall architecture diagram](architecture/index.md).

## Autonomous Agents

Anima has a definition file and status, and is executed from chat, received messages, heartbeat, cron, and tasks. For process structure, see [process](architecture/process.md); for individual files, see [Anima files](architecture/anima-files.md).

## Memory

Conversation history, current work status, long-term memory, and shared knowledge are handled by purpose. For reflection into prompts and the structure of Anima data, see [Anima files](architecture/anima-files.md) and [prompt construction](architecture/prompt.md).

## Multi-Model Execution

Six execution modes are selected from model names and per-anima configuration, and invoked via SDK, CLI, or API. For an overview of engines and fallback, see [model execution](architecture/execution.md).

## Organization

Anima has a company, role, and supervisor, delegates work within the organization, and uses shared workspaces and channels. For permissions and organization boundaries, see [security](security.md); for the messaging mechanism, see [messaging](architecture/messaging.md).

## Messaging

Handles DMs between Anima, shared Boards, and contact via external services. The Inbox starts on file change notifications, and message aggregation and deduplication are described in [Messaging](architecture/messaging.md).

## Task Management

Tasks are persisted in a shared TaskStore, and delegation, attempts, leases, and the Web Task Board are handled centrally. For the roles of CLI and background tasks, see [task management](architecture/tasks.md).

## Skills and Tools

MCP narrows down available tools based on triggers, and the skill catalog for each Anima is selected according to requests. For prompt and tool guide structure, see [prompt construction](architecture/prompt.md); for arguments of `animaworks-tool`, see [tool CLI reference](reference/tool-cli.md).

## Web UI

The Web UI has pages for Home, Chat, Animas, Activity, Logs, Settings, Board, Task Board, and Users. For the API list, see [API reference](reference/api.md).

## Character Assets

Anima has assets such as character images and expressions, used in the Web UI and conversation expressions. For file layout and Anima definitions, see [Anima files](architecture/anima-files.md).

## Security

Combines individual and overall permission settings, checks during tool execution, and authentication for external integrations. For permission boundaries and configuration methods, see [security](security.md).

## Process Management

The server and supervisor manage Anima roots, task runner startup, communication, and restarts. For details, see [process structure](architecture/process.md).

## CLI

`animaworks` provides management for initialization, configuration, Anima, tasks, memory, and more. For the list of command names and arguments, see [CLI reference](reference/cli.md).

## Configuration Management

Overall settings, model selection, per-anima `status.json`, and permission settings are managed separately. For the list of configuration values and defaults, see [configuration reference](reference/config.md).

## Operations

Provides startup preparation, log checking, data maintenance, and troubleshooting procedures. For operational procedures, see [operations guide](operations/index.md).

## MCP

The MCP allowlist `MCP_TOOL_NAMES` and the trigger- and role-based `resolve_tool_surface` are consolidated in `core/tooling/surface.py`. The list that is actually published varies depending on the trigger and Anima execution conditions.
