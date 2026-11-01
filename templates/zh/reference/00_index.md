# 参考 — 技术参考目录

AnimaWorks 的详细技术规格与面向管理员的配置指南列表。
不在 RAG 搜索范围内。需要时通过 `read_memory_file(path="reference/...")` 直接查阅。

## Reference files

### anatomy/ — 构成・架构

| File | Title |
|------|-------|
| `anatomy/anima-anatomy.md` | Anima 配置文件完全指南 |
| `anatomy/environment-layout.md` | 运行时目录结构与权限 |
| `anatomy/memory-system.md` | 记忆系统指南 |
| `anatomy/priming-channels.md` | Priming 频道技术参考 |
| `anatomy/working-memory.md` | 工作记忆（state/）技术参考 |

### communication/ — 消息传递・协作

| File | Title |
|------|-------|
| `communication/instruction-patterns.md` | 指令下达模式集 |
| `communication/messaging-guide.md` | 消息发送完全指南 |
| `communication/reporting-guide.md` | 报告・升级的方法 |
| `communication/slack-bot-token-guide.md` | Slack 机器人令牌配置指南 |

### internals/ — 框架内部规格

| File | Title |
|------|-------|
| `internals/common-knowledge-access-paths.md` | common_knowledge 的引用路径 |

### operations/ — 管理・运维配置

| File | Title |
|------|-------|
| `operations/browser-automation-guide.md` | 浏览器操作指南 |
| `operations/heartbeat-cron-guide.md` | 定期执行的配置与运维 |
| `operations/memory-writing-guide.md` | 记忆的记录位置与定期执行的选择 |
| `operations/mode-s-auth-guide.md` | Mode S（Agent SDK）认证模式配置指南 |
| `operations/model-guide.md` | 模型选择・配置指南 |
| `operations/project-setup.md` | 项目配置方法 |
| `operations/task-management.md` | 任务管理的方法 |
| `operations/tool-usage-overview.md` | 工具使用指南 |
| `operations/voice-chat-guide.md` | 语音聊天（Voice Chat）指南 |

### organization/ — 组织结构

| File | Title |
|------|-------|
| `organization/roles.md` | 角色与责任范围 |
| `organization/structure.md` | 组织结构的机制 |

### troubleshooting/ — 故障排查

| File | Title |
|------|-------|
| `troubleshooting/common-issues.md` | 常见问题与处理方法 |
| `troubleshooting/escalation-flowchart.md` | 遇到困难时的流程图 |
| `troubleshooting/gmail-credential-setup.md` | Gmail Tool 认证配置指南 |

### usecases/ — 用例指南

| File | Title |
|------|-------|
| `usecases/usecase-communication.md` | 用例: 沟通自动化 |
| `usecases/usecase-customer-support.md` | 用例: 客户支持 |
| `usecases/usecase-development.md` | 用例: 软件开发支持 |
| `usecases/usecase-knowledge.md` | 用例: 知识管理与文档整理 |
| `usecases/usecase-monitoring.md` | 用例: 基础设施・服务监控 |
| `usecases/usecase-overview.md` | AnimaWorks 用例指南 |
| `usecases/usecase-research.md` | 用例: 调查・研究・分析 |
| `usecases/usecase-secretary.md` | 用例: 秘书・事务支持 |

## Related

- 日常实用指南 → `common_knowledge/00_index.md`
- 通用技能 → `common_skills/`
