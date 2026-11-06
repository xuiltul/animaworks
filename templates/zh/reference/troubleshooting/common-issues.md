# 常见问题与处理方法

汇总了业务中容易遇到的问题及其处理步骤的参考文档。
每个问题按“症状 → 原因 → 处理步骤”的形式记载。

遇到困难时，首先阅读本文档，并按照对应项目的步骤操作。
如果此处无法解决，请参阅 `troubleshooting/escalation-flowchart.md` 并妥善升级处理。

---

## 消息无法送达

### 症状

- 本应发送的消息没有回复
- 被告知对方没有收到消息
- 执行了 `send_message`，但对方没有反应

### 原因

1. 收件人指定错误（Anima 正式名、用户别名、`slack:` / `chatwork:` 前缀等），或指定不符合解析顺序
2. 服务器已停止
3. 对方的 Anima 已停止、被禁用，或 Inbox 正处于 Provider 错误等待状态
4. 发生了收件人解析、权限、外部频道配送等实际发送错误
5. `intent` 未指定或不正确。DM 仅支持 `report` / `question`。任务委派使用 `delegate_task`（在 `send_message` 上附加 `intent="delegation"` 会返回已弃用消息）
6. 同一 run 中已向同一收件人发送过 DM（仅拒绝向同一收件人发送的第二条。收件人数量没有上限）

### 处理步骤

1. **确认发送目标名称与收件人格式**
   - 确认 `send_message` 的 `to` 参数是否会解析为预期对象
   - 实现（`core/messaging/outbound.py` `resolve_recipient`）的解析顺序大致如下:
     1. 与已知 Anima 名的**完全一致**（区分大小写）→ 内部
     2. `config.json` `external_messaging.user_aliases` 的**别名**（不区分大小写）→ 外部（preferred_channel）
     3. `slack:USERID` / `chatwork:ROOMID` → 外部直接
     4. 裸 Slack 用户 ID（`U` + 8 个以上英数字符）→ Slack 直接
     5. 已知 Anima 名的**不区分大小写一致** → 内部
     6. 上述以外 → 解析失败
   - 要确保送达内部 Anima，请使用 `~/.animaworks/animas/<名前>/` 或 `reference/organization/structure.md` 上的**正式名**
   - 确认方法:
     ```
     search_memory(query="组织", scope="common_knowledge")
     ```
     或通过 `read_memory_file(path="reference/organization/structure.md")` 确认组织中所有 Anima 名
   - **注意**: 在聊天中面向人类时不能使用 `send_message`。直接用文本回复即可送达人类。在聊天以外（heartbeat 等）联系人类时，使用 `call_human`

2. **确认服务器运行状态**
   - 自己正在运行时，服务器应当处于运行中
   - 如果仍不放心，向上司报告“消息无法送达”

3. **等待对方响应**
   - 对方会按心跳间隔（例如每 30 分钟）检查收件箱
   - 即使没有立即回复，也会在下次心跳时处理
   - 紧急情况下向上司报告“想尽快联系”，并请求手动启动

4. **出现发送错误时**
   - 记录错误消息
   - 在 `state/current_state.md` 中记载情况
   - 向上司报告

### 具体示例

```
# 名前を間違えていた場合
send_message(to="Aoi", content="...", intent="report")   # OK
send_message(to="aoi", content="...", intent="report")  # 名前が異なればエラーになる可能性あり

# DM は intent 必須（report / question のみ）。委譲は delegate_task
# 同一 run 内で同一宛先へ送れる DM は1通まで。宛先数の上限はない
send_message(
    to="aoi",
    content="了解しました。作業を開始します。",
    intent="report",           # 必須: report / question
    reply_to="msg-abc123",     # 任意: 元メッセージのID
    thread_id="thread-xyz789"  # 任意: スレッドID
)

# 確認・お礼・称賛だけのメッセージには返信しない。全体共有が必要なら `animaworks-tool board post <channel> <text>` を使用
```

---

## 无法推进任务

不存在“等待条件齐备”“阻塞中”这样的状态（`blocked` 已废止）。可选的只有推进、关闭、商量这三种。

### 症状

- 想推进作业，但必要的信息或权限不足
- 正在等待其他 Anima 完成作业
- 外部服务返回错误

### 原因

1. 依赖任务未完成
2. 权限不足（`permissions.json` 中未允许的操作。即使在仅有 `permissions.md` 的环境中，首次 `load_permissions` 也会生成 JSON，MD 会退避到 `.bak`）
3. 必要信息不足
4. 外部服务故障

### 处理步骤

1. **不要重复相同操作**
   - 具体确定缺少什么
   - 整理“谁的”“什么作业”“到何时”需要

2. **判断能否自行解决**
   - 考虑能否用其他方法绕过
   - 搜索记忆，确认过去是否有类似问题:
     ```
     search_memory(query="错误内容或关键词", scope="episodes")
     search_memory(query="规避", scope="knowledge")
     ```

3. **无法解决时向委托者报告**（参阅 `troubleshooting/escalation-flowchart.md`）。报告中应包含以下内容:
   - 想做什么
   - 缺少什么、正在等待什么
   - 自己尝试过的处理与建议
   ```
   send_message(
       to="上司の名前",
       content="【進捗報告】\nタスク: XXXの実装\n事実: YYYのAPI権限が不足\n試行: permissions.jsonを確認したが該当設定なし\n推奨: API権限の追加をお願いします",
       intent="report"
   )
   ```

4. **决定任务的处理方式**
   - 如果可能继续，则无需做任何事（任务保持 `pending`。即使不声明就结束会话，也会自动回到 `pending`）
   - 如果不再需要，则设为 `update_task(status="cancelled", summary="理由")`

5. **确认等待期间是否有可推进的作业**
   - 持久任务队列: 如果可使用工具，通过 `list_tasks` 或 `Bash: animaworks-tool task list` 确认
   - 通过 `list_tasks(detail=true)` 确认未着手、依赖关系、需处理理由。不要复制既有任务或手动重新投入
   - 着手其他任务

---

## 找不到记忆

### 症状

- 想不起过去做过的事
- 本应有步骤书却找不到
- 搜索也没有返回对应结果

### 原因

1. 搜索关键词不合适
2. 搜索范围（scope）过窄
3. 尚未作为记忆写入（首次作业）
4. 文件路径错误

### 处理步骤

1. **扩大范围重新搜索**
   - 首先用 `all` 范围广泛搜索:
     ```
     search_memory(query="想搜索的关键词", scope="all")
     ```
   - 如果结果过多，则缩小范围:
     ```
     search_memory(query="Slack配置", scope="procedures")    # 限定于步骤书
     search_memory(query="Slack故障", scope="episodes")      # 限定于过去事件
     search_memory(query="Slack", scope="knowledge")         # 限定于学到的知识
     ```

2. **更换关键词重新搜索**
   - 尝试同义词或相关词（例:“发送”“消息”“通知”“联系”）
   - 也尝试英文关键词（例:"slack", "message", "send"）
   - 注意部分匹配（例:“Chatwork”→“chatwork”“聊天工作”）

3. **搜索共享知识**
   - 如果个人记忆中没有，可能存在于共享知识中:
     ```
     search_memory(query="搜索关键词", scope="common_knowledge")
     ```
   - 确认共享知识的目录:
     ```
     read_memory_file(path="common_knowledge/00_index.md")
     ```

4. **直接确认目录**
   - 在 Mode S（Claude Agent SDK）等中，可通过内置 `Glob` 列出 Anima 目录下的内容。在 Mode A 等中，用 `read_memory_file` 打开已知路径，或用 `search_memory` 广泛查找
   - 如果知道文件名，可直接读取:
     ```
     read_memory_file(path="procedures/slack-setup.md")
     read_memory_file(path="knowledge/xxx-findings.md")
     ```

5. **记忆不存在时**
   - 可能是首次作业
   - 确认共享知识（`common_knowledge/`）中是否有相关指南
   - 向上司或同事询问是否有见解
   - 作业完成后 MUST 作为记忆记录（为了下次）
   - 旧记忆、重复记忆可通过 `archive_memory_file(path="...", reason="...")` 退避到 archive/（不是删除而是移动。`reason` 为必需）

### 搜索范围列表

| scope | 搜索对象 | 用途 |
|-------|---------|------|
| `knowledge` | 学到的知识、经验技巧 | 应对方针、技术备忘 |
| `episodes` | 过去的行动日志 | 确认“何时做了什么”的事实 |
| `procedures` | 步骤书 | 确认“怎么做”的步骤 |
| `common_knowledge` | 全 Anima 共享的知识 | 组织规则、系统指南 |
| `skills` | 技能、共通技能（向量搜索） | 技能的发现、搜索 |
| `activity_log` | 最近的行动日志（工具执行结果、消息等） | 确认“刚才读的邮件”“刚才的搜索结果”等近期事实 |
| `all` | 上述全部（向量搜索 + activity_log BM25 通过 RRF 整合） | 关键词存在性确认、广泛搜索 |

---

## 没有权限

### 症状

- 执行工具时返回“没有权限”“Permission denied”等错误
- 想读写文件但无法访问
- 想执行命令但被拒绝

### 原因

1. `permissions.json` 中未允许的操作（在仅有 `permissions.md` 的情况下，`load_permissions` 在读取时会规范化为 JSON 等价形式。无效 JSON 可能会在警告后回退到开放默认值）
2. 外部工具类别未在注册表中启用（`check_permissions` 的 `available_but_not_enabled`）
3. 文件路径超出允许范围（写入保护文件、`file_roots` 外等）。此外，**全局**拒绝有 `permissions.global.json` 和框架侧模式

### 处理步骤

1. **确认自己的权限**
   ```
   check_permissions()
   ```
   - 以 JSON 返回。确认 `internal_tools`、`external_tools.enabled` / `available_but_not_enabled`、`file_access`（read/write）・`restrictions`（命令 deny 等）
   - 原始配置可通过 `read_memory_file(path="permissions.json")`（不存在时使用 `permissions.md`）确认
   - 注入到系统提示词中的权限说明，是运行时从 JSON 格式化后的文本

2. **确认是否为被允许的操作**
   - 自己的 `anima_dir` 内原则上可读写（`IDENTITY.md` / `SOUL.md` 等保护文件除外）。上司、同事的 `activity_log` 及下属的 `state/` 读取会依角色反映到 `check_permissions` 的 `file_access`
   - Shell 命令: `permissions.json` 的 `commands`（遵循 allow/deny）。全局危险模式也会在框架侧被阻止

3. **需要权限时的应对**
   - 重新考虑该操作是否真的必要
   - 考虑能否用其他方法（允许范围内的操作）替代
   - 无法替代时向上司请求追加权限:
   ```
   send_message(
       to="上司の名前",
       content="【権限追加依頼】\n目的: XXXの作業のため\n必要な権限: /path/to/dir の読み取り\n理由: YYYの情報を参照する必要があるため",
       intent="question"
   )
   ```

4. **绝对不能做的事**
   - 试图绕过权限检查
   - 试图用其他方法执行未允许的命令
   - 试图利用其他 Anima 的权限

---

## 工具无法使用

### 症状

- 调用工具时返回“找不到工具”等错误
- 外部工具（Slack、Gmail 等）无法使用

### 原因

1. 该工具未在 `permissions.json`（或读取时规范化后的 MD 来源配置）中被允许，或带门控的动作未被明确允许
2. 找不到技能文件
3. 未配置外部服务的认证信息

### 处理步骤

1. **通过技能确认工具的使用方法**
   - 在 `read_memory_file` 中指定系统提示词技能目录中所示的路径（例: `skills/foo/SKILL.md`, `common_skills/bar/SKILL.md`），获取步骤全文
   - 在 B-mode 中允许外部工具时，可通过 `Bash: animaworks-tool <ツール> <サブコマンド>` 调用

2. **确认权限**
   ```
   check_permissions()
   ```
   - `external_tools.enabled`: 此 Anima 的工具注册表中列出的外部工具类别（实际传递给会话的内容）
   - `external_tools.available_but_not_enabled`: 框架中已实现，但此 Anima 的注册表中未包含的类别。与 `permissions.json` 的许可、门控操作、执行模式一并确认

3. **无法使用时**
   - 在 `permissions.json` 中确认相应工具/操作是否被允许，以及是否有认证信息（`shared/credentials.json` 等）
   - 若仍不可用，则请求上司（明确说明“为什么需要该工具”）

4. **MCP 集成模式（S/C/D/G: Claude Agent SDK / Codex CLI / Cursor Agent / Gemini CLI）的情况**
   - 内置工具无需前缀即可使用（例: `send_message`）。若找不到，则需要重启进程
   - 外部工具通过 `read_memory_file` 读取技能正文并确认 CLI 用法，然后通过 **Bash** 执行 `animaworks-tool <ツール> <サブコマンド>`（使用代理的 Bash 工具）
   - 长时间工具（图像生成、本地 LLM 等）通过 `animaworks-tool submit` 异步执行

5. **D-mode（Cursor Agent）特有的常见问题**
   - **找不到 CLI**: 确认主机上是否安装了 `cursor-agent` CLI
   - **认证错误**: 在终端执行 `agent login` 进行登录
   - **回退**: 若无法解决，将 `execution_mode` 设为 `A`，或将模型切换为经由 LiteLLM（Mode A）运行

6. **G-mode（Gemini CLI）特有的常见问题**
   - **找不到 CLI**: 确认主机上是否安装了 `gemini` CLI
   - **认证错误**: 执行 `gemini auth login`，或设置环境变量 `GEMINI_API_KEY`
   - **回退**: 若无法解决，将 `execution_mode` 设为 `A`，或将模型切换为经由 LiteLLM（Mode A）。`gemini/` 前缀可能会针对 Google 提供商被重映射为 `google/`

7. **A-mode（LiteLLM）的情况**
   - 外部工具通过 `read_memory_file` 读取技能正文并确认用法，然后通过 **Bash** 执行 `animaworks-tool <ツール> <サブコマンド>`

8. **工具返回错误时**
   - 准确记录错误消息
   - 若为认证错误，则向上司报告（认证信息的配置是管理者的职责）
   - 若为临时超时或速率限制，则短暂等待后重试（次数与间隔取决于工具实现和服务器配置）
   - 若未改善，则向请求者报告事实及已尝试的事项

工具体系的整体概况请参照 `operations/tool-usage-overview.md`。

---

## 上下文变得过长

### 症状

- 会话持续了很长时间
- 响应变慢了
- 系统发出“接近上下文上限”的通知

### 原因

- 长时间作业或大量工具调用消耗了上下文窗口
- 读取了大量文件内容

### 处理步骤

1. **将作业状态保存到短期记忆**（MUST）
   - 将当前作业状态写入 `shortterm/`（聊天会话时写入 `shortterm/chat/`）:
   ```
   write_memory_file(
       path="shortterm/chat/session_state.md",
       content="## 作業状態\n\n### 実行中のタスク\n- XXXの実装（50%完了）\n\n### 次のステップ\n1. YYYを完了する\n2. ZZZをテストする\n\n### 重要な中間結果\n- AAAの調査結果: BBB\n- CCCの設定値: DDD",
       mode="overwrite"
   )
   ```
   - 心跳会话时使用 `shortterm/heartbeat/session_state.md`

2. **更新 `state/current_state.md`**（MUST）
   ```
   write_memory_file(
       path="state/current_state.md",
       content="## 現在のタスク\n\nXXXの実装\n\n### 進捗\n- 50%完了\n- 次回はYYYから再開\n\n### メモ\n- 重要な発見事項をここに記載",
       mode="overwrite"
   )
   ```

3. **将重要见解保存到持久记忆**（SHOULD）
   - 将作业中获得的见解保存到 `knowledge/`:
   ```
   write_memory_file(
       path="knowledge/xxx-findings.md",
       content="# XXXに関する知見\n\n## 発見事項\n...",
       mode="overwrite"
   )
   ```

4. **等待会话继续**
   - 系统会自动开始新会话
   - 新会话中会包含 `shortterm/chat/`（或 `shortterm/heartbeat/`）的内容
   - 重新读取 `state/current_state.md` 并恢复作业

### 预防措施

- 大文件不要整体读取，只搜索必要的部分
- 长时间作业要定期更新 `state/current_state.md`
- 将中间结果及时写入记忆

---

## 发送消息时返回错误

### 症状与原因

- `RecipientResolutionError`、外部频道的 `DeliveryFailed`、权限或公司边界错误等，都会返回实际的发送错误
- `send_message` 在同一 run 中尝试向同一收件人发送第二条消息时会被拒绝（防止重复）。收件人数量没有上限
- 不会因每日或每小时发送预算或对话深度而拒绝发送。内部 Anima 之间的深度有时会记录在诊断日志中

### 处理步骤

1. **确认错误内容**: 确认 `to` 的解析目标、intent（`report` / `question`）、频道 ACL、公司边界、外部 API 的响应
2. **确认重复发送**: 同一 run 中已发送过的收件人无法再发送追加 DM。无需因收件人数量而等待
3. **调查配送失败**: 若为 Slack / Chatwork 等外部频道，则确认返回的配送错误和连接配置
4. **Inbox 的 Provider 错误的情况**: 在 `rate_guard` 指定的恢复时间后，未读消息会被重新处理

详情请参照 `communication/sending-limits.md`。

---

## 命令被阻止

### 症状

- 尝试执行命令时返回“PermissionDenied”“Command blocked”等错误
- 只有特定命令无法执行

### 原因

1. 符合框架/`permissions.global.json` 全局拒绝模式的命令（例: `rm -rf /` 等）
2. `permissions.json` 的 `commands.deny` 中列出的命令

### 处理步骤

1. **确认自己的权限**
   ```
   check_permissions()
   ```
   - `restrictions` 中会列出被 deny 的命令。同时通过 `read_memory_file(path="permissions.json")` 直接确认配置（遗留环境中为 `permissions.md`）

2. **考虑替代手段**
   - 思考能否用被允许的工具实现与被阻止命令同等的操作
   - 例: 若 `rm -rf` 被阻止，则删除单个文件可能被允许

3. **需要变更权限时**
   - 请求上司解除阻止
   - 请求时须明确说明“为什么需要该命令”

---

## 提示词被缩短

### 症状

- 系统提示词比平时更薄，Priming（自动想起）接近为空
- 在长对话或大型用户消息之后，响应前提示词似乎被重新构建的行为

### 原因

大致分为 2 层。

**1. Priming（自动想起）的层级** — `core/prompt/builder.py` 的 `resolve_prompt_tier(context_window)` 根据估算的上下文窗口决定层级。窗口的解析顺序为 `core/prompt/context.py` `resolve_context_window`: **`~/.animaworks/models.json`（SSoT）** → 已弃用的 `config.json` `model_context_windows` → `MODEL_CONTEXT_WINDOWS` 等代码内回退 → 默认 128k。

| 层级 | 条件（`context_window`） | Priming 的处理（`core/agent/priming.py`） |
|--------|--------------------------|---------------------------------------------|
| full | **≥ 128_000** | 将 compact 的常规获取在 `priming.max_tokens` 范围内整形后载入 |
| standard | **≥ 32_000 且 < 128_000** | 相同的 compact 路径，但获取预算限制为最多 1000 token |
| light | **≥ 16_000 且 < 32_000** | 获取 compact 的基本上下文，抑制相关知识/情节搜索 |
| minimal | **< 16_000** | 维持 compact 的基本上下文，抑制相关知识/情节搜索 |

心跳/cron 用的查询文本，是从 activity_log 中收集最近 `[REFLECTION]` 的文本（不是长模板全文）。

**2. 系统提示词本体的收缩** — `core/agent/priming.py` `_fit_prompt_to_context_window`: 当系统＋用户的估算 token + 工具 schema overhead 超过 **上下文窗口的约 80%** 时，将 `build_system_prompt` 按 **系统预算 75% → 50% → 25%** 逐步缩小并重新构建。在 **25% 以下的阶段**，会先将 **Priming 块和面向人类的通知块清空**后再应用。若仍无法容纳，则对系统提示词进行**按字节的硬截断**。

### 处理步骤

1. **显式获取不足的上下文**: 通过 `search_memory` / `read_memory_file` 读取组织、步骤、共享知识（特别是在 `minimal` / `light` 中 Priming 较弱）
2. **将作业状态留在磁盘上**: 在 `state/current_state.md` 或 `shortterm/` 中写入摘要，以便会话中断后也能恢复
3. **与上司/管理者商量**: 若实际运行中过于局促，则考虑 `models.json` 的 `context_window` 或更换模型

---

## 其他常见问题

### 找不到文件

- **原因**: 路径指定错误、文件不存在
- **处理**: 在 Mode S 中使用 `Glob`，其他情况使用 `search_memory` 或已知路径的 `read_memory_file` 来查找
- **注意**: `read_memory_file` 除了相对于 Anima 目录（例: `knowledge/xxx.md`）之外，还可以通过 `common_knowledge/`・`reference/`・`common_skills/` 前缀读取共享目录。`Read`（代理内置）的路径由另一套规则决定

### 无法在 Board read 中指定 inbox

- **原因**：`animaworks-tool board read` 用于共享频道。inbox（收件箱）不是频道
- **处理方法**：系统会自动处理 inbox 中的消息。指定 `animaworks-tool board read inbox` 或 `inbox/` 会导致错误

### 命令超时

- **原因**: 处理时间超过了 `timeout`
- **处理**: 增大 Bash 执行时的 `timeout` 参数（默认: 30 秒）
- **注意**: 对长时间执行的命令应设置合适的超时值

### 对方的 Anima 不存在

- **原因**: Anima 名称错误，或该 Anima 尚未创建
- **处理**: 向上司确认。组织结构请参照 `reference/organization/structure.md`

### 重建 venv 后 SEGV 频发

- **原因**: 当 ChromaDB 或 PyTorch 的主版本升级后，会与现有 vectordb（HNSW 段）不兼容，读取时发生 SEGV（Segmentation Fault）。ChromaDB 1.5.x 的 Rust 绑定对损坏的索引存在引发 SEGV 而非 Python 异常的已知问题（chromadb #6852, #6949, #6979）
- **处理**: 重建 venv 后务必也全量重建 vectordb
  1. 停止服务器
  2. 删除 `~/.animaworks/vectordb/` 和各 `~/.animaworks/animas/*/vectordb/`（或备份后删除）
  3. 删除 `~/.animaworks/index_meta.json`
  4. 启动服务器（启动时会自动重新索引）
- **预防**: 更新包时确认 chromadb・torch・sentence-transformers 的版本变更，若有变更则重建 vectordb
