# AnimaWorks — Organization-as-Code

### 像发微信/LINE一样和团队聊天，幕后24小时自主交付软件的AI组织。

AnimaWorks 是一个开源框架，它将拥有独立长期记忆、人格、职位与日程表的持久化 AI 智能体（**Digital Anima**）凝聚在一起，作为**“活的组织”**全天候 24 小时自主运转。

人类面对的前台界面，是像 LINE / 微信一样轻快直观的聊天窗口。
而在幕后，运行在独立操作系统进程中的智能体们会自动拆解目标、在并行的 `git worktree` 沙箱中编写代码、互相执行测试与多模型代码审查、创建 Pull Request、自主修复 CI 故障，甚至在部署后持续观测日志——全程无需人类插手。

```text
人类的一句话 / GitHub 事件 / 定时触发（Heartbeat · Cron）
   ↓
主管 Anima 拆解与委派（SQLite TaskStore · 互斥锁键 · 委派契约）
   ↓
工程师 Anima 在隔离 worktree 中实现 → 测试 → 多模型 PR 审查 → CI 自愈 → 部署后观测
   ↓
仅在遇到需要决策的异常或审批门禁时，通过 LINE 风格聊天、语音通话或通知升级给人类
```

AnimaWorks 并没有硬编码死板的构建流水线。框架提供的是**“组织的基础设施”**：基于脑科学的长期记忆、进程级故障隔离、原子任务看板、权限与隐私隔离（**Enclave**）以及多渠道通信网。正因如此，同一个组织既能交付代码，也能做竞品调研、处理客户工单、整理会议纪要、通过 Slack 或电话沟通。它不是工作流引擎，**它本身就是一个组织**。

**[English README](README.md)** | **[日本語 README](README_ja.md)** | **[한국어 README](README_ko.md)**

---

## 生产环境真实运行实绩

这不是概念演示。在过去 6 个月里，由 8 个 Anima 组成的 AnimaWorks 组织一直在真实运营并维护着线上生产级 SaaS 产品的日常研发、运维与支持。

| 指标（2026年3月～8月 · 基于原始执行日志统计） | 真实运行数据 |
|---|---|
| 智能体自主创建的 Pull Request | **302 个**（其中 **267 个**已合并） |
| 智能体运维处理的 Pull Request（审查、CI 修复、冲突解决） | **752 个**（其中 **721 个**已合并） |
| 组织自发立项并执行的任务占比 | **99.7%**（共 31,215 项任务中，人类发起仅 92 项） |
| 由 GitHub 事件自动生成的任务数（仅 8 月单月） | **2,508 项** |

<p align="center">
  <img src="docs/images/chat-memory.png" alt="AnimaWorks LINE风格聊天界面与记忆弹窗" width="760">
  <br><em>全新 LINE 风格聊天界面：左侧 Anima 纵列一览成员状态，右侧绿色气泡支持引用回复，左侧白色气泡内嵌审批通知卡与成果文件卡片，右下角常驻半身像随情绪实时变化。</em>
</p>

<p align="center">
  <img src="docs/images/workspace-dashboard.gif" alt="AnimaWorks Workspace — 实时组织树与活动流" width="760">
  <br><em>3D Workspace：实时可视化组织树、各 Anima 状态以及最近执行的动作。</em>
</p>

<p align="center">
  <img src="docs/images/pixel-workspace.gif" alt="AnimaWorks 像素办公室 — 运行中组织的实时视图" width="760">
  <br><em>像素办公室（Pixel Workspace）并非单纯的装饰动画——每一个对话框都对应着此时此刻正在后台运行的真实任务。</em>
</p>

---

## :rocket: 60 秒快速体验（演示环境）

**如果已安装 Claude Code CLI 或已登录 Codex，甚至无需配置任何 API 密钥。**

一行命令完成克隆与安装，立即启动包含 3 天完整活动历史与 3 人团队（产品经理 `kaito`、主任工程师 `sora`、助理 `hina`）的演示组织。演示环境使用完全隔离的数据目录（`~/.animaworks-demo`），不会污染正式的 `~/.animaworks` 环境。

```bash
curl -sSL https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.sh | bash
cd animaworks
uv run animaworks demo --preset ja-business
```

打开 **http://localhost:18501**，即可在 LINE 风格界面中与已经在运转的团队开始对话（[演示环境详情 →](demo/README.md)）。

> **4 种内置预设**：`en-business`（英文·写实，默认）/ `en-anime`（英文·动漫）/ `ja-business`（日文·写实）/ `ja-anime`（日文·动漫）。切换预设时请附带 `--reset` 重新运行（例如：`uv run animaworks demo --reset --preset ja-anime`）。

---

## 快速上手（组建你自己的组织）

### macOS / Linux / WSL

```bash
curl -sSL https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.sh | bash
cd animaworks
uv run animaworks start
```

### Windows (PowerShell)

```powershell
irm https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.ps1 | iex
cd animaworks
uv run animaworks start
```

> **Windows 说明**：AnimaWorks 依赖 Unix 系统接口（`fcntl`、`pty`、`os.killpg` 等），不支持原生 Windows 直接运行。在 PowerShell 中运行 `setup.ps1` 会自动安装 WSL 并在 WSL Linux 环境中完成部署。

在浏览器中打开 **http://localhost:18500**，按向导完成 3 步设置：
1. **选择界面语言与主题**（`zh` / `ja` / `en` / `ko`）
2. **配置 LLM 连接**（支持 Claude Code 订阅登录、Codex 登录、各类 API 密钥或本地 Ollama）
3. **创建第一位 Anima**（从 `dev-lead`、`full-stack`、`assistant`、`marketer`、`translator` 等模板中选择，或自定义角色）

也可以完全不使用浏览器，直接通过 CLI 初始化并操控一切：

```bash
uv run animaworks init
uv run animaworks start
uv run animaworks chat alice "检查今天的迭代目标，并把任务分配给团队成员"
```

<details>
<summary><strong>Docker Compose / pip 手动安装步骤</strong></summary>

#### Docker（推荐：零宿主机环境污染）

`docker compose up` 会启动一个预装了 Node.js、GitHub CLI (`gh`)、Playwright、Bubblewrap 以及 Claude Code / Codex / Gemini CLI 的隔离 Linux 容器。

```bash
git clone https://github.com/xuiltul/animaworks.git && cd animaworks
# 在 .env 中配置认证信息（ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN / GH_TOKEN 等）
docker compose up -d --build
```

无需浏览器向导的无头初始化示例：

```bash
docker exec -it <container> animaworks init --skip-anima
docker exec -it <container> animaworks anima create --name alice --template dev-lead
docker exec -it <container> animaworks config set setup_complete true
docker exec -it <container> animaworks send <your-name> alice "你好"
```

#### 使用 pip 手动安装（推荐 Python 3.12 / 3.13）

```bash
git clone https://github.com/xuiltul/animaworks.git && cd animaworks
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e ".[all-tools]"
animaworks start
```

</details>

---

## 台前体验：对人类而言是“像聊天软件一样好用的团队”

许多自主智能体框架要求人类盯着枯燥的控制台日志或操作复杂的节点连线图。AnimaWorks 的理念恰恰相反：**日常沟通像 LINE 或微信一样轻快自然，需要深挖时只需轻轻一点即可展开大脑记忆与后台作业**。

### 1. LINE 风格对话界面、左侧 Anima 纵列与右下角常驻半身像
- **直观的气泡对话流**：人类发言显示为右侧绿色气泡，Anima 回复显示为左侧白色气泡，流式输出时平滑展示思考预览与工具调用进度。
- **左侧 Anima 纵列（Anima Rail）**：屏幕最左侧排列着团队成员头像，谁正在工作（蓝圈）、谁在等待你的回复（琥珀色待回复角标）一目了然。可以用 `×` 收起不常联系的成员、用 `+` 重新打开，还能一键切换最多 4 分屏视图或最多 5 名成员参与的**会议模式（带主持人的群组讨论）**。
- **右下角常驻半身像（Presence Panel）**：对话窗口右下角常驻当前 Anima 的半身立绘，根据回复情绪（`smile` / `thinking` / `troubled` / `surprised` 等）实时切换表情，并在后台工作时用小气泡轻声告诉你“正在做什么”。

### 2. 以人类为中心的信息流降噪、“一键审批与引用回复”
- **无噪音的对话时间线**：在 24 小时运转的组织里，后台会产生数百次 Heartbeat 巡检、Anima 之间的私信（DM）与定时 Cron 任务。新聊天界面会自动将这些后台活动**折叠为不打扰人类对话的单行摘要束**（随时可通过顶部 `后台作业` 开关一键展开）。
- **审批通知卡与引用回复**：当 Anima 遇到越权操作需要请示人类（`call_human`）时，聊天流中会直接出现带**批准 / 拒绝 / 评论**按钮的通知卡。你还可以对任意历史消息或通知发起**引用回复（Reply Chip）**，让 Anima 准确识别你的指令针对哪一件事。

### 3. 双向文件附件、内嵌全屏预览器与 Slack 镜像同步
- **拖拽上传任意文件**：不仅支持图片，还可直接将 PDF、Markdown、CSV、代码、Office 文档、音频、视频拖入聊天框发送。
- **Anima 交付物文件卡片 (`attach_file`)**：Anima 会把生成的调研报告、CSV 表格或图片直接作为附件卡片发回聊天窗口。点击卡片即可唤起浏览器内置的**全屏文件查看器**（安全渲染 Markdown、代码高亮、音视频播放）。
- **Slack 对话双向镜像**：外出时在 Slack 上与 Anima 交流的内容，会自动同步并呈现在 Web 聊天时间线中。

### 4. 点击头像展开“记忆弹窗”与全屏语音通话
- **D3 实体关联气泡图**：点击聊天顶部栏的 `记忆` 按钮或头像，即可以气泡弹窗形式打开该 Anima 的大脑。内含可视化人物、服务与话题关联的 **D3 实体关联气泡图（`关联`）**，以及当前状态（`现状`）、积累的知识与流程文档（`记忆文件`）、每日日志（`日记`）、工具活动记录（`活动`）和定时任务（`例行作业`）。
- **全屏语音通话 (`Call Screen`)**：点击电话图标即可进入低延迟前排模型 + VOICEVOX / Style-BERT-VITS2 / ElevenLabs / Irodori TTS + `faster-whisper` 流式语音识别驱动的全屏通话界面，支持口型同步、眨眼动画与实时字幕。遇到重大线上事故时，结合 Twilio（`core/phone/`），**Anima 甚至能直接拨打人类的手机把你叫醒**。

<p align="center">
  <img src="docs/images/memory-popover.png" alt="AnimaWorks 记忆弹窗 — D3实体关联气泡图" width="760">
  <br>
  <em>从聊天顶部“记忆”按钮或头像一键唤起的记忆弹窗：当场查看事实、知识、流程、日记以及人物·服务·话题的关联气泡图。</em>
</p>

<p align="center">
  <img src="docs/images/dashboard.png" alt="AnimaWorks 仪表盘 — 实时组织图与任务状态" width="760">
  <br>
  <em>Dashboard（首页）：集中总览组织架构图、待办事项、模型配额消耗、进程健康度与近期团队动态。</em>
</p>

---

## 幕后引擎：24 小时运转软件与业务的自主组织内核

在亲切的前台界面背后，是一套经过高强度生产验证、可连续数月无故障运行的分布式多进程架构。

```text
FastAPI Server（轻量控制面 / WebSocket & SSE / Enclave 客户端）
  ├── MemoryService Process（独立 ChromaDB + Embedding HTTP 宿主进程、SQLite WAL）
  ├── Anima Root Process: kaito (Manager)
  │     ├── Front Lane（低延迟即时响应 / 语音通话 / 意图路由）
  │     ├── Main Chat / Inbox / Heartbeat / Cron Lanes
  │     └── Task Runner Subprocesses（IPC v2 — 每个重度任务独立子进程隔离执行）
  ├── Anima Root Process: sora (Lead Engineer)
  └── Anima Root Process: hina (Assistant)
```

### 1. 永远“接着昨天的话聊”——多层记忆架构与自主睡眠维护
不同于每次会话都失忆的普通聊天封装，每位 Anima 都拥有文件系统优先、模拟人脑机制的多层记忆：
- **6 通道自动记忆激活 (`core/memory/priming/`)**：每次推理前，在动态 Token 预算内自动从 6 个通道检索上下文：(A) 发送者画像、(B) 近期活动、(C) 相关知识、(E) 待办任务、(F) 情景记忆、(G) 图谱社区与近期原子事实。
- **混合检索流水线 (`core/memory/retrieval/`)**：融合向量检索（ChromaDB）+ BM25 词法检索 + 原子事实（`FactRecord`）+ 实体注册表 + RRF 倒数排名融合 + 置信度门控。支持按全局或单个 Anima 切换 `legacy`（默认）或 **Neo4j 图数据库后端 (`neo4j`)**。
- **睡眠记忆巩固与主动遗忘 (`core/memory/maintenance/`)**：每天夜间自动将零散的日间活动提炼为结构化知识（Consolidation），并按衰减评分清理低价值陈旧记忆（Active Forgetting），同时严格保护核心身份、流程与未完成任务。

### 2. 按自己的时钟行动——Heartbeat、Cron、SQLite TaskStore 与 GitHub 网关
- **Heartbeat 与 Cron**：支持自适应间隔与免打扰时段的定期自主巡检（`HEARTBEAT.md`）以及精确到分钟的定时任务（`cron.md`）。
- **SQLite TaskStore (`shared/taskboard.sqlite3`)**：基于 SQLite WAL 的组织级原子任务看板，支持任务依赖、互斥锁键（`mutex_key`，防止多名 Anima 同时修改同一资源）、委派契约（`delegated_tasks`，下属完成后自动触发上级验收）以及卡死任务的自动回收重试。
- **GitHub 事件网关**：自动接收 CI 失败、PR 审查评论、合并冲突等 Webhook 事件，去重后转化为任务并精准派发给负责该模块的工程师 Anima。

### 3. 永不雪崩的进程隔离——Phase 3 进程树与 IPC v2
- **按 Anima 隔离的根进程 + Task Runner 子进程**：每位 Anima 运行在由 `ProcessSupervisor` 监管的独立操作系统进程中；繁重的后台任务与委派任务会进一步通过 IPC v2（`STDIN/STDOUT` JSON-RPC + 二进制帧）派生独立的 `TaskRunner` 子进程执行。即便某个工具卡死或崩溃，也只会回收该子进程，绝不拖垮主服务或其他成员。
- **独立 MemoryService 进程**：向量库与 Embedding 模型常驻于单一宿主进程（非宿主进程全部通过 `HttpVectorStore` 走本地 HTTP 访问），彻底消除多进程并发读写 ChromaDB 导致的索引损坏与内存翻倍问题。

### 4. 硬件级隐私与凭据隔离——Enclave 与多层信任边界
- **Enclave (`core/enclave/`)**：将原始 API 密钥、Cookie 与客户隐私（PII）隔离在独立的操作系统用户/权限域中，Anima 进程只能通过 Unix 域套接字调用白名单内的受控操作（如 `gmail_list`、`gmail_send`、`git_push`、`sql_query`）。即便智能体执行任意 Bash 命令也无法读取原始密钥；返回结果还会经过 **Egress PII 自动脱敏**（自动掩码邮箱、电话、卡号及自定义正则）。
- **多层权限门禁 (`ToolHandler`)**：结合全局 `permissions.global.json`、单智能体 `permissions.json`、危险命令模式拦截以及防提示词注入的信任边界（`trusted` vs `external`），确保安全可控。

### 5. 多引擎执行与混合模型路由
根据职位自由组合最合适的执行引擎与模型，并支持为 Heartbeat/Cron 配置低成本的 `background_model`：

| 模式 | 执行引擎 | 主要适用模型 | 特性与工具集成 |
|---|---|---|---|
| **S** | Claude Agent SDK | `claude-*`（推荐） | 内置 Claude Code 工具链（Read/Write/Edit/Bash/Grep/Glob）+ **stdio MCP (`mcp__aw__*`)** 直连内部工具。支持 Claude Code 订阅登录 |
| **C** | Codex SDK / CLI | `codex/*`, `openai-codex/*` | Codex 沙箱执行 + AnimaWorks MCP 集成。支持 `codex login` |
| **D** | Cursor Agent CLI | `cursor/*` | Cursor CLI 会话与 MCP 集成智能体循环 |
| **G** | Gemini CLI | `gemini/*` | Google Gemini CLI 流式 JSON 与工具循环集成 |
| **X** | Grok Build CLI | `grok/*` | 基于 ACP stdio 的 Grok Build 智能体循环 |
| **A** | LiteLLM Autonomous | GPT, Gemini, DeepSeek, Bedrock, Vertex, Azure, Ollama, vLLM 等 | 在任意云端 API 或本地大模型上运行 CC 兼容工具、记忆检索、任务管理与技能创建 |

<p align="center">
  <img src="docs/images/asset-management.png" alt="AnimaWorks 资产管理 — 角色立绘与表情差分" width="760">
  <br><em>资产管理：根据性格设定一键生成全身立绘、7 种表情半身像、Q 版角色、像素精灵图与绑定骨骼的 3D 模型，并支持继承上级画风的 Vibe Transfer。</em>
</p>

---

## 与其他框架的对比

| 维度 | AnimaWorks | CrewAI | LangGraph | OpenClaw | OpenAI Agents SDK |
|---|---|---|---|---|---|
| **核心单元与设计理念** | **持久化自主智能体组成的“组织”** | 基于角色的任务小队 | 状态图工作流 | 个人 AI 助手 | 轻量多智能体 SDK |
| **人机交互界面** | **LINE 风格聊天 + 常驻半身像 + D3 记忆图谱 + 3D/像素办公室 + 语音通话** | CLI / 简易日志 | LangSmith Studio（面向开发者） | 聊天 UI | 无（纯代码） |
| **长期记忆系统** | **脑科学架构**：6 通道预激活、向量+BM25+原子事实+Neo4j、夜间巩固与受保护主动遗忘 | Cognitive Memory | Checkpointer + Store | SuperMemory | 仅限单次会话 |
| **自主运行触发器** | **Heartbeat + Cron + TaskStore + GitHub 事件网关（24/7 全天候运行）** | 人工触发为主 | 人工/API 触发 | Cron + Heartbeat | 人工/API 触发 |
| **进程与容错隔离** | **按 Anima 独立根进程 + 隔离 task runner（IPC v2）、自动重启、RAG 索引自愈** | 单一共享进程 | 单一共享进程 | 单一进程 | 单一共享进程 |
| **凭据与隐私隔离** | **Enclave（独立 OS 用户 + Unix 套接字 + Egress PII 脱敏 + 多层信任边界）** | 无 | 无 | 无 | 仅输入输出护栏 |
| **多模型执行** | **7 大引擎（Claude SDK / Codex / Cursor / Gemini CLI / Grok / LiteLLM 等）+ 跨提供商限流保护** | LiteLLM | LangChain 模型 | OpenAI 兼容 | OpenAI 为主 |

---

## 为什么做 AnimaWorks

**一个人什么都做不了。所以我创建了一个组织。**

这个项目诞生于三段职业经历的交汇点。

**作为经营者**——我深知“一个人什么都做不了”。既需要优秀的工程师，也需要擅长沟通的员工。有默默干活的执行者，也有偶尔提出犀利想法的人。仅靠天才，组织是无法运转的。当多元的力量汇聚在一起时，才能完成一个人无法做到的事。

**作为精神科医生**——当我观察 LLM 的内部结构时，我发现它与人脑有着惊人的相似之处。回忆、学习、遗忘、巩固——如果把大脑处理记忆的机制原样实现为 LLM 的记忆系统，也许就能重现人脑。既然如此，只要把 LLM 当作“拟似人类”来对待，就应该也能像人类一样组建组织。

**作为工程师**——我写了 30 年代码。我了解构建逻辑的乐趣，也了解自动化带来的快感。只要把所有理想都写进代码，就能创建出我理想中的组织。

优秀的“单人 AI 秘书”框架已经有很多了。但我一直觉得，用代码构建接近人类的单元，并让这些单元作为组织运作的项目仍然不多。AnimaWorks 是一个 AI 组织，我亲自将它融入业务，并在日常使用中不断完善。

> *不完美个体的协作，能构建出比单一全能者更坚韧的组织。*

三项原则为此提供支撑：

- **封装**——内部的思考和记忆不会对外公开。与他人仅通过文本对话相连。这与现实中的组织相同。
- **RAG记忆（档案库型）**——不会把所有内容都塞进窗口。Priming 会通过 RAG 提取相关片段，智能体会通过 `search_memory` 等内容主动回忆。
- **自主性**——不等待指示。按照自己的时钟行动，并依据自己的价值观做判断。

---

<details>
<summary><strong>API 密钥、认证与外部服务集成参考</strong></summary>

#### LLM 提供商与 CLI 订阅登录

| 认证方式 / 密钥 | 对应服务 | 执行模式 | 说明 |
|---|---|---|---|
| `claude` CLI 登录 | Claude Code 订阅 | S | 无需 API 密钥，在安装向导或设置中启用 |
| `codex login` | OpenAI Codex CLI | C | 无需 API 密钥，支持 ChatGPT Plus/Pro/Team 账号 |
| `ANTHROPIC_API_KEY` | Anthropic API | S / A | [console.anthropic.com](https://console.anthropic.com/) |
| `OPENAI_API_KEY` | OpenAI API | A / C | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `GOOGLE_API_KEY` | Google AI (Gemini) | A / G | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `grok login` | Grok Build CLI | X | 通过 ACP stdio 运行 `grok/*` 模型 |
| `cursor-agent` 登录 | Cursor Agent CLI | D | 运行 `cursor/*` 模型 |
| `OLLAMA_SERVERS` | Ollama / vLLM 等本地大模型 | A | 无需 API 密钥（默认：`http://localhost:11434`） |

- **Azure OpenAI / AWS Bedrock / Google Vertex AI** 同样可通过 `config.json` 的 `credentials` 完整配置。
- 认证凭据按以下优先级解析：`config.json` → 加密保险库（`animaworks vault`） → 共享凭据 → 环境变量。

#### 图像与 3D 生成（可选）

| 密钥 / 配置 | 服务 | 用途 |
|---|---|---|
| `NOVELAI_TOKEN` | NovelAI | 动漫风格头像、表情差分、Vibe Transfer 画风继承 |
| `FAL_KEY` | fal.ai (Flux) | 写实风格 / 风格化头像生成 |
| Codex 图像生成 / 本地 Diffusers | OpenAI / Local | 全身图、半身像与像素精灵图（`pixel_sheet.png`）生成 |
| `MESHY_API_KEY` | Meshy | 绑定骨骼的 3D 角色模型生成（`.glb`） |

#### 语音通话与紧急电话告警（可选）

| 组件 | 支持引擎 / 服务 |
|---|---|
| **STT（语音识别）** | `faster-whisper`（LocalAgreement-2 流式确认，`pip install animaworks[transcribe]`） |
| **TTS（语音合成）** | VOICEVOX (`:50021`) / Style-BERT-VITS2 · AivisSpeech (`:5000`) / Irodori (`:7861`) / ElevenLabs API |
| **紧急电话呼叫** | Twilio Voice（`core/phone/` — 发生严重故障时直接拨打人类手机号并语音播报） |

#### 外部消息平台与 SaaS 集成（可选）

| 服务 | 功能 |
|---|---|
| **GitHub** | Webhook 网关（CI 失败、PR 审查、合并冲突自动转任务）+ 按 Anima 独立身份执行 `gh` 操作 |
| **Slack** | Socket Mode 双向对话、共享 Board 同步、**Slack 对话镜像至 Web 聊天界面**、人类通知 |
| **Discord / Chatwork** | Gateway / Webhook 接收、双向消息、人类通知 |
| **Zoom RTMS** | 实时采集会议音频与字幕，自动生成会议纪要与待办事项 |
| **Google Workspace / Notion** | 读写 Gmail、Google Calendar、Google Sheets、Google Tasks 与 Notion 页面 |
| **通知渠道** | Slack / Discord / Chatwork / LINE / Telegram / ntfy / Twilio 电话呼叫 |

</details>

<details>
<summary><strong>组织层级、角色模板与多公司（Company）管理</strong></summary>

仅凭 `status.json` 中的 `supervisor` 字段即可定义汇报线（上级与下属）。你甚至可以在聊天中直接吩咐主管 Anima“招聘一位具备某项技能的成员”，由主管自行设计角色、性格与权限并创建新的 Anima。

| 角色模板 | 主要职责 | 自动赋予的能力 |
|---|---|---|
| `manager` | 决策制定、任务拆解与委派、进度审计 | 主管工具集（`delegate_task`、`org_dashboard`、查看/重启/招聘下属） |
| `engineer` | 代码实现、测试编写、CI 修复、创建 PR | 隔离 worktree 与临时环境开发规范、Git/GitHub 集成 |
| `researcher` | 技术调研、竞品分析、文档审查 | Web 搜索、知识结构化、报告生成 |
| `writer` | 内容创作、文档与发布说明撰写 | 文档质量管理、成果物文件附件输出 |
| `ops` | 日志监控、例行巡检、基础设施健康检查 | 基于轻量模型（本地大模型等）的低成本全天候监控 |
| `general` | 助理、日程协调、前台窗口联络 | 跨团队沟通枢纽、会议纪要整理 |

此外，借助 `animaworks company` 命令集，还可以在同一运行时内定义**多家公司（Company）或事业部**，支持频道作用域隔离、公司拆分（`split`）与导出（`export`）。

</details>

<details>
<summary><strong>多层安全防御与信任边界</strong></summary>

正因为赋予了自主智能体 Shell 与外部工具权限，AnimaWorks 实现了不依赖“性善论”的纵深防御体系（详见 [安全文档](docs/en/security.md) / [Enclave](docs/en/enclave.md)）：

1. **Enclave（PII 与生产数据隔离运行时）**：将个人隐私与生产 DB/S3 凭据封闭在独立操作系统用户的隔离空间内，仅通过 Unix 域套接字（`SO_PEERCRED`）与 Egress PII 脱敏流水线向主组织返回脱敏后的安全事实。
2. **信任边界标签（Provenance & Trust）**：来自 Web 搜索、Slack、邮件等外部输入均打上不可信（`untrusted`）来源标签，并在会话与 RAG 记忆元数据中全程追踪来源，防止提示词注入导致权限提升。
3. **5 级命令与路径检查**：Shell 注入检测 → 全局黑名单（若缺失 `permissions.global.json` 则服务端拒绝启动） → 单 Anima 禁用命令 → 单 Anima 白名单 → 包含相对路径在内的目录穿越拦截与符号链接越界防护。
4. **文件沙箱与门禁工具（Action Gate）**：每个 Anima 仅能写入允许的 `file_roots`，且禁止篡改自身的 `IDENTITY.md` / `SOUL.md` 与 `permissions.json`；高风险外部发送工具支持事前规则门禁校验。
5. **认证、SSRF 防护与 Webhook 验签**：Argon2id 密码哈希、48 字节会话令牌、带防重放保护的 Slack/Chatwork/Zoom/GitHub HMAC 签名验证，以及拦截内网 IP 与 DNS 重绑定的媒体代理。

</details>

<details>
<summary><strong>CLI 命令参考</strong></summary>

日常对话与监控均可直接在 Web UI 中完成，同时框架也提供了强大的 CLI（`animaworks` 与供智能体内部调用的 `animaworks-tool`）用于自动化与运维管理（完整说明见 [CLI 参考](docs/en/reference/cli.md)）：

| 类别 | 常用命令 | 说明 |
|---|---|---|
| **服务 / 演示** | `animaworks start [-f] [--port 18500]`<br>`animaworks stop` / `restart`<br>`animaworks demo [--preset ja-business] [--reset]` | 启动、停止或重启服务端<br>启动内置 3 天活动历史的 3 人演示团队（默认端口 18501） |
| **Anima 管理** | `animaworks anima create / list / info / status`<br>`animaworks anima set-model / set-background-model / set-role`<br>`animaworks anima restart / reload / enable / disable`<br>`animaworks anima audit / permissions / repair-bootstrap` | 创建 Anima、查看状态、切换模型、热重载提示词、权限与安全审计 |
| **任务管理** | `animaworks task board / list / show / add`<br>`animaworks task claim / release / done / cancel / note`<br>`animaworks task autonomy [--days N]` | 操作 SQLite TaskStore、管理租约、生成**组织自主执行率统计报告** |
| **对话与联络** | `animaworks chat ANIMA "消息"`<br>`animaworks send FROM TO "消息"`<br>`animaworks board read / post / dm-history`<br>`animaworks heartbeat ANIMA` | 从 CLI 发起聊天、发送智能体间 DM、读写共享看板、手动触发 Heartbeat |
| **记忆、RAG 与技能** | `animaworks memory status / migrate / backup / cleanup`<br>`animaworks index [--anima NAME] [--full]`<br>`animaworks repair-rag --anima NAME --full`<br>`animaworks skills list / install / inspect / remove` | 管理记忆后端（`legacy` / `neo4j`）、重建 RAG 索引、管理 Skill Hub |
| **Enclave、Vault 与公司** | `animaworks enclave doctor / status / audit`<br>`animaworks vault init / store / get / list`<br>`animaworks company create / list / assign / split / export`<br>`animaworks cost [--today]` / `models list` | 诊断 Enclave 隔离环境、管理加密凭据、管理多公司架构、统计 LLM 成本 |

</details>

<details>
<summary><strong>技术栈与项目目录结构</strong></summary>

```text
animaworks/
├── main.py                  # CLI 入口
├── core/                    # Digital Anima 核心框架
│   ├── anima/               # DigitalAnima 外观类、生命周期、Heartbeat、Inbox、附件管理
│   ├── agent/               # AgentCore、对话循环、Executor 工厂、前置上下文构建
│   ├── runtime/             # Anima 根进程、task runner 监管器、IPC / IPC v2、流式传输
│   ├── execution/           # 7 种执行引擎（claude, codex, cursor, gemini, grok, litellm）+ watchdog 与会话存储
│   ├── memory/              # 长期记忆（priming, retrieval, rag, facts/connections, consolidation, activity）
│   ├── tasks/               # SQLite TaskStore（board, attempts, leases, wakeups, delegation）与外部任务采集
│   ├── enclave/             # 机密与 PII 隔离运行时（gateway, egress masker, raw_store, SSM tunnel, guards）
│   ├── tooling/             # ToolHandler、权限策略、动作门禁、MCP/外部工具分发
│   ├── integrations/        # 自动发现的外部集成工具（GitHub, Slack, Discord, Notion, Google, 图像生成等）
│   ├── mcp/                 # 面向 Mode S 与外部客户端的 stdio MCP 服务（mcp__aw__*）
│   ├── skills/              # Skill Hub、路由器、策展器、从流程记忆提炼技能
│   ├── prompt/              # 系统提示词、组织上下文与工具指南组装
│   ├── messaging/           # Anima 间 DM、共享频道与外部发送路由
│   ├── notification/        # 人类通知（call_human）、内嵌审批与回复路由
│   ├── voice/               # 语音对话（faster-whisper STT + 多引擎 TTS + 低延迟前排通道）
│   ├── phone/               # Twilio 紧急电话呼叫告警
│   ├── org/                 # 多公司（Company）、组织层级与工作区管理
│   ├── config/              # Pydantic 配置模型、模型解析与全局权限缓存
│   ├── auth/                # Argon2id 认证与会话管理
│   └── i18n/                # 多语言消息目录（t()）
├── server/                  # FastAPI 服务端 + WebSocket/SSE + 网关 + Web UI
│   ├── app.py               # ASGI 应用、生命周期、Readiness 门禁与基础路径控制
│   ├── supervisor/          # ProcessSupervisor、健康检查、指数退避重启与自动更新
│   ├── gateways/            # GitHub Webhook, Slack Socket Mode, Discord Gateway, Zoom RTMS
│   ├── routes/              # REST / WebSocket / SSE 路由（chat, sessions, memory, assets, attachment_preview 等）
│   └── static/              # Web UI（LINE 风格聊天、左侧纵列、常驻立绘、D3 记忆弹窗、通话界面、3D/像素办公室）
├── cli/                     # animaworks / animaworks-tool CLI 实现、演示运行器与 TUI
├── templates/               # 多语言（ja / en / ko）初始提示词、角色规范与 systemd 模板
└── demo/                    # 4 种预设（ja/en × business/anime）与内置 3 天活动历史
```

</details>

---

## 文档导航

可通过 **[文档总索引 (`docs/en/README.md`)](docs/en/README.md)**（[日本語](docs/ja/README.md) / [한국어](docs/ko/README.md)）查阅设计理念、架构详解、运维指南与自动生成的参考手册：

| 文档 | 内容概要 |
|---|---|
| **[设计理念 (Vision)](docs/en/vision.md)** | “不完美个体的协作”核心思想与组织设计原则 |
| **[功能概览 (Overview)](docs/en/overview.md)** | AnimaWorks 完整功能地图与各详细章节入口 |
| **[脑科学映射 (Brain Mapping)](docs/en/brain-mapping.md)** | 海马体、杏仁核、前额叶皮层等大脑机制与各模块的对应关系 |
| **[整体架构 (Architecture)](docs/en/architecture/index.md)** | Phase 3 进程架构、执行模式、生命周期、提示词构建与任务管理 |
| **[记忆系统 (Memory System)](docs/en/memory/index.md)** | 自动记忆预激活（Priming）、混合检索、日/周记忆巩固与主动遗忘 |
| **[安全机制 (Security)](docs/en/security.md)** | 信任边界、命令防护、文件沙箱、认证与审计 |
| **[Enclave（隔离运行时）](docs/en/enclave.md)** | 隔离个人隐私与生产数据的 Enclave 构建、SSM 隧道与 Egress 脱敏运维 |
| **[参考手册 (References)](docs/en/reference/cli.md)** | [CLI](docs/en/reference/cli.md) / [工具 CLI](docs/en/reference/tool-cli.md) / [HTTP API](docs/en/reference/api.md) / [配置项](docs/en/reference/config.md) / [模块列表](docs/en/reference/modules.md) |

---

## 开源许可 (License)

Apache License 2.0 — 详情请参阅 [LICENSE](LICENSE)。

