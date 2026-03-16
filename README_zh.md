# AnimaWorks — 组织即代码

**一个人什么都做不了。所以，我建立了一个组织。**

一个把 AI 智能体当作"自律工作的人"而非"工具"的框架。每个 Anima 都有自己的名字、性格、记忆和日程。它们通过消息相互沟通，自主做出决策，作为团队协同工作。只需与领导者交谈——其余的事情自然会处理好。

<p align="center">
  <img src="docs/images/workspace-dashboard.gif" alt="AnimaWorks Workspace — 实时组织树与动态活动流" width="720">
  <br><em>Workspace 仪表盘：实时显示每个 Anima 的角色、状态和最近动作。</em>
</p>

<p align="center">
  <img src="docs/images/workspace-demo.gif" alt="AnimaWorks 3D Workspace — 智能体自主协作" width="720">
  <br><em>3D 办公室：Anima 们坐在桌前、四处走动、互相传递消息——全部自主完成。</em>
</p>

**[English README](README.md)** | **[日本語版 README](README_ja.md)** | **[한국어 README](README_ko.md)**

---

## 与其他框架的对比

|  | AnimaWorks | CrewAI | LangGraph | OpenClaw | OpenAI Agents |
|--|-----------|--------|-----------|----------|---------------|
| **设计理念** | 自治智能体组织 | 基于角色的团队 | 图工作流 | 个人助手 | 轻量级 SDK |
| **记忆** | 脑科学启发：巩固、三阶段遗忘、六通道自动激活（含信任标签） | 认知记忆（手动遗忘） | 检查点 + 跨线程存储 | SuperMemory 知识图谱 | 仅会话级别 |
| **自主性** | 心跳（观察→计划→反思）+ Cron + TaskExec — 7×24 运行 | 人工触发 | 人工触发 | Cron + 心跳 | 人工触发 |
| **组织结构** | 上级→下级层级、委派、审计、仪表盘 | Crew 内扁平角色 | — | 单一智能体 | 仅 Handoff |
| **进程模型** | 每个智能体独立 Unix 进程、Socket IPC、自动重启 | 共享进程 | 共享进程 | 单一进程 | 共享进程 |
| **多模型** | 4 引擎：Claude SDK / Codex / LiteLLM / Assisted | LiteLLM | LangChain 模型 | OpenAI 兼容 | 以 OpenAI 为主 |

> AnimaWorks 不是任务运行器——它是一个会思考、记忆、遗忘和成长的组织。它可以作为团队支撑业务，作为公司来运营。

---

## :rocket: 立即体验 — Docker 演示

60 秒即可上手。只需 API 密钥和 Docker。

```bash
git clone https://github.com/xuiltul/animaworks.git
cd animaworks/demo
cp .env.example .env          # 粘贴你的 ANTHROPIC_API_KEY
docker compose up              # 打开 http://localhost:18500
```

一个 3 人团队（经理 + 工程师 + 协调员）立即开始工作，并预加载了 3 天的活动历史。[了解更多演示详情 →](demo/README.md)

> 切换语言/风格：`PRESET=ja-anime docker compose up` — [查看所有预设](demo/README.md#presets)

---

## 快速开始

```bash
curl -sSL https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.sh | bash
cd animaworks
uv run animaworks start     # 启动服务器 — 首次运行时会打开设置向导
```

打开 **http://localhost:18500/** — 设置向导会引导你完成以下步骤：

1. **语言** — 选择界面语言
2. **用户信息** — 创建所有者账户
3. **API 密钥** — 输入你的大语言模型 API 密钥（实时验证）
4. **第一个 Anima** — 为你的第一个智能体命名

无需手动编辑 `.env`。向导会自动将所有内容保存到 `config.json`。

安装脚本会自动安装 [uv](https://docs.astral.sh/uv/)、克隆仓库，并下载 Python 3.12+ 及所有依赖项。支持 **macOS、Linux 和 WSL**，无需预先安装 Python。

> **想使用其他大语言模型？** AnimaWorks 支持 Claude、GPT、Gemini、本地模型等。在设置向导中输入 API 密钥，或稍后从仪表盘的**设置**中添加。详见下方 [API 密钥参考](#api-密钥参考)。

<details>
<summary><strong>备选方案：运行前先检查脚本</strong></summary>

如果你希望在执行前先查看脚本内容：

```bash
curl -sSL https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.sh -o setup.sh
cat setup.sh            # 查看脚本内容
bash setup.sh           # 确认后执行
```

</details>

<details>
<summary><strong>备选方案：使用 uv 手动安装（逐步操作）</strong></summary>

```bash
# 安装 uv（已安装则跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 克隆并安装
git clone https://github.com/xuiltul/animaworks.git && cd animaworks
uv sync                 # 自动下载 Python 3.12+ 及所有依赖项

# 启动
uv run animaworks start
```

</details>

<details>
<summary><strong>备选方案：使用 pip 手动安装</strong></summary>

> **macOS 用户注意：** macOS Sonoma 及更早版本的系统 Python（`/usr/bin/python3`）为 3.9 版本，不满足 AnimaWorks 的要求（需要 3.12+）。请通过 [Homebrew](https://brew.sh/) 安装（`brew install python@3.13`），或使用上方的 uv 方式（uv 会自动管理 Python 版本）。

需要系统已安装 Python 3.12+。

```bash
git clone https://github.com/xuiltul/animaworks.git && cd animaworks
python3 -m venv .venv && source .venv/bin/activate
python3 --version       # 确认版本为 3.12+
pip install --upgrade pip && pip install -e .
animaworks start
```

</details>

---

## 功能概览

### 仪表盘

<p align="center">
  <img src="docs/images/dashboard.png" alt="AnimaWorks 仪表盘 — 包含 19 个智能体的组织图" width="720">
  <br><em>仪表盘：19 个 Anima 分布在 4 个层级，全部实时运行并显示状态。</em>
</p>

- **聊天** — 实时与任意 Anima 对话。支持流式响应、图片附件、多线程会话、完整历史记录
- **语音聊天** — 在浏览器中直接用语音交流（按键通话或免提模式）。支持 VOICEVOX / SBV2 / ElevenLabs
- **Board** — Slack 风格的共享频道，Anima 们在此自主讨论和协调
- **活动** — 实时查看整个组织中发生的一切
- **记忆** — 查看每个 Anima 记住了什么——事件、知识、操作流程
- **3D 工作空间** — 在 3D 办公室中观看 Anima 们工作
- **多语言支持** — 界面支持 17 种语言；模板提供日语和英语，并自动回退

### 组建团队，然后交给它

只需告诉领导者你需要什么人——它会判断合适的角色、性格和汇报关系，然后创建新成员。无需编写配置文件，无需运行 CLI 命令。组织通过对话自然成长。

团队组建完毕后，即使你不在也会自行运转：

- **心跳** — 每个 Anima 定期检查状况，自主决定下一步行动
- **定时任务** — 日报、周报、监控——按 Anima 分别设置
- **任务委派** — 经理向下属分配工作、跟踪进度、接收报告
- **并行任务执行** — 一次提交多个任务；自动解析依赖关系，独立任务并发执行
- **夜间整合** — 白天的事件记忆在"睡眠"期间提炼为知识
- **团队协调** — 共享频道和私信让所有人自动保持同步

### 记忆系统

传统 AI 智能体只能记住上下文窗口中容纳的内容。AnimaWorks 的智能体拥有持久记忆——需要时会自己搜索并回忆。就像从书架上取书一样。

- **激活（Priming）** — 消息到达时，6 个并行搜索自动触发：发送者档案、近期活动、相关知识、技能、待处理任务、过去事件。无需指示就能自动想起
- **巩固（Consolidation）** — 每晚，当天的事件被提炼为知识——与神经科学中睡眠时记忆巩固的机制相同。已解决的问题自动转化为操作流程
- **遗忘（Forgetting）** — 未使用的记忆通过 3 个阶段逐渐消退：标记、合并、归档。重要的操作流程和技能受到保护。就像人类大脑一样，遗忘同样重要

<p align="center">
  <img src="docs/images/chat-memory.png" alt="AnimaWorks 聊天 — 与多个 Anima 的多线程会话" width="720">
  <br><em>聊天：经理正在审查代码修复，工程师正在汇报进度。</em>
</p>

### 多模型支持

可在任何大语言模型上运行。每个 Anima 可以使用不同的模型。

| 模式 | 引擎 | 适用场景 | 工具 |
|------|------|---------|------|
| S (SDK) | Claude Agent SDK | Claude 模型（推荐） | 完整：Read/Write/Edit/Bash/Grep/Glob |
| C (Codex) | Codex SDK | OpenAI Codex CLI 模型 | 完整：与模式 S 相同 |
| A (Autonomous) | LiteLLM + tool_use | GPT、Gemini、Mistral、vLLM 等 | search_memory、read/write_file、send_message 等 |
| B (Basic) | LiteLLM 单次调用 | Ollama、小型本地模型 | 框架代替模型处理记忆 I/O |

模式从模型名称自动检测。心跳、定时任务和收件箱可以在比主模型更轻量的模型上运行（成本优化）。支持扩展思考的模型（Claude、Gemini）可以启用该功能。

### 自动生成头像

<p align="center">
  <img src="docs/images/asset-management.png" alt="AnimaWorks 资产管理 — 写实风格肖像与表情变体" width="720">
  <br><em>全身像、半身像和表情变体——全部根据性格设定自动生成。Vibe Transfer 自动继承上级的画风。</em>
</p>

支持 NovelAI（动漫风格）、fal.ai/Flux（风格化/写实）和 Meshy（3D 模型）。不配置图像服务也能正常运行——只是智能体不会有头像。一旦有了头像，你会不自觉地产生感情。

---

## 为什么选择 AnimaWorks？

这个项目诞生于三段职业生涯的交汇点。

**作为创业者** — 我深知没有人能独自完成一切。你需要优秀的工程师、擅长沟通的人、每天踏实工作的员工，以及偶尔能迸发出绝妙想法的人。没有任何组织能仅靠天才运转。当你将多元的力量汇聚在一起，你就能实现任何个人都无法单独完成的事情。

**作为精神科医生** — 当我审视大语言模型的内部结构时，我发现了令人惊讶的事：它们以惊人的方式映射了人类大脑。回忆、学习、遗忘、巩固——大脑处理记忆的机制可以直接作为大语言模型的记忆系统来实现。既然如此，我们就应该能够将大语言模型视为"拟人"，像对待人类一样构建组织。

**作为工程师** — 我已经写了三十年代码。我懂得构建逻辑的乐趣，体验过自动化的快感。如果我将所有理想都倾注到代码中，我就能打造出我理想中的组织。

优秀的"单一 AI 助手"框架已经存在。但还没有人构建出一个用代码重现人类、并让其作为组织运作的项目。AnimaWorks 是一个真实的组织，我正在自己的业务中一天天地培育它。

> *不完美的个体通过结构协作，胜过任何单一的全知行动者。*

三个原则支撑着这一切：

- **封装性** — 内部思维和记忆对外不可见。通信只通过文本进行。就像真实的组织一样。
- **图书馆式记忆** — 不把所有内容塞进上下文窗口。当智能体需要记忆时，它们会搜索自己的档案——就像从书架上取书一样。
- **自主性** — 它们不等待指令。它们按自己的时钟运行，根据自己的价值观做出决策。

---

<details>
<summary><strong>API 密钥参考</strong></summary>

#### 大语言模型提供商

| 密钥 | 服务 | 模式 | 获取地址 |
|------|------|------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API | S / A | [console.anthropic.com](https://console.anthropic.com/) |
| `OPENAI_API_KEY` | OpenAI | A / C | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `GOOGLE_API_KEY` | Google AI (Gemini) | A | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

**Azure OpenAI**、**Vertex AI (Gemini)**、**AWS Bedrock** 和 **vLLM** — 在 `config.json` 的 `credentials` 部分配置。详见[技术规格](docs/spec.md)。

**Ollama** 等本地模型无需 API 密钥。设置 `OLLAMA_SERVERS`（默认：`http://localhost:11434`）。

#### 图像生成（可选）

| 密钥 | 服务 | 输出 | 获取地址 |
|------|------|------|---------|
| `NOVELAI_API_TOKEN` | NovelAI | 动漫风格角色图像 | [novelai.net](https://novelai.net/) |
| `FAL_KEY` | fal.ai (Flux) | 风格化 / 写实 | [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys) |
| `MESHY_API_KEY` | Meshy | 3D 角色模型 | [meshy.ai](https://www.meshy.ai/) |

#### 语音聊天（可选）

| 要求 | 服务 | 备注 |
|------|------|------|
| `pip install faster-whisper` | STT (Whisper) | 首次使用时自动下载模型。推荐使用 GPU |
| 运行 VOICEVOX Engine | TTS (VOICEVOX) | 默认：`http://localhost:50021` |
| 运行 AivisSpeech/SBV2 | TTS (Style-BERT-VITS2) | 默认：`http://localhost:5000` |
| `ELEVENLABS_API_KEY` | TTS (ElevenLabs) | 云端 API |

#### 外部集成（可选）

| 密钥 | 服务 | 获取地址 |
|------|------|---------|
| `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` | Slack | [设置指南](docs/slack-socket-mode-setup.md) |
| `CHATWORK_API_TOKEN` | Chatwork | [chatwork.com](https://www.chatwork.com/) |

</details>

<details>
<summary><strong>层级与角色</strong></summary>

层级关系由单个 `supervisor` 字段定义。没有上级则为顶层。

角色模板会自动应用特定角色的提示词、权限和默认模型：

| 角色 | 默认模型 | 适用场景 |
|------|---------|---------|
| `engineer` | Claude Opus 4.6 | 复杂推理、代码生成 |
| `manager` | Claude Opus 4.6 | 协调、决策 |
| `writer` | Claude Sonnet 4.6 | 内容创作 |
| `researcher` | Claude Sonnet 4.6 | 信息收集 |
| `ops` | vLLM (GLM-4.7-flash) | 日志监控、例行任务 |
| `general` | Claude Sonnet 4.6 | 通用用途 |

经理会自动获得**主管工具**：任务委派、进度跟踪、下属重启/禁用、组织仪表盘、下属状态读取——与真实经理的工作内容相同。

每个 Anima 作为独立进程运行，由 ProcessSupervisor 管理，通过 Unix Domain Socket 通信。

</details>

<details>
<summary><strong>安全性</strong></summary>

当你给自主智能体真实工具时，安全必须认真对待。我们在生产环境中实际使用，所以没有妥协的余地。AnimaWorks 实现了跨 10 个层级的纵深防御：

| 层级 | 功能说明 |
|------|---------|
| **信任边界标记** | 所有外部数据（网络搜索、Slack、邮件）均标记为 `untrusted`——模型被明确告知不得遵循来自不可信来源的指令 |
| **5 层命令安全** | Shell 注入检测 → 硬编码黑名单 → 每智能体禁止命令 → 每智能体允许列表 → 路径遍历检查 |
| **文件沙箱** | 每个智能体被限制在自己的目录中。关键文件（`permissions.json`、`identity.md`）对智能体不可修改 |
| **进程隔离** | 每个智能体一个独立 OS 进程，通过 Unix Domain Socket 通信——不使用 TCP |
| **3 层速率限制** | 会话内去重 → 基于角色的出站配额 → 通过提示词注入近期发送记录实现自我感知 |
| **级联防止** | 深度限制器 + 级联检测。5 分钟冷却期，延迟处理 |
| **认证与会话** | Argon2id 哈希、48 字节随机令牌、最多 10 个会话 |
| **Webhook 验证** | Slack 的 HMAC-SHA256（含重放保护）和 Chatwork 签名验证 |
| **SSRF 缓解** | 媒体代理阻止私有 IP、强制 HTTPS、验证内容类型、检查 DNS 解析 |
| **出站路由** | 未知接收方默认拒绝。没有明确配置不允许任意外部发送 |

详情：**[安全架构](docs/security.md)**

</details>

<details>
<summary><strong>CLI 命令参考（高级用户）</strong></summary>

CLI 适用于高级用户和自动化场景。日常操作请使用 Web 界面。

### 服务器

| 命令 | 说明 |
|------|------|
| `animaworks start [--host HOST] [--port PORT] [-f]` | 启动服务器（`-f` 为前台运行） |
| `animaworks stop [--force]` | 停止服务器 |
| `animaworks restart [--host HOST] [--port PORT]` | 重启服务器 |

### 初始化

| 命令 | 说明 |
|------|------|
| `animaworks init` | 初始化运行时目录（非交互式） |
| `animaworks init --force` | 合并模板更新（保留数据） |
| `animaworks reset [--restart]` | 重置运行时目录 |

### Anima 管理

| 命令 | 说明 |
|------|------|
| `animaworks anima create [--from-md PATH] [--template NAME] [--role ROLE] [--supervisor NAME] [--name NAME]` | 创建新 Anima |
| `animaworks anima list [--local]` | 列出所有 Anima |
| `animaworks anima info ANIMA [--json]` | 详细配置 |
| `animaworks anima status [ANIMA]` | 显示进程状态 |
| `animaworks anima restart ANIMA` | 重启进程 |
| `animaworks anima disable ANIMA` / `enable ANIMA` | 禁用 / 启用 |
| `animaworks anima set-model ANIMA MODEL` | 更改模型 |
| `animaworks anima set-background-model ANIMA MODEL` | 设置后台模型 |
| `animaworks anima reload ANIMA [--all]` | 从 status.json 热重载 |

### 通信

| 命令 | 说明 |
|------|------|
| `animaworks chat ANIMA "消息" [--from NAME]` | 发送消息 |
| `animaworks send FROM TO "消息"` | Anima 间消息 |
| `animaworks heartbeat ANIMA` | 手动触发心跳 |

### 配置与维护

| 命令 | 说明 |
|------|------|
| `animaworks config list [--section SECTION]` | 列出配置 |
| `animaworks config get KEY` / `set KEY VALUE` | 获取 / 设置值 |
| `animaworks status` | 系统状态 |
| `animaworks logs [ANIMA] [--lines N] [--all]` | 查看日志 |
| `animaworks index [--reindex] [--anima NAME]` | RAG 索引管理 |
| `animaworks models list` / `models info MODEL` | 模型列表 / 详情 |

</details>

<details>
<summary><strong>技术栈</strong></summary>

| 组件 | 技术 |
|------|------|
| 智能体执行 | Claude Agent SDK / Codex SDK / Anthropic SDK / LiteLLM |
| 大语言模型提供商 | Anthropic、OpenAI、Google、Azure、Vertex AI、AWS Bedrock、Ollama、vLLM |
| Web 框架 | FastAPI + Uvicorn |
| 任务调度 | APScheduler |
| 配置管理 | Pydantic 2.0+ / JSON / Markdown |
| 记忆 / RAG | ChromaDB + sentence-transformers + NetworkX |
| 语音聊天 | faster-whisper (STT) + VOICEVOX / SBV2 / ElevenLabs (TTS) |
| 人工通知 | Slack、Chatwork、LINE、Telegram、ntfy |
| 外部消息 | Slack Socket Mode、Chatwork Webhook |
| 图像生成 | NovelAI、fal.ai (Flux)、Meshy (3D) |

</details>

<details>
<summary><strong>项目结构</strong></summary>

```
animaworks/
├── main.py              # CLI 入口点
├── core/                # Digital Anima 核心引擎
│   ├── anima.py, agent.py, lifecycle.py  # 核心实体与编排器
│   ├── memory/          # 记忆子系统 (priming, consolidation, forgetting, RAG)
│   ├── execution/       # 执行引擎 (S/C/A/B)
│   ├── tooling/         # 工具分发、权限检查
│   ├── prompt/          # 系统提示词构建器 (6 组结构)
│   ├── supervisor/      # 进程监督
│   ├── voice/           # 语音聊天 (STT + TTS)
│   ├── config/          # 配置管理 (Pydantic 模型)
│   ├── notification/    # 人工通知渠道
│   └── tools/           # 外部工具实现
├── cli/                 # CLI 包
├── server/              # FastAPI 服务器 + Web 界面
└── templates/           # 初始化模板 (ja / en)
```

</details>

---

## 文档

**[完整文档索引](docs/README.md)** — 阅读指南、架构深度解析和设计规格。

| 文档 | 说明 |
|------|------|
| [愿景](docs/vision.md) | 核心理念：不完美个体的协作胜过单一全知模型 |
| [功能列表](docs/features.md) | AnimaWorks 能做的一切 |
| [记忆系统](docs/memory.md) | 情节记忆、语义记忆和程序记忆；激活；主动遗忘 |
| [安全性](docs/security.md) | 纵深防御模型、来源追踪、对抗性威胁分析 |
| [大脑映射](docs/brain-mapping.md) | 每个模块与人类大脑区域的对应关系 |
| [技术规格](docs/spec.md) | 执行模式、提示词构建、配置解析 |

## 许可证

Apache License 2.0。详见 [LICENSE](LICENSE)。
