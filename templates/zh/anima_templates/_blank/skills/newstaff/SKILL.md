---
name: newstaff
description: >-
  在AnimaWorks组织中雇用・创建新的Digital Anima的技能。
  基于访谈创建角色表（Markdown），并通过CLI命令（animaworks anima create）
  IDENTITY/SOUL/AGENTS/permissions等一次性生成。创建后通过bootstrap进行自我整备。
  “创建新员工”“雇人”“新员工”“雇用”“创建Anima”“招聘”“增加团队成员”“增加人员”“创建部下”“添加员工”“hire”“recruit”“team member”
---


# 技能: 雇用新员工

## 前提条件

- 要创建的员工的角色方向已确定（若不明则进行访谈）

## 步骤

### 1. 访谈（最小限度即可）

向委托者访谈以下信息。**仅粗体项目为必填**，其他未指定则自动生成:

**必填:**
- **英文名**（仅半角英数字小写。将成为目录名）
- **角色/专业领域**: 负责什么（例: 研究、开发、沟通、基础设施监控）
- **性格方向**: 不是职能而是温度（例: 元气、辣妹、天然、热血、大小姐、姐姐）。若组织中已有2个酷性格则不要选酷
- **脸型**: 从 `{data_dir}/prompts/face_types.md` 中选择1个。不要与现有成员过度重复

**可选（有指定则反映，无则自动生成）:**
- 日文名
- 年龄
- 其他任何讲究

**技术配置（无指定则使用默认）:**
- 角色: `commander`（可委派给其他员工）或 `worker`（接受委派的一方）
- supervisor: 作为上司的 Anima 的英文名（worker 情况下必填。未指定则为自己）

**头脑（LLM模型）配置:**

不要让用户看模型表来选择（因为会混入不可用的模型）。仅在用户指定模型时，参考下表进行配置。若无指定则省略角色表的模型行，使用默认模型推进:

| 级别 | 执行模式 | 使用模型示例 | 特点 | credential |
|--------|-----------|-------------|------|------------|
| S | autonomous | `claude-opus-5-5`, `claude-sonnet-5-5` | Claude Agent SDK。功能最强 | anthropic |
| A | autonomous | `openai/gpt-4.1`, `google/gemini-2.5-pro`, `vertex_ai/gemini-2.5-flash` | 经由LiteLLM。可使用工具 | openai / google / azure / vertex |
| B | assisted | `ollama/gemma3:27b`, `ollama/qwen2.5-coder:32b` | 无工具。本地执行・低成本 | ollama |

※ 若无指定则省略模型・执行模式・credential 行（使用默认值）。

### 2. 角色设计（自动生成）

根据访谈获得的信息充实角色。**人格在先，职能在后。**

务必按此顺序 Read 以下内容:
1. `{data_dir}/prompts/face_types.md`
2. `{data_dir}/prompts/character_design_guide.md`

不要从角色联想到酷美人。爱好不要作为工作的延伸。确认是否与现有成员的头发颜色・脸型・口吻重复。

### 3. 创建角色表，并通过CLI一次性创建

按照**角色表规范**将访谈和设计的结果写入角色表文件，并通过CLI命令创建:

1. 将角色表写入文件（例: `/tmp/{english_name}.md`）
2. 执行以下命令:

```bash
animaworks anima create --from-md /tmp/{english_name}.md --name {english_name} --supervisor {supervisor_english_name}
```

**supervisor的配置:**
- 通过 `supervisor` 参数显式指定（推荐）
- 若省略: 从角色表的 `| 上司 |` 栏获取
- 若两者都没有: 自己（调用方Anima）成为supervisor

**角色表规范:**

```markdown
# キャラクターシート: {Japanese name}

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | {lowercase alphanumeric} |
| 日本語名 | {Japanese full name} |
| 役職/専門 | {Role description} |
| 上司 | {supervisor English name} |
| 役割 | {commander / worker} |
| モデル | {model name — ユーザーが指定した時だけ書く} |
| credential | {ユーザーがモデルを指定した時だけ書く} |

## 人格 (→ SOUL.md)

{Personality, speaking style, values, backstory, appearance, etc.}

## 役割・行動方針 (→ AGENTS.md)

{Responsible areas, decision criteria, reporting rules, conduct standards, etc.}

## 権限 (→ permissions.json) [省略可]

{If omitted: default template applied}

## 定期業務 (→ HEARTBEAT.md, cron.md) [省略可]

{If omitted: generic template applied. New Anima self-adjusts in bootstrap}

## 初回起動指示 (→ BOOTSTRAP.md 追加指示) [省略可]

{If omitted: standard bootstrap only}
```

**必填部分**: 基本信息、人格、角色・行动方针
**可省略部分**: 权限、定期业务、首次启动指令

由此自动执行以下内容:
- 一次性创建目录结构
- 放置 skeleton 文件
- 放置 BOOTSTRAP.md
- 创建 status.json（含supervisor）
- 注册到 config.json（model, supervisor等）
- 对省略部分应用默认值

### 4. 确认 config.json 的模型配置

`animaworks anima create` 会自动注册到config.json，但需确认・补全以下内容:

- `model`: 访谈中决定的模型名
- `credential`: 使用的credential名
- `execution_mode`: autonomous 或 assisted
- `speciality`: 职位/专业

### 4.5. heartbeat频率的建议・配置

根据新员工工作的驱动类型，向委托者建议定期heartbeat的间隔，并配置达成一致的值:

| 驱动类型 | 推荐间隔 | 示例 |
|--------|---------|-----|
| 以自发的状况判断・调整为主业务 | 30分钟（默认） | 团队协调角色、进度监督 |
| 以消息・事件驱动为主体 | 60〜120分钟 | 评审担当、开发担当 |
| 以cron定期业务为主体 | 120〜240分钟 | 监控、会计、法务 |

配置是在 `{data_dir}/animas/{英名}/status.json` 中追加 `"heartbeat_interval_minutes": <分>`（1〜1440）。
消息触发的heartbeat和cron与此配置无关地运行，因此即使延长间隔响应性也不会下降。若犹豫则选择较长（空跑的heartbeat会浪费token）。

### 5. 反映到服务器

```
Bash: curl -s -X POST "${ANIMAWORKS_SERVER_URL:-http://localhost:18500}"/api/system/reload
```

### 6. 向委托者报告

报告雇用完成:
- 新员工的名字和角色
- 配置的技术栈（模型、执行模式）

- 到任后立即着手第一个小工作，结果由你汇总传达

⚠️ 不报告头像图片的生成（由新Anima自身在bootstrap中生成）

### 之后由新Anima自身自主执行:
- IDENTITY.md / SOUL.md / AGENTS.md 的充实化
- HEARTBEAT.md / cron.md 的自我设计
- 头像图片的生成（附带上司参照）
- 向上司的到任报告
- 着手第一个小工作（对用户有用的东西），并向上司报告结果
