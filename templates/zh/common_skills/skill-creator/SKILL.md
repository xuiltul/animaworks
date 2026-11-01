---
name: skill-creator
description: >-
  创建 Markdown 技能的元技能。处理 SKILL.md 的 frontmatter 和正文、Progressive Disclosure 以及 create_skill 的程序。
  Use when: 需要新增技能、确认面向 read_memory_file 的记述规则、生成带 references 或 templates 的技能时。
---


# skill-creator

## 与实现的对应关系

| 角色 | 模块 |
|------|------------|
| `read_memory_file` 工具（指定技能／程序的相对路径来读取正文） | 经由 `ToolHandler`（记忆树内文件） |
| 系统提示词中的技能目录（带路径列表・预算） | 提示词构建（例: `skills/foo/SKILL.md`, `common_skills/bar/SKILL.md`, `procedures/baz.md`） |
| `create_skill` 工具（生成目录） | `core/tooling/skill_creator.py` |
| 模式（参数定义） | `core/tooling/schemas/skill.py` |
| 前置元数据解析（基于行・不会因正文侧的 `---` 而错误分割） | `core/memory/frontmatter.py` 的 `parse_frontmatter()` |
| 元数据类型・提取（`SkillMeta`、程序路径推定） | `core/schemas.py`、`core/memory/skill_metadata.py` 的 `SkillMetadataService.extract_skill_meta()` |
| 基于描述文本的技能元数据提取（目录・搜索辅助） | `core/memory/skill_metadata.py` 的 `SkillMetadataService` 等 |
| `*-tool` 正文的门控行移除 | `core/tooling/guide.py` 的 `filter_gated_from_guide()` |
| 允许工具集合（permissions） | `core/config/models.load_permissions()` + `core/tooling/permissions.get_permitted_tools()` |

## 技能的类型与路径

技能与程序按**不同布局**管理。

| 类型 | 路径 | 备注 |
|------|------|------|
| 个人技能 | `skills/{name}/SKILL.md` | 目录 + `SKILL.md` |
| 共通技能 | `common_skills/{name}/SKILL.md` | 运行时为 `~/.animaworks/common_skills/` 等 |
| 程序（procedure） | `procedures/{name}.md` | **扁平单文件**。不是目录 |

`create_skill` 生成的仅是上表中的**技能**（个人或共通）。程序通过 `write_memory_file` 等另行创建为 `procedures/*.md`。

### 不放置 symlink（2026-09-04 规则化）

不要在 `common_skills/` 或 `skills/` 之下，用指向外部（如 `~/.claude/skills`）的 **symlink 进行“再掲”**。`read_memory_file` 会在实体所在位置检查边界，因此若 symlink 目标在外部，会因“Path traversal detected”而无法读取。此外，目录会将那个 symlink 作为 native 技能载入，并隐藏同名的 external 候选，于是 Anima 只会看到无法读取的路径。

- 宿主侧技能（`~/.claude/skills`, `~/.codex/skills` 等）会从配置 `skills.external_roots` 自动注入，并可通过 `external/<engine>/<name>/SKILL.md` 读取。无需在共通技能中再掲。
- 若无论如何都想作为共通技能放置，请复制实体文件，并将正本确定在其中一方。

## 使用 read_memory_file 读取技能

技能・程序的正文通过 **`read_memory_file(path="...")`** 指定记忆树相对路径来读取。系统提示词的技能目录中会显示可用路径（例: `skills/foo/SKILL.md`, `common_skills/bar/SKILL.md`, `procedures/baz.md`）。

- **个人技能**: `skills/{name}/SKILL.md`
- **共通技能**: `common_skills/{name}/SKILL.md`
- **程序**: `procedures/{name}.md`

`path` 以 Anima 目录为基准的相对路径传入（共享树使用 `common_skills/` 等前缀）。

### 前置元数据与正文

SKILL.md 开头的 YAML 会由 `core/memory/frontmatter.parse_frontmatter()` 移除后读取。**分隔线 `---` 仅按行单位**识别，因此即使 YAML 值或正文中包含 `---`，也不易发生错误分割。

### 正文中的占位符（`*-tool` 指南）

技能正文中可能使用 `{{now_local}}`, `{{anima_name}}`, `{{anima_dir}}` 等占位符。面向外部工具的技能（名称以 `*-tool` 结尾）会根据允许配置对 `animaworks-tool` 行进行门控处理（`core/tooling/guide.py` 的 `filter_gated_from_guide()` 等）。

### 目录与 description

技能目录的 **Level 1** 中，`name` + `description` 会在预算内载入。技能数量越多越容易被省略，因此**`description` 应保持简短具体**。需要全文时，用 `read_memory_file` 打开目录中的路径。

**旧版兼容**: 当 `description` 为空时，`extract_skill_meta()` 会将正文中 **`## 概要` 章节的第一个非空行**作为回退使用。新技能以 frontmatter 为准。

### 布局上的注意事项

个人 `skills/` 直下的 **`*.md`（扁平单文件）**可能成为目录或元数据提取的对象，但**推荐路径是 `skills/{name}/SKILL.md`**。运维上以 `create_skill` 的目录形式为准。

## 技能文件的结构

SKILL.md 由 YAML 前置元数据和 Markdown 正文构成。
前置元数据中 `name` 和 `description` 为**必填**。

`create_skill` 不仅可写 `allowed_tools`，还可写出信任・出处・分类・路由辅助的元数据。必填为 `name` / `description` / `body`，可选键仅在用途明确时使用。

```yaml
---
name: skill-name
description: >-
  スキルが行うことの簡潔な説明（三人称）。
  Use when: このスキルを使う具体的な場面をカンマ区切りで列挙する。
allowed_tools:
  - read_memory_file
  - web_search
trust_level: trusted
source:
  type: anima
  origin: manual
category: communication
use_when:
  - drafting partner emails
trigger_phrases:
  - draft a partner email
negative_phrases:
  - personal diary
domains:
  - gmail
routing_examples:
  - Prepare a reply draft for the bank thread
---
```

主要可选字段: `allowed_tools`, `trust_level`, `source_type`, `source_origin`, `category`, `promotion_status`, `skill_policy`, `use_when`, `trigger_phrases`, `negative_phrases`, `domains`, `routing_examples`。在 `create_skill` 的参数名中，`source.type` 作为 `source_type`、`source.origin` 作为 `source_origin` 传入。

### `description` 的角色

**新技能的记述**遵循 Agent Skills 标准，用 **`Use when:`** 写使用场景（详情见 `references/description_guide.md`）。创建后可用 **`python scripts/lint_skill.py`** 验证格式。

- **通过 `read_memory_file` 指定路径读取时**: 若文件存在，则**与 description 的匹配无关**即可获得正文（前置元数据处理・`*-tool` 相关的处理取决于读取路径）。
- **系统提示词的技能目录**: 作为 Level 1 载入 `name` + `description`，并带预算。不会载入全文，因此若需要步骤，请用 **`read_memory_file(path="skills/.../SKILL.md")` 等**打开。

**旧版兼容**: 当 `description` 为空时，`extract_skill_meta()` 会将正文开头附近的 **`## 概要` 章节的第一个非空行**作为 description 的回退使用。新技能以 frontmatter 为准，避免依赖 `## 概要`。

**description 的写法**（**Use when:** 模式・lint）请参照 `references/description_guide.md`。

## Progressive Disclosure（渐进式披露）

技能信息大致按以下阶段披露。

| Level | 内容 | 显示时机 |
|-------|------|----------------|
| Level 1 | `name` + `description` | 系统提示词的技能目录（预算内）的材料 |
| Level 2 | body（正文） | 智能体通过 `read_memory_file(path="skills/.../SKILL.md")` 等读取时 |
| Level 3 | 外部资源 | 按正文指令，在需要时通过 `read_memory_file` 等读取 `references/` 或 `templates/` |

Level 1 在目录中容易消耗上下文，因此 **description 应简洁**。Level 2 是步骤的核心。Level 3 分离冗长的参照。

※ Priming 的技能正文注入路径已废止。若需要正文，请用 **`read_memory_file`** 打开技能路径。

## 创建步骤

### Step 1: 需求访谈

理解用户的要求。确认以下内容:

- 想自动化・程序化什么
- 对象是个人技能还是共通技能（若为程序，则另行设计到 `procedures/`）
- 写入 **Use when:** 的使用场景（何时选择此技能）

### Step 2: 设计

决定以下内容:

- **name**: 技能名（kebab-case，例: `my-skill`）。若为外部工具指南，考虑 `*-tool` 规范
- **description**: 第三人称摘要 + **`Use when:`** 行（参照 `references/description_guide.md`）
- **body**: 步骤的构成（分节）。如有需要，使用 `{{now_local}}` 等占位符
- **references** / **templates**: 如有需要，设计外部文件
- **allowed_tools**: 仅在想限定推荐工具时
- **trust/source/category/policy/routing**: 信任级别、出处、分类、prompt policy、`use_when` / `trigger_phrases` / `negative_phrases` / `domains` / `routing_examples` 按需设计

### Step 3: 创建

用 `create_skill` 工具将技能创建为目录结构。

**基本（个人技能）**:

```
create_skill(skill_name="{name}", description="{description}", body="{body}")
```

**共通技能**:

```
create_skill(skill_name="{name}", description="{description}", body="{body}", location="common")
```

**包含 references 与 templates 时**:

```
create_skill(
  skill_name="{name}",
  description="{description}",
  body="{body}",
  location="personal",
  references=[
    {"filename": "description_guide.md", "content": "..."},
  ],
  templates=[
    {"filename": "skill_template.md", "content": "..."},
  ],
  allowed_tools=["read_memory_file", "write_memory_file"]
)
```

| 参数 | 必填 | 说明 |
|-----------|------|------|
| skill_name | ✓ | 技能名（kebab-case）。不可为 `/`, `\`, `..` |
| description | ✓ | frontmatter description（**Use when:** 推荐。`references/description_guide.md`） |
| body | ✓ | SKILL.md正文（Markdown）。内置替换对象 |
| location | | `personal`（默认）或 `common` |
| references | | 放置到 `references/` 的文件群。`[{filename, content}, ...]` |
| templates | | 放置到 `templates/` 的文件群。`[{filename, content}, ...]` |
| allowed_tools | | frontmatter 的 `allowed_tools`（可选） |
| trust_level | | `trusted` / `community` 等信任级别 |
| source_type / source_origin | | 出处（例: `anima`, `manual`, `auto_created`） |
| category | | 分类标签 |
| promotion_status | | `probation` / `trusted` 等晋升状态 |
| skill_policy | | prompt 注入方针（`use_mode`, injection 系配置） |
| use_when / trigger_phrases / negative_phrases / domains / routing_examples | | 帮助技能路由器候选选择的辅助元数据 |

`references` / `templates` 的 `filename` 中不要包含路径成分。用 `_validate_filename()` 确认不会解析到父目录之外，若非法则**静默跳过**（该文件不会被创建）。

※ 新技能务必使用 `create_skill`。用 `write_memory_file` 仅创建 `skills/foo.md` 这类**直下单文件**的方法**不推荐**。该形式在 **`read_memory_file(path="skills/foo/SKILL.md")` 中无法打开**，因此推荐 `skills/foo/SKILL.md` 的目录形式。

※ 程序为 `procedures/{name}.md`（扁平单文件）。前置元数据与技能同样推荐 `name` / `description`（＋可选 `allowed_tools`）。

### Step 4: 确认

- **个人技能**: 用 `read_memory_file(path="skills/{name}/SKILL.md")` 确认内容
- **共通技能**: 用 `read_memory_file(path="common_skills/{name}/SKILL.md")` 等目录中显示的路径确认
- **程序**: 确认可通过 `read_memory_file(path="procedures/{name}.md")` 解析
- **用 `python scripts/lint_skill.py`** 验证 `SKILL.md` 的 frontmatter / description（可选但推荐）

## 检查清单

保存前请确认以下事项：

- [ ] 有以 `---` 开头、以 `---` 结尾的 YAML frontmatter
- [ ] 有 `name` 字段
- [ ] 有 `description` 字段
- [ ] description 中包含 **`Use when:`**，并且包含领域特定的具体词（不要使用旧的 `「」` 枚举）
- [ ] **description 具有领域特定性且具体**（避免“进行管理”“确认”之类的泛化表达，明确写出工具名、操作名、对象）
- [ ] body 中记载了具体步骤
- [ ] 新建时在 frontmatter 中放置 `description`，**不要只依赖 `## 概要` 进行说明**（也避免 `## 発動条件` 等旧模板形式）
- [ ] 可选元数据（`trust_level`, `source`, `category`, `skill_policy`, `use_when`, `trigger_phrases`, `negative_phrases`, `domains`, `routing_examples`）与实际用途一致
- [ ] 技能是使用 `create_skill` 创建 `{name}/SKILL.md` 的（程序是否按 `procedures/*.md` 的意图运行）

## 模板

请参照本技能同捆的 `templates/skill_template.md`。或者复制以下内容使用：

```markdown
---
name: {スキル名}
description: >-
  {具体的な対象}の{具体的な操作}スキル（三人称の短い要約）。
  Use when: {利用シーンをカンマ区切り}
---

# {スキル名}

## 手順

1. ...
2. ...

## 注意事項

- ...
```

## 注意事项

- 技能是 Markdown 程序书，与 Python 代码（工具）不同
- frontmatter 的必填字段是 `name` 和 `description`
- `create_skill` 还可以设置信任、出处、分类、policy、routing 辅助元数据。不要增加不必要的键，如果仅靠说明文就足够，则保持简单
- body 过长会压迫上下文，因此以 150 行以内为目标
- 外部资源参照（Level 3）应利用 `references/` 保持正文简洁
