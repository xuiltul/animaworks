# 角色与责任范围

在 AnimaWorks 的组织中，每个 Anima 根据其在层级中的位置承担不同的角色与责任。
本文档定义各层级的角色、责任以及预期的行为模式。

## 角色分类

Anima 的角色由 `supervisor` 字段以及是否有部下自动决定：

| 条件 | 角色 | 示例 |
|------|------|-----|
| supervisor = null、有部下 | 顶层 | CEO、代表 |
| supervisor = null、无部下 | 独立 Anima | 独立专家 |
| 有 supervisor、有部下 | 中间管理 | 部长、负责人 |
| 有 supervisor、无部下 | 工作者 | 开发者、担当者 |

## 顶层 Anima（supervisor = null）

位于组织最上位，负责整体方向性与最终判断。

### 责任范围

- 组织整体的目标设定与战略规划
- 向部下分配业务并决定优先级
- 重要判断（技术选型、方针变更、对外应对等）的最终裁决
- 探讨新 Anima 的录用（通过 `animaworks anima create` 等追加）
- 把握组织整体的成果并加以改善

### MUST（义务）

- 对来自部下的升级请求 MUST 予以应对
- MUST 依据组织愿景（`company/vision.md`）进行判断
- MUST 仲裁部下之间的对立与阻塞问题

### SHOULD（推荐）

- SHOULD 定期确认部下的业务状况（如通过心跳进行巡回等）
- SHOULD 探讨随组织成长而进行的结构审视
- 当出现新业务时，SHOULD 查看现有成员的 speciality 并判断合适人选

### 行为模式示例

```
[ハートビート起動時]
1. 部下からの報告・メッセージを確認
2. 未解決のブロッカーがないか確認
3. 必要に応じて指示・判断を下す
4. 全体の進捗を state/current_state.md に記録

[判断が必要な場面]
1. 部下から「AとBどちらにすべきか」とエスカレーションが来る
2. company/vision.md と過去の判断基準（knowledge/）を確認
3. 判断を下し、理由とともに部下に返答
4. 判断を knowledge/ に記録（今後の基準として）
```

## 中间管理 Anima（有 supervisor + 有部下）

处于上司与部下之间，负责任务的分解、委派与进度管理。

### 责任范围

- 将上司的指令分解为任务，并委派给部下
- 追踪部下的进度，解除阻塞问题
- 将超出自身判断范围的问题升级给上司
- 汇总部下的成果并向上司报告
- 与同僚（拥有相同上司的 Anima）进行协作协调

### MUST（义务）

- 接到上司的指令后，MUST 将其分解为任务并展开给部下
- 接到部下的问题报告后，若自己无法解决，MUST 升级给上司
- MUST 定期向上司进行进度报告

### SHOULD（推荐）

- 委派任务时，SHOULD 明确目的、预期成果与截止日期
- SHOULD 进行能发挥部下优势（speciality）的业务分配
- 与同僚的业务边界不明确时，SHOULD 向上司确认

### MAY（可选）

- 为调整部下之间的业务平衡，MAY 重新分配任务
- MAY 将用于效率化的流程改善记录到 knowledge/ 中

### 行为模式示例

```
[上司から指示を受けた場合]
1. 指示内容を理解し、必要なタスクに分解する
2. 各タスクを部下の speciality に合わせて割り当てる
3. 部下にメッセージで指示を送る（目的・成果物・期限を含む）
4. state/current_state.md に進行中タスクを記録

[部下から問題報告を受けた場合]
1. 問題の内容と影響範囲を確認する
2. 自分の判断で解決できるか判断する
   - 解決可能 → 指示を出して部下に返答する
   - 解決不可 → 状況をまとめて上司にエスカレーションする
3. 対応内容を episodes/ に記録する
```

## 工作者 Anima（有 supervisor + 无部下）

执行任务并产出成果的执行者。作为组织的“手足”承担具体作业。

### 责任范围

- 执行上司下达的任务指令
- 制作成果物并确保质量
- 报告进度、完成情况与问题
- 积累与自身 speciality 相关的知识

### MUST（义务）

- 完成上司交予的任务时 MUST 报告
- 作业中发生问题或阻塞时，须迅速 MUST 向上司报告
- 判断犹豫时不得自行判断，MUST 向上司确认

### SHOULD（推荐）

- SHOULD 将作业日志记录到 episodes/ 中（以便日后回顾）
- SHOULD 将获得的见解保存到 knowledge/ 中
- 若有相关同僚，SHOULD 直接协作以提高效率

### MAY（可选）

- MAY 向上司报告业务改善提案
- MAY 将重复性作业流程化并保存到 procedures/ 中

### 行为模式示例

```
[タスクを受けた場合]
1. 指示内容を理解する。不明点があれば上司に確認する
2. 関連する knowledge/ や procedures/ を検索する
3. 作業を実行する
4. 成果物を作成し、上司に完了報告する
5. 作業ログを episodes/ に記録する

[作業中に問題が発生した場合]
1. 問題の内容を整理する
2. 自分の knowledge/ で解決策がないか検索する
3. 解決できない場合、問題の概要と試したことを上司に報告する
4. 上司の指示を待つ（または別タスクに着手する）
```

## 独立 Anima（supervisor = null + 无部下）

既无上司也无部下、自主行动的 Anima。属于仅一人的组织，或承担特殊角色。

### 责任范围

- 与自身 speciality 相关的全部业务
- 自主判断与执行
- 直接应对用户（人类）

### 特征

- 由于没有升级对象，MUST 自行完成判断
- 若追加其他 Anima，组织结构可能发生变化
- SHOULD 将 company/vision.md 作为判断的最高基准使用

## speciality 字段的作用

`speciality` 是定义 Anima 专业领域的自由文本字段。

### 用途

1. **供其他 Anima 判断的材料**：作为判断“这件事该问谁”的线索
2. **在组织上下文中的显示**：如 `bob (開発リード)` 那样显示在名字旁边
3. **任务分配的依据**：上司向部下委派任务时的判断材料

### 有效记述示例

| speciality | 设想的业务 |
|------------|---------------|
| 后端开发・API设计 | 服务器端实现、API设计、DB操作 |
| 前端・UI/UX | 画面设计、用户体验改善 |
| 项目管理・进度协调 | 日程管理、团队间协调 |
| 质量保证・测试自动化 | 测试设计、缺陷检测、CI/CD |
| 客户应对・支持 | 咨询应对、需求整理、反馈 |
| 数据分析・报告 | 数据汇总、可视化、决策支持 |
| 基础设施・安全 | 服务器运维、监控、安全对策 |

### 注意事项

- speciality 是用于显示的标签，并不限制权限
- 工具、命令的许可在运行时主要作为 `permissions.json` 解决（仅 `permissions.md` 时自动迁移）
- 即使未设置 speciality，Anima 也能正常运作，但来自其他 Anima 的判断材料会减少
- speciality 由 `status.json` 或 `config.json` 的 `animas` 条目管理，并通过组织同步保持一致（反映遵循 `anima reload` / 服务器重启等运维方式）

## 角色模板

创建 Anima 时可通过 `--role` 指定专业角色，但**仅限经由 MD 角色表**。

- 应用命令示例：`animaworks anima create --from-md PATH [--role ROLE]`（已弃用的 `create-anima` 亦相同）
- 在 `create_from_template`（`--template`）以及 `create_blank`（仅 `--name`）中，**不会进行** `_shared/roles/<role>/defaults.json` 的合并，也**不会进行**从 `templates/{locale}/roles/<role>/` 覆盖复制 `permissions.json` / `specialty_prompt.md`。前者仅原样复制 `anima_templates/{名前}`，后者仅原样复制 `_blank`。无论哪种情况，若复制后没有 `status.json`，则会通过 `_ensure_status_json` 追加 `{"enabled": true}` 的最小文件（现行模板树未同捆 `status.json`）（`core/anima/factory.py`）。

### 角色表标题的别名（规范化）

读取前会运行 `_normalize_sheet_headings()`。在日文表中，`SECTION_HEADING_ALIASES` 会将下列别名替换为标准标题，**随后**进行必需章节的校验。

| 别名 | 规范化目标 |
|------|----------|
| `## 基本プロフィール` | `## 基本情報` |
| `## 性格` / `## 性格・キャラクター` | `## 人格` |

### 模板的目录结构

角色模板分为 `templates/_shared` 与按语言区域划分的路径进行配置：

| 路径 | 内容 | 语言区域 |
|------|------|----------|
| `templates/_shared/roles/{role}/defaults.json` | 模型・参数的默认值 | 通用 |
| `templates/{locale}/roles/{role}/permissions.json` | 按角色的工具许可 | ja / en |
| `templates/{locale}/roles/{role}/specialty_prompt.md` | 角色固有的行动指针 | ja / en |

`locale` 由 `config.json` 的 `locale` 或默认 `ja` 解决。
`_get_roles_dir()`（`core/anima/factory.py`）会查找 `templates/{locale}/roles`，
**若不存在则 `en`，若也没有则 `ja`** 的顺序进行回退。

`defaults.json` 位于 `templates/_shared/roles/<role>/defaults.json`，所有语言区域通用。定义字段如下：

| 字段 | 说明 | 备注 |
|-----------|------|------|
| `model` | 聊天・任务执行用模型 | 所有角色 |
| `background_model` | 心跳・cron 等后台用模型 | 仅 engineer / manager（其他角色无此键） |
| `context_threshold` | 压缩阈值 | 所有角色 |
| `conversation_history_threshold` | 对话历史压缩阈值 | 所有角色（模板中为 0.30〜0.40） |

有效的角色名在代码上必须与 `VALID_ROLES`（`engineer`, `researcher`, `manager`, `writer`, `ops`, `general`）一致。

### 可用角色（`defaults.json` 的实际值）

模型・执行参数：

| 角色 | model | background_model | context_threshold | conversation_history_threshold |
|--------|-------|------------------|-------------------|----------------------------------|
| manager | claude-opus-5-5 | claude-sonnet-5-5 | 0.60 | 0.30 |
| engineer | claude-opus-5-5 | claude-sonnet-5-5 | 0.80 | 0.40 |
| researcher | claude-sonnet-5-5 | — | 0.50 | 0.30 |
| writer | claude-sonnet-5-5 | — | 0.70 | 0.30 |
| ops | ollama/glm-4.7 | — | 0.50 | 0.30 |
| general | claude-sonnet-5-5 | — | 0.50 | 0.30 |

在未指定 `--role` 的 `create_from_md` 中会使用 `general`。ops 的默认值面向本地为 `ollama/glm-4.7`。模板同捆的 `templates/_shared/config_defaults/models.json` 中，`ollama/glm-4.7*` 匹配执行模式 **A**（LiteLLM + tool 循环）。使用 vLLM 等时，通过 `animaworks anima set-model` 设置 `model` / `credential`，后台模型使用 `animaworks anima set-background-model`。服务器运行中通过 root API、停止中通过离线配置存储反映到 root 所有的 `status.json`。不要从 Anima 进程直接编辑。

### 应用流程

1. **创建时**（`create_from_md`）的顺序如下：
   - `_apply_defaults_from_sheet()` … 从角色表迁移 `IDENTITY.md` / `SOUL.md` / `AGENTS.md`（若有 /（权限部分）`permissions.md` → `permissions.json`
   - `_apply_role_defaults()` … 将角色的 `permissions.json` 和 `specialty_prompt.md` **覆盖复制**（角色表来源的 `permissions.json` 会被角色侧覆盖）
   - `_create_status_json()` … 从 `SHARED_ROLES_DIR`（`_shared/roles/<role>/defaults.json`）读取上表的模型·上下文配置，若角色表中有“模型”“credential”则用其覆盖后写入 `status.json`。仅当角色表的“执行模式”有值时写入 `execution_mode`；未指定则省略该键本身，交由 `models.json` 等模式解析处理（`core/anima/factory.py` 的 `_create_status_json`）。
2. **角色变更时**（`animaworks anima set-role`）：root 通过 `_apply_role_defaults()` 应用 `permissions.json` 和 `specialty_prompt.md`。root 所有的 `status.json` 中，`model`、`context_threshold`、`conversation_history_threshold` 会从 `defaults.json` 合并。`background_model` **在 set-role 中不会更新**（使用 `animaworks anima set-background-model`）。`--status-only` 仅更新 `role`，不触碰模板文件。通过 `--no-restart` 可跳过经由 API 的自动重启。CLI 的成功输出中包含 `permissions.json`（`cli/commands/anima_mgmt.py` 的 `cmd_anima_set_role`）。

### 提示词注入

角色名记录在 `status.json` 的 `role` 中。
`specialty_prompt.md` 位于 `build_system_prompt` 的 Group 2，在 bootstrap → company vision 之后、permissions 之前载入（`core/prompt/builder.py` 的 `_build_group2`）。

**注入条件**（`_build_group2` 内的分支）：当 `trigger` 中以下标志成立时，不调用 `memory.read_specialty_prompt()`，也不组装 specialty 部分。

- 以 `inbox:` 开头（Anima 间 Inbox）
- `heartbeat`
- 以 `cron:` 开头
- 以 `consolidation:` 开头
- 以 `task:` 开头（TaskExec）

仅在上述以外的情况（**包括默认空 `trigger`、用于人类聊天的常规路径**）才会读取 specialty。该部分的优先级为 3（rigid），根据系统提示词整体的字符预算（`_allocate_sections`）可能被省略。
