# 组织结构机制

在 AnimaWorks 中，组织结构以每个 Anima 的 `status.json`（或 `IDENTITY.md`）作为 Single Source of Truth（SSoT）来构建。
`core/org/org_sync.py` 将磁盘上的 **supervisor** 同步到 `config.json`，并在构建提示词时使用。
本文档说明组织结构如何被定义、解释和显示。

## 数据源与优先级

### supervisor（上司）

组织的上下级关系由每个 Anima 的 `supervisor` 定义。读取优先级：

1. **status.json** — `"supervisor"` 键（推荐）
2. **IDENTITY.md** — 表格形式 `| 上司 | name |` 的行（仅日语。`core/config/models.py` 的 `read_anima_supervisor` 进行解析）

当 `supervisor` 未设置、为空、或为“无”“(无)”“（无）”“-”“---”时，将成为顶层（最上位）。
`animaworks config set animas.<name>.supervisor <supervisor>` 会统一更新 root 所有的 status/config。如果直接只修改 `config.json`，会被 org_sync 从磁盘覆盖，因此请使用 CLI/API。

### speciality（专业）

专业领域由 `core/prompt/builder.py` 的 `_scan_all_animas()` 按以下优先级解析：

1. **status.json** — `"speciality"` 键（自由文本）
2. **config.json** — `animas.<name>.speciality`（当 status.json 中没有 `speciality` 键时的回退）
3. **status.json** — `"role"` 键（上述均无法解析时的最终回退。角色名：engineer, researcher, manager, writer, ops, general）

**注意：** org_sync **不同步 speciality**。speciality 在构建提示词时从磁盘和 config 每次解析。
通过 `animaworks anima create --from-md` 创建的 Anima 会在 `status.json` 中写入 `role`，但不会写入 `speciality`。
如需自定义显示（例如“开发负责人”），请使用 `animaworks config set animas.<name>.speciality "開発リード"`。root 所有的 status/config 两者都会被更新。

## 通过 org_sync 进行 config.json 同步

`core/org/org_sync.py` 的 `sync_org_structure()` 执行以下操作：

1. 从各 Anima 目录（仅存在 `IDENTITY.md` 的目录）读取 `status.json` / `IDENTITY.md`，提取 supervisor（`read_anima_supervisor`）
2. 检测循环引用（检测到的 Anima 不纳入同步对象）
3. 将 `config.json` 的 `animas.<name>.supervisor` 按磁盘值更新（**仅 supervisor**）
4. 删除磁盘上不存在的 Anima 的 config 条目（prune）

**同步项目：** 仅 supervisor。speciality 不会由 org_sync 更新。

**执行时机：**

- 服务器启动时（`animaworks start` 的 Anima 进程启动后）
- Anima 通过 reconciliation 被添加时（`on_anima_added` 回调）

## 通过 supervisor 定义层级

- `supervisor: null` 或未设置 → 该 Anima 为顶层（最上位）
- `supervisor: "alice"` → alice 是上司

root 所有的 status 值示例（用于显示。不要直接编辑，请通过 CLI/API 设置）：

```json
{
  "enabled": true,
  "supervisor": null,
  "speciality": "経営戦略・全体統括"
}
```

```json
{
  "enabled": true,
  "supervisor": "alice",
  "speciality": "開発リード"
}
```

通过此设置将构建以下层级：

```
alice（経営戦略・全体統括）
├── bob（開発リード）
│   └── dave（バックエンド開発）
└── carol（デザイン・UX）
```

重要约束：
- supervisor 中指定的名称必须是已知的 Anima 名（英文名）
- 循环引用（alice → bob → alice）会被检测并排除在同步对象之外
- 一个 Anima 只能拥有一个 supervisor

## 组织上下文的构建过程

`core/prompt/builder.py` 的 `_build_org_context()` 从目录扫描与 config.json 的合并结果中计算出以下信息：

1. **上司（supervisor）**：自己的 supervisor 值。未设置则为“你是顶层”
2. **部下（subordinates）**：supervisor 为自己的名字的所有 Anima
3. **同僚（peers）**：与自己拥有相同 supervisor 的 Anima（不含自己）

计算结果会作为“你的组织位置”注入系统提示词：

```
## あなたの組織上の位置

あなたの専門: 開発リード

上司: alice (経営戦略・全体統括)
部下: dave (バックエンド開発)
同僚（同じ上司を持つメンバー）: carol (デザイン・UX)
```

## 如何读取自己的位置

从系统提示词的“你的组织位置”部分，可以确认以下内容：

| 项目 | 含义 | 对行动的影响 |
|------|------|-------------|
| 你的专业 | speciality 的值 | 该领域的提问和判断由自己负责 |
| 上司 | 汇报对象的 Anima | 进度汇报、问题升级的对象 |
| 部下 | 自己下属的 Anima | 任务委派对象、进度确认的对象 |
| 同僚 | 拥有相同上司的伙伴 | 在相关业务中直接协作的对象 |

### 应确认的要点

- 如果上司为“(无 — 你是顶层)”，则你作为组织顶层承担整体责任
- 如果部下为“(无)”，则你作为任务执行者自己动手
- 如果有同僚，可以在相关业务中直接协调

## 组织变更时的行为

组织结构的变更按以下步骤反映：

1. 组织设置通过 root 所有的 CLI/API 修改（例如：`animaworks config set animas.<name>.supervisor <supervisor>` / `animaworks config set animas.<name>.speciality <speciality>`）。
2. CLI/API 会同时更新 `status.json` 和 `config.json`，org_sync 同步层级。下次 reconciliation / prompt 将使用新值。
3. **speciality 的变更：** 由于在构建提示词时读取，无需重启 Anima。会在下次聊天/心跳时反映。

注意事项：
- 不要从 Anima 进程直接编辑 `status.json` / `config.json`。写入者仅为 root 和服务器停止期间的 CLI。
- 组织变更后，应（SHOULD，推荐）通过消息通知受影响的 Anima

## 组织结构模式示例

以下是组织设置的示例。通过 root 所有的 CLI/API 设置，org_sync 同步 `supervisor`。`speciality` 在构建提示词时解析。

### 模式1：扁平组织

所有人均为顶层。无上下级关系。

各 Anima 的 status.json：
```json
{ "supervisor": null, "speciality": "企画" }
{ "supervisor": null, "speciality": "開発" }
{ "supervisor": null, "speciality": "デザイン" }
```

```
alice（企画）
bob（開発）
carol（デザイン）
```

特点：
- 所有人以对等立场直接交流
- 适合小规模团队或各自拥有独立业务的情况
- 所有人的同僚均为“(无)”（因为没有共享相同的 supervisor）

### 模式2：层级型组织

有明确的上下级关系。最常见的模式。

各层级字段通过 root 所有的 CLI/API 设置（例如：`animaworks config set animas.dave.supervisor bob`）：

```
alice（CEO・全体統括）
├── bob（開発部長）
│   ├── dave（バックエンド）
│   └── eve（フロントエンド）
└── carol（営業部長）
    └── frank（顧客対応）
```

特点：
- bob 和 carol 是同僚（相同 supervisor = alice）
- dave 和 eve 是同僚（相同 supervisor = bob）
- 从 dave 到 frank 的联系需经过 bob → alice → carol → frank 的路径（其他部门规则）

### 模式3：专家＋管理者型

少数管理者统管多数专家。

```
manager（プロジェクト管理）
├── dev1（API開発）
├── dev2（DB設計）
├── dev3（インフラ）
└── qa（品質保証）
```

特点：
- 所有成员均为同僚关系。直接协作容易
- manager 负责整体任务分配和进度管理
- 适合初创公司或项目团队

## speciality 的运用

`speciality` 以自由文本保存在 root 所有的 `status.json` 中。通过 `animaworks config set animas.<name>.speciality <value>` 设置。未设置时，`role`（角色名）会作为回退显示。

- 在组织上下文中显示在各 Anima 名字旁边（例如：`bob (開発リード)` 或 `bob (engineer)`）
- 成为其他 Anima 判断任务咨询对象或委派对象的线索
- 未设置时显示为“(未设置)”

**创建 Anima 时的行为（`core/anima/factory.py`）：**
- 通过 `animaworks anima create --from-md PATH [--role ROLE] [--supervisor NAME] [--name NAME]` 创建时，会在 `status.json` 中写入 `supervisor` 和 `role`
- **supervisor**：如果指定了 `--supervisor` 选项则优先使用。未指定时从角色表的基本信息表格（`| 上司 | name |`）解析
- **speciality**：不包含在角色表的基本信息表格中，`_create_status_json` 也不会写入 speciality，因此创建时不会自动设置
- 如需自定义专业显示，请在创建后通过 `animaworks config set animas.<name>.speciality "開発リード"` 设置。不要直接编辑配置文件
- 通过 `create_from_template` / `create_blank` 创建时同样，speciality 不会自动设置到 status.json（如果模板中包含 status.json，则其内容会被复制）

有效的 speciality 写法：
- 具体且简短：`バックエンド開発` `顧客サポート` `データ分析`
- 不过于模糊：`いろいろ` → `企画・調整・進行管理`
- 有多个专业时用中黑点分隔：`UI設計・フロントエンド開発`
