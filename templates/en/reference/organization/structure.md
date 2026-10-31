# How the Organization Structure Works

In AnimaWorks, the organization structure is built using each Anima's `status.json` (or `IDENTITY.md`) as the Single Source of Truth (SSoT).
`core/org/org_sync.py` synchronizes the **supervisor** on disk to `config.json`, which is then used during prompt construction.
This document explains how the organization structure is defined, interpreted, and displayed.

## Data Sources and Priority Order

### supervisor

The hierarchical relationships within the organization are defined by each Anima's `supervisor`. Read priority:

1. **status.json** — `"supervisor"` key (recommended)
2. **IDENTITY.md** — Row in the table format `| 上司 | name |` (Japanese only. The `read_anima_supervisor` of `core/config/models.py` is parsed)

If `supervisor` is unset, empty, "none", "(none)", "（none）", "-", or "---", the Anima is at the top level (highest rank).
`animaworks config set animas.<name>.supervisor <supervisor>` updates the root-owned status/config collectively. Directly modifying only `config.json` will be overwritten from disk by org_sync, so use CLI/API.

### speciality

The area of expertise is resolved by `_scan_all_animas()` in `core/prompt/builder.py` with the following priority:

1. **status.json** — `"speciality"` key (free text)
2. **config.json** — `animas.<name>.speciality` (fallback when status.json has no `speciality` key)
3. **status.json** — `"role"` key (final fallback when the above are not resolved. Role names: engineer, researcher, manager, writer, ops, general)

**Note:** org_sync does **not synchronize speciality**. Speciality is resolved each time from disk and config during prompt construction.
Anima created with `animaworks anima create --from-md` will have `role` in `status.json` but not `speciality`.
To use a custom display (e.g., "development lead"), use `animaworks config set animas.<name>.speciality "開発リード"`. Both root-owned status/config are updated.

## config.json Synchronization via org_sync

The `sync_org_structure()` of `core/org/org_sync.py` performs the following:

1. Reads `status.json` / `IDENTITY.md` from each Anima directory (only those where `IDENTITY.md` exists) and extracts the supervisor (`read_anima_supervisor`)
2. Detects circular references (Animas with circular references are excluded from synchronization)
3. Updates the `animas.<name>.supervisor` of `config.json` to match the values on disk (**supervisor only**)
4. Removes config entries for Animas that no longer exist on disk (prune)

**Items synchronized:** supervisor only. Speciality is not updated by org_sync.

**Execution timing:**

- At server startup (after the Anima process of `animaworks start` starts)
- When an Anima is added via reconciliation (`on_anima_added` callback)

## Hierarchy definition via supervisor

- `supervisor: null` or unset → that Anima is at the top level (highest rank)
- `supervisor: "alice"` → alice is the supervisor

Example of root-owned status values (for display. Do not edit directly; set via CLI/API):

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

With this configuration, the following hierarchy is built:

```
alice（経営戦略・全体統括）
├── bob（開発リード）
│   └── dave（バックエンド開発）
└── carol（デザイン・UX）
```

Important constraints:
- The name specified in supervisor must be a known Anima name (English name)
- Circular references (alice → bob → alice) are detected and excluded from synchronization
- Each Anima can have only one supervisor

## Organization Context Construction Process

The `_build_org_context()` of `core/prompt/builder.py` calculates the following information from the directory scan and the merge result of config.json:

1. **Supervisor**: The value of your own supervisor. If unset, "You are at the top"
2. **Subordinates**: All Animas whose supervisor is your name
3. **Peers**: Animas that share the same supervisor as you (excluding yourself)

The result is injected into the system prompt as "Your position in the organization":

```
## あなたの組織上の位置

あなたの専門: 開発リード

上司: alice (経営戦略・全体統括)
部下: dave (バックエンド開発)
同僚（同じ上司を持つメンバー）: carol (デザイン・UX)
```

## How to Read Your Position

From the "Your position in the organization" section of the system prompt, you can check the following:

| Item | Meaning | Impact on Behavior |
|------|---------|-------------------|
| Your specialty | The value of speciality | You are responsible for questions and decisions in this area |
| Supervisor | The Anima you report to | The destination for progress reports and issue escalation |
| Subordinates | Animas under your command | Delegation targets for tasks and progress confirmation |
| Peers | Colleagues sharing the same supervisor | People you can directly coordinate with on related work |

### Points to Check

- If your supervisor is "(なし — あなたがトップです)", you are at the top of the organization and bear overall responsibility
- If your subordinates are "(なし)", you are a task executor and should do the work yourself
- If you have peers, you can directly coordinate with them on related work

## Behavior during organizational changes

Changes to the organizational structure are reflected in the following steps:

1. Organizational settings are changed in the root-owned CLI/API (e.g., `animaworks config set animas.<name>.supervisor <supervisor>` / `animaworks config set animas.<name>.speciality <speciality>`).
2. CLI/API updates both `status.json` and `config.json`, and org_sync synchronizes the hierarchy. The new values are used at the next reconciliation / prompt.
3. **Speciality changes:** Since it is read during prompt construction, no Anima restart is needed. It takes effect at the next chat/heartbeat.

Notes:
- Do not edit `status.json` / `config.json` directly from the Anima process. Only root and the CLI while the server is stopped can write.
- After organizational changes, it is SHOULD (recommended) to notify affected Anima via message.

## Example organizational structure patterns

Below are examples of organizational settings. Configure in the root-owned CLI/API, and org_sync synchronizes `supervisor`. `speciality` is resolved during prompt construction.

### Pattern 1: Flat Organization

Everyone is at the top level. No reporting hierarchy.

Root-owned status values (for illustration):
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

Characteristics:
- Everyone can interact directly on equal footing
- Suitable for small teams or when each person has independent work
- Everyone's peers are "(なし)" (because they do not share the same supervisor)

### Pattern 2: Hierarchical organization

There is a clear superior-subordinate relationship. This is the most common pattern.

Each hierarchy field is set in the root-owned CLI/API (e.g., `animaworks config set animas.dave.supervisor bob`):

```
alice（CEO・全体統括）
├── bob（開発部長）
│   ├── dave（バックエンド）
│   └── eve（フロントエンド）
└── carol（営業部長）
    └── frank（顧客対応）
```

Characteristics:
- bob and carol are colleagues (same supervisor = alice)
- dave and eve are colleagues (same supervisor = bob)
- Contact from dave to frank follows the path bob → alice → carol → frank (cross-department rule)

### Pattern 3: Specialist + Manager Type

A small number of managers oversee a larger number of specialists.

```
manager（プロジェクト管理）
├── dev1（API開発）
├── dev2（DB設計）
├── dev3（インフラ）
└── qa（品質保証）
```

Characteristics:
- All members are peers. Direct coordination is easy
- The manager handles overall task allocation and progress management
- Suitable for startups and project teams

## Using speciality

`speciality` is stored as free text in the root-owned `status.json`. Set it via `animaworks config set animas.<name>.speciality <value>`. When unset, `role` (role name) is displayed as a fallback.

- Displayed next to each Anima's name in the organizational context (e.g., `bob (開発リード)` or `bob (engineer)`)
- Serves as a clue for other Anima when deciding who to consult or delegate tasks to
- If unset, "(unset)" is displayed

**Behavior when creating an Anima (`core/anima/factory.py`):**
- When created with `animaworks anima create --from-md PATH [--role ROLE] [--supervisor NAME] [--name NAME]`, `supervisor` and `role` are written to `status.json`
- **supervisor**: If the `--supervisor` option is specified, it takes priority. If not specified, it is parsed from the basic information table of the character sheet (`| 上司 | name |`)
- **speciality**: Not included in the basic information table of the character sheet, and `_create_status_json` also does not write speciality, so it is not automatically set at creation
- If a custom specialty display is needed, set it via `animaworks config set animas.<name>.speciality "開発リード"` after creation. Do not edit the configuration file directly
- When created with `create_from_template` / `create_blank`, speciality is likewise not automatically set in status.json (if the template contains status.json, its content is copied)

Tips for writing effective speciality:
- Be specific and short: `バックエンド開発` `顧客サポート` `データ分析`
- Avoid being too vague: `いろいろ` → `企画・調整・進行管理`
- If there are multiple areas of expertise, separate them with a middle dot: `UI設計・フロントエンド開発`
