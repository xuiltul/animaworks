# Roles and Responsibilities

In the AnimaWorks organization, each Anima has different roles and responsibilities depending on its position in the hierarchy.
This documentation defines the roles, responsibilities, and expected behavior patterns for each level of the hierarchy.

## Role Classification

An Anima's role is automatically determined by the `supervisor` field and whether it has subordinates:

| Condition | Role | Example |
|------|------|-----|
| supervisor = null, has subordinates | Top-level | CEO, representative |
| supervisor = null, no subordinates | Independent Anima | Solo specialist |
| supervisor present, has subordinates | Middle management | Department head, leader |
| supervisor present, no subordinates | Worker | Developer, staff member |

## Top-level Anima (supervisor = null)

Positioned at the top of the organization, responsible for overall direction and final decision-making.

### Scope of Responsibility

- Setting organization-wide goals and strategic planning
- Allocating work to subordinates and determining priorities
- Final approval of important decisions (technology selection, policy changes, external communications, etc.)
- Considering the addition of new Anima (via `animaworks anima create`, etc.)
- Monitoring and improving organization-wide results

### MUST (Obligations)

- MUST respond to escalations from subordinates
- MUST make decisions aligned with the organization's vision (`company/vision.md`)
- MUST mediate conflicts and resolve blockers among subordinates

### SHOULD (Recommended)

- SHOULD regularly check subordinates' work status (e.g., through heartbeat rounds)
- SHOULD consider restructuring as the organization grows
- When new work arises, SHOULD assess existing members' specialities to determine the right person

### Example Behavior Patterns

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

## Middle Management Anima (supervisor present + has subordinates)

Stands between supervisors and subordinates, responsible for task decomposition, delegation, and progress management.

### Scope of Responsibility

- Breaking down supervisor instructions into tasks and delegating them to subordinates
- Tracking subordinate progress and resolving blockers
- Escalating problems beyond their decision-making scope to supervisors
- Consolidating subordinate results and reporting to supervisors
- Coordinating with peers (Anima sharing the same supervisor)

### MUST (Obligations)

- Upon receiving instructions from a supervisor, MUST break them into tasks and deploy them to subordinates
- Upon receiving problem reports from subordinates, MUST escalate to the supervisor if unable to resolve independently
- MUST regularly report progress to the supervisor

### SHOULD (Recommended)

- When delegating tasks, SHOULD clearly state the purpose, expected outcomes, and deadlines
- SHOULD assign work that leverages subordinates' strengths (specialities)
- If boundaries with peers are unclear, SHOULD confirm with the supervisor

### MAY (Optional)

- MAY redistribute tasks to balance workloads among subordinates
- MAY record process improvements for efficiency in knowledge/

### Example Behavior Patterns

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

## Worker Anima (supervisor present + no subordinates)

Executes tasks and delivers results. Acts as the organization's "hands and feet" for concrete work.

### Scope of Responsibility

- Executing task instructions from supervisors
- Creating deliverables and ensuring quality
- Reporting progress, completion, and problems
- Accumulating knowledge related to their speciality

### MUST (Obligations)

- MUST report upon completion of tasks received from supervisors
- MUST promptly report problems or blockers encountered during work to supervisors
- When uncertain about a decision, MUST confirm with the supervisor rather than deciding independently

### SHOULD (Recommended)

- SHOULD record work logs in episodes/ (for later review)
- SHOULD save insights gained in knowledge/
- If relevant peers exist, SHOULD coordinate directly with them for efficiency

### MAY (Optional)

- MAY report improvement suggestions to supervisors
- MAY standardize repetitive work and save procedures in procedures/

### Example Behavior Patterns

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

## Independent Anima (supervisor = null + no subordinates)

An Anima with neither supervisor nor subordinates, operating autonomously. Represents a single-person organization or a special role.

### Scope of Responsibility

- All work related to their speciality
- Autonomous decision-making and execution
- Direct interaction with users (humans)

### Characteristics

- With no escalation path, MUST complete decisions independently
- If other Anima are added, the organizational structure may change
- SHOULD use company/vision.md as the highest-priority basis for decisions

## Role of the speciality Field

`speciality` is a free-text field defining an Anima's area of expertise.

### Uses

1. **Reference for other Anima**: A clue for determining "who should be asked about this matter"
2. **Display in organizational context**: Shown next to the name as in `bob (開発リード)`
3. **Basis for task assignment**: Reference material when supervisors delegate tasks to subordinates

### Effective Description Examples

| speciality | Expected Work |
|------------|---------------|
| Backend development and API design | Server-side implementation, API design, database operations |
| Frontend and UI/UX | Screen design, user experience improvement |
| Project management and coordination | Schedule management, cross-team coordination |
| Quality assurance and test automation | Test design, bug detection, CI/CD |
| Customer support | Handling inquiries, organizing requests, feedback |
| Data analysis and reporting | Data aggregation, visualization, decision support |
| Infrastructure and security | Server operations, monitoring, security measures |

### Notes

- speciality is a display label and does not restrict permissions
- Tool and command permissions are primarily resolved at runtime as `permissions.json` (automatic migration if only `permissions.md` is present)
- An Anima works normally even without a speciality, but other Anima have less basis for decision-making
- speciality is managed via `status.json` or `config.json` `animas` entries and is reconciled through organization synchronization (reflected according to operational practices such as `anima reload` / server restart)

## Role Templates

When creating an Anima, specifying a specialized role via `--role` is only possible **through MD character sheets**.

- Example command: `animaworks anima create --from-md PATH [--role ROLE]` (also works with the deprecated `create-anima`)
- With `create_from_template` (`--template`) and `create_blank` (`--name` only), neither the merge of `_shared/roles/<role>/defaults.json` nor the overwrite copy of `permissions.json` / `specialty_prompt.md` from `templates/{locale}/roles/<role>/` is **performed**. The former simply copies `anima_templates/{名前}`, and the latter copies `_blank` as-is. In both cases, if `status.json` is absent after copying, the minimal file for `{"enabled": true}` is added via `_ensure_status_json` (the current template tree does not include `status.json`) (`core/anima/factory.py`).

### Character Sheet Heading Aliases (Normalization)

Before loading, `_normalize_sheet_headings()` runs. In Japanese sheets, the following aliases are replaced with standard headings via `SECTION_HEADING_ALIASES`, and **after that**, validation of required sections is performed.

| Alias | Normalized To |
|------|----------|
| `## 基本プロフィール` | `## 基本情報` |
| `## 性格` / `## 性格・キャラクター` | `## 人格` |

### Template directory structure

Role templates are organized into `templates/_shared` and locale-specific paths:

| Path | Contents | Locale |
|------|----------|--------|
| `templates/_shared/roles/{role}/defaults.json` | Model and parameter default values | Common |
| `templates/{locale}/roles/{role}/permissions.json` | Role-specific tool permissions | ja / en |
| `templates/{locale}/roles/{role}/specialty_prompt.md` | Role-specific behavioral guidelines | ja / en |

`locale` is resolved via `config.json`'s `locale` or the default `ja`.
`_get_roles_dir()` (`core/anima/factory.py`) looks for `templates/{locale}/roles` and falls back in the order of **`en` if it does not exist, then `ja`**.

`defaults.json` is located in `templates/_shared/roles/<role>/defaults.json` and is common to all locales. The definition fields are as follows:

| Field | Description | Notes |
|-----------|------|------|
| `model` | Model for chat and task execution | All roles |
| `background_model` | Model for background tasks such as heartbeat and cron | engineer / manager only (other roles have no key) |
| `context_threshold` | Compaction threshold | All roles |
| `conversation_history_threshold` | Conversation history compression threshold | All roles (0.30–0.40 in templates) |

Valid role names must match `VALID_ROLES` (`engineer`, `researcher`, `manager`, `writer`, `ops`, `general`) in the code.

### Available Roles (actual values for `defaults.json`)

Model and execution parameters:

| Role | model | background_model | context_threshold | conversation_history_threshold |
|--------|-------|------------------|-------------------|----------------------------------|
| manager | claude-opus-5-5 | claude-sonnet-5-5 | 0.60 | 0.30 |
| engineer | claude-opus-5-5 | claude-sonnet-5-5 | 0.80 | 0.40 |
| researcher | claude-sonnet-5-5 | — | 0.50 | 0.30 |
| writer | claude-sonnet-5-5 | — | 0.70 | 0.30 |
| ops | ollama/glm-4.7 | — | 0.50 | 0.30 |
| general | claude-sonnet-5-5 | — | 0.50 | 0.30 |

For `create_from_md` where `--role` is not specified, `general` is used. The default for ops is `ollama/glm-4.7` for local use. In the template's bundled `templates/_shared/config_defaults/models.json`, `ollama/glm-4.7*` matches execution mode **A** (LiteLLM + tool loop). When using vLLM, etc., set `model` / `credential` via `animaworks anima set-model`, and use `animaworks anima set-background-model` for the background model. While the server is running, changes are reflected to the root-owned `status.json` via the root API; during shutdown, they are applied through the offline configuration store. Do not edit directly from the Anima process.

### Application Flow

1. **At creation** (`create_from_md`), the order is as follows:
   - `_apply_defaults_from_sheet()` … Migrate from the character sheet to `IDENTITY.md` / `SOUL.md` / `AGENTS.md` (if there is a /（ permission section) `permissions.md` → `permissions.json`
   - `_apply_role_defaults()` … **Overwrite-copy** the role's `permissions.json` and `specialty_prompt.md` (the `permissions.json` derived from the character sheet is overwritten on the role side)
   - `_create_status_json()` … Read the model and context configuration from the table above (`SHARED_ROLES_DIR`, `_shared/roles/<role>/defaults.json`), and if the character sheet has "model" or "credential", overwrite with those and write `status.json`. Only write `execution_mode` when the character sheet's "execution mode" has a value; if unspecified, omit the key itself and leave it to pattern resolution such as `models.json` (`_create_status_json` of `core/anima/factory.py`).
2. **At role change** (`animaworks anima set-role`): root applies `permissions.json` and `specialty_prompt.md` via `_apply_role_defaults()`. Into the root-owned `status.json`, `model`, `context_threshold`, and `conversation_history_threshold` are merged from `defaults.json`. `background_model` is **not updated by set-role** (use `animaworks anima set-background-model`). `--status-only` updates only `role` and does not touch template files. Automatic restart via API can be skipped with `--no-restart`. The CLI success output includes `permissions.json` (`cmd_anima_set_role` of `cli/commands/anima_mgmt.py`).

### Prompt Injection

The role name is recorded in `role` of `status.json`.
`specialty_prompt.md` is Group 2 of `build_system_prompt`, placed after bootstrap → company vision and immediately before permissions (`_build_group2` of `core/prompt/builder.py`).

**Injection conditions** (branch within `_build_group2`): When the following flags are set from `trigger`, do not call `memory.read_specialty_prompt()` and do not assemble the specialty section.

- Starts with `inbox:` (Anima-to-Anima Inbox)
- `heartbeat`
- Starts with `cron:`
- Starts with `consolidation:`
- Starts with `task:` (TaskExec)

Only in cases other than the above (**the normal path for human chat, including the default empty `trigger`**) is the specialty loaded. The section's priority is 3 (rigid), and it may be omitted depending on the overall system prompt character budget (`_allocate_sections`).
