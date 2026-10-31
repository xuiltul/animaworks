<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/operations/dev-team.md -->
<!-- i18n: source-sha256=33c335399f699ae80e7322151309867c19a8b99faaf967e288e2a3e7f863b69f generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> Confirmed commit: 581e20f1

# Development Team Structure

AnimaWorks includes role templates for development teams. Templates serve as a starting point for launching a team, and specific development procedures and knowledge should be tailored to each team's operations.

## Role Examples

| Template | Primary Role |
|---|---|
| `dev-lead` | Break down work, delegate to implementation and investigation assignees, and perform quality checks and escalation. |
| `dev-engineer` | Implement in an isolated work environment, and report tests and changes made. |
| `dev-researcher` | Investigate in read-only mode, and report evidence and unconfirmed items separately. |

The number of roles and the hierarchy are not fixed. In small teams, one person should not hold multiple responsibilities; create only the necessary roles.

## Creation

```sh
animaworks anima create --name lead --template dev-lead
animaworks anima create --name engineer --template dev-engineer --supervisor lead
animaworks anima create --name researcher --template dev-researcher --supervisor lead
```

The name is an example. After creation, review the organization chart, model, permissions, and heartbeat/cron configuration, and adjust the template instructions to align with the team's approval and review standards. For how to create an Anima, refer to the [CLI reference](../reference/cli.md).

## Connecting to the Development Pipeline

Enabling the GitHub webhook allows notifications to be sent to a designated dispatcher Anima for target events. Notification handling and the implementation/review process are separate responsibilities, and received events do not automatically guarantee safe changes or merges. Repository permissions, CI, review approvals, and branch protection should also be configured on the GitHub side.

For work handled by the team, as a rule, separate the work tree, clarify the scope of changes, and run tests. Investigation reports should include the referenced paths and symbols, and implementation results should include validation details and any remaining constraints.
