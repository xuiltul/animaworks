<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/security.md -->
<!-- i18n: source-sha256=34c34a676a45cd171ae5f13268b722e6309e16be7ca5692e7d681c63756cba30 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> Verified commit: 581e20f1

# Security

AnimaWorks does not trust external input as instructions and enforces permission boundaries along the execution path. The basics are to never broaden permissions when configuration cannot be read, and to use the same judgment across multiple execution engines. This chapter explains the main boundaries in the implementation. For the list of configuration items, see the [Configuration Reference](reference/config.md); for authentication categories per API, see the [API Reference](reference/api.md).

## Threat Model

The assets to protect are Anima's data and credentials, other Anima and company data, commands executed on the host, and the ability to send to external services. Threat sources include malicious content arriving via the web, email, chat, and other channels; instructions that attempt to induce operations beyond permission; broken configuration; and compromised child processes or external services.

Rather than relying solely on the model's response, the host records the origin of input, applies permission checks to operations, and validates paths and callers. However, this does not prevent arbitrary host compromise. Managing the runtime directory, OS, credentials, and network is also the operator's responsibility.

## Trust Boundaries and Data Provenance

`core/execution/_sanitize.py` resolves trust from provenance. When a chain of provenance exists, the lowest trust level in the chain is inherited. Unknown provenance is also classified on the low side, and data with omitted provenance is not considered trusted.

| Trust Level | Example of Provenance | Handling |
|---|---|---|
| `trusted` | System-originated, internal data between Anima instances | Include in the prompt with a trust boundary tag. |
| `medium` | Human input, output from integration processing | Treat the content as data and perform permission checks separately. |
| `untrusted` | Web, external platforms, unknown or mixed provenance | Do not execute as instructions; explicitly maintain the trust boundary. |

`wrap_tool_result`, `wrap_external_message`, and `wrap_priming` wrap content in boundary tags and neutralize any notation in the input body that spoofs the same boundary tag. Tool result trust is resolved through a centralized table, and unregistered tools are treated on the low side. The trust of tools used during a session is also tracked, and the lower trust level is inherited by subsequent processing.

## Permission Checks

### External Tools and Operations with Side Effects

`check_tool_access` of `core/tooling/permissions.py` handles access to external tools and per-operation permissions through a common decision path. The per-Anima `permissions.json` defines allow and deny rules for commands and external tools, as well as file access scope. Deny takes precedence over allow. Even when `allow_all` is enabled, operations individually restricted by the execution profile require explicit permission. The same decision applies to core, shared, and personal tools.

The `permissions.global.json` applied to all Anima instances defines command deny patterns and input detection patterns. `GlobalPermissionsCache` of `core/config/global_permissions.py` loads at startup, and when periodic integrity checks detect runtime changes, it reverts to the startup content. It is also protected from file operations intended for Anima.

If problems occur in configuration loading, schema validation, or profile loading, the operation is denied. It never falls back to allowing when the decision cannot be confirmed.

### Bash Commands

The pure function `evaluate_command` of `core/tooling/command_policy.py` makes centralized command decisions. ToolHandler, the Claude Agent SDK hook, and the Codex hook all call the same policy. The evaluation includes shell injection detection, common deny rules for all Anima instances, a recursive search guard (Layer 2.6), per-Anima deny and allowlists, path traversal checks, and prevention of writes to other Anima directories. The recursive search guard rejects operations that scan the entire runtime or large log areas in one pass and encourages targeted searches.

If the common policy cannot be loaded, the caller is also rejected as an error. The decision must not differ when the execution engine changes.

### Files and Company Boundaries

`core/config/file_access_policy.py` checks read and write targets resolved to real paths and applies per-Anima allow, read-only, and deny routes. It also inspects the targets of resolved symbolic links and hides protected paths such as runtime control information from model-facing file operations.

When the affiliated company differs, other companies' areas are excluded from access. Limited write routes are provided for the company's `shared/`. Operations whose company boundary cannot be confirmed are denied on the safe side. An Anima without an affiliation does not have the same boundaries as a company-affiliated Anima, so configure an affiliation when company separation is required.

## Skill Trust Model

Skills distinguish between their registration source and inspection status. Skills outside the runtime are enumerated read-only from the configured host-side external roots. They do not have the same write permissions as skills copied into Anima-managed areas. Skill creation and promotion are separate permission operations and follow trust level, scan results, and explicit human-initiated operation conditions.

## Processes and Internal API

Each Anima runs in a separate child process managed by the supervisor. Inter-process requests go through the internal API, which validates per-Anima or per-operator call tokens derived from a secret generated at each server startup. Tokens are generated in-process, and no private key is stored on disk. After validation, it confirms whether the requested operation can target the caller itself or only its descendants in the hierarchy.

When authentication configuration is required, `auth_guard` of `server/app.py` checks the session cookie. For exception conditions such as secure localhost connections and the list of paths that do not require authentication, see the [authentication categories in the API Reference](reference/api.md). Webhooks use signature validation per service rather than session cookies.

## External Sending and Receiving

External sending verifies recipient resolution, operation permissions, and delivery results from external services. Within the same run, a second message to the same destination is prevented, but there is no rejection based on the number of destinations, time-based or daily sending budgets, or conversation depth. On the receiving side, startup is triggered by Inbox file change notifications, and only one message is processed at a time. In the event of a Provider error, `rate_guard` adjusts the wait time before retry. Do not include authentication information in code or documentation, and enable only permitted integrations. For details, refer to [Messaging](architecture/messaging.md).

## Adversarial Threat Analysis

| Threat | Defense and Limitations |
|---|---|
| Instructions embedded in external messages | Mark provenance as low trust with boundary tags. The model may still misinterpret, so external requests are never treated as permissions. |
| Compound shell commands or broad searches | Split commands into syntactic units and evaluate the same policy across all execution paths. This is not a guarantee of semantically understanding every obfuscated shell expression. |
| Access to other Anima or company data | Validate canonical paths, company affiliation, and caller identity. An Anima without company configuration is not isolated by company boundaries. |
| Broken permission configuration or credentials | Treat failures in JSON loading, permission checks, and internal caller validation as denials. Threats where the host administrator can modify configuration or the execution environment itself require separate management. |

## Applying Defense in Depth

This combines input origin display via boundary tags, per-Anima tool permissions, common command evaluation, file path and company affiliation checks, caller validation for internal APIs, HTTP authentication, and prevention of duplicate destinations within the same run. Each layer protects a different operation point, so safety is not achieved by any single layer alone. Operate in conjunction with configuration management, runtime environment protection, and least-privilege access to external services.

## Residual Risks and Operations

- Trust tags cannot completely prevent model misinterpretation. For requests contained in externally sourced content, confirm the origin of the content before executing them.
- `permissions.json` and `permissions.global.json` are permission boundaries. Take a backup before editing, and validate the JSON and schema after changes. If a read error occurs, do not loosen permissions; fix the cause instead.
- When enabling localhost trust, correctly restrict the reverse proxy's forwarding headers and public exposure. When exposing the server to the internet, configure authentication and TLS.
- The log mode of the internal API is for migration verification and differs from operations that enforce caller validation. Check the production environment configuration.
- Prevention of duplicate destinations within the same run is an aid to reduce the impact of misdirected messages; it does not replace human review of destinations and content or least-privilege settings on the external service side.

## Related Documents

- [Configuration Reference](reference/config.md)
- [API Reference](reference/api.md)
- [Messaging](architecture/messaging.md)
- [Anima Files](architecture/anima-files.md)
- [Process Architecture](architecture/process.md)
