<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/security.md -->
<!-- i18n: source-sha256=ee45e223668f7d5fc4077d6187f1bf3e606e8dafa0fca4557bf71e7ad61943c6 generated=2026-11-10 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

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

`core/tooling/permissions.py`’s `check_tool_access` handles external tool access and operation-level permissions through a shared decision path. Each Anima’s `permissions.json` defines permitted and denied commands and external tools, as well as file access scopes. Deny rules take precedence over allow rules. Even when `allow_all` is enabled, operations individually restricted by the execution profile require explicit permission. The same checks apply to core, shared, and personal tools.

The `permissions.global.json` applies to all Anima and defines command-denial patterns and input-detection patterns. `core/config/global_permissions.py`’s `GlobalPermissionsCache` loads it at startup and restores its startup contents if periodic integrity checks detect runtime changes. It is also protected from file operations initiated by Anima.

If a problem occurs while loading configuration, validating the schema, or loading a profile, the operation is denied. When a decision cannot be verified, the system does not fail open.

### Bash Commands

The pure function `evaluate_command` in `core/tooling/command_policy.py` evaluates commands centrally. ToolHandler, the Claude Agent SDK hook, and the Codex hook all call the same policy. Evaluation includes shell-injection detection, rules that prohibit actions across all Anima, a recursive-search guard (Layer 2.6), Anima-specific deny rules and allowlists, path-traversal checks, and protection against writes to other Anima’s directories. The recursive-search guard denies operations that scan the entire runtime or large log areas at once, encouraging more targeted searches.

If the shared policy cannot be loaded, the caller also denies the request with an error. This ensures that changing the execution engine does not change the decision.

### File and Company Boundaries

`core/config/file_access_policy.py` checks read and write targets after resolving them to their actual paths, then applies each Anima’s allowed, read-only, and denied routes. It also checks the targets of resolved symbolic links and hides protected paths, such as runtime control information, from model-directed file operations.

When Anima belong to different companies, other companies’ company areas are excluded from access. Company `shared/` has limited write routes. Operations whose company boundaries cannot be verified are denied as a precaution. Anima without a company affiliation do not share the same boundaries as company-affiliated Anima, so set an affiliation when company separation is required.

## Skill Trust Model

Skills distinguish their source from their inspection status. Skills outside the runtime are listed read-only from the configured host-side external roots. They do not have the same write permissions as skills copied into areas managed by Anima. Creating and promoting skills are separate permissioned operations, subject to trust level, scan results, and explicit human-initiated operation requirements.

## Processes and Internal APIs

Each Anima runs in its own child process managed by the supervisor. Inter-process requests go through an internal API, which validates caller tokens specific to an Anima or operator using a secret generated each time the server starts. Tokens are generated in-process, and no secret keys are stored on disk. After validation, the system checks whether the caller may act only on itself or also on its descendants in the hierarchy, according to the requested operation.

When authentication configuration is required, `server/app.py`’s `auth_guard` checks the session cookie. For exception conditions such as secure localhost connections, and for a list of paths that do not require authentication, see [the authentication categories in the API reference](reference/api.md). Webhooks use each service’s signature verification instead of a session cookie.

## Outbound Sending and Receiving

For outbound sending, the system checks the destination resolution, operation permissions, and delivery result from the external service. It prevents a second message to the same destination within the same run, but does not deny sends based on recipient counts, time-based or daily send budgets, or conversation depth. On the receiving side, processing starts on an Inbox file-change notification, with only one item processed at a time. On Provider errors, `rate_guard` adjusts the wait before retrying. Do not include credentials in code or documentation, and enable only approved integrations. See [Messaging](architecture/messaging.md) for details.

## 1Password Integration

- Keys in the `shared` section are read one-way from 1Password’s `op://` references. animaworks does not overwrite 1Password.
- The mapping table is stored in `vault_sources.json`, update times and comparison results in `vault_meta.json`, and previous values encrypted in `vault.previous.json`. None contain secret values or hashes of values (comparisons are performed in memory).
- New items are created only when a person explicitly requests `animaworks vault register-1p --commit`; existing items are not changed.
- Synchronization, comparison, and registration are operator-only and cannot be performed from Anima’s tool context.

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
