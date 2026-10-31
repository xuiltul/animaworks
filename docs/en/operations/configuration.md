<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/operations/configuration.md -->
<!-- i18n: source-sha256=7af4d2067fc6d903b130aec1b7be2e9f06d27f3968f81963e44a6b4116d85779 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> Verified commit: 581e20f1

# Configuration Management

AnimaWorks configuration is divided into runtime-wide behavior, model resolution, per-Anima status, and permission policies. Rather than consolidating files with different purposes into one, choose what to change and how to apply it.

## Role of Configuration Files

| File | Primary Role | Scope |
|---|---|---|
| `config.json` | Runtime settings such as server, integration, memory, GPU, and common defaults | Entire runtime. |
| `models.json` | Execution mode and model metadata per model name pattern | Model resolution. |
| `animas/<name>/status.json` | Model, execution mode, affiliation, status, etc. per Anima | Individual Anima. |
| `permissions.global.json` | Command prohibition rules and input detection settings applied to all Anima | Runtime-wide security boundary. |
| `animas/<name>/permissions.json` | File, command, and external tool permissions for individual Anima | Individual Anima. |

The actual file location is the directory specified by `ANIMAWORKS_DATA_DIR` if set; otherwise, it is the default data directory. For a complete list of paths and items, see the [configuration reference](../reference/config.md).

## Priority Order

Common runtime configuration values are held by `config.json`, and Anima-specific specifications are applied as overrides. Model execution mode is resolved in the order of `execution_mode` explicitly specified in the Anima's `status.json`, `models.json` patterns, the `model_modes` of the compatibility `config.json`, and code defaults. If the model name does not match any pattern, `A` is used.

`permissions.global.json` and `permissions.json` are permission policies separate from general settings. Individual deny rules take precedence over allow rules. Global settings are loaded at server startup and cached as the active policy.

## Editing Methods

Use `animaworks config get <dot.path>`, `animaworks config set <dot.path> <value>`, and `animaworks config list` to view and edit `config.json`. Values are interpreted as booleans, numbers, or `null`. Avoid inadvertently displaying secret values on the terminal or in logs.

Use `animaworks anima set-model <name> <model>` to change an Anima's model. `models.json` is the model name mapping table. If manually changing items without a corresponding edit command, back up the file, verify syntax and values, then save. Since permission files are access boundaries, always review for unintended expansion of allowed scope.

## Applying Changes

- For `status.json` Anima configuration changes, use `animaworks anima reload <name>` to reload the running Anima. For changes involving process or execution environment initialization, use `animaworks anima restart <name>`.
- For `config.json` general settings or connection settings, use hot reload if supported by the target configuration, or restart the server. If unsure how to apply, restart.
- `permissions.global.json` is cached at startup. Restart the server to apply changes.
- `models.json` model patterns are cached but are reloaded upon detecting file updates. After changes, verify resolution on the next execution of the target Anima.

If the restart target is chosen incorrectly, only the configuration file changes without being applied to running connections or processes. After changes, check `animaworks anima status <name>` or the server's health/status to confirm the intended values are active.
