<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/architecture/prompt.md -->
<!-- i18n: source-sha256=4efd5e6044c341500cee440726a5e0ae86fc499c7ff7c73f6109d91d97138208 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> Confirmed commit: b304b7dc

# Building the system prompt

`core/prompt/builder.py` builds the system prompt based on memory, execution mode, trigger, and the current request, and `core/prompt/assembler.py` assembles sections with XML boundary tags to fit within the budget. Fixed parts are grouped at the beginning, with status information that changes each turn placed afterward.

## Section Structure and Order

The builder creates six logical groups. They are assembled in this order: groups 1, 2, 4, 5, and 6, followed by the dynamic group 3.

1. **Environment and behavior rules** — Runtime information, workspace, Anima’s `IDENTITY.md` / `SOUL.md`, `AGENTS.md`, and behavior rules.
2. **Information about Anima itself** — Bootstrap status, the company vision, speciality, and permissions.
3. **Memory and capabilities** — Memory guide, tool guide tailored to the trigger / mode, available external tools, active skill context, and skill catalog.
4. **Organization and communication** — Organization relationships, internal messages, and how to notify humans.
5. **Meta configuration** — Instructions for emotion during chat and response rules for specific engines.
6. **Current situation** — Current time, `current_state.md`, recent resolutions, priming, relevant human notifications, and short-term memory.

Group 3 collects information that changes from turn to turn. Treating the preceding groups as a stable prefix makes it easier to reuse the provider’s prompt cache. When the context window is small, some environment information and optional sections are omitted according to the prompt tier.

## Skills and tool guide

The skill list is assembled after indexing Anima-specific, common, and procedure skills. In chat, the request text is passed to the skill router to prioritize relevant candidates; if no suitable candidate exists, the normal catalog is used. The catalog is limited to a maximum of 3 items by default. In chat, the body of enabled skills is also added to the context. In automatic execution, skills requiring human approval are excluded from the list.

The tool guide switches depending on heartbeat, engines using MCP, and other engines. Skill bodies and actual tool inputs/outputs are loaded on demand, while the system prompt contains available entry points and necessary rules.

## Handling Budget and Bloat

The assembler allocates the budget based on section priority and trims elastic sections by Markdown paragraph. It never cuts off an individual paragraph mid-way, and removes lower-priority rigid sections if the hard ceiling would otherwise be exceeded. The default normal target is 6,000 tokens, with a cap of 35% of the context window; these values can be changed through configuration.

Size warnings for `AGENTS.md` are added only during consolidation, when the configured character-count threshold is exceeded. The skill catalog also has an item limit. For a list of global and individual settings, see the [Configuration Reference](../reference/config.md).

## Template resolution

Markdown templates are placed under `templates/{locale}/`. The `resolve_template_path` of `core/paths.py` is searched in the order of the specified locale, `en`, `ja`, and `_shared`. Runtime template paths based on the distribution format are treated as a compatibility fallback after this search.
