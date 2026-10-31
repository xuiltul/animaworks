<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/architecture/prompt.md -->
<!-- i18n: source-sha256=4efd5e6044c341500cee440726a5e0ae86fc499c7ff7c73f6109d91d97138208 generated=2026-10-16 engine=luna model=gpt-6-luna translator=2 -->

> Confirmed commit: b304b7dc

# Building the system prompt

`core/prompt/builder.py` builds the system prompt based on memory, execution mode, trigger, and the current request, and `core/prompt/assembler.py` assembles sections with XML boundary tags to fit within the budget. Fixed parts are grouped at the beginning, with status information that changes each turn placed afterward.

## Section structure and order

The builder creates six logical groups. The assembly order is groups 1, 2, 4, 5, 6, and finally the dynamic group 3.

1. **Environment and behavior rules** — runtime information, workspace, Anima's `IDENTITY.md` / `SOUL.md`, `AGENTS.md`, behavior rules.
2. **Anima's own information** — bootstrap status, company vision, speciality, permissions.
3. **Memory and capabilities** — memory guide, tool guide tailored to trigger / mode, available external tools, active skill context, skill catalog.
4. **Organization and communication** — organizational relationships, internal messages, notification methods for humans.
5. **Meta configuration** — emotion instructions during chat and response rules for specific engines.
6. **Current situation** — current time, `current_state.md`, recent resolutions, priming, relevant human notifications, short-term memory.

Group 3 contains information that changes per turn. By treating the preceding groups as a stable prefix, the provider's prompt cache can be reused more easily. If the context window is small, some environment information and optional sections are omitted depending on the prompt tier.

## Skills and tool guide

The skill list is assembled after indexing Anima-specific, common, and procedure skills. In chat, the request text is passed to the skill router to prioritize relevant candidates; if no suitable candidate exists, the normal catalog is used. The catalog is limited to a maximum of 3 items by default. In chat, the body of enabled skills is also added to the context. In automatic execution, skills requiring human approval are excluded from the list.

The tool guide switches depending on heartbeat, engines using MCP, and other engines. Skill bodies and actual tool inputs/outputs are loaded on demand, while the system prompt contains available entry points and necessary rules.

## Budget and bloat handling

The assembler allocates budget based on section priority, and elastic sections are trimmed at the Markdown paragraph level. Individual paragraphs are not cut mid-way; if the hard ceiling is exceeded, lower-priority rigid sections are also removed. The default normal target is 6,000 tokens, with a cap of 35% of the context window, adjustable via configuration.

The `AGENTS.md` size warning is added only during consolidation, when the configured character threshold is exceeded. The skill catalog also has a count limit. See the [configuration reference](../reference/config.md) for the full list of global and individual settings.

## Template resolution

Markdown templates are placed under `templates/{locale}/`. The `resolve_template_path` of `core/paths.py` is searched in the order of the specified locale, `en`, `ja`, and `_shared`. Runtime template paths based on the distribution format are treated as a compatibility fallback after this search.
