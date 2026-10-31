<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/memory/index.md -->
<!-- i18n: source-sha256=ff2e049713f751fce015b910c2fab2c4ee9c63821df9794c5156bdf2c5dbe844 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> Confirmed commit: 193a5e72

# Memory System

AnimaWorks memory is a file-based archive that retrieves only what is needed into the runtime context. Anima reads and writes memory, and decides when to consolidate or organize records. The framework provides indexes and candidates for search, but it is not designed to move or delete old records under uniform mechanical rules. For the role of automatic indexing, see [Intentional Recall and Search](retrieval.md).

## Correspondence with the Human Memory Model

| Human Memory | AnimaWorks | Role |
|---|---|---|
| Working memory | LLM context | Temporarily holds information needed for current decisions. |
| Episodic memory | `episodes/` | Records what happened and when, in chronological order. |
| Semantic memory | `knowledge/` | Holds knowledge, lessons, and policies derived from experience. |
| Procedural memory | `procedures/`, `skills/` | Holds how to perform tasks and reusable procedures. |
| Structured facts | `facts/` | Stores atomic facts with entities and timestamps extracted from conversations and records. |

The authoritative source of long-term memory is files; search vector indexes and BM25 indexes are derived data that can be regenerated. For the distinction between automatic recall and explicit search, see [Automatic Recall](priming.md) and [Search](retrieval.md).

## Memory Directory

Each Anima's memory area is located under `~/.animaworks/animas/{name}/`.

| Directory | Role |
|---|---|
| `episodes/` | Episodic records of daily events and work. |
| `knowledge/` | Semantic memory. Holds knowledge, lessons, policies, etc., in Markdown. |
| `procedures/` | Holds procedures in Markdown, and tracks confidence and versions using execution results. |
| `facts/` | Stores atomic facts in date-based JSONL files. |
| `skills/` | Holds reusable skill documents for procedures and tool usage. |
| `state/` | Holds status data such as current status and entity registry that can be reconstructed from facts. |
| `shortterm/` | Holds short-term data and streaming journals during sessions. |

Interpersonal profiles are placed in the shared area at `shared/users/` and are referenced separately from Anima-specific long-term memory.

## Frontmatter

Markdown files in `knowledge/` and `procedures/` can have YAML frontmatter. `core/memory/frontmatter.py` handles common read/write, repair, and default value completion. It is not a closed schema with uniformly required keys, but metadata that depends on content and purpose.

In `knowledge/`, `created_at`, `updated_at`, `confidence`, `source_episodes`, `auto_consolidated`, and `version` are used during creation and maintenance. `success_count` and `failure_count` track reported usefulness together with `confidence`, `valid_from` and `valid_until` indicate the validity period of facts, and `description` can be used for summarizing search results. `always_prime: true` is an explicit opt-in for automatic recall. For meaning and scope, see [Automatic Recall](priming.md).

In `procedures/`, `description`, `confidence`, `success_count`, `failure_count`, `version`, `created_at`, and `updated_at` are used to describe procedures and track usage results. `auto_distilled` indicates procedures created by automatic distillation, and `protected` is a protection designation. `core/memory/skill_metadata.py` handles procedure metadata in a common format with the skill catalog. If `description` is absent, it can be supplemented from the document's `## 概要`. `core/skills/loader.py` handles loading metadata for skill documents.

For the list of configuration items and default values, see [Configuration Reference](../reference/config.md). Daily integration and procedure usage tracking are described in [Integration and Forgetting](consolidation.md).
