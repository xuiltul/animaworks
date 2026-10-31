<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/vision.md -->
<!-- i18n: source-sha256=648d7564f35cd1ef72ce7d941f78e3e67170e7d1e96203125376a6cc080e6d6b generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> Confirmed commit: 581e20f1

# Design Philosophy of Digital Anima

**Organization-as-Code for LLM Agents**

AnimaWorks is a system that defines LLM agents as an organization, entrusts them with work, and supports ongoing collaboration. Each Digital Anima has its own role and memory, and collaborates with others through messages.

## What We Aim For

Collaboration among imperfect individuals creates a more robust organization than a single omnipotent one. And by accumulating experience in memory, it leads to judgments that transcend individual limitations.

Rather than concentrating all knowledge and work in a single model, each Anima makes decisions based on its own expertise and shares what is necessary in its own words. The differences in perspective and capability among the individuals that make up the organization are design elements for dividing roles and collaborating.

## Three Principles

### Encapsulated Individual

Each Anima has its own memory and status. Collaboration with others is conducted through explicit interfaces such as messages and tasks, and the internal state of others is not directly shared. This boundary allows roles and responsibilities to be separated.

### Memory That Leverages Experience

Anima does not load the entire conversation history as-is, but recalls memory as needed. It searches different types of information—activity logs, episodes, knowledge, procedures—and uses them according to context. In daily and weekly consolidation, candidates and verification materials are prepared, and Anima's own processing is also used for content judgment. See the [Memory System](memory/index.md) for memory structure and indexing.

### Collaboration as an Organization

Combines hierarchical supervision and delegation, shared channels, DMs, and task management. See [Lifecycle](architecture/lifecycle.md) for execution paths, [Process Configuration](architecture/process.md) for process separation, and [Messaging](architecture/messaging.md) for collaboration methods.

## Choosing the Right Model for the Job

You can select different models and execution engines for each Anima. The current execution modes are Claude Agent SDK `S`, Codex `C`, Grok Build `X`, Cursor Agent `D`, Gemini CLI `G`, and LiteLLM-based `A`. In addition to automatic resolution from model names, you can configure the execution mode per Anima. For details and priority order, see [Model Execution](architecture/execution.md) and the [Configuration Reference](reference/config.md).

Rather than building your organization with only the most powerful models, balance capability, response time, and cost according to the nature of the work. You can also set a different background model for heartbeats and cron jobs.

## Design Evaluation

This design is evaluated by whether Anima can recall necessary experience, learn from results, delegate to the right counterpart, and share progress and outcomes. It aims not merely to advocate memory and autonomy, but to improve judgment and collaboration in daily operations.

AnimaWorks is not a standalone chat assistant, but a foundation for multiple individuals with limitations to create value as an organization.
