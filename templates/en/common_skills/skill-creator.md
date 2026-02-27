---
name: skill-creator
description: >-
  Meta-skill for creating Markdown Skill files (.md) with correct YAML frontmatter format.
  Provides description rules for personal skills (skills/) and common skills (common_skills/),
  keyword design with 「」, Progressive Disclosure structure, and save procedure via write_memory_file.
  Use for: "create skill", "create a skill", "new skill", "create procedure", "skill file".
---

# skill-creator

## Skill File Structure

A Skill file consists of YAML frontmatter and Markdown body.
Include only `name` and `description` in the frontmatter.

```yaml
---
name: skill_name
description: >-
  Skill description.
  「keyword1」「keyword2」
---
```

`description` is the most important field for Skill activation.
The body is injected into the system prompt only when the user's message matches the description.
In other words, description acts as the "primary trigger mechanism."

## Writing description

### Critical Rule: Be Domain-Specific and Concrete

description is used for vector search (semantic similarity) matching.
**Generic or abstract descriptions cause false matches with other Skills.**
Always include terms, operations, and targets specific to that Skill.

**Bad example (generic → causes false positives)**:

```yaml
description: >-
  Manages AnimaWorks server. Handles reload after code updates and system state checks.
```

→ "management", "check", "state" are too generic and match unrelated messages.

**Good example (domain-specific → accurate match)**:

```yaml
description: >-
  AnimaWorks server process operations skill.
  Executes hot reload after code update (server reload), Anima process restart,
  and server status check (running Anima list, memory usage).
  「reload」「reflect update」「reload」「system status」「server restart」「process check」
```

→ "hot reload", "server reload", "Anima process", "memory usage" are specific vocabulary.

### Basic Structure

1. **First sentence**: State the Skill's purpose with **specific target and operations** (not "manage ~" but "hot reload, restart, status check for ~")
2. **Second sentence onward**: Include tool names, API names, specific procedure steps
3. **End**: List keywords in `「」` format

### description Concreteness Check

If any of these appear alone, replace with more concrete wording:

| Avoid | Better |
|-------|--------|
| perform management | hot reload, process restart, status check |
| perform check | slack_search tool for DM retrieval, unread check |
| perform response | chatwork_search for room unread retrieval, send reply |
| mechanism and usage | RAG/Priming/Consolidation, execution mode (S/A/B) |
| meta-skill for creating | create in YAML frontmatter format, 「」keyword design |

### Keyword Design Tips

- Choose short phrases (2–5 chars) users naturally say
- Include synonyms and variants (e.g., "periodic execution", "cron setup", "schedule")
- For English descriptions, comma-separated or use-case examples are fine
- Too many keywords cause false triggers; keep to about 3–6
- Make keywords domain-specific ("cron check", "Slack check" instead of "check")

### Example

```yaml
description: >-
  Skill for AnimaWorks cron job (APScheduler) setup and management.
  Provides procedures for adding, editing, and removing periodic tasks in crontab format,
  checking next run time, and viewing execution logs.
  「cron setup」「periodic execution」「schedule」「scheduled task」「cron check」
```

## Progressive Disclosure

Skill information is disclosed in 3 levels.

| Level | Content | When Shown |
|-------|---------|------------|
| Level 1 | description | Always shown in Skill table. Used for Skill selection |
| Level 2 | body | Injected into system prompt when description matches |
| Level 3 | External resources | Read files as instructed in body when needed |

Level 1 always uses context; keep description concise.
Write concrete procedures in Level 2 body.
Use Level 3 for long reference materials or code examples in external files.

## Creation Procedure

### Step 1: Clarify

Understand the user's request. Confirm:

- What to automate or document
- Personal Skill or common Skill
- Trigger keywords (how they want to invoke it)

### Step 2: Design

Decide:

- **name**: Skill name (kebab-case, e.g., `my-skill`)
- **description**: Trigger text and keywords
- **body**: Section structure

### Step 3: Create

Save the Skill file:

```
write_memory_file("skills/{name}.md", content)
```

For common Skills:

```
write_memory_file("common_skills/{name}.md", content)
```

### Step 4: Verify

After saving, re-read and verify:

```
read_memory_file("skills/{name}.md")
```

## Checklist

Before saving, verify:

- [ ] YAML frontmatter starts and ends with `---`
- [ ] `name` field exists
- [ ] `description` field exists
- [ ] description has at least one `「」` keyword
- [ ] **description is domain-specific and concrete** (avoid generic phrases like "perform management", "perform check"; include tool names, operation names, targets)
- [ ] body contains concrete procedure steps
- [ ] Old format `## Overview` / `## Trigger Conditions` is not used

## Template

Use this as a starting point:

```markdown
---
name: {skill_name}
description: >-
  Skill for {specific target}'s {specific operation}.
  Executes {specific procedure summary} via {tool/API name}.
  「{domain keyword 1}」「{domain keyword 2}」「{domain keyword 3}」
---

# {skill_name}

## Procedure

1. ...
2. ...

## Notes

- ...
```

## Notes

- Skills are Markdown procedure documents, distinct from Python code (tools)
- Do not add fields other than `name` and `description` to the frontmatter
- Keep body under about 150 lines to avoid context pressure
- Use external resource references (Level 3) to keep the body concise
