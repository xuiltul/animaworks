---
name: skill-creator
description: >-
  Meta-skill for authoring Markdown Skill files with YAML frontmatter and progressive disclosure via create_skill.
  Use when: adding a new skill, generating SKILL.md with references or templates, or checking description rules.
---

# skill-creator

## Skill file structure

A Skill file consists of YAML frontmatter and Markdown body.
Required frontmatter fields are `name` and `description`.
Optional fields: `allowed_tools` (permitted tool list), `tags` (classification tags).

```yaml
---
name: skill-name
description: >-
  Concise third-person summary of what the skill does.
  Use when: comma-separated scenarios where this skill applies.
---
```

`description` is the primary field for discovery and selection: the model uses it to decide relevance.
The body is injected only when the skill is loaded via the `skill` tool by name.

**Authoring format**: follow **`Use when:`** as described in `references/description_guide.md` (Agent Skills standard).
After editing, validate with **`python scripts/lint_skill.py path/to/SKILL.md`**.

## Writing `description`

Do not use legacy **`「」` keyword lists**. Use a short third-person capability line plus **`Use when:`** with concrete verbs and nouns.

See **`references/description_guide.md`** for rules (250 characters, no XML tags, examples, checklist).

### Domain-specific and concrete

Generic wording causes false positives. Prefer tool names, operations, and targets specific to the skill.

## Progressive disclosure

Skill information is disclosed in three levels.

| Level | Content | When shown |
|-------|---------|------------|
| Level 1 | `name` + `description` | Skill catalog / tool descriptions (budgeted) |
| Level 2 | body | Injected when `skill(skill_name=...)` runs |
| Level 3 | External files | Loaded per body instructions (`references/`, `templates/`) |

Keep Level 1 concise; put procedures in Level 2; offload long material to Level 3.

## Creation procedure

### Step 1: Clarify

- What to automate or document
- Personal vs common Skill (procedures use `procedures/*.md` separately)
- **Use when:** scenarios (when to choose this skill)

### Step 2: Design

- **name**: kebab-case (e.g. `my-skill`); use `*-tool` naming for external tool guides when applicable
- **description**: third-person summary + **`Use when:`** (see `references/description_guide.md`)
- **body**: section structure; optional `{{now_local}}` and other builtins
- **references** / **templates**: optional
- **allowed_tools**: optional soft constraint

### Step 3: Create

```
create_skill(skill_name="{name}", description="{description}", body="{body}")
```

Common skills:

```
create_skill(skill_name="{name}", description="{description}", body="{body}", location="common")
```

Prefer `create_skill` for new skills; flat `skills/foo.md` alone may not resolve via the skill tool.

### Step 4: Verify

- Re-read `skills/{name}/SKILL.md` or use `skill(skill_name="{name}")`
- Run **`python scripts/lint_skill.py`** on the file (recommended)

## Checklist

- [ ] YAML frontmatter delimited by `---`
- [ ] `name` and `description` present
- [ ] **`Use when:`** present; no **`「」`** keyword enumeration
- [ ] Domain-specific, concrete wording (avoid vague “manage” / “check” alone)
- [ ] Body has actionable steps
- [ ] Avoid relying only on `## Overview` for description; prefer frontmatter
- [ ] Created via `create_skill` with `{name}/SKILL.md` layout where applicable

## Template

Use `templates/skill_template.md` bundled with this skill, or:

```markdown
---
name: {{skill_name}}
description: >-
  {{Line 1: concise capability summary}}
  Use when: {{comma-separated usage scenarios}}
---

# {{skill_name}}

## Procedure

1. ...
2. ...

## Notes

- ...
```

## Notes

- Skills are Markdown playbooks, not Python tools
- Required frontmatter: `name`, `description`
- Optional: `allowed_tools`, `tags` (metadata usage may vary)
- Keep body around 150 lines when practical; use `references/` for long material
