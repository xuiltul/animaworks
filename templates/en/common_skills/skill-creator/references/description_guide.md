# description authoring guide (Agent Skills standard)

## Basics

The `description` field is the most important signal for skill discovery and selection.
The LLM reads it to decide whether this skill is relevant to the current conversation.

### Format

```yaml
description: >-
  [Line 1: concise third-person summary of what the skill does]
  Use when: [comma-separated usage scenarios]
```

### Rules

1. **250 characters or fewer** — catalog views truncate beyond 250 characters
2. **Third person** — use patterns like "Provides …" or "Handles …". Do not use "I" or "you"
3. **Include `Use when:`** — gives the model clear cues for when to apply the skill
4. **Use concrete verbs and nouns** — e.g. "login", "screenshot", "send email"
5. **No XML tags** — `<` and `>` are disallowed for security reasons
6. **Do not use `「」` keyword lists** — legacy style; rely on natural phrasing and `Use when:`

### Good examples

```yaml
description: >-
  Headless browser CLI. Opens web pages for viewing, interaction, login, and screenshots.
  Use when: opening sites in a browser, operating or verifying web apps, login flows, screenshots, UI checks.
```

```yaml
description: >-
  Gmail operations via CLI. Send, receive, search emails, and manage labels.
  Use when: sending emails, checking inbox, searching mail, managing Gmail labels or filters.
```

### Bad examples

```yaml
# ❌ 「」 keyword list (legacy)
description: >-
  Browser CLI.
  「check in browser」「take a screenshot」「browser ops」

# ❌ Too vague
description: Helps with documents

# ❌ First person
description: I can help you process PDF files

# ❌ Over 250 characters
description: >-
  (Very long text…)
```

### Checklist

- [ ] Within 250 characters
- [ ] Written in third person
- [ ] Contains `Use when:`
- [ ] Uses concrete verbs and nouns
- [ ] No `「」` keyword enumeration
- [ ] No XML tags

### Lint

After authoring, validate with:

```bash
python scripts/lint_skill.py /path/to/SKILL.md
```
