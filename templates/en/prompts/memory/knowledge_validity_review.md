You are {anima_name}. Review your activity today and self-assess whether the following knowledge/procedure files are still accurate.

【Today's episode summary】
{episodes_summary}

【Files to review】
{review_files}

For each file, choose one of these verdicts:
- **valid**: Content is still accurate; no changes needed
- **stale**: Outdated (resolved issues still described as unresolved, deprecated procedures, changed specs, etc.)
- **needs_update**: Generally correct but needs partial updates based on today's activity

Output only the following JSON array (no explanatory text):
```json
[
  {{"file": "filename", "verdict": "valid|stale|needs_update", "reason": "verdict rationale (1-2 sentences)", "correction": "correction text for stale/needs_update (null for valid)"}}
]
```

Guidelines:
- If problems resolved today are still described as "unresolved", mark stale
- If procedures contradict today's experience, mark stale or needs_update
- If information is old but harmless, mark valid (judge conservatively)
- When correction is needed, provide concrete correction text in correction
