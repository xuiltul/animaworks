# Episode Extraction from Activity Log

{anima_name}, organize your activity records into a structured timeline.

## Target period: {time_range}

## Activity Log

{activity_chunk}

---

## Output Format

Use the following Markdown format. Separate sections by time period using `## HH:MM-HH:MM Title` headers, with bullet points for events.

```
## HH:MM-HH:MM Section Title

- HH:MM Event summary
  - Details, results, related information
- HH:MM Next event
```

## Rules

1. **Group by time period**: Cluster related activities into 30-minute to 2-hour blocks
2. **Preserve specific information**: Keep key details from email bodies, command outputs, file changes, and message contents that could serve as future knowledge references
3. **Eliminate redundant repetition**: Deduplicate repeated `current_state.md` dumps or duplicate REFLECTION blocks — keep only one instance
4. **Tool execution results**: Record result summaries for successes and error details for failures
5. **Communication content**: Record the key points of sent/received messages (who, to whom, about what)
6. **No speculation**: Record only facts from the activity log. Do not add inferences or interpretations
7. **Use only `##` markdown headers**: Do not use `#` or `###`
