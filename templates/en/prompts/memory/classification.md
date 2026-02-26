Classify the following episodes (action records) and extract knowledge and procedures.

【Classification criteria】
- **knowledge**: Lessons learned, policies, facts, pattern recognition, principles ("why" and "what" knowledge)
- **procedures**: Steps, workflows, checklists, work flows ("how to" procedures)
- **skip**: No need to consolidate (small talk, transient information, greetings only)

【Priority classification rules】
- **Prioritize procedures for fix/problem-resolution episodes**: Records of error fixes, configuration changes, and troubleshooting should be extracted as procedures when they contain concrete steps. The pattern "cause was X → resolved by Y" has value for procedure extraction even for a single occurrence
- The same episode may yield both knowledge and procedure (e.g., causal knowledge + resolution steps)

【Episodes】
{episodes_text}

【Existing procedures (avoid duplicates)】
{existing_procedures}

【Output format】
Output in the following sections. Use "(none)" if nothing applies.

## knowledge extraction
- Filename: knowledge/xxx.md
  Content: (Knowledge body in Markdown format)

## procedure extraction
- Filename: procedures/zzz.md
  description: Concrete one-line that clearly states the procedure's purpose (e.g., "Chatwork critical case escalation decision and notification"). Must match the first heading
  tags: tag1, tag2
  Content: (Procedure body in Markdown format. First heading matches description. Include concrete steps)

【Rules】
- Skip if duplicate of existing procedure
- Extract only generic, reusable procedures
- However, problem-resolution and fix procedures should be extracted even if low reusability (valuable when the same issue recurs)
- Do not extract if the procedure is vague
- Empty results are acceptable (write "(none)")
- Do not wrap output in code fences (```)
