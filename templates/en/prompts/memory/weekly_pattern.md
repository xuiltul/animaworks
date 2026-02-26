The following are clusters of similar behavioral patterns extracted from 7 days of activity logs.
Each cluster contains similar actions that were repeated 3 or more times.

Identify repetitive patterns and distill them into reusable procedures.

【Behavioral patterns】
{clusters_text}

【Existing procedures】
{existing_procedures}

Output format (JSON array):
[
  {{
    "title": "Procedure name (for English filename)",
    "description": "Brief procedure overview (1-2 sentences)",
    "tags": ["tag1", "tag2"],
    "content": "# Procedure name\\n\\n## Overview\\n...\\n\\n## Steps\\n1. ...\\n2. ...\\n\\n## Notes\\n..."
  }}
]

Rules:
- Return an empty array [] if there are no repetitive patterns
- Skip if it duplicates an existing procedure
- Include concrete procedure steps
- An empty array [] is acceptable
