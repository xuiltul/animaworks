Compare the following two knowledge files and verify whether they contradict each other.

【File A: {file_a}】
{text_a}

【File B: {file_b}】
{text_b}

Task:
1. Determine whether there are contradictory statements between the two files
2. If contradictions exist, propose one of the following resolutions:
   - "supersede": One piece of information is outdated and should be replaced by the newer one
   - "merge": Both pieces of information should be consolidated into a single knowledge item
   - "coexist": Both statements are correct depending on context (they can coexist)

Output your answer only in the following JSON format:
{{"is_contradiction": true/false, "resolution": "supersede"/"merge"/"coexist", "reason": "explanation of the reason", "merged_content": "merged text when resolution is merge (null otherwise)"}}
