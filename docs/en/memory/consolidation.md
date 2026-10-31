<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/memory/consolidation.md -->
<!-- i18n: source-sha256=a05463694132540a497b99625f9dc933a6b34890c941860e8c6bc5269d5e45a0 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> Confirmed commit: 193a5e72

# Memory Consolidation and Forgetting

Memory consolidation consists of judgment by Anima and candidate collection and index maintenance performed by the framework. The timing is configurable, with the default daily consolidation at 02:00 and weekly consolidation on Sunday at 03:00. Weekly consolidation runs when `weekly_enabled` is enabled.

## Daily Processing

`server/supervisor/_mgr_scheduler.py` registers a daily job, and `core/lifecycle/system_consolidation.py`'s `_handle_daily_consolidation` selects the target Anima. Depending on the period of inactivity or the number of records in the last 24 hours, execution may be skipped.

Anima's `run_consolidation` summarizes the previous day's activity log into dated episodes and extracts atomic facts from the summary. Subsequent framework post-processing proceeds in the following order:

1. `ForgettingEngine.synaptic_downscaling` marks low-activity candidates on the index.
2. `knowledge_self_correction` reviews the relevant knowledge and procedures with an LLM, and creates procedures from resolved issues.

This correction is executed by `core/lifecycle/knowledge_correction.py` while managing the upper limit. Detailed re-fixation conditions are described later.

## Weekly Processing

The weekly Anima cycle is handled by `core/anima/lifecycle.py`'s `_run_weekly_consolidation`. `ConsolidationEngine` prepares merges, conflicts between facts, and forgetting candidates, while `hygiene.py` compiles a report of knowledge file format and size check candidates. These are included in the weekly prompt, and Anima judges the content to update and archive memory files. The hygiene scan itself does not move or delete memory files.

After weekly processing, `ProceduralDistiller.weekly_pattern_distill` distills recurring patterns from activity logs into procedure candidates. Additionally, if `skill_autolearn_enabled` is enabled, eligible procedures are used to create low-risk probation skills.

## Forgetting Candidates and Thresholds

`core/memory/maintenance/forgetting.py`'s daily downscaling marks regular knowledge and episodes as low-activity if more than 90 days have passed since last use and usage count is fewer than 3. Procedures use a different criterion: they are marked as low-activity if unused for more than 180 days with total usage below 3, or if there are 3 or more failures and utility is below 0.3. Protected or reused memories are excluded from consideration.

Weekly forgetting candidates include memories that have been in a low-activity state for more than 90 days with usage count of 2 or fewer. This is not a deletion command but a candidate list for Anima to decide whether to retain based on the content and surrounding context. Candidate presentation scope and protection rule implementation are handled by `ForgettingEngine`.

## Procedure Usage Tracking and Reconsolidation

After using a procedure, `report_procedure_outcome` records success or failure. The results are used to update `success_count`, `failure_count`, and `confidence` in the frontmatter.

`core/memory/maintenance/reconsolidation.py` marks a procedure as a reconsolidation candidate if it meets `failure_count >= 1` **or** `confidence < 0.6`. When the LLM revises it, the old version is archived, `version` is advanced, and the success and failure counters and confidence are reset to initial values.

## Periodic Update of the RAG Index

The RAG index update is a single schedule separate from the weekly integration. `server/supervisor/_mgr_scheduler.py`'s daily indexing job reflects memory files, facts, and shared data at the default time of 04:00, and also updates the BM25 and entity indexes. There is no path where weekly post-processing separately rebuilds the entire RAG. For the index and recovery paths, refer to [Intentional Recall and Search](retrieval.md).
