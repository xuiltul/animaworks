from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Post-consolidation knowledge self-correction hooks."""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("animaworks.lifecycle")


@dataclass(frozen=True)
class KnowledgeCorrectionLimits:
    """Cost controls for one anima's nightly self-correction batch."""

    max_contradiction_pairs: int = 20
    max_reconsolidation_files: int = 5
    timeout_seconds: float = 300.0
    recent_hours: int = 24
    contradiction_batch_size: int = 20
    contradiction_nli_prefilter_threshold: float | None = 0.70


def _empty_summary() -> dict[str, Any]:
    return {
        "timed_out": False,
        "errors": 0,
        "contradiction": {
            "targets": 0,
            "llm_checks": 0,
            "detected": 0,
            "limit_reached": False,
            "resolved": {"superseded": 0, "merged": 0, "coexisted": 0, "errors": 0},
        },
        "reconsolidation": {
            "knowledge": {"targets_found": 0, "updated": 0, "skipped": 0, "errors": 0, "updated_files": []},
            "procedures": {"targets_found": 0, "updated": 0, "skipped": 0, "errors": 0},
            "procedure_creation": {"created": 0, "skipped": 0, "errors": 0},
        },
    }


async def run_post_consolidation_knowledge_correction(
    anima_dir: Path,
    anima_name: str,
    *,
    model: str,
    limits: KnowledgeCorrectionLimits | None = None,
) -> dict[str, Any]:
    """Run nightly contradiction resolution and reconsolidation.

    The caller owns scheduling. This function commits each successful file
    update immediately, so a timeout preserves already-applied corrections and
    leaves the remaining work for the next nightly batch.
    """
    limits = limits or KnowledgeCorrectionLimits()
    summary = _empty_summary()
    deadline = time.monotonic() + max(0.0, limits.timeout_seconds)

    try:
        await _run_with_remaining_time(
            _run_contradiction_stage(anima_dir, anima_name, model, limits, summary),
            deadline,
        )
        await _run_with_remaining_time(
            _run_reconsolidation_stage(anima_dir, anima_name, model, limits, summary),
            deadline,
        )
    except TimeoutError:
        summary["timed_out"] = True
        logger.warning(
            "Knowledge self-correction timed out for anima=%s; partial summary=%s",
            anima_name,
            summary,
        )
    except Exception:
        summary["errors"] = int(summary["errors"]) + 1
        logger.exception("Knowledge self-correction failed for anima=%s", anima_name)

    logger.info("Knowledge self-correction for %s: %s", anima_name, summary)
    return summary


async def _run_with_remaining_time(coro, deadline: float):  # noqa: ANN001
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        coro.close()
        raise TimeoutError
    return await asyncio.wait_for(coro, timeout=remaining)


async def _run_contradiction_stage(
    anima_dir: Path,
    anima_name: str,
    model: str,
    limits: KnowledgeCorrectionLimits,
    summary: dict[str, Any],
) -> None:
    from core.memory.activity import ActivityLogger
    from core.memory.contradiction import ContradictionDetector, ContradictionPair
    from core.memory.manager import MemoryManager

    mm = MemoryManager(anima_dir)
    detector = ContradictionDetector(
        anima_dir,
        anima_name,
        ActivityLogger(anima_dir),
        memory_manager=mm,
        batch_size=limits.contradiction_batch_size,
        nli_prefilter_threshold=limits.contradiction_nli_prefilter_threshold,
    )

    targets = _recent_active_knowledge_files(mm, anima_dir, recent_hours=limits.recent_hours)
    contradiction_summary = summary["contradiction"]
    contradiction_summary["targets"] = len(targets)
    if not targets or limits.max_contradiction_pairs <= 0:
        contradiction_summary["limit_reached"] = bool(targets and limits.max_contradiction_pairs <= 0)
        return

    pairs_to_resolve: list[ContradictionPair] = await detector.scan_contradictions(
        target_files=targets,
        model=model,
        max_llm_checks=limits.max_contradiction_pairs,
    )
    stats = detector.last_scan_stats
    contradiction_summary["llm_checks"] = int(stats.get("llm_checks", 0) or 0)
    contradiction_summary["limit_reached"] = bool(stats.get("limit_reached", False))

    contradiction_summary["detected"] = len(pairs_to_resolve)
    if pairs_to_resolve:
        contradiction_summary["resolved"] = await detector.resolve_contradictions(pairs_to_resolve, model)


async def _run_reconsolidation_stage(
    anima_dir: Path,
    anima_name: str,
    model: str,
    limits: KnowledgeCorrectionLimits,
    summary: dict[str, Any],
) -> None:
    from core.memory.activity import ActivityLogger
    from core.memory.manager import MemoryManager
    from core.memory.reconsolidation import ReconsolidationEngine

    mm = MemoryManager(anima_dir)
    engine = ReconsolidationEngine(
        anima_dir,
        anima_name,
        memory_manager=mm,
        activity_logger=ActivityLogger(anima_dir),
    )

    remaining_files = max(0, limits.max_reconsolidation_files)
    recon_summary = summary["reconsolidation"]

    knowledge_result = await engine.reconsolidate_knowledge(
        model=model,
        max_files=remaining_files,
    )
    recon_summary["knowledge"] = knowledge_result
    remaining_files = max(0, remaining_files - int(knowledge_result.get("targets_found", 0)))

    procedure_targets = await engine.find_reconsolidation_targets(max_files=remaining_files)
    recon_summary["procedures"]["targets_found"] = len(procedure_targets)
    if procedure_targets:
        recon_summary["procedures"] = {
            "targets_found": len(procedure_targets),
            **await engine.apply_reconsolidation(procedure_targets, model=model),
        }

    recon_summary["procedure_creation"] = await engine.create_procedures_from_resolved(
        model=model,
        days=1,
    )


def _recent_active_knowledge_files(
    memory_manager: Any,
    anima_dir: Path,
    *,
    recent_hours: int,
) -> list[Path]:
    knowledge_dir = anima_dir / "knowledge"
    if not knowledge_dir.exists():
        return []

    cutoff = time.time() - max(0, recent_hours) * 3600
    files: list[Path] = []
    for path in sorted(knowledge_dir.rglob("*.md")):
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                continue
            meta = memory_manager.read_knowledge_metadata(path)
        except OSError:
            logger.warning("Failed to inspect knowledge file for self-correction: %s", path)
            continue
        if meta.get("valid_until"):
            continue
        files.append(path)
    return files
