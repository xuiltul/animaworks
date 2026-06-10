from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import load_config
from core.config.models import ConsolidationConfig

logger = logging.getLogger("animaworks.lifecycle")


@dataclass(frozen=True)
class DailyConsolidationGate:
    """Decision details for daily consolidation eligibility."""

    should_run: bool
    activity_count: int
    episode_count: int
    carryover_count: int
    threshold: int


def evaluate_daily_consolidation_gate(
    anima_dir: Path,
    anima_name: str,
    *,
    threshold: int,
    hours: int = 24,
) -> DailyConsolidationGate:
    """Return whether daily consolidation should run for one anima."""
    from core.memory.consolidation import ConsolidationEngine

    engine = ConsolidationEngine(anima_dir, anima_name)
    episode_count = 0
    activity_count = 0
    try:
        episode_count = len(engine._collect_recent_episodes(hours=hours))
    except Exception:
        logger.debug("Failed to count recent episodes for %s", anima_name, exc_info=True)
    try:
        _target_date, window_start, window_end = engine.previous_local_day_window()
        activity_count = engine.count_recent_activity_entries(
            hours=hours,
            since=window_start,
            until=window_end,
        )
    except Exception:
        logger.debug("Failed to count recent activity entries for %s", anima_name, exc_info=True)
    carryover_count = engine.count_pending_phase_b_carryover()
    return DailyConsolidationGate(
        should_run=activity_count >= threshold or episode_count >= threshold or carryover_count > 0,
        activity_count=activity_count,
        episode_count=episode_count,
        carryover_count=carryover_count,
        threshold=threshold,
    )


async def run_daily_consolidation_post_processing(
    anima_name: str,
    anima_dir: Path,
    *,
    consolidation_cfg: Any,
    model: str,
) -> None:
    """Run framework-side daily consolidation post-processing."""
    try:
        from core.memory.forgetting import ForgettingEngine

        forgetter = ForgettingEngine(anima_dir, anima_name)
        downscaling_result = forgetter.synaptic_downscaling()
        logger.info("Synaptic downscaling for %s: %s", anima_name, downscaling_result)
    except Exception:
        logger.exception("Synaptic downscaling failed for anima=%s", anima_name)

    await run_knowledge_self_correction_if_enabled(
        anima_dir,
        anima_name,
        consolidation_cfg,
        model=model,
    )

    try:
        from core.memory.consolidation import ConsolidationEngine

        engine = ConsolidationEngine(anima_dir, anima_name)
        engine._rebuild_rag_index()
    except Exception:
        logger.exception("RAG index rebuild failed for anima=%s", anima_name)

    try:
        from core.memory.consolidation import ConsolidationEngine

        engine = ConsolidationEngine(anima_dir, anima_name)
        await engine.ingest_recent_to_backend(hours=48)
    except Exception:
        logger.exception("Neo4j ingest failed for anima=%s", anima_name)

    await detect_communities_if_neo4j(anima_dir, anima_name)


async def run_weekly_integration_post_processing(
    anima_name: str,
    anima_dir: Path,
    *,
    consolidation_cfg: Any,
    model: str,
) -> None:
    """Run framework-side weekly integration post-processing."""
    try:
        from core.memory.forgetting import ForgettingEngine

        forgetter = ForgettingEngine(anima_dir, anima_name)
        reorg_result = await forgetter.neurogenesis_reorganize(model=model)
        logger.info("Neurogenesis reorganization for %s: %s", anima_name, reorg_result)
    except Exception:
        logger.exception("Neurogenesis reorganization failed for anima=%s", anima_name)

    try:
        from core.memory.consolidation import ConsolidationEngine

        engine = ConsolidationEngine(anima_dir, anima_name)
        engine._rebuild_rag_index()
    except Exception:
        logger.exception("RAG index rebuild failed for anima=%s", anima_name)

    try:
        from core.memory.consolidation import ConsolidationEngine

        engine = ConsolidationEngine(anima_dir, anima_name)
        await engine.ingest_recent_to_backend(hours=168)
    except Exception:
        logger.exception("Neo4j ingest failed for anima=%s", anima_name)

    await detect_communities_if_neo4j(anima_dir, anima_name)


async def run_knowledge_self_correction_if_enabled(
    anima_dir: Path,
    anima_name: str,
    consolidation_cfg: Any,
    *,
    model: str,
) -> None:
    """Run post-consolidation knowledge correction when enabled."""
    default_cfg = ConsolidationConfig()
    if consolidation_cfg is not None:
        enabled = getattr(consolidation_cfg, "knowledge_self_correction_enabled", True)
    else:
        enabled = default_cfg.knowledge_self_correction_enabled
    if not enabled:
        return

    try:
        from core.lifecycle.knowledge_correction import (
            KnowledgeCorrectionLimits,
            run_post_consolidation_knowledge_correction,
        )

        limits = KnowledgeCorrectionLimits(
            max_contradiction_pairs=getattr(
                consolidation_cfg,
                "knowledge_self_correction_max_contradiction_pairs",
                default_cfg.knowledge_self_correction_max_contradiction_pairs,
            ),
            max_reconsolidation_files=getattr(
                consolidation_cfg,
                "knowledge_self_correction_max_reconsolidation_files",
                default_cfg.knowledge_self_correction_max_reconsolidation_files,
            ),
            timeout_seconds=float(
                getattr(
                    consolidation_cfg,
                    "knowledge_self_correction_timeout_seconds",
                    default_cfg.knowledge_self_correction_timeout_seconds,
                )
            ),
            recent_hours=getattr(
                consolidation_cfg,
                "knowledge_self_correction_recent_hours",
                default_cfg.knowledge_self_correction_recent_hours,
            ),
        )
        result = await run_post_consolidation_knowledge_correction(
            anima_dir,
            anima_name,
            model=model,
            limits=limits,
        )
        logger.info("Knowledge self-correction post-processing for %s: %s", anima_name, result)
    except Exception:
        logger.exception("Knowledge self-correction failed for anima=%s", anima_name)


async def detect_communities_if_neo4j(anima_dir: Path, anima_name: str) -> None:
    """Run batch community detection if the anima uses the Neo4j backend."""
    backend = None
    try:
        from core.memory.backend.registry import get_backend, resolve_backend_type

        backend_type = resolve_backend_type(anima_dir)
        if backend_type != "neo4j":
            return

        backend = get_backend(backend_type, anima_dir)
        driver = await backend._ensure_driver()

        from core.memory.graph.community import CommunityDetector

        detector = CommunityDetector(
            driver,
            backend._group_id,
            model=backend._resolve_background_model(),
            locale=backend._resolve_locale(),
        )
        communities = await detector.detect_and_store()
        stats = await detector.get_community_stats()
        logger.info(
            "Community detection for %s: detected=%d stored=%d memberships=%d",
            anima_name,
            len(communities),
            stats["communities"],
            stats["memberships"],
        )
    except Exception:
        logger.exception("Community detection failed for anima=%s", anima_name)
    finally:
        if backend is not None:
            try:
                await backend.close()
            except Exception:
                logger.debug("Failed to close Neo4j backend after community detection", exc_info=True)


class SystemConsolidationMixin:
    """Mixin providing daily/weekly/monthly consolidation handlers."""

    async def _handle_daily_consolidation(self) -> None:
        """Run daily consolidation for all animas.

        New flow: Anima-driven consolidation via run_consolidation(),
        followed by framework-side post-processing.
        """
        logger.info("Starting system-wide daily consolidation")

        config = load_config()
        consolidation_cfg = getattr(config, "consolidation", None)

        # Default config if not present
        enabled = True
        min_episodes = ConsolidationConfig().min_episodes_threshold
        max_turns = ConsolidationConfig().max_turns
        model = ConsolidationConfig().llm_model

        if consolidation_cfg:
            enabled = getattr(consolidation_cfg, "daily_enabled", True)
            min_episodes = getattr(consolidation_cfg, "min_episodes_threshold", min_episodes)
            max_turns = getattr(consolidation_cfg, "max_turns", max_turns)
            model = getattr(consolidation_cfg, "llm_model", model)

        if not enabled:
            logger.info("Daily consolidation is disabled in config")
            return

        for anima_name, anima in self.animas.items():
            gate = evaluate_daily_consolidation_gate(
                anima.memory.anima_dir,
                anima_name,
                threshold=min_episodes,
                hours=24,
            )
            if not gate.should_run:
                logger.info(
                    "Daily consolidation skipped for %s: activity=%d episodes=%d carryover=%d threshold=%d",
                    anima_name,
                    gate.activity_count,
                    gate.episode_count,
                    gate.carryover_count,
                    gate.threshold,
                )
                continue

            result = None
            should_retry = False
            try:
                result = await anima.run_consolidation(
                    consolidation_type="daily",
                    max_turns=max_turns,
                )

                if result.duration_ms < 10_000:
                    logger.warning(
                        "Daily consolidation too short for %s (%dms), scheduling retry",
                        anima_name,
                        result.duration_ms,
                    )
                    should_retry = True

                logger.info(
                    "Daily consolidation for %s: duration_ms=%d",
                    anima_name,
                    result.duration_ms,
                )
            except TimeoutError:
                should_retry = True
                logger.warning(
                    "consolidation_timeout anima=%s phase=phase_b type=daily",
                    anima_name,
                )
            except Exception:
                should_retry = True
                logger.exception("Daily consolidation failed for anima=%s", anima_name)
            finally:
                await run_daily_consolidation_post_processing(
                    anima_name,
                    anima.memory.anima_dir,
                    consolidation_cfg=consolidation_cfg,
                    model=model,
                )

            if should_retry:
                self._schedule_consolidation_retry(anima_name, max_turns)

            if self._ws_broadcast:
                await self._ws_broadcast(
                    {
                        "type": "system.consolidation",
                        "data": {
                            "anima": anima_name,
                            "type": "daily",
                            "summary": result.summary[:500] if result else "",
                            "duration_ms": result.duration_ms if result else 0,
                        },
                    }
                )

    async def _handle_weekly_integration(self) -> None:
        """Run weekly integration for all animas.

        New flow: Anima-driven consolidation via run_consolidation(),
        followed by framework-side post-processing.
        """
        logger.info("Starting system-wide weekly integration")

        config = load_config()
        consolidation_cfg = getattr(config, "consolidation", None)

        # Default config
        enabled = True
        model = ConsolidationConfig().llm_model
        max_turns = ConsolidationConfig().max_turns

        if consolidation_cfg:
            enabled = getattr(consolidation_cfg, "weekly_enabled", True)
            model = getattr(consolidation_cfg, "llm_model", model)
            max_turns = getattr(consolidation_cfg, "max_turns", max_turns)

        if not enabled:
            logger.info("Weekly integration is disabled in config")
            return

        for anima_name, anima in self.animas.items():
            result = None
            try:
                result = await anima.run_consolidation(
                    consolidation_type="weekly",
                    max_turns=max_turns,
                )

                logger.info(
                    "Weekly integration for %s: duration_ms=%d",
                    anima_name,
                    result.duration_ms,
                )
            except TimeoutError:
                logger.warning(
                    "consolidation_timeout anima=%s phase=phase_b type=weekly",
                    anima_name,
                )
            except Exception:
                logger.exception("Weekly integration failed for anima=%s", anima_name)
            finally:
                await run_weekly_integration_post_processing(
                    anima_name,
                    anima.memory.anima_dir,
                    consolidation_cfg=consolidation_cfg,
                    model=model,
                )

            if self._ws_broadcast:
                await self._ws_broadcast(
                    {
                        "type": "system.consolidation",
                        "data": {
                            "anima": anima_name,
                            "type": "weekly",
                            "summary": result.summary[:500] if result else "",
                            "duration_ms": result.duration_ms if result else 0,
                        },
                    }
                )

    async def _handle_monthly_forgetting(self) -> None:
        """Run monthly forgetting for all animas."""
        logger.info("Starting system-wide monthly forgetting")

        config = load_config()
        consolidation_cfg = getattr(config, "consolidation", None)

        # Default config
        enabled = True

        if consolidation_cfg:
            enabled = getattr(consolidation_cfg, "monthly_forgetting_enabled", True)

        if not enabled:
            logger.info("Monthly forgetting is disabled in config")
            return

        # Run forgetting for each anima
        for anima_name, anima in self.animas.items():
            try:
                from core.memory.consolidation import ConsolidationEngine

                engine = ConsolidationEngine(
                    anima_dir=anima.memory.anima_dir,
                    anima_name=anima_name,
                )

                result = await engine.monthly_forget()

                logger.info(
                    "Monthly forgetting for %s: forgotten=%d archived=%d",
                    anima_name,
                    result.get("forgotten_chunks", 0),
                    len(result.get("archived_files", [])),
                )

                # Broadcast result
                if self._ws_broadcast:
                    await self._ws_broadcast(
                        {
                            "type": "system.consolidation",
                            "data": {
                                "anima": anima_name,
                                "type": "monthly_forgetting",
                                "result": result,
                            },
                        }
                    )

            except Exception:
                logger.exception("Monthly forgetting failed for anima=%s", anima_name)

    # ── Community detection helper ────────────────────────────

    @staticmethod
    async def _run_knowledge_self_correction_if_enabled(
        anima,  # noqa: ANN001
        anima_name: str,
        consolidation_cfg,
        *,
        model: str,
    ) -> None:
        await run_knowledge_self_correction_if_enabled(
            anima.memory.anima_dir,
            anima_name,
            consolidation_cfg,
            model=model,
        )

    @staticmethod
    async def _detect_communities_if_neo4j(anima) -> None:  # noqa: ANN001
        """Run batch community detection if Neo4j backend is active."""
        await detect_communities_if_neo4j(anima.memory.anima_dir, anima.name)
