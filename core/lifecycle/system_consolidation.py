from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.
import logging

from core.config import load_config
from core.config.models import ConsolidationConfig

logger = logging.getLogger("animaworks.lifecycle")


class SystemConsolidationMixin:
    """Mixin providing daily/weekly/monthly consolidation handlers."""

    async def _handle_daily_consolidation(self) -> None:
        """Run daily consolidation for all animas.

        New flow: Anima-driven consolidation via run_consolidation(),
        followed by metadata-based synaptic downscaling and RAG rebuild.
        """
        logger.info("Starting system-wide daily consolidation")

        config = load_config()
        consolidation_cfg = getattr(config, "consolidation", None)

        # Default config if not present
        enabled = True
        min_episodes = ConsolidationConfig().min_episodes_threshold
        max_turns = ConsolidationConfig().max_turns

        if consolidation_cfg:
            enabled = getattr(consolidation_cfg, "daily_enabled", True)
            min_episodes = getattr(consolidation_cfg, "min_episodes_threshold", min_episodes)
            max_turns = getattr(consolidation_cfg, "max_turns", max_turns)

        if not enabled:
            logger.info("Daily consolidation is disabled in config")
            return

        # Run consolidation for each anima
        for anima_name, anima in self.animas.items():
            try:
                # Check if anima has recent episodes worth consolidating
                episode_count = anima.count_recent_episodes(hours=24)
                if episode_count < min_episodes:
                    logger.info(
                        "Daily consolidation skipped for %s: episodes=%d < threshold=%d",
                        anima_name,
                        episode_count,
                        min_episodes,
                    )
                    continue

                # Anima-driven consolidation (tool-call loop)
                result = await anima.run_consolidation(
                    consolidation_type="daily",
                    max_turns=max_turns,
                )

                # Failure detection: extremely short execution suggests rate limit or error
                if result.duration_ms < 10_000:
                    logger.warning(
                        "Daily consolidation too short for %s (%dms), scheduling retry",
                        anima_name,
                        result.duration_ms,
                    )
                    self._schedule_consolidation_retry(anima_name, max_turns)
                    continue

                logger.info(
                    "Daily consolidation for %s: duration_ms=%d",
                    anima_name,
                    result.duration_ms,
                )

                # Post-processing: Synaptic downscaling (metadata-based, no LLM)
                try:
                    from core.memory.forgetting import ForgettingEngine

                    forgetter = ForgettingEngine(anima.memory.anima_dir, anima_name)
                    downscaling_result = forgetter.synaptic_downscaling()
                    logger.info(
                        "Synaptic downscaling for %s: %s",
                        anima_name,
                        downscaling_result,
                    )
                except Exception:
                    logger.exception(
                        "Synaptic downscaling failed for anima=%s",
                        anima_name,
                    )

                # Post-processing: Rebuild RAG index
                try:
                    from core.memory.consolidation import ConsolidationEngine

                    engine = ConsolidationEngine(anima.memory.anima_dir, anima_name)
                    engine._rebuild_rag_index()
                except Exception:
                    logger.exception(
                        "RAG index rebuild failed for anima=%s",
                        anima_name,
                    )

                # Post-processing: Neo4j backend ingest
                try:
                    from core.memory.consolidation import ConsolidationEngine as _CE

                    neo4j_engine = _CE(anima.memory.anima_dir, anima_name)
                    await neo4j_engine.ingest_recent_to_backend(hours=48)
                except Exception:
                    logger.exception(
                        "Neo4j ingest failed for anima=%s",
                        anima_name,
                    )

                # Post-processing: Community detection (Neo4j only)
                await self._detect_communities_if_neo4j(anima)

                # Broadcast result via WebSocket
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

            except Exception:
                logger.exception("Daily consolidation failed for anima=%s", anima_name)
                self._schedule_consolidation_retry(anima_name, max_turns)

    async def _handle_weekly_integration(self) -> None:
        """Run weekly integration for all animas.

        New flow: Anima-driven consolidation via run_consolidation(),
        followed by neurogenesis reorganization and RAG rebuild.
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

        # Run integration for each anima
        for anima_name, anima in self.animas.items():
            try:
                # Anima-driven consolidation (tool-call loop)
                result = await anima.run_consolidation(
                    consolidation_type="weekly",
                    max_turns=max_turns,
                )

                logger.info(
                    "Weekly integration for %s: duration_ms=%d",
                    anima_name,
                    result.duration_ms,
                )

                # Post-processing: Neurogenesis reorganization (LLM-based merge)
                try:
                    from core.memory.forgetting import ForgettingEngine

                    forgetter = ForgettingEngine(anima.memory.anima_dir, anima_name)
                    reorg_result = await forgetter.neurogenesis_reorganize(model=model)
                    logger.info(
                        "Neurogenesis reorganization for %s: %s",
                        anima_name,
                        reorg_result,
                    )
                except Exception:
                    logger.exception(
                        "Neurogenesis reorganization failed for anima=%s",
                        anima_name,
                    )

                # Post-processing: Rebuild RAG index
                try:
                    from core.memory.consolidation import ConsolidationEngine

                    engine = ConsolidationEngine(anima.memory.anima_dir, anima_name)
                    engine._rebuild_rag_index()
                except Exception:
                    logger.exception(
                        "RAG index rebuild failed for anima=%s",
                        anima_name,
                    )

                # Post-processing: Neo4j backend ingest
                try:
                    from core.memory.consolidation import ConsolidationEngine as _CE

                    neo4j_engine = _CE(anima.memory.anima_dir, anima_name)
                    await neo4j_engine.ingest_recent_to_backend(hours=168)
                except Exception:
                    logger.exception(
                        "Neo4j ingest failed for anima=%s",
                        anima_name,
                    )

                # Post-processing: Community detection (Neo4j only)
                await self._detect_communities_if_neo4j(anima)

                # Broadcast result via WebSocket
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

            except Exception:
                logger.exception("Weekly integration failed for anima=%s", anima_name)

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
    async def _detect_communities_if_neo4j(anima) -> None:  # noqa: ANN001
        """Run batch community detection if Neo4j backend is active."""
        try:
            from core.config.models import load_config

            cfg = load_config()
            mem_cfg = getattr(cfg, "memory", None)
            if not mem_cfg or getattr(mem_cfg, "backend", "legacy") != "neo4j":
                return

            from core.memory.backend.registry import get_backend

            backend = get_backend("neo4j", anima.memory.anima_dir)
            driver = await backend._ensure_driver()

            from core.memory.graph.community import CommunityDetector

            detector = CommunityDetector(
                driver,
                backend._group_id,
                model=backend._resolve_background_model(),
                locale=backend._resolve_locale(),
            )
            communities = await detector.detect_and_store()
            logger.info(
                "Community detection for %s: %d communities",
                anima.name,
                len(communities),
            )
            await backend.close()
        except Exception:
            logger.exception("Community detection failed for anima=%s", anima.name)
