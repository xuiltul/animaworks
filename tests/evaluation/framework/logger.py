# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Experiment logging and result recording.

This module provides structured logging for memory performance experiments,
recording all metrics to JSON files for later analysis.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ConversationMetrics, TurnMetrics

logger = logging.getLogger(__name__)

# ── Experiment Logger ───────────────────────────────────────────────────────


class ExperimentLogger:
    """Logs experiment results in a structured format.

    This logger creates a hierarchical directory structure:
        output_dir/
        ├── {experiment_id}/
        │   ├── metadata.json         # Experiment configuration
        │   ├── conversations/
        │   │   ├── conv_001.json     # Per-conversation metrics
        │   │   ├── conv_002.json
        │   │   └── ...
        │   ├── turns/
        │   │   ├── turn_001.json     # Per-turn metrics
        │   │   ├── turn_002.json
        │   │   └── ...
        │   └── summary.json          # Aggregated results

    Each JSON file contains:
    - Timestamp
    - Metrics data
    - Metadata (condition, participant_id, etc.)

    Examples:
        >>> logger = ExperimentLogger(
        ...     experiment_id="exp_001",
        ...     output_dir=Path("results/raw"),
        ...     condition="D",
        ...     participant_id=1
        ... )
        >>> logger.start_conversation("conv_001")
        >>> logger.log_turn_metrics(turn_metrics)
        >>> logger.end_conversation(conv_metrics)
    """

    def __init__(
        self,
        experiment_id: str,
        output_dir: Path,
        condition: str,
        participant_id: int,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize experiment logger.

        Args:
            experiment_id: Unique experiment identifier
            output_dir: Base directory for results
            condition: Experimental condition (A/B/C/D)
            participant_id: Participant (agent) ID
            metadata: Additional metadata to include in all logs
        """
        self.experiment_id = experiment_id
        self.condition = condition
        self.participant_id = participant_id
        self.metadata = metadata or {}

        # Create directory structure
        self.experiment_dir = output_dir / experiment_id
        self.conversations_dir = self.experiment_dir / "conversations"
        self.turns_dir = self.experiment_dir / "turns"

        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir.mkdir(exist_ok=True)
        self.turns_dir.mkdir(exist_ok=True)

        # Current conversation tracking
        self.current_conversation_id: str | None = None
        self.current_conversation_start: float | None = None
        self.current_turn_metrics: list[TurnMetrics] = []

        # Save metadata
        self._save_metadata()

        logger.info(
            f"ExperimentLogger initialized: {experiment_id}, "
            f"condition={condition}, participant={participant_id}"
        )

    def _save_metadata(self) -> None:
        """Save experiment metadata to file."""
        metadata_file = self.experiment_dir / "metadata.json"
        data = {
            "experiment_id": self.experiment_id,
            "condition": self.condition,
            "participant_id": self.participant_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **self.metadata,
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _create_log_entry(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create a log entry with standard metadata.

        Args:
            data: Log data

        Returns:
            Log entry with metadata
        """
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_id": self.experiment_id,
            "condition": self.condition,
            "participant_id": self.participant_id,
            **data,
        }

    # ── Conversation-level Logging ──────────────────────────────────────────

    def start_conversation(self, conversation_id: str) -> None:
        """Start a new conversation session.

        Args:
            conversation_id: Unique conversation identifier
        """
        if self.current_conversation_id is not None:
            logger.warning(
                f"Starting new conversation {conversation_id} "
                f"while {self.current_conversation_id} is still active"
            )

        self.current_conversation_id = conversation_id
        self.current_conversation_start = datetime.now(timezone.utc).timestamp()
        self.current_turn_metrics = []

        logger.info(f"Started conversation: {conversation_id}")

    def end_conversation(self, conv_metrics: ConversationMetrics) -> None:
        """End current conversation and save metrics.

        Args:
            conv_metrics: Aggregated conversation metrics
        """
        if self.current_conversation_id is None:
            logger.warning("end_conversation called without active conversation")
            return

        # Save conversation metrics
        conv_file = self.conversations_dir / f"{self.current_conversation_id}.json"
        entry = self._create_log_entry(
            {
                "conversation_id": conv_metrics.conversation_id,
                "scenario_type": conv_metrics.scenario_type.value,
                "turn_count": conv_metrics.turn_count,
                "total_latency_ms": conv_metrics.total_latency,
                "avg_turn_latency_ms": conv_metrics.avg_turn_latency,
                "p50_latency_ms": conv_metrics.p50_latency,
                "p95_latency_ms": conv_metrics.p95_latency,
                "p99_latency_ms": conv_metrics.p99_latency,
                "total_tool_calls": conv_metrics.total_tool_calls,
                "avg_tool_calls_per_turn": conv_metrics.avg_tool_calls_per_turn,
                "total_tokens": conv_metrics.total_tokens,
                "avg_precision": conv_metrics.avg_precision,
                "avg_recall": conv_metrics.avg_recall,
                "avg_f1": conv_metrics.avg_f1,
            }
        )

        with open(conv_file, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Ended conversation: {self.current_conversation_id}, "
            f"turns={conv_metrics.turn_count}, "
            f"avg_latency={conv_metrics.avg_turn_latency:.1f}ms"
        )

        # Reset state
        self.current_conversation_id = None
        self.current_conversation_start = None
        self.current_turn_metrics = []

    # ── Turn-level Logging ──────────────────────────────────────────────────

    def log_turn_metrics(self, turn_metrics: TurnMetrics) -> None:
        """Log metrics for a single conversation turn.

        Args:
            turn_metrics: Turn-level metrics
        """
        if self.current_conversation_id is None:
            logger.warning("log_turn_metrics called without active conversation")
            return

        # Save turn metrics to file
        turn_file = self.turns_dir / f"{turn_metrics.turn_id}.json"
        entry = self._create_log_entry(
            {
                "conversation_id": self.current_conversation_id,
                "turn_id": turn_metrics.turn_id,
                "priming_latency_ms": turn_metrics.priming_latency,
                "priming_tokens": turn_metrics.priming_tokens,
                "response_latency_ms": turn_metrics.response_latency,
                "llm_latency_ms": turn_metrics.llm_latency,
                "tool_call_count": turn_metrics.tool_call_count,
                "total_tokens": turn_metrics.total_tokens,
                "search_tokens": turn_metrics.search_tokens,
                "search_precision": turn_metrics.search_precision,
                "search_recall": turn_metrics.search_recall,
                "search_f1": turn_metrics.search_f1,
            }
        )

        with open(turn_file, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)

        # Track for conversation summary
        self.current_turn_metrics.append(turn_metrics)

        logger.debug(
            f"Logged turn {turn_metrics.turn_id}: "
            f"latency={turn_metrics.response_latency:.1f}ms, "
            f"tools={turn_metrics.tool_call_count}"
        )

    # ── Component-level Logging ─────────────────────────────────────────────

    def log_priming(self, turn_id: str, latency_ms: float, tokens: int) -> None:
        """Log priming layer metrics.

        Args:
            turn_id: Turn identifier
            latency_ms: Priming latency in milliseconds
            tokens: Number of tokens in priming result
        """
        entry = self._create_log_entry(
            {
                "conversation_id": self.current_conversation_id,
                "turn_id": turn_id,
                "component": "priming",
                "latency_ms": latency_ms,
                "tokens": tokens,
            }
        )

        # Append to priming log
        priming_log = self.experiment_dir / "priming.jsonl"
        with open(priming_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_search(
        self,
        turn_id: str,
        latency_ms: float,
        precision: float,
        recall: float,
        f1: float,
        retrieved_count: int,
    ) -> None:
        """Log search operation metrics.

        Args:
            turn_id: Turn identifier
            latency_ms: Search latency in milliseconds
            precision: Precision@k
            recall: Recall@k
            f1: F1@k
            retrieved_count: Number of documents retrieved
        """
        entry = self._create_log_entry(
            {
                "conversation_id": self.current_conversation_id,
                "turn_id": turn_id,
                "component": "search",
                "latency_ms": latency_ms,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "retrieved_count": retrieved_count,
            }
        )

        # Append to search log
        search_log = self.experiment_dir / "search.jsonl"
        with open(search_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_error(self, error_type: str, error_message: str, context: dict[str, Any] | None = None) -> None:
        """Log an error that occurred during the experiment.

        Args:
            error_type: Type of error
            error_message: Error message
            context: Additional context information
        """
        entry = self._create_log_entry(
            {
                "conversation_id": self.current_conversation_id,
                "error_type": error_type,
                "error_message": error_message,
                "context": context or {},
            }
        )

        # Append to error log
        error_log = self.experiment_dir / "errors.jsonl"
        with open(error_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.error(f"Logged error: {error_type} - {error_message}")

    # ── Summary Generation ──────────────────────────────────────────────────

    def generate_summary(self) -> dict[str, Any]:
        """Generate summary statistics for all logged data.

        Returns:
            Summary dictionary

        Notes:
            This should be called after all conversations are complete.
        """
        # Load all conversation metrics
        conv_files = list(self.conversations_dir.glob("*.json"))
        conversations = []
        for conv_file in conv_files:
            with open(conv_file, encoding="utf-8") as f:
                conversations.append(json.load(f))

        if not conversations:
            logger.warning("No conversations to summarize")
            return {}

        # Calculate aggregated statistics
        summary = {
            "experiment_id": self.experiment_id,
            "condition": self.condition,
            "participant_id": self.participant_id,
            "total_conversations": len(conversations),
            "total_turns": sum(c["turn_count"] for c in conversations),
            "avg_latency_ms": sum(c["avg_turn_latency_ms"] for c in conversations) / len(conversations),
            "avg_tool_calls": sum(c["avg_tool_calls_per_turn"] for c in conversations) / len(conversations),
            "total_tokens": sum(c["total_tokens"] for c in conversations),
        }

        # Save summary
        summary_file = self.experiment_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Generated summary: {summary_file}")
        return summary
