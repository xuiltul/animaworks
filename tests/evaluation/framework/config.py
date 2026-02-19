# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Experiment configuration data models.

This module defines the configuration structures for memory performance experiments
as specified in the evaluation protocol.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

# ── Enumerations ────────────────────────────────────────────────────────────


class MemorySize(str, Enum):
    """Memory base size categories."""

    SMALL = "small"  # 50 files, ~50K tokens
    MEDIUM = "medium"  # 500 files, ~500K tokens
    LARGE = "large"  # 5000 files, ~5M tokens


class SearchMethod(str, Enum):
    """Search method types."""

    BM25 = "bm25"  # BM25 only (baseline)
    VECTOR = "vector"  # Vector search only
    HYBRID = "hybrid"  # Hybrid search (BM25 + Vector + Recency)
    HYBRID_PRIMING = "hybrid_priming"  # Hybrid + Priming layer


class ConversationLength(str, Enum):
    """Conversation length categories."""

    SHORT = "short"  # 5 turns
    MEDIUM = "medium"  # 20 turns
    LONG = "long"  # 50 turns


class ScenarioType(str, Enum):
    """Conversation scenario types."""

    FACTUAL_RECALL = "factual_recall"
    EPISODIC_RECALL = "episodic_recall"
    MULTIHOP_REASONING = "multihop_reasoning"
    LONG_CONVERSATION = "long_conversation"


class Domain(str, Enum):
    """Memory domain types."""

    BUSINESS = "business"
    TECH_SUPPORT = "tech_support"
    EDUCATION = "education"


# ── Configuration Classes ───────────────────────────────────────────────────


@dataclass
class SearchConfig:
    """Search configuration for each experimental condition.

    Attributes:
        method: Search method type (bm25, vector, hybrid, hybrid_priming)
        top_k: Number of top results to retrieve
        priming_enabled: Whether to enable priming layer (Condition D only)
        priming_budget: Token budget for priming (default: 2000)
        weights: Score weights for hybrid search (vector, bm25, recency)
        vector_model: Embedding model name (default: multilingual-e5-small)
    """

    method: SearchMethod
    top_k: int = 3
    priming_enabled: bool = False
    priming_budget: int = 2000
    weights: dict[str, float] = field(
        default_factory=lambda: {"vector": 0.5, "bm25": 0.3, "recency": 0.2}
    )
    vector_model: str = "multilingual-e5-small"

    def __post_init__(self):
        """Validate configuration."""
        if self.method == SearchMethod.HYBRID_PRIMING:
            self.priming_enabled = True

        if self.priming_enabled and self.priming_budget <= 0:
            raise ValueError("priming_budget must be positive when priming is enabled")

        if self.method in [SearchMethod.HYBRID, SearchMethod.HYBRID_PRIMING]:
            # Validate weights sum to 1.0
            total = sum(self.weights.values())
            if not (0.99 <= total <= 1.01):
                raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class ExperimentConfig:
    """Main experiment configuration.

    Attributes:
        experiment_id: Unique identifier for this experiment
        condition: Experimental condition (A/B/C/D)
        participants: Number of agent participants
        memory_size: Size of memory base (small/medium/large)
        conversation_length: Length of conversations (short/medium/long)
        domain: Memory domain (business/tech_support/education)
        search_config: Search configuration for this condition
        model_name: LLM model name (default: claude-sonnet-4)
        temperature: LLM temperature (default: 0.7)
        context_window: Context window size in tokens (default: 200000)
        random_seed: Random seed for reproducibility
    """

    experiment_id: str
    condition: Literal["A", "B", "C", "D"]
    participants: int
    memory_size: MemorySize
    conversation_length: ConversationLength
    domain: Domain
    search_config: SearchConfig
    model_name: str = "claude-sonnet-4"
    temperature: float = 0.7
    context_window: int = 200000
    random_seed: int | None = None

    def __post_init__(self):
        """Validate experiment configuration."""
        if self.participants <= 0:
            raise ValueError("participants must be positive")

        if not (0.0 <= self.temperature <= 1.0):
            raise ValueError("temperature must be in [0.0, 1.0]")

        if self.context_window <= 0:
            raise ValueError("context_window must be positive")

        # Validate condition matches search method
        condition_method_map = {
            "A": SearchMethod.BM25,
            "B": SearchMethod.VECTOR,
            "C": SearchMethod.HYBRID,
            "D": SearchMethod.HYBRID_PRIMING,
        }
        expected_method = condition_method_map[self.condition]
        if self.search_config.method != expected_method:
            raise ValueError(
                f"Condition {self.condition} requires method {expected_method.value}, "
                f"got {self.search_config.method.value}"
            )

    @classmethod
    def create_condition_a(
        cls,
        experiment_id: str,
        participants: int = 30,
        memory_size: MemorySize = MemorySize.MEDIUM,
        conversation_length: ConversationLength = ConversationLength.MEDIUM,
        domain: Domain = Domain.BUSINESS,
    ) -> ExperimentConfig:
        """Create configuration for Condition A (BM25 only)."""
        return cls(
            experiment_id=experiment_id,
            condition="A",
            participants=participants,
            memory_size=memory_size,
            conversation_length=conversation_length,
            domain=domain,
            search_config=SearchConfig(method=SearchMethod.BM25, top_k=3),
        )

    @classmethod
    def create_condition_b(
        cls,
        experiment_id: str,
        participants: int = 30,
        memory_size: MemorySize = MemorySize.MEDIUM,
        conversation_length: ConversationLength = ConversationLength.MEDIUM,
        domain: Domain = Domain.BUSINESS,
    ) -> ExperimentConfig:
        """Create configuration for Condition B (Vector only)."""
        return cls(
            experiment_id=experiment_id,
            condition="B",
            participants=participants,
            memory_size=memory_size,
            conversation_length=conversation_length,
            domain=domain,
            search_config=SearchConfig(method=SearchMethod.VECTOR, top_k=3),
        )

    @classmethod
    def create_condition_c(
        cls,
        experiment_id: str,
        participants: int = 30,
        memory_size: MemorySize = MemorySize.MEDIUM,
        conversation_length: ConversationLength = ConversationLength.MEDIUM,
        domain: Domain = Domain.BUSINESS,
    ) -> ExperimentConfig:
        """Create configuration for Condition C (Hybrid)."""
        return cls(
            experiment_id=experiment_id,
            condition="C",
            participants=participants,
            memory_size=memory_size,
            conversation_length=conversation_length,
            domain=domain,
            search_config=SearchConfig(
                method=SearchMethod.HYBRID,
                top_k=3,
                weights={"vector": 0.5, "bm25": 0.3, "recency": 0.2},
            ),
        )

    @classmethod
    def create_condition_d(
        cls,
        experiment_id: str,
        participants: int = 30,
        memory_size: MemorySize = MemorySize.MEDIUM,
        conversation_length: ConversationLength = ConversationLength.MEDIUM,
        domain: Domain = Domain.BUSINESS,
        priming_budget: int = 2000,
    ) -> ExperimentConfig:
        """Create configuration for Condition D (Hybrid + Priming)."""
        return cls(
            experiment_id=experiment_id,
            condition="D",
            participants=participants,
            memory_size=memory_size,
            conversation_length=conversation_length,
            domain=domain,
            search_config=SearchConfig(
                method=SearchMethod.HYBRID_PRIMING,
                top_k=3,
                priming_enabled=True,
                priming_budget=priming_budget,
                weights={"vector": 0.5, "bm25": 0.3, "recency": 0.2},
            ),
        )


@dataclass
class TurnMetrics:
    """Metrics collected for a single conversation turn.

    Attributes:
        turn_id: Unique turn identifier
        priming_latency: Time spent on priming (ms), None if not applicable
        priming_tokens: Tokens used in priming, None if not applicable
        response_latency: Total response generation time (ms)
        llm_latency: LLM inference time (ms)
        tool_call_count: Number of search_memory tool calls
        total_tokens: Total tokens in context
        search_tokens: Tokens from search results
        search_precision: Precision@k, None if no search
        search_recall: Recall@k, None if no search
        search_f1: F1@k, None if no search
    """

    turn_id: str
    priming_latency: float | None = None
    priming_tokens: int | None = None
    response_latency: float = 0.0
    llm_latency: float = 0.0
    tool_call_count: int = 0
    total_tokens: int = 0
    search_tokens: int = 0
    search_precision: float | None = None
    search_recall: float | None = None
    search_f1: float | None = None


@dataclass
class ConversationMetrics:
    """Aggregated metrics for a complete conversation.

    Attributes:
        conversation_id: Unique conversation identifier
        scenario_type: Type of scenario
        turn_count: Number of turns
        total_latency: Total conversation time (ms)
        avg_turn_latency: Average latency per turn (ms)
        p50_latency: Median latency (ms)
        p95_latency: 95th percentile latency (ms)
        p99_latency: 99th percentile latency (ms)
        total_tool_calls: Total search_memory calls
        avg_tool_calls_per_turn: Average tool calls per turn
        total_tokens: Total tokens consumed
        avg_precision: Average precision across turns with search
        avg_recall: Average recall across turns with search
        avg_f1: Average F1 across turns with search
    """

    conversation_id: str
    scenario_type: ScenarioType
    turn_count: int
    total_latency: float
    avg_turn_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    total_tool_calls: int
    avg_tool_calls_per_turn: float
    total_tokens: int
    avg_precision: float | None = None
    avg_recall: float | None = None
    avg_f1: float | None = None
