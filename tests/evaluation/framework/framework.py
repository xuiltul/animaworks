# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Main experiment framework for memory performance evaluation.

This module provides the MemoryExperiment class that orchestrates the entire
experimental workflow including:
- Agent initialization with different search configurations
- Scenario execution
- Metrics collection
- Result logging
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    ConversationLength,
    ConversationMetrics,
    ExperimentConfig,
    ScenarioType,
    TurnMetrics,
)
from .logger import ExperimentLogger
from .metrics import MetricsCollector

logger = logging.getLogger(__name__)

# ── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class Turn:
    """A single conversation turn.

    Attributes:
        turn_id: Unique turn identifier
        message: User message
        relevant_memories: Ground truth relevant memory file paths
        expected_answer: Expected answer content (for validation)
    """

    turn_id: str
    message: str
    relevant_memories: list[str]
    expected_answer: str | None = None


@dataclass
class Scenario:
    """A conversation scenario.

    Attributes:
        scenario_id: Unique scenario identifier
        scenario_type: Type of scenario (factual, episodic, etc.)
        domain: Domain (business, tech_support, education)
        turns: List of conversation turns
        metadata: Additional scenario metadata
    """

    scenario_id: str
    scenario_type: ScenarioType
    domain: str
    turns: list[Turn]
    metadata: dict[str, Any] | None = None


@dataclass
class AgentResponse:
    """Response from agent processing a turn.

    Attributes:
        content: Response text
        tool_calls: Number of search_memory tool calls made
        search_results: List of retrieved memory file paths (if search occurred)
        context_tokens: Total tokens in context
        latency_ms: Total response latency in milliseconds
        llm_latency_ms: LLM inference latency in milliseconds
    """

    content: str
    tool_calls: int = 0
    search_results: list[str] | None = None
    context_tokens: int = 0
    latency_ms: float = 0.0
    llm_latency_ms: float = 0.0


# ── Memory Experiment Framework ─────────────────────────────────────────────


class MemoryExperiment:
    """Main experiment framework for memory performance evaluation.

    This class orchestrates the entire experimental workflow:
    1. Initialize agents with different search configurations
    2. Execute conversation scenarios
    3. Collect metrics (latency, precision, recall, tokens)
    4. Log results

    The framework supports all four experimental conditions:
    - Condition A: BM25 only
    - Condition B: Vector search only
    - Condition C: Hybrid search
    - Condition D: Hybrid search + Priming

    Examples:
        >>> config = ExperimentConfig.create_condition_d(
        ...     experiment_id="exp_001",
        ...     participants=30
        ... )
        >>> experiment = MemoryExperiment(config, output_dir=Path("results/raw"))
        >>> await experiment.run_all()
    """

    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: Path,
        scenarios: list[Scenario] | None = None,
    ):
        """Initialize memory experiment.

        Args:
            config: Experiment configuration
            output_dir: Directory for output files
            scenarios: List of conversation scenarios (if None, will be loaded)
        """
        self.config = config
        self.output_dir = output_dir
        self.scenarios = scenarios or []

        self.metrics_collector = MetricsCollector()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"MemoryExperiment initialized: {config.experiment_id}, "
            f"condition={config.condition}, "
            f"participants={config.participants}"
        )

    # ── Agent Management ────────────────────────────────────────────────────

    async def create_agent(self, participant_id: int) -> Any:
        """Create and initialize an agent for the experiment.

        Args:
            participant_id: Participant ID

        Returns:
            Initialized DigitalAnima agent

        Notes:
            This is a placeholder that should be replaced with actual
            AnimaWorks DigitalAnima initialization in integration testing.
        """
        # TODO: Replace with actual DigitalAnima initialization
        # from core.anima import DigitalAnima
        # anima_dir = self._create_anima_environment(participant_id)
        # agent = DigitalAnima(
        #     anima_dir=anima_dir,
        #     search_config=self.config.search_config
        # )
        # return agent

        logger.warning(
            f"create_agent is a placeholder (participant_id={participant_id}). "
            "Replace with actual DigitalAnima initialization."
        )
        return None

    def _create_anima_environment(self, participant_id: int) -> Path:
        """Create an anima environment for the experiment.

        Args:
            participant_id: Participant ID

        Returns:
            Path to anima directory

        Notes:
            This should:
            1. Create ~/.animaworks/animas/{experiment_id}_{participant_id}/
            2. Copy memory base files (knowledge, episodes, skills)
            3. Configure search settings
        """
        # TODO: Implement anima environment creation
        anima_id = f"{self.config.experiment_id}_{self.config.condition}_{participant_id}"
        anima_dir = Path.home() / ".animaworks" / "animas" / anima_id

        logger.info(f"Would create anima environment at: {anima_dir}")
        return anima_dir

    # ── Scenario Execution ──────────────────────────────────────────────────

    async def run_scenario(
        self,
        agent: Any,
        scenario: Scenario,
        exp_logger: ExperimentLogger,
    ) -> ConversationMetrics:
        """Execute a single conversation scenario.

        Args:
            agent: DigitalAnima agent
            scenario: Conversation scenario
            exp_logger: Experiment logger

        Returns:
            Aggregated conversation metrics
        """
        exp_logger.start_conversation(scenario.scenario_id)

        turn_latencies: list[float] = []
        turn_tool_calls: list[int] = []
        total_tokens = 0
        precision_values: list[float] = []
        recall_values: list[float] = []
        f1_values: list[float] = []

        for turn in scenario.turns:
            turn_metrics = await self._execute_turn(agent, turn, scenario.scenario_id, exp_logger)

            turn_latencies.append(turn_metrics.response_latency)
            turn_tool_calls.append(turn_metrics.tool_call_count)
            total_tokens += turn_metrics.total_tokens

            if turn_metrics.search_precision is not None:
                precision_values.append(turn_metrics.search_precision)
                recall_values.append(turn_metrics.search_recall)
                f1_values.append(turn_metrics.search_f1)

            exp_logger.log_turn_metrics(turn_metrics)

        # Calculate conversation-level metrics
        percentiles = self.metrics_collector.calculate_percentiles(turn_latencies, [50, 95, 99])

        conv_metrics = ConversationMetrics(
            conversation_id=scenario.scenario_id,
            scenario_type=scenario.scenario_type,
            turn_count=len(scenario.turns),
            total_latency=sum(turn_latencies),
            avg_turn_latency=sum(turn_latencies) / len(turn_latencies) if turn_latencies else 0.0,
            p50_latency=percentiles["p50"],
            p95_latency=percentiles["p95"],
            p99_latency=percentiles["p99"],
            total_tool_calls=sum(turn_tool_calls),
            avg_tool_calls_per_turn=sum(turn_tool_calls) / len(turn_tool_calls) if turn_tool_calls else 0.0,
            total_tokens=total_tokens,
            avg_precision=sum(precision_values) / len(precision_values) if precision_values else None,
            avg_recall=sum(recall_values) / len(recall_values) if recall_values else None,
            avg_f1=sum(f1_values) / len(f1_values) if f1_values else None,
        )

        exp_logger.end_conversation(conv_metrics)
        return conv_metrics

    async def _execute_turn(
        self,
        agent: Any,
        turn: Turn,
        conversation_id: str,
        exp_logger: ExperimentLogger,
    ) -> TurnMetrics:
        """Execute a single conversation turn.

        Args:
            agent: DigitalAnima agent
            turn: Turn to execute
            conversation_id: Parent conversation ID
            exp_logger: Experiment logger

        Returns:
            Turn metrics
        """
        turn_metrics = TurnMetrics(turn_id=turn.turn_id)

        try:
            # Measure priming (Condition D only)
            if self.config.search_config.priming_enabled:
                priming_result, priming_latency = await self._measure_priming(agent, turn.message)
                turn_metrics.priming_latency = priming_latency
                turn_metrics.priming_tokens = self.metrics_collector.count_tokens(priming_result or "")

                exp_logger.log_priming(
                    turn_id=turn.turn_id,
                    latency_ms=priming_latency,
                    tokens=turn_metrics.priming_tokens,
                )

            # Measure response generation
            response, response_latency = await self._measure_response(agent, turn.message)
            turn_metrics.response_latency = response_latency
            turn_metrics.llm_latency = response.llm_latency_ms
            turn_metrics.tool_call_count = response.tool_calls
            turn_metrics.total_tokens = response.context_tokens

            # Calculate search metrics if search occurred
            if response.search_results and turn.relevant_memories:
                precision, recall, f1 = self.metrics_collector.calculate_precision_recall(
                    retrieved=response.search_results,
                    relevant=turn.relevant_memories,
                    k=self.config.search_config.top_k,
                )
                turn_metrics.search_precision = precision
                turn_metrics.search_recall = recall
                turn_metrics.search_f1 = f1
                turn_metrics.search_tokens = sum(
                    self.metrics_collector.count_tokens(r) for r in response.search_results
                )

                exp_logger.log_search(
                    turn_id=turn.turn_id,
                    latency_ms=response_latency,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    retrieved_count=len(response.search_results),
                )

        except Exception as e:
            logger.error(f"Error executing turn {turn.turn_id}: {e}")
            exp_logger.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                context={"turn_id": turn.turn_id, "conversation_id": conversation_id},
            )
            raise

        return turn_metrics

    async def _measure_priming(self, agent: Any, message: str) -> tuple[str | None, float]:
        """Measure priming layer execution.

        Args:
            agent: DigitalAnima agent
            message: User message

        Returns:
            Tuple of (priming_result, latency_ms)
        """
        # TODO: Replace with actual priming layer call
        # result, latency = await self.metrics_collector.measure_latency_async(
        #     agent.priming_engine.prime_memories,
        #     message=message,
        #     sender="user"
        # )
        # return result, latency

        # Placeholder
        await asyncio.sleep(0.05)  # Simulate 50ms priming
        return "Primed memories: ...", 50.0

    async def _measure_response(self, agent: Any, message: str) -> tuple[AgentResponse, float]:
        """Measure agent response generation.

        Args:
            agent: DigitalAnima agent
            message: User message

        Returns:
            Tuple of (response, latency_ms)
        """
        # TODO: Replace with actual agent.process_message call
        # result, latency = await self.metrics_collector.measure_latency_async(
        #     agent.process_message,
        #     message=message
        # )
        # response = AgentResponse(
        #     content=result.content,
        #     tool_calls=result.tool_calls,
        #     search_results=result.search_results,
        #     context_tokens=result.context_tokens,
        #     latency_ms=latency,
        #     llm_latency_ms=result.llm_latency_ms
        # )
        # return response, latency

        # Placeholder
        await asyncio.sleep(0.2)  # Simulate 200ms response
        response = AgentResponse(
            content="This is a test response.",
            tool_calls=1,
            search_results=["knowledge/test.md"],
            context_tokens=500,
            latency_ms=200.0,
            llm_latency_ms=150.0,
        )
        return response, 200.0

    # ── Main Execution Loop ─────────────────────────────────────────────────

    async def run_participant(self, participant_id: int) -> dict[str, Any]:
        """Run experiment for a single participant.

        Args:
            participant_id: Participant ID

        Returns:
            Summary results for this participant
        """
        logger.info(
            f"Running participant {participant_id}/{self.config.participants} "
            f"(condition={self.config.condition})"
        )

        # Initialize logger
        exp_logger = ExperimentLogger(
            experiment_id=self.config.experiment_id,
            output_dir=self.output_dir,
            condition=self.config.condition,
            participant_id=participant_id,
            metadata={
                "memory_size": self.config.memory_size.value,
                "conversation_length": self.config.conversation_length.value,
                "domain": self.config.domain.value,
                "model_name": self.config.model_name,
                "temperature": self.config.temperature,
            },
        )

        # Create agent
        agent = await self.create_agent(participant_id)

        # Run all scenarios
        conversation_metrics: list[ConversationMetrics] = []
        for scenario in self.scenarios:
            try:
                conv_metrics = await self.run_scenario(agent, scenario, exp_logger)
                conversation_metrics.append(conv_metrics)
            except Exception as e:
                logger.error(f"Error in scenario {scenario.scenario_id}: {e}")
                exp_logger.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={"scenario_id": scenario.scenario_id},
                )

        # Generate summary
        summary = exp_logger.generate_summary()
        logger.info(f"Participant {participant_id} complete: {summary}")

        return summary

    async def run_all(self) -> list[dict[str, Any]]:
        """Run experiment for all participants.

        Returns:
            List of summary results for each participant
        """
        logger.info(
            f"Starting experiment {self.config.experiment_id}: "
            f"condition={self.config.condition}, "
            f"participants={self.config.participants}, "
            f"scenarios={len(self.scenarios)}"
        )

        results = []
        for participant_id in range(1, self.config.participants + 1):
            try:
                summary = await self.run_participant(participant_id)
                results.append(summary)
            except Exception as e:
                logger.error(f"Error running participant {participant_id}: {e}")

        logger.info(f"Experiment {self.config.experiment_id} complete")
        return results

    # ── Utility Methods ─────────────────────────────────────────────────────

    def load_scenarios(self, scenarios_dir: Path) -> None:
        """Load conversation scenarios from directory.

        Args:
            scenarios_dir: Directory containing scenario YAML files
        """
        # TODO: Implement scenario loading from YAML
        logger.warning("load_scenarios not yet implemented")

    def validate_setup(self) -> bool:
        """Validate experimental setup.

        Returns:
            True if setup is valid

        Checks:
        - Scenarios are loaded
        - Output directory exists
        - Configuration is valid
        """
        if not self.scenarios:
            logger.error("No scenarios loaded")
            return False

        if not self.output_dir.exists():
            logger.error(f"Output directory does not exist: {self.output_dir}")
            return False

        logger.info("Experimental setup validated")
        return True
