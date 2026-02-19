#!/usr/bin/env python3

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Example usage of the memory evaluation framework.

This script demonstrates how to:
1. Configure an experiment
2. Create scenarios
3. Run the experiment
4. Analyze results
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import framework
sys.path.insert(0, str(Path(__file__).parent.parent))

from framework import (
    ConversationLength,
    Domain,
    ExperimentConfig,
    MemoryExperiment,
    MemorySize,
    Scenario,
    ScenarioType,
    Turn,
)


def create_sample_scenarios() -> list[Scenario]:
    """Create sample conversation scenarios for testing.

    Returns:
        List of Scenario objects
    """
    scenarios = [
        # Scenario 1: Factual recall
        Scenario(
            scenario_id="fact_001",
            scenario_type=ScenarioType.FACTUAL_RECALL,
            domain="business",
            turns=[
                Turn(
                    turn_id="turn_001",
                    message="山田さんのプロジェクトの締め切りはいつでしたか?",
                    relevant_memories=[
                        "knowledge/clients/yamada-project.md",
                        "episodes/2026-02-10.md",
                    ],
                    expected_answer="2026年3月15日",
                ),
                Turn(
                    turn_id="turn_002",
                    message="そのプロジェクトの予算は?",
                    relevant_memories=["knowledge/clients/yamada-project.md"],
                    expected_answer="500万円",
                ),
            ],
        ),
        # Scenario 2: Episodic recall
        Scenario(
            scenario_id="episode_001",
            scenario_type=ScenarioType.EPISODIC_RECALL,
            domain="business",
            turns=[
                Turn(
                    turn_id="turn_003",
                    message="先週の山田さんとのミーティングで何を話しましたか?",
                    relevant_memories=["episodes/2026-02-07.md"],
                    expected_answer="プロジェクトの進捗と次のマイルストーン",
                ),
            ],
        ),
        # Scenario 3: Multi-hop reasoning
        Scenario(
            scenario_id="multihop_001",
            scenario_type=ScenarioType.MULTIHOP_REASONING,
            domain="business",
            turns=[
                Turn(
                    turn_id="turn_004",
                    message="山田さんのプロジェクトで使っている技術スタックは何ですか?",
                    relevant_memories=[
                        "knowledge/clients/yamada-project.md",
                        "episodes/2026-02-05.md",
                        "knowledge/tech/react-guidelines.md",
                    ],
                    expected_answer="React + TypeScript + FastAPI",
                ),
            ],
        ),
    ]

    return scenarios


async def run_single_participant_example():
    """Example: Run experiment for a single participant."""
    print("=" * 80)
    print("Example: Single Participant Experiment")
    print("=" * 80)

    # Configure Condition D (Hybrid + Priming)
    config = ExperimentConfig.create_condition_d(
        experiment_id="example_001",
        participants=1,  # Single participant for this example
        memory_size=MemorySize.SMALL,
        conversation_length=ConversationLength.SHORT,
        domain=Domain.BUSINESS,
        priming_budget=2000,
    )

    print(f"\nConfiguration:")
    print(f"  Experiment ID: {config.experiment_id}")
    print(f"  Condition: {config.condition} ({config.search_config.method.value})")
    print(f"  Priming enabled: {config.search_config.priming_enabled}")
    print(f"  Priming budget: {config.search_config.priming_budget} tokens")
    print(f"  Memory size: {config.memory_size.value}")
    print(f"  Domain: {config.domain.value}")

    # Create experiment
    output_dir = Path("results/raw")
    experiment = MemoryExperiment(config=config, output_dir=output_dir)

    # Load scenarios
    scenarios = create_sample_scenarios()
    experiment.scenarios = scenarios

    print(f"\nScenarios loaded: {len(scenarios)}")
    for scenario in scenarios:
        print(f"  - {scenario.scenario_id}: {scenario.scenario_type.value} ({len(scenario.turns)} turns)")

    # Validate setup
    if not experiment.validate_setup():
        print("\n❌ Setup validation failed!")
        return

    print("\n✓ Setup validated")

    # Run single participant
    print(f"\nRunning participant 1...")
    summary = await experiment.run_participant(participant_id=1)

    print(f"\n✓ Participant complete!")
    print(f"\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print(f"\nResults saved to: {output_dir / config.experiment_id}")


async def run_condition_comparison_example():
    """Example: Compare all four conditions with same scenarios."""
    print("\n" + "=" * 80)
    print("Example: Condition Comparison (A, B, C, D)")
    print("=" * 80)

    output_dir = Path("results/raw")
    scenarios = create_sample_scenarios()

    conditions = [
        ("A", ExperimentConfig.create_condition_a),
        ("B", ExperimentConfig.create_condition_b),
        ("C", ExperimentConfig.create_condition_c),
        ("D", ExperimentConfig.create_condition_d),
    ]

    results = {}

    for condition_name, factory in conditions:
        print(f"\nRunning Condition {condition_name}...")

        config = factory(
            experiment_id=f"comparison_{condition_name}",
            participants=1,  # Single participant for demo
            memory_size=MemorySize.SMALL,
        )

        experiment = MemoryExperiment(config=config, output_dir=output_dir, scenarios=scenarios)

        if not experiment.validate_setup():
            print(f"  ❌ Setup validation failed for condition {condition_name}")
            continue

        summary = await experiment.run_participant(participant_id=1)
        results[condition_name] = summary

        print(f"  ✓ Complete - Avg latency: {summary.get('avg_latency_ms', 'N/A'):.1f}ms")

    # Display comparison
    print("\n" + "=" * 80)
    print("Comparison Results")
    print("=" * 80)
    print(f"{'Condition':<15} {'Avg Latency (ms)':<20} {'Tool Calls':<15} {'Total Tokens':<15}")
    print("-" * 80)

    for condition_name, summary in results.items():
        print(
            f"{condition_name:<15} "
            f"{summary.get('avg_latency_ms', 0):<20.1f} "
            f"{summary.get('avg_tool_calls', 0):<15.1f} "
            f"{summary.get('total_tokens', 0):<15}"
        )


async def main():
    """Main entry point."""
    print("\n🔬 AnimaWorks Memory Performance Evaluation Framework")
    print("     Phase 1: Core Implementation Demo\n")

    # Example 1: Single participant
    await run_single_participant_example()

    # Example 2: Condition comparison
    # Uncomment to run full comparison:
    # await run_condition_comparison_example()

    print("\n" + "=" * 80)
    print("✅ Examples complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Implement actual agent integration (replace placeholders in framework.py)")
    print("  2. Generate memory bases and scenarios (Phase 2)")
    print("  3. Run full experiments with N=30 participants")
    print("  4. Perform statistical analysis")


if __name__ == "__main__":
    asyncio.run(main())
