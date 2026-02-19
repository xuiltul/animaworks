#!/usr/bin/env python3

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""
Generate a small sample dataset for testing.

This script generates a small business domain dataset without requiring LLM calls.
It serves as a demonstration and test of the Phase 2 implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.evaluation.framework import (
    DatasetGenerator,
    GroundTruthManager,
)


def main():
    """Generate small business dataset."""
    print("=" * 70)
    print("AnimaWorks Memory Performance Evaluation - Phase 2")
    print("Small Dataset Generation (Business Domain)")
    print("=" * 70)
    print()

    # Setup paths
    output_root = project_root / "tests" / "evaluation"
    dataset_dir = output_root / "datasets"
    scenario_dir = output_root / "scenarios"
    gt_dir = output_root / "ground_truth"

    # Initialize generator (template mode, no LLM)
    print("Initializing dataset generator (template mode)...")
    generator = DatasetGenerator(
        output_dir=dataset_dir,
        use_llm=False  # Use templates for reproducible generation
    )
    print("✓ Generator initialized")
    print()

    # Step 1: Generate memory base
    print("Step 1: Generating small business memory base...")
    print("-" * 70)
    memory_base = generator.generate_memory_base(
        domain="business",
        size="small"
    )
    print()
    print(f"✓ Memory base generated:")
    print(f"  - Domain: {memory_base.domain}")
    print(f"  - Size: {memory_base.size}")
    print(f"  - Total files: {memory_base.total_files}")
    print(f"  - Knowledge files: {len(memory_base.knowledge_files)}")
    print(f"  - Episode files: {len(memory_base.episode_files)}")
    print(f"  - Skill files: {len(memory_base.skill_files)}")
    print(f"  - Total tokens: {memory_base.total_tokens:,}")
    print()

    # Step 2: Generate scenarios
    print("Step 2: Generating conversation scenarios...")
    print("-" * 70)
    scenarios = generator.generate_scenarios(
        domain="business",
        memory_base=memory_base,
        total_count=20  # Small number for demo
    )
    print(f"✓ Generated {len(scenarios)} scenarios")
    print()

    # Show scenario distribution
    type_counts = {}
    for scenario in scenarios:
        type_counts[scenario.scenario_type] = type_counts.get(scenario.scenario_type, 0) + 1

    print("Scenario distribution:")
    for scenario_type, count in sorted(type_counts.items()):
        print(f"  - {scenario_type}: {count}")
    print()

    # Step 3: Save scenarios
    print("Step 3: Saving scenarios to YAML files...")
    print("-" * 70)
    saved_paths = generator.save_scenarios(scenarios, scenario_dir)
    print(f"✓ Saved {len(saved_paths)} scenario files to:")
    print(f"  {scenario_dir}")
    print()

    # Step 4: Create ground truth annotations
    print("Step 4: Creating ground truth annotations...")
    print("-" * 70)
    gt_manager = GroundTruthManager(output_dir=gt_dir)

    annotation_set = gt_manager.create_annotations(
        scenarios=scenarios,
        memory_base=memory_base,
        annotator_id="auto_generator"
    )

    print(f"✓ Created {len(annotation_set.annotations)} annotations")
    print()

    # Step 5: Save annotations
    print("Step 5: Saving annotations...")
    print("-" * 70)
    saved_gt_path = gt_manager.save_annotations(annotation_set)
    print(f"✓ Saved annotations to:")
    print(f"  {saved_gt_path}")
    print()

    # Summary
    print("=" * 70)
    print("Dataset Generation Complete!")
    print("=" * 70)
    print()
    print("Generated files:")
    print(f"  Memory base: {dataset_dir / 'business' / 'small'}")
    print(f"  Scenarios: {scenario_dir}")
    print(f"  Ground truth: {gt_dir}")
    print()
    print("Next steps:")
    print("  1. Review generated files")
    print("  2. Optionally refine scenarios manually")
    print("  3. Proceed to Phase 3: Metrics Collection")
    print()


if __name__ == "__main__":
    main()
