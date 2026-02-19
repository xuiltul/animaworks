# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for dataset generator.

Tests the generation of memory bases and scenarios without requiring LLM calls.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from tests.evaluation.framework import (
    DatasetGenerator,
    MemoryBase,
    SizeConfig,
    ScenarioTypeConfig
)


@pytest.fixture
def temp_dataset_dir(tmp_path):
    """Temporary directory for dataset generation."""
    return tmp_path / "datasets"


@pytest.fixture
def generator(temp_dataset_dir):
    """Dataset generator without LLM (template mode)."""
    return DatasetGenerator(
        output_dir=temp_dataset_dir,
        use_llm=False  # Use templates for faster testing
    )


# ── Size Configuration Tests ─────────────────────────────────────────────────


def test_size_config_small():
    """Test small dataset configuration."""
    config = SizeConfig.get_config("small")

    assert config.knowledge_count == 50
    assert config.episode_count == 30
    assert config.skill_count == 10
    assert config.target_tokens == 50_000


def test_size_config_medium():
    """Test medium dataset configuration."""
    config = SizeConfig.get_config("medium")

    assert config.knowledge_count == 500
    assert config.episode_count == 300
    assert config.skill_count == 100
    assert config.target_tokens == 500_000


def test_size_config_large():
    """Test large dataset configuration."""
    config = SizeConfig.get_config("large")

    assert config.knowledge_count == 5000
    assert config.episode_count == 3000
    assert config.skill_count == 1000
    assert config.target_tokens == 5_000_000


# ── Scenario Type Configuration Tests ────────────────────────────────────────


def test_scenario_type_distribution():
    """Test scenario type distribution calculation."""
    config = ScenarioTypeConfig()
    counts = config.get_counts(100)

    assert counts["factual"] == 40
    assert counts["episodic"] == 30
    assert counts["multihop"] == 20
    assert counts["long"] == 10
    assert sum(counts.values()) == 100


# ── Knowledge File Generation Tests ──────────────────────────────────────────


def test_generate_knowledge_file(generator):
    """Test single knowledge file generation."""
    content = generator.generate_knowledge_file(
        domain="business",
        topic="Company Vision",
        index=0
    )

    assert isinstance(content, str)
    assert len(content) > 100
    assert "Company Vision" in content
    assert "#" in content  # Has markdown headers


def test_knowledge_file_size_range(generator):
    """Test that knowledge files are within expected size range."""
    content = generator.generate_knowledge_file(
        domain="tech_support",
        topic="API Reference",
        index=1
    )

    # Should be at least 300 chars for minimal content
    assert len(content) >= 300
    # Template should not exceed 5000 chars
    assert len(content) <= 5000


# ── Episode File Generation Tests ────────────────────────────────────────────


def test_generate_episode_file(generator):
    """Test single episode file generation."""
    content = generator.generate_episode_file(
        domain="education",
        date="2026-02-14",
        index=0
    )

    assert isinstance(content, str)
    assert len(content) > 100
    assert "2026-02-14" in content
    assert "#" in content  # Has markdown headers


def test_episode_file_has_timeline(generator):
    """Test that episode files have timeline structure."""
    content = generator.generate_episode_file(
        domain="business",
        date="2026-02-14",
        index=0
    )

    # Should have time-based sections
    assert "##" in content  # Has subsections


# ── Skill File Generation Tests ──────────────────────────────────────────────


def test_generate_skill_file(generator):
    """Test single skill file generation."""
    content = generator.generate_skill_file(
        domain="tech_support",
        skill_name="Log Analysis",
        index=0
    )

    assert isinstance(content, str)
    assert len(content) > 100
    assert "Log Analysis" in content
    assert "#" in content  # Has markdown headers


def test_skill_file_has_steps(generator):
    """Test that skill files have procedural steps."""
    content = generator.generate_skill_file(
        domain="education",
        skill_name="Lesson Planning",
        index=0
    )

    # Should have numbered or bulleted steps
    assert any(char in content for char in ["1.", "-", "*"])


# ── Memory Base Generation Tests ─────────────────────────────────────────────


def test_generate_small_memory_base(generator):
    """Test generation of small memory base."""
    memory_base = generator.generate_memory_base(
        domain="business",
        size="small"
    )

    assert isinstance(memory_base, MemoryBase)
    assert memory_base.domain == "business"
    assert memory_base.size == "small"

    # Check file counts (small = 50 knowledge + 30 episodes + 10 skills)
    assert len(memory_base.knowledge_files) == 50
    assert len(memory_base.episode_files) == 30
    assert len(memory_base.skill_files) == 10
    assert memory_base.total_files == 90

    # Check that files exist
    for file in memory_base.all_files:
        assert file.path.exists()
        assert file.tokens > 0


def test_memory_base_file_structure(generator, temp_dataset_dir):
    """Test that memory base creates correct directory structure."""
    memory_base = generator.generate_memory_base(
        domain="tech_support",
        size="small"
    )

    base_dir = temp_dataset_dir / "tech_support" / "small"

    assert (base_dir / "knowledge").exists()
    assert (base_dir / "episodes").exists()
    assert (base_dir / "skills").exists()


def test_memory_base_token_estimation(generator):
    """Test token estimation for memory base."""
    memory_base = generator.generate_memory_base(
        domain="education",
        size="small"
    )

    # Total tokens should be reasonable (not exact due to templates)
    # Small target is 50K tokens, templates should be in ballpark
    assert memory_base.total_tokens > 10_000
    assert memory_base.total_tokens < 200_000


# ── Scenario Generation Tests ────────────────────────────────────────────────


def test_generate_scenarios(generator):
    """Test scenario generation."""
    # First create a small memory base
    memory_base = generator.generate_memory_base(
        domain="business",
        size="small"
    )

    # Generate scenarios
    scenarios = generator.generate_scenarios(
        domain="business",
        memory_base=memory_base,
        total_count=20  # Smaller for test speed
    )

    assert len(scenarios) > 0
    assert all(s.domain == "business" for s in scenarios)

    # Check type distribution (should have multiple types)
    types = {s.scenario_type for s in scenarios}
    assert len(types) > 1


def test_factual_scenario_structure(generator):
    """Test structure of factual recall scenarios."""
    memory_base = generator.generate_memory_base(
        domain="business",
        size="small"
    )

    scenarios = generator.generate_scenarios(
        domain="business",
        memory_base=memory_base,
        total_count=10
    )

    # Find a factual scenario
    factual = [s for s in scenarios if s.scenario_type == "factual"]
    if factual:
        scenario = factual[0]
        assert scenario.num_turns == 5  # Factual scenarios have 5 turns
        assert all(len(turn.relevant_memories) > 0 for turn in scenario.turns)


def test_long_scenario_structure(generator):
    """Test structure of long conversation scenarios."""
    memory_base = generator.generate_memory_base(
        domain="tech_support",
        size="small"
    )

    scenarios = generator.generate_scenarios(
        domain="tech_support",
        memory_base=memory_base,
        total_count=50  # Need more to ensure we get a long scenario
    )

    # Find a long scenario
    long_scenarios = [s for s in scenarios if s.scenario_type == "long"]
    if long_scenarios:
        scenario = long_scenarios[0]
        assert scenario.num_turns == 20  # Long scenarios have 20 turns


# ── Scenario Storage Tests ───────────────────────────────────────────────────


def test_save_scenarios(generator, temp_dataset_dir):
    """Test saving scenarios to YAML files."""
    memory_base = generator.generate_memory_base(
        domain="education",
        size="small"
    )

    scenarios = generator.generate_scenarios(
        domain="education",
        memory_base=memory_base,
        total_count=5
    )

    scenario_dir = temp_dataset_dir / "scenarios"
    saved_paths = generator.save_scenarios(scenarios, scenario_dir)

    assert len(saved_paths) == len(scenarios)
    assert all(p.exists() for p in saved_paths)
    assert all(p.suffix == ".yaml" for p in saved_paths)


# ── Integration Tests ────────────────────────────────────────────────────────


def test_full_dataset_generation_workflow(generator, temp_dataset_dir):
    """Test complete workflow from memory base to scenarios."""
    # Step 1: Generate memory base
    memory_base = generator.generate_memory_base(
        domain="business",
        size="small"
    )

    assert memory_base.total_files == 90

    # Step 2: Generate scenarios
    scenarios = generator.generate_scenarios(
        domain="business",
        memory_base=memory_base,
        total_count=10
    )

    assert len(scenarios) > 0

    # Step 3: Save scenarios
    scenario_dir = temp_dataset_dir / "scenarios"
    saved_paths = generator.save_scenarios(scenarios, scenario_dir)

    assert len(saved_paths) == len(scenarios)

    # Verify everything exists
    assert (temp_dataset_dir / "business" / "small").exists()
    assert scenario_dir.exists()


def test_multiple_domains_generation(generator):
    """Test generating datasets for multiple domains."""
    domains = ["business", "tech_support", "education"]

    for domain in domains:
        memory_base = generator.generate_memory_base(
            domain=domain,  # type: ignore
            size="small"
        )

        assert memory_base.domain == domain
        assert memory_base.total_files == 90


# ── Edge Cases ───────────────────────────────────────────────────────────────


def test_empty_scenario_generation(generator):
    """Test scenario generation with zero count."""
    memory_base = generator.generate_memory_base(
        domain="business",
        size="small"
    )

    scenarios = generator.generate_scenarios(
        domain="business",
        memory_base=memory_base,
        total_count=0
    )

    # Should return empty list, not crash
    assert scenarios == []


def test_generator_output_dir_creation(tmp_path):
    """Test that generator creates output directory if not exists."""
    new_dir = tmp_path / "nonexistent" / "nested" / "path"
    generator = DatasetGenerator(output_dir=new_dir, use_llm=False)

    memory_base = generator.generate_memory_base(
        domain="business",
        size="small"
    )

    # Directory should be created
    assert new_dir.exists()
    assert (new_dir / "business" / "small").exists()
