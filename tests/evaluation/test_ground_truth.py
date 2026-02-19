# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ground truth management.

Tests annotation creation, storage, and inter-annotator agreement calculation.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from tests.evaluation.framework import (
    AnnotationSet,
    ConversationTurn,
    DatasetGenerator,
    GroundTruth,
    GroundTruthManager,
    RelevantMemory,
    Scenario
)


@pytest.fixture
def temp_gt_dir(tmp_path):
    """Temporary directory for ground truth files."""
    return tmp_path / "ground_truth"


@pytest.fixture
def gt_manager(temp_gt_dir):
    """Ground truth manager instance."""
    return GroundTruthManager(output_dir=temp_gt_dir)


@pytest.fixture
def sample_memory_base(tmp_path):
    """Generate a small sample memory base for testing."""
    generator = DatasetGenerator(
        output_dir=tmp_path / "datasets",
        use_llm=False
    )
    return generator.generate_memory_base(
        domain="business",
        size="small"
    )


@pytest.fixture
def sample_scenarios(sample_memory_base):
    """Generate sample scenarios for testing."""
    generator = DatasetGenerator(
        output_dir=Path("/tmp/test_datasets"),
        use_llm=False
    )
    return generator.generate_scenarios(
        domain="business",
        memory_base=sample_memory_base,
        total_count=5
    )


# ── Annotation Creation Tests ────────────────────────────────────────────────


def test_create_annotations(gt_manager, sample_scenarios, sample_memory_base):
    """Test creating annotations from scenarios."""
    annotation_set = gt_manager.create_annotations(
        scenarios=sample_scenarios,
        memory_base=sample_memory_base,
        annotator_id="test_annotator"
    )

    assert isinstance(annotation_set, AnnotationSet)
    assert annotation_set.annotator_id == "test_annotator"
    assert len(annotation_set.annotations) > 0

    # Should have one annotation per turn across all scenarios
    total_turns = sum(s.num_turns for s in sample_scenarios)
    assert len(annotation_set.annotations) == total_turns


def test_annotation_has_metadata(gt_manager, sample_scenarios, sample_memory_base):
    """Test that annotation set includes metadata."""
    annotation_set = gt_manager.create_annotations(
        scenarios=sample_scenarios,
        memory_base=sample_memory_base,
        annotator_id="test_annotator"
    )

    assert "created_at" in annotation_set.metadata
    assert "num_scenarios" in annotation_set.metadata
    assert annotation_set.metadata["domain"] == "business"
    assert annotation_set.metadata["size"] == "small"


def test_ground_truth_structure(gt_manager, sample_scenarios, sample_memory_base):
    """Test structure of individual ground truth annotations."""
    annotation_set = gt_manager.create_annotations(
        scenarios=sample_scenarios,
        memory_base=sample_memory_base,
        annotator_id="test_annotator"
    )

    # Get first annotation
    query_id = list(annotation_set.annotations.keys())[0]
    gt = annotation_set.annotations[query_id]

    assert isinstance(gt, GroundTruth)
    assert gt.query != ""
    assert gt.scenario_id != ""
    assert gt.turn_index >= 0
    assert isinstance(gt.relevant_memories, list)
    assert isinstance(gt.irrelevant_memories, list)
    assert gt.annotator_id == "test_annotator"
    assert gt.timestamp != ""


def test_relevant_memory_structure(gt_manager, sample_scenarios, sample_memory_base):
    """Test structure of RelevantMemory objects."""
    annotation_set = gt_manager.create_annotations(
        scenarios=sample_scenarios,
        memory_base=sample_memory_base,
        annotator_id="test_annotator"
    )

    # Find an annotation with relevant memories
    for gt in annotation_set.annotations.values():
        if gt.relevant_memories:
            rm = gt.relevant_memories[0]
            assert isinstance(rm, RelevantMemory)
            assert isinstance(rm.file_path, Path)
            assert rm.relevance in ["high", "medium", "low"]
            break


# ── Annotation Storage Tests ─────────────────────────────────────────────────


def test_save_annotations(gt_manager, sample_scenarios, sample_memory_base):
    """Test saving annotations to JSON file."""
    annotation_set = gt_manager.create_annotations(
        scenarios=sample_scenarios,
        memory_base=sample_memory_base,
        annotator_id="annotator1"
    )

    saved_path = gt_manager.save_annotations(annotation_set)

    assert saved_path.exists()
    assert saved_path.suffix == ".json"
    assert "annotator1" in saved_path.name


def test_save_annotations_custom_filename(gt_manager, sample_scenarios, sample_memory_base):
    """Test saving annotations with custom filename."""
    annotation_set = gt_manager.create_annotations(
        scenarios=sample_scenarios,
        memory_base=sample_memory_base,
        annotator_id="annotator1"
    )

    saved_path = gt_manager.save_annotations(
        annotation_set,
        filename="custom_annotations.json"
    )

    assert saved_path.name == "custom_annotations.json"


def test_load_annotations(gt_manager, sample_scenarios, sample_memory_base):
    """Test loading annotations from JSON file."""
    # Create and save annotations
    annotation_set = gt_manager.create_annotations(
        scenarios=sample_scenarios,
        memory_base=sample_memory_base,
        annotator_id="annotator1"
    )

    saved_path = gt_manager.save_annotations(annotation_set)

    # Load annotations
    loaded_set = gt_manager.load_annotations(saved_path.name)

    assert loaded_set.annotator_id == annotation_set.annotator_id
    assert len(loaded_set.annotations) == len(annotation_set.annotations)


def test_save_load_roundtrip(gt_manager, sample_scenarios, sample_memory_base):
    """Test that save/load preserves all data."""
    # Create annotations
    original_set = gt_manager.create_annotations(
        scenarios=sample_scenarios,
        memory_base=sample_memory_base,
        annotator_id="annotator1"
    )

    # Save and load
    saved_path = gt_manager.save_annotations(original_set)
    loaded_set = gt_manager.load_annotations(saved_path.name)

    # Compare
    assert loaded_set.annotator_id == original_set.annotator_id
    assert loaded_set.metadata == original_set.metadata

    # Compare annotations
    for query_id in original_set.annotations:
        original_gt = original_set.annotations[query_id]
        loaded_gt = loaded_set.annotations[query_id]

        assert loaded_gt.query == original_gt.query
        assert loaded_gt.scenario_id == original_gt.scenario_id
        assert loaded_gt.turn_index == original_gt.turn_index
        assert len(loaded_gt.relevant_memories) == len(original_gt.relevant_memories)
        assert len(loaded_gt.irrelevant_memories) == len(original_gt.irrelevant_memories)


# ── Inter-Annotator Agreement Tests ──────────────────────────────────────────


def test_calculate_agreement_identical(gt_manager, sample_scenarios, sample_memory_base):
    """Test agreement calculation with identical annotations."""
    # Create identical annotations
    annotation_set1 = gt_manager.create_annotations(
        scenarios=sample_scenarios,
        memory_base=sample_memory_base,
        annotator_id="annotator1"
    )

    annotation_set2 = gt_manager.create_annotations(
        scenarios=sample_scenarios,
        memory_base=sample_memory_base,
        annotator_id="annotator2"
    )

    agreement = gt_manager.calculate_agreement(annotation_set1, annotation_set2)

    # Identical annotations should have perfect agreement
    assert agreement["agreement_rate"] == 1.0
    assert agreement["cohens_kappa"] == 1.0
    assert agreement["num_queries"] > 0


def test_calculate_agreement_disjoint(gt_manager, sample_scenarios, sample_memory_base):
    """Test agreement calculation with completely different annotations."""
    # Create first annotation set
    annotation_set1 = gt_manager.create_annotations(
        scenarios=sample_scenarios,
        memory_base=sample_memory_base,
        annotator_id="annotator1"
    )

    # Create second annotation set (manually modify to be different)
    annotation_set2 = gt_manager.create_annotations(
        scenarios=sample_scenarios,
        memory_base=sample_memory_base,
        annotator_id="annotator2"
    )

    # Manually swap relevant/irrelevant for all annotations
    for query_id, gt in annotation_set2.annotations.items():
        # Swap relevant and irrelevant: what was relevant becomes irrelevant and vice versa
        original_relevant = gt.relevant_memories
        original_irrelevant = gt.irrelevant_memories
        gt.relevant_memories = [
            RelevantMemory(file_path=p, relevance="high")
            for p in original_irrelevant
        ]
        gt.irrelevant_memories = [rm.file_path for rm in original_relevant]

    agreement = gt_manager.calculate_agreement(annotation_set1, annotation_set2)

    # Should have low agreement
    assert agreement["agreement_rate"] < 0.5
    assert agreement["cohens_kappa"] < 0.5


def test_agreement_with_empty_overlap(gt_manager):
    """Test agreement calculation with no overlapping queries."""
    # Create two annotation sets with different query IDs
    set1 = AnnotationSet(annotator_id="annotator1")
    set1.add_annotation("query_1", GroundTruth(
        query="test1",
        scenario_id="s1",
        turn_index=0
    ))

    set2 = AnnotationSet(annotator_id="annotator2")
    set2.add_annotation("query_2", GroundTruth(
        query="test2",
        scenario_id="s2",
        turn_index=0
    ))

    agreement = gt_manager.calculate_agreement(set1, set2)

    # No overlap should result in zero metrics
    assert agreement["cohens_kappa"] == 0.0
    assert agreement["agreement_rate"] == 0.0
    assert agreement["num_queries"] == 0


# ── Agreement Report Tests ───────────────────────────────────────────────────


def test_save_agreement_report(gt_manager, sample_scenarios, sample_memory_base):
    """Test saving agreement report to file."""
    annotation_set1 = gt_manager.create_annotations(
        scenarios=sample_scenarios,
        memory_base=sample_memory_base,
        annotator_id="annotator1"
    )

    annotation_set2 = gt_manager.create_annotations(
        scenarios=sample_scenarios,
        memory_base=sample_memory_base,
        annotator_id="annotator2"
    )

    report_path = gt_manager.save_agreement_report(
        annotation_set1,
        annotation_set2
    )

    assert report_path.exists()
    assert report_path.suffix == ".json"

    # Verify report structure
    with open(report_path) as f:
        report = json.load(f)

    assert report["annotator_1"] == "annotator1"
    assert report["annotator_2"] == "annotator2"
    assert "agreement_metrics" in report
    assert "interpretation" in report


def test_kappa_interpretation(gt_manager):
    """Test Cohen's kappa interpretation."""
    # Test various kappa values
    assert "Almost perfect" in gt_manager._interpret_kappa(0.85)
    assert "Substantial" in gt_manager._interpret_kappa(0.70)
    assert "Moderate" in gt_manager._interpret_kappa(0.50)
    assert "Fair" in gt_manager._interpret_kappa(0.30)
    assert "Slight" in gt_manager._interpret_kappa(0.10)
    assert "Poor" in gt_manager._interpret_kappa(-0.10)


# ── GroundTruth Helper Method Tests ──────────────────────────────────────────


def test_high_relevance_files(sample_memory_base):
    """Test extracting high relevance files from ground truth."""
    gt = GroundTruth(
        query="test query",
        scenario_id="s1",
        turn_index=0,
        relevant_memories=[
            RelevantMemory(
                file_path=sample_memory_base.knowledge_files[0].path,
                relevance="high"
            ),
            RelevantMemory(
                file_path=sample_memory_base.knowledge_files[1].path,
                relevance="medium"
            ),
            RelevantMemory(
                file_path=sample_memory_base.knowledge_files[2].path,
                relevance="high"
            ),
        ]
    )

    high_rel = gt.high_relevance_files
    assert len(high_rel) == 2


# ── Integration Tests ────────────────────────────────────────────────────────


def test_full_annotation_workflow(gt_manager, sample_scenarios, sample_memory_base, temp_gt_dir):
    """Test complete annotation workflow."""
    # Step 1: Create annotations
    annotation_set = gt_manager.create_annotations(
        scenarios=sample_scenarios,
        memory_base=sample_memory_base,
        annotator_id="annotator1"
    )

    # Step 2: Save annotations
    saved_path = gt_manager.save_annotations(annotation_set)
    assert saved_path.exists()

    # Step 3: Load annotations
    loaded_set = gt_manager.load_annotations(saved_path.name)
    assert len(loaded_set.annotations) == len(annotation_set.annotations)

    # Step 4: Create second annotator's set
    annotation_set2 = gt_manager.create_annotations(
        scenarios=sample_scenarios,
        memory_base=sample_memory_base,
        annotator_id="annotator2"
    )

    gt_manager.save_annotations(annotation_set2)

    # Step 5: Calculate agreement
    agreement = gt_manager.calculate_agreement(annotation_set, annotation_set2)
    assert "cohens_kappa" in agreement

    # Step 6: Save agreement report
    report_path = gt_manager.save_agreement_report(annotation_set, annotation_set2)
    assert report_path.exists()


def test_annotation_set_operations(gt_manager):
    """Test AnnotationSet add/get operations."""
    annotation_set = AnnotationSet(annotator_id="test")

    gt1 = GroundTruth(
        query="query1",
        scenario_id="s1",
        turn_index=0
    )

    # Add annotation
    annotation_set.add_annotation("q1", gt1)
    assert len(annotation_set.annotations) == 1

    # Get annotation
    retrieved = annotation_set.get_annotation("q1")
    assert retrieved == gt1

    # Get non-existent annotation
    none_result = annotation_set.get_annotation("nonexistent")
    assert none_result is None
