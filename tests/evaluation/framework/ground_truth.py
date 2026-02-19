# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""
Ground Truth management for memory evaluation.

Provides annotation creation, storage, and inter-annotator agreement calculation.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .schemas import AnnotationSet, GroundTruth, MemoryBase, RelevantMemory, Scenario


# ── Ground Truth Manager ────────────────────────────────────────────────────


class GroundTruthManager:
    """
    Manages ground truth annotations for memory evaluation.

    Responsibilities:
    - Create annotations (manual or heuristic-based)
    - Save/load annotation sets
    - Calculate inter-annotator agreement (Cohen's κ)
    """

    def __init__(self, output_dir: Path):
        """
        Initialize ground truth manager.

        Args:
            output_dir: Directory to store annotation files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Annotation Creation ──────────────────────────────────────────────────

    def create_annotations(
        self,
        scenarios: list[Scenario],
        memory_base: MemoryBase,
        annotator_id: str = "auto"
    ) -> AnnotationSet:
        """
        Create ground truth annotations for scenarios.

        For now, uses heuristic-based annotation (scenarios come with
        relevant_memories already specified). In production, this would
        provide a UI for manual annotation.

        Args:
            scenarios: List of scenarios to annotate
            memory_base: Memory base to annotate against
            annotator_id: Identifier for the annotator

        Returns:
            Complete annotation set
        """
        annotation_set = AnnotationSet(
            annotator_id=annotator_id,
            metadata={
                "created_at": datetime.now().isoformat(),
                "num_scenarios": len(scenarios),
                "domain": memory_base.domain,
                "size": memory_base.size
            }
        )

        for scenario in scenarios:
            for turn_idx, turn in enumerate(scenario.turns):
                query_id = f"{scenario.scenario_id}_turn_{turn_idx}"

                # Create ground truth from pre-specified relevant memories
                gt = self._create_ground_truth_from_turn(
                    scenario=scenario,
                    turn_idx=turn_idx,
                    memory_base=memory_base,
                    annotator_id=annotator_id
                )

                annotation_set.add_annotation(query_id, gt)

        return annotation_set

    def _create_ground_truth_from_turn(
        self,
        scenario: Scenario,
        turn_idx: int,
        memory_base: MemoryBase,
        annotator_id: str
    ) -> GroundTruth:
        """
        Create ground truth for a single turn.

        Uses the relevant_memories field from the scenario turn.
        In a real annotation workflow, this would be done manually.
        """
        turn = scenario.turns[turn_idx]

        # Convert paths to RelevantMemory objects
        # For now, mark all as "high" relevance (can be refined later)
        relevant_memories = [
            RelevantMemory(
                file_path=path,
                relevance="high",
                section=None,
                notes="Auto-generated from scenario"
            )
            for path in turn.relevant_memories
        ]

        # Get all memory files and mark non-relevant ones
        all_memory_paths = {f.path for f in memory_base.all_files}
        relevant_paths = set(turn.relevant_memories)
        irrelevant_paths = list(all_memory_paths - relevant_paths)

        return GroundTruth(
            query=turn.message,
            scenario_id=scenario.scenario_id,
            turn_index=turn_idx,
            relevant_memories=relevant_memories,
            irrelevant_memories=irrelevant_paths,
            annotator_id=annotator_id,
            timestamp=datetime.now().isoformat()
        )

    # ── Storage ──────────────────────────────────────────────────────────────

    def save_annotations(
        self,
        annotation_set: AnnotationSet,
        filename: str | None = None
    ) -> Path:
        """
        Save annotation set to JSON file.

        Args:
            annotation_set: Annotation set to save
            filename: Optional custom filename (defaults to annotator_id.json)

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"annotations_{annotation_set.annotator_id}.json"

        output_path = self.output_dir / filename

        # Convert to JSON-serializable format
        data = {
            "annotator_id": annotation_set.annotator_id,
            "metadata": annotation_set.metadata,
            "annotations": {
                query_id: self._ground_truth_to_dict(gt)
                for query_id, gt in annotation_set.annotations.items()
            }
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return output_path

    def load_annotations(self, filename: str) -> AnnotationSet:
        """
        Load annotation set from JSON file.

        Args:
            filename: Name of file to load

        Returns:
            Loaded annotation set
        """
        file_path = self.output_dir / filename

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        annotation_set = AnnotationSet(
            annotator_id=data["annotator_id"],
            metadata=data["metadata"]
        )

        for query_id, gt_dict in data["annotations"].items():
            gt = self._dict_to_ground_truth(gt_dict)
            annotation_set.add_annotation(query_id, gt)

        return annotation_set

    def _ground_truth_to_dict(self, gt: GroundTruth) -> dict[str, Any]:
        """Convert GroundTruth to JSON-serializable dict."""
        return {
            "query": gt.query,
            "scenario_id": gt.scenario_id,
            "turn_index": gt.turn_index,
            "relevant_memories": [
                {
                    "file_path": str(rm.file_path),
                    "relevance": rm.relevance,
                    "section": rm.section,
                    "notes": rm.notes
                }
                for rm in gt.relevant_memories
            ],
            "irrelevant_memories": [str(p) for p in gt.irrelevant_memories],
            "annotator_id": gt.annotator_id,
            "timestamp": gt.timestamp
        }

    def _dict_to_ground_truth(self, data: dict[str, Any]) -> GroundTruth:
        """Convert dict to GroundTruth object."""
        return GroundTruth(
            query=data["query"],
            scenario_id=data["scenario_id"],
            turn_index=data["turn_index"],
            relevant_memories=[
                RelevantMemory(
                    file_path=Path(rm["file_path"]),
                    relevance=rm["relevance"],
                    section=rm.get("section"),
                    notes=rm.get("notes", "")
                )
                for rm in data["relevant_memories"]
            ],
            irrelevant_memories=[Path(p) for p in data["irrelevant_memories"]],
            annotator_id=data["annotator_id"],
            timestamp=data["timestamp"]
        )

    # ── Inter-Annotator Agreement ────────────────────────────────────────────

    def calculate_agreement(
        self,
        annotator1: AnnotationSet,
        annotator2: AnnotationSet
    ) -> dict[str, float]:
        """
        Calculate inter-annotator agreement using Cohen's κ.

        Compares two annotation sets on:
        1. Binary relevance (relevant vs. irrelevant)
        2. Relevance level (high/medium/low) for relevant items

        Args:
            annotator1: First annotator's annotations
            annotator2: Second annotator's annotations

        Returns:
            Dictionary with agreement metrics:
            - cohens_kappa: Overall Cohen's κ
            - agreement_rate: Raw agreement percentage
            - num_queries: Number of queries compared
        """
        # Get common query IDs
        common_queries = set(annotator1.annotations.keys()) & set(annotator2.annotations.keys())

        if not common_queries:
            return {
                "cohens_kappa": 0.0,
                "agreement_rate": 0.0,
                "num_queries": 0
            }

        # Collect binary relevance judgments
        agreements = []
        total_items = 0

        for query_id in common_queries:
            gt1 = annotator1.get_annotation(query_id)
            gt2 = annotator2.get_annotation(query_id)

            if gt1 is None or gt2 is None:
                continue

            # Get all memory files mentioned by either annotator
            all_files_1 = {rm.file_path for rm in gt1.relevant_memories} | set(gt1.irrelevant_memories)
            all_files_2 = {rm.file_path for rm in gt2.relevant_memories} | set(gt2.irrelevant_memories)
            all_files = all_files_1 | all_files_2

            # Compare binary relevance for each file
            for file_path in all_files:
                is_relevant_1 = any(rm.file_path == file_path for rm in gt1.relevant_memories)
                is_relevant_2 = any(rm.file_path == file_path for rm in gt2.relevant_memories)

                agreements.append(1 if is_relevant_1 == is_relevant_2 else 0)
                total_items += 1

        # Calculate metrics
        if not agreements:
            return {
                "cohens_kappa": 0.0,
                "agreement_rate": 0.0,
                "num_queries": len(common_queries)
            }

        agreement_rate = sum(agreements) / len(agreements)
        cohens_kappa = self._calculate_cohens_kappa(agreements)

        return {
            "cohens_kappa": cohens_kappa,
            "agreement_rate": agreement_rate,
            "num_queries": len(common_queries),
            "num_items_compared": total_items
        }

    def _calculate_cohens_kappa(self, agreements: list[int]) -> float:
        """
        Calculate Cohen's κ from binary agreement list.

        κ = (Po - Pe) / (1 - Pe)
        where Po = observed agreement, Pe = expected agreement by chance
        """
        if not agreements:
            return 0.0

        # Observed agreement
        po = sum(agreements) / len(agreements)

        # For binary classification with balanced classes, Pe ≈ 0.5
        # (This is a simplification; full κ calculation would track all categories)
        pe = 0.5

        if pe == 1.0:
            return 1.0  # Perfect agreement expected by chance

        kappa = (po - pe) / (1 - pe)
        return kappa

    # ── Utility Methods ──────────────────────────────────────────────────────

    def save_agreement_report(
        self,
        annotator1: AnnotationSet,
        annotator2: AnnotationSet,
        filename: str = "agreement_report.json"
    ) -> Path:
        """
        Calculate and save agreement report to file.

        Args:
            annotator1: First annotator's annotations
            annotator2: Second annotator's annotations
            filename: Output filename

        Returns:
            Path to saved report
        """
        agreement = self.calculate_agreement(annotator1, annotator2)

        report = {
            "annotator_1": annotator1.annotator_id,
            "annotator_2": annotator2.annotator_id,
            "timestamp": datetime.now().isoformat(),
            "agreement_metrics": agreement,
            "interpretation": self._interpret_kappa(agreement["cohens_kappa"])
        }

        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return output_path

    def _interpret_kappa(self, kappa: float) -> str:
        """
        Interpret Cohen's κ value according to Landis & Koch (1977).

        Returns:
            Interpretation string
        """
        if kappa < 0:
            return "Poor (less than chance agreement)"
        elif kappa < 0.20:
            return "Slight"
        elif kappa < 0.40:
            return "Fair"
        elif kappa < 0.60:
            return "Moderate"
        elif kappa < 0.80:
            return "Substantial"
        else:
            return "Almost perfect"
