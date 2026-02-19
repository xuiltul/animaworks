# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Memory Performance Evaluation Framework.

This package provides the core framework for evaluating AnimaWorks' memory system
performance according to the research protocol defined in:
docs/research/memory-performance-evaluation-protocol.md
"""

# Phase 2: Dataset Generation (standalone, no numpy dependency)
from .dataset_generator import DatasetGenerator
from .ground_truth import GroundTruthManager
from .schemas import (
    AnnotationSet,
    ConversationTurn,
    GroundTruth,
    MemoryBase,
    MemoryFile,
    RelevantMemory,
    ScenarioTypeConfig,
    SizeConfig,
)

# Phase 1: Experiment framework (requires numpy for metrics)
try:
    from .config import (
        ConversationLength,
        ConversationMetrics,
        Domain,
        ExperimentConfig,
        MemorySize,
        ScenarioType,
        SearchConfig,
        SearchMethod,
        TurnMetrics,
    )
    from .framework import MemoryExperiment, Scenario, Turn
    from .logger import ExperimentLogger
    from .metrics import MetricsCollector

    _PHASE1_AVAILABLE = True
except ImportError:
    # Phase 1 components not available (missing numpy or other deps)
    _PHASE1_AVAILABLE = False
    ConversationLength = None  # type: ignore
    ConversationMetrics = None  # type: ignore
    Domain = None  # type: ignore
    ExperimentConfig = None  # type: ignore
    MemorySize = None  # type: ignore
    ScenarioType = None  # type: ignore
    SearchConfig = None  # type: ignore
    SearchMethod = None  # type: ignore
    TurnMetrics = None  # type: ignore
    MemoryExperiment = None  # type: ignore
    Scenario = None  # type: ignore
    Turn = None  # type: ignore
    ExperimentLogger = None  # type: ignore
    MetricsCollector = None  # type: ignore

# Phase 3-4: Statistical Analysis and Visualization
try:
    from .analysis import StatisticalAnalyzer
    from .visualization import ExperimentVisualizer

    _PHASE34_AVAILABLE = True
except ImportError:
    # Phase 3-4 components not available (missing scipy/matplotlib etc)
    _PHASE34_AVAILABLE = False
    StatisticalAnalyzer = None  # type: ignore
    ExperimentVisualizer = None  # type: ignore

__all__ = [
    # Phase 2: Dataset Generation (always available)
    "DatasetGenerator",
    "GroundTruthManager",
    "MemoryFile",
    "MemoryBase",
    "ConversationTurn",
    "GroundTruth",
    "RelevantMemory",
    "AnnotationSet",
    "SizeConfig",
    "ScenarioTypeConfig",
]

# Add Phase 1 components if available
if _PHASE1_AVAILABLE:
    __all__.extend([
        # Config
        "ConversationLength",
        "ConversationMetrics",
        "Domain",
        "ExperimentConfig",
        "MemorySize",
        "ScenarioType",
        "SearchConfig",
        "SearchMethod",
        "TurnMetrics",
        # Framework
        "MemoryExperiment",
        "Scenario",
        "Turn",
        # Tools
        "ExperimentLogger",
        "MetricsCollector",
    ])

# Add Phase 3-4 components if available
if _PHASE34_AVAILABLE:
    __all__.extend([
        # Analysis & Visualization
        "StatisticalAnalyzer",
        "ExperimentVisualizer",
    ])
