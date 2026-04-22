from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for MemoryConfig schema."""

import pytest
from pydantic import ValidationError

from core.config.schemas import AnimaWorksConfig, MemoryConfig


def test_memory_config_default() -> None:
    cfg = MemoryConfig()
    assert cfg.backend == "legacy"


def test_memory_config_neo4j() -> None:
    cfg = MemoryConfig(backend="neo4j")
    assert cfg.backend == "neo4j"


def test_memory_config_invalid() -> None:
    with pytest.raises(ValidationError):
        MemoryConfig(backend="invalid")


def test_animaworks_config_has_memory() -> None:
    cfg = AnimaWorksConfig()
    assert hasattr(cfg, "memory")
    assert isinstance(cfg.memory, MemoryConfig)
    assert cfg.memory.backend == "legacy"
