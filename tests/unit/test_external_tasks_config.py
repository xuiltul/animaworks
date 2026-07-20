# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ExternalTasksConfig defaults and AnimaWorksConfig wiring."""

from __future__ import annotations

from core.config.schemas import AnimaWorksConfig, ExternalTasksConfig, ExternalTasksSourcesConfig


def test_external_tasks_config_defaults_disabled() -> None:
    cfg = ExternalTasksConfig()
    assert cfg.enabled is False
    assert cfg.interval_minutes == 5
    assert isinstance(cfg.sources, ExternalTasksSourcesConfig)
    assert cfg.sources.github is True
    assert cfg.sources.slack is True
    assert cfg.sources.chatwork is True
    assert cfg.sources.gmail is True


def test_anima_works_config_includes_external_tasks() -> None:
    root = AnimaWorksConfig()
    assert isinstance(root.external_tasks, ExternalTasksConfig)
    assert root.external_tasks.enabled is False
    assert root.external_tasks.interval_minutes == 5
    assert root.external_tasks.sources.github is True


def test_anima_works_config_loads_external_tasks_from_dict() -> None:
    root = AnimaWorksConfig.model_validate(
        {
            "external_tasks": {
                "enabled": True,
                "interval_minutes": 10,
                "sources": {"github": False, "slack": True},
            }
        }
    )
    assert root.external_tasks.enabled is True
    assert root.external_tasks.interval_minutes == 10
    assert root.external_tasks.sources.github is False
    assert root.external_tasks.sources.slack is True
    # unspecified source flags keep defaults
    assert root.external_tasks.sources.chatwork is True
    assert root.external_tasks.sources.gmail is True
