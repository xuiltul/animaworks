"""delegate_task model override validation."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.config.models import AnimaModelConfig, AnimaWorksConfig
from core.tooling.handler_delegation import _pr_key_from_text


def _make_config(animas: dict[str, dict]) -> AnimaWorksConfig:
    cfg = AnimaWorksConfig()
    cfg.animas = {name: AnimaModelConfig(**data) for name, data in animas.items()}
    return cfg


@pytest.fixture()
def handler_with_sub(tmp_path: Path):
    from core.tooling.handler import ToolHandler

    animas_dir = tmp_path / "animas"
    boss_dir = animas_dir / "boss"
    alice_dir = animas_dir / "alice"
    (boss_dir / "state").mkdir(parents=True)
    (alice_dir / "state").mkdir(parents=True)
    (boss_dir / "status.json").write_text("{}", encoding="utf-8")
    (alice_dir / "status.json").write_text("{}", encoding="utf-8")

    handler = ToolHandler(
        anima_dir=boss_dir,
        memory=MagicMock(),
        messenger=MagicMock(),
        tool_registry=[],
    )
    handler._org_context = SimpleNamespace(subordinates=["alice"], descendants=["alice"])
    cfg = _make_config({"boss": {"supervisor": None}, "alice": {"supervisor": "boss"}})
    return handler, animas_dir, cfg


class TestDelegateTaskModelValidation:
    def test_valid_model_succeeds(self, handler_with_sub):
        handler, animas_dir, cfg = handler_with_sub
        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.tooling.handler_delegation.build_outgoing_origin_chain", return_value=["anima"]),
            patch("core.config.model_catalog.validate_model_override", return_value=None),
        ):
            result = handler.handle(
                "delegate_task",
                {
                    "name": "alice",
                    "instruction": "Do the thing",
                    "summary": "Thing",
                    "deadline": "1d",
                    "model": "openai/gpt-4.1",
                },
            )
        pending = list((animas_dir / "alice" / "state" / "pending").glob("*.json"))
        assert len(pending) == 1, result
        data = json.loads(pending[0].read_text(encoding="utf-8"))
        assert data["model"] == "openai/gpt-4.1"
        assert "InvalidArguments" not in result

    def test_unknown_model_returns_invalid_arguments(self, handler_with_sub):
        handler, animas_dir, cfg = handler_with_sub
        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.tooling.handler_delegation.build_outgoing_origin_chain", return_value=["anima"]),
            patch(
                "core.config.model_catalog.validate_model_override",
                return_value="unknown model override 'not-a-real-model'",
            ),
        ):
            result = handler.handle(
                "delegate_task",
                {
                    "name": "alice",
                    "instruction": "Do the thing",
                    "summary": "Thing",
                    "deadline": "1d",
                    "model": "not-a-real-model",
                },
            )
        parsed = json.loads(result)
        assert parsed["error_type"] == "InvalidArguments"
        assert "not-a-real-model" in parsed["message"]
        assert "available-models" in parsed["message"]
        pending = list((animas_dir / "alice" / "state" / "pending").glob("*.json"))
        assert pending == []

    def test_omitted_model_keeps_legacy_behavior(self, handler_with_sub):
        handler, animas_dir, cfg = handler_with_sub
        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.tooling.handler_delegation.build_outgoing_origin_chain", return_value=["anima"]),
            patch("core.config.model_catalog.validate_model_override") as validate,
        ):
            result = handler.handle(
                "delegate_task",
                {
                    "name": "alice",
                    "instruction": "Do the thing",
                    "summary": "Thing",
                    "deadline": "1d",
                },
            )
            validate.assert_not_called()
        pending = list((animas_dir / "alice" / "state" / "pending").glob("*.json"))
        assert len(pending) == 1, result
        data = json.loads(pending[0].read_text(encoding="utf-8"))
        assert data.get("model") == ""

    def test_empty_model_skips_validation(self, handler_with_sub):
        handler, animas_dir, cfg = handler_with_sub
        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.tooling.handler_delegation.build_outgoing_origin_chain", return_value=["anima"]),
            patch("core.config.model_catalog.validate_model_override") as validate,
        ):
            result = handler.handle(
                "delegate_task",
                {
                    "name": "alice",
                    "instruction": "Do the thing",
                    "summary": "Thing",
                    "deadline": "1d",
                    "model": "  ",
                },
            )
            validate.assert_not_called()
        pending = list((animas_dir / "alice" / "state" / "pending").glob("*.json"))
        assert len(pending) == 1, result


def test_pr_key_from_text() -> None:
    assert _pr_key_from_text("fix CI on PR #5008 now") == "pr-5008"
    assert _pr_key_from_text("see https://github.com/o/r/pull/4985 please") == "pr-4985"
    assert _pr_key_from_text("no reference here") == ""
    assert _pr_key_from_text("item #1 of the list") == ""
