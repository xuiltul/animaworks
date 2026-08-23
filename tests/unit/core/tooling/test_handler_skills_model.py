"""submit_tasks model override validation."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.tooling.handler import ToolHandler


def _handler(tmp_path: Path) -> ToolHandler:
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    (anima_dir / "state").mkdir()
    return ToolHandler(anima_dir=anima_dir, memory=MagicMock(), tool_registry=[])


def _tasks(*models: str | None) -> list[dict]:
    tasks = []
    for i, model in enumerate(models, start=1):
        item = {
            "task_id": f"task-{i}",
            "title": f"Task {i}",
            "description": f"Do thing {i}",
        }
        if model is not None:
            item["model"] = model
        tasks.append(item)
    return tasks


class TestSubmitTasksModelValidation:
    def test_valid_model_succeeds(self, tmp_path: Path) -> None:
        handler = _handler(tmp_path)
        with patch("core.config.model_catalog.validate_model_override", return_value=None):
            result = handler.handle(
                "submit_tasks",
                {"batch_id": "b1", "tasks": _tasks("openai/gpt-4.1")},
            )
        parsed = json.loads(result)
        assert parsed.get("status") == "submitted"
        pending = handler._anima_dir / "state" / "pending" / "task-1.json"
        assert pending.exists()
        assert json.loads(pending.read_text(encoding="utf-8"))["model"] == "openai/gpt-4.1"

    def test_unknown_model_returns_invalid_arguments(self, tmp_path: Path) -> None:
        handler = _handler(tmp_path)
        with patch(
            "core.config.model_catalog.validate_model_override",
            return_value="unknown model override 'not-a-real-model'",
        ):
            result = handler.handle(
                "submit_tasks",
                {"batch_id": "b1", "tasks": _tasks("not-a-real-model")},
            )
        parsed = json.loads(result)
        assert parsed["error_type"] == "InvalidArguments"
        assert "not-a-real-model" in parsed["message"]
        assert "task-1" in parsed["message"]
        assert "available-models" in parsed["message"]
        assert not (handler._anima_dir / "state" / "pending" / "task-1.json").exists()

    def test_one_invalid_in_batch_rejects_all(self, tmp_path: Path) -> None:
        handler = _handler(tmp_path)

        def _validate(_anima: str, model: str):
            if model == "bad-model":
                return "unknown model override 'bad-model'"
            return None

        with patch("core.config.model_catalog.validate_model_override", side_effect=_validate):
            result = handler.handle(
                "submit_tasks",
                {
                    "batch_id": "b1",
                    "tasks": _tasks("openai/gpt-4.1", "bad-model"),
                },
            )
        parsed = json.loads(result)
        assert parsed["error_type"] == "InvalidArguments"
        assert "task-2" in parsed["message"]
        assert "bad-model" in parsed["message"]
        pending_dir = handler._anima_dir / "state" / "pending"
        assert list(pending_dir.glob("*.json")) == []

    def test_omitted_model_keeps_legacy_behavior(self, tmp_path: Path) -> None:
        handler = _handler(tmp_path)
        with patch("core.config.model_catalog.validate_model_override") as validate:
            result = handler.handle(
                "submit_tasks",
                {"batch_id": "b1", "tasks": _tasks(None)},
            )
            validate.assert_not_called()
        parsed = json.loads(result)
        assert parsed.get("status") == "submitted"
        pending = handler._anima_dir / "state" / "pending" / "task-1.json"
        assert json.loads(pending.read_text(encoding="utf-8")).get("model") == ""
