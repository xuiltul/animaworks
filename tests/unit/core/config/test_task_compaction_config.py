"""Per-anima task compaction configuration propagation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from core.config.model_config import load_model_config
from core.config.resolver import resolve_anima_config
from core.config.schemas import AnimaWorksConfig
from core.memory.config_reader import ConfigReader


def test_status_json_task_compaction_settings_reach_model_config(tmp_path: Path) -> None:
    anima_dir = tmp_path / "sora"
    anima_dir.mkdir()
    (anima_dir / "status.json").write_text(
        json.dumps({"task_compaction_tokens": 80_000, "task_compaction_max": 4}),
        encoding="utf-8",
    )
    config = AnimaWorksConfig()
    resolved, _ = resolve_anima_config(config, "sora", anima_dir=anima_dir)

    config_path = tmp_path / "config.json"
    config_path.touch()
    with (
        patch("core.config.models.get_config_path", return_value=config_path),
        patch("core.config.models.load_config", return_value=config),
        patch("core.config.get_config_path", return_value=config_path),
        patch("core.config.load_config", return_value=config),
        patch("core.config.resolve_execution_mode", return_value="S"),
    ):
        model_config = load_model_config(anima_dir)
        reader_config = ConfigReader(anima_dir).read_model_config()

    assert resolved.task_compaction_tokens == 80_000
    assert resolved.task_compaction_max == 4
    assert model_config.task_compaction_tokens == 80_000
    assert model_config.task_compaction_max == 4
    assert reader_config.task_compaction_tokens == 80_000
    assert reader_config.task_compaction_max == 4


def test_task_compaction_defaults_are_opt_in() -> None:
    defaults, _ = resolve_anima_config(AnimaWorksConfig(), "sora")

    assert defaults.task_compaction_tokens == 0
    assert defaults.task_compaction_max == 6
