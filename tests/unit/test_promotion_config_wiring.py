from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for promotion config wiring and deprecated flag handling."""

import json
import logging
from pathlib import Path
from unittest.mock import patch

from core.config.schemas import AnimaWorksConfig, SkillPromotionConfig
from core.skills.autolearn import AutonomousSkillLearner
from core.skills.promotion import PromotionPolicy
from core.time_utils import now_iso


def _write_procedure(anima_dir: Path, name: str, *, success_count: int) -> Path:
    procedures = anima_dir / "procedures"
    procedures.mkdir(parents=True, exist_ok=True)
    path = procedures / f"{name}.md"
    path.write_text(
        "---\n"
        f"name: {name}\n"
        f"description: Procedure for {name}\n"
        f"success_count: {success_count}\n"
        "failure_count: 0\n"
        "confidence: 0.95\n"
        f"last_used: {now_iso()}\n"
        "domains: [operations]\n"
        "trigger_phrases: [run deploy]\n"
        "---\n\n"
        f"# {name}\n\nStep 1: Do the safe thing.\n",
        encoding="utf-8",
    )
    return path


def _config_with_threshold(threshold: int) -> AnimaWorksConfig:
    config = AnimaWorksConfig()
    config.skills.promotion = SkillPromotionConfig(success_count_threshold=threshold)
    return config


def test_from_config_maps_threshold_fields() -> None:
    pcfg = SkillPromotionConfig(
        success_count_threshold=7,
        confidence_threshold=0.6,
        failure_count_max=2,
        last_used_within_days=30,
    )
    policy = PromotionPolicy.from_config(pcfg)
    assert policy.success_count_threshold == 7
    assert policy.confidence_threshold == 0.6
    assert policy.failure_count_max == 2
    assert policy.last_used_within_days == 30


def test_config_threshold_reflected_in_autolearner(tmp_path: Path) -> None:
    """A procedure with success_count=4 is eligible at threshold 3 but not 5."""
    anima_dir = tmp_path / "alice"
    _write_procedure(anima_dir, "deploy-flow", success_count=4)

    with patch("core.config.models.load_config", return_value=_config_with_threshold(3)):
        learner_lenient = AutonomousSkillLearner(anima_dir)
    candidates_lenient = learner_lenient.converter.find_candidates(eligible_only=True)
    assert any(c.name == "deploy-flow" for c in candidates_lenient)

    with patch("core.config.models.load_config", return_value=_config_with_threshold(5)):
        learner_strict = AutonomousSkillLearner(anima_dir)
    candidates_strict = learner_strict.converter.find_candidates(eligible_only=True)
    assert not any(c.name == "deploy-flow" for c in candidates_strict)


def test_autolearner_falls_back_to_default_policy_on_config_error(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    with patch("core.config.models.load_config", side_effect=RuntimeError("boom")):
        learner = AutonomousSkillLearner(anima_dir)
    assert learner.converter.policy.success_count_threshold == PromotionPolicy().success_count_threshold


def test_explicit_converter_takes_precedence(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    from core.skills.promotion import ProcedureToSkillConverter

    explicit = ProcedureToSkillConverter(anima_dir, policy=PromotionPolicy(success_count_threshold=99))
    with patch("core.config.models.load_config", return_value=_config_with_threshold(3)):
        learner = AutonomousSkillLearner(anima_dir, converter=explicit)
    assert learner.converter is explicit
    assert learner.converter.policy.success_count_threshold == 99


def test_deprecated_flag_warns_once_and_continues(tmp_path: Path, caplog) -> None:
    import core.config.io as io

    config_path = tmp_path / "config.json"
    data = AnimaWorksConfig().model_dump(mode="json")
    data["skills"]["promotion"]["auto_activate"] = True
    config_path.write_text(json.dumps(data), encoding="utf-8")

    io.invalidate_cache()
    with caplog.at_level(logging.WARNING, logger="animaworks.config"):
        config = io.load_config(config_path)
    # Startup continues: config is loaded normally, flag value preserved but no-op.
    assert config.skills.promotion.auto_activate is True
    warnings = [r for r in caplog.records if "deprecated no-op" in r.getMessage()]
    assert len(warnings) == 1

    # Second load (same process, cache hit) must not re-warn.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="animaworks.config"):
        io.load_config(config_path)
    assert not [r for r in caplog.records if "deprecated no-op" in r.getMessage()]
    io.invalidate_cache()


def test_default_flags_do_not_warn(tmp_path: Path, caplog) -> None:
    import core.config.io as io

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(AnimaWorksConfig().model_dump(mode="json")), encoding="utf-8")

    io.invalidate_cache()
    with caplog.at_level(logging.WARNING, logger="animaworks.config"):
        io.load_config(config_path)
    assert not [r for r in caplog.records if "deprecated no-op" in r.getMessage()]
    io.invalidate_cache()
