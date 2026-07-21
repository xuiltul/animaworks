from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from unittest.mock import patch

from cli.commands.cost_cmd import cmd_cost


def _args(*, anima: str | None = None, json_output: bool = False) -> argparse.Namespace:
    return argparse.Namespace(anima=anima, days=30, today=False, json_output=json_output)


def test_cost_json_includes_monthly_budget_fields_for_every_anima(tmp_path, capsys) -> None:
    data_dir = tmp_path / "data"
    alice = data_dir / "animas" / "alice"
    bob = data_dir / "animas" / "bob"
    alice.mkdir(parents=True)
    bob.mkdir(parents=True)
    (alice / "token_usage").mkdir()
    (bob / "token_usage").mkdir()

    def budget_status(anima_dir, *, now):
        if anima_dir.name == "alice":
            return SimpleNamespace(budget=1_000, consumed=1_100, remaining=0, exceeded=True)
        return SimpleNamespace(budget=None, consumed=25, remaining=None, exceeded=False)

    with (
        patch("core.paths.get_data_dir", return_value=data_dir),
        patch("core.memory.token_budget.read_token_budget_status", side_effect=budget_status),
    ):
        cmd_cost(_args(json_output=True))

    result = json.loads(capsys.readouterr().out)
    assert {key: result["alice"][key] for key in ("budget", "consumed", "remaining", "exceeded")} == {
        "budget": 1_000,
        "consumed": 1_100,
        "remaining": 0,
        "exceeded": True,
    }
    assert result["bob"]["budget"] is None
    assert result["bob"]["consumed"] == 25
    assert result["bob"]["remaining"] is None
    assert result["bob"]["exceeded"] is False


def test_cost_text_displays_unlimited_budget_as_dash(tmp_path, capsys) -> None:
    data_dir = tmp_path / "data"
    anima_dir = data_dir / "animas" / "alice"
    anima_dir.mkdir(parents=True)
    status = SimpleNamespace(budget=None, consumed=25, remaining=None, exceeded=False)

    with (
        patch("core.paths.get_data_dir", return_value=data_dir),
        patch("core.memory.token_budget.read_token_budget_status", return_value=status),
    ):
        cmd_cost(_args(anima="alice"))

    output = capsys.readouterr().out
    assert "Monthly budget: -" in output
    assert "Consumed: 25" in output
    assert "Remaining: -" in output
    assert "Exceeded: no" in output


def test_cost_all_preserves_unconfigured_zero_usage_filter(tmp_path, capsys) -> None:
    data_dir = tmp_path / "data"
    anima_dir = data_dir / "animas" / "alice"
    anima_dir.mkdir(parents=True)

    with (
        patch("core.paths.get_data_dir", return_value=data_dir),
        patch("core.config.model_config.load_model_config", return_value=SimpleNamespace(token_budget_monthly=None)),
    ):
        cmd_cost(_args())

    assert capsys.readouterr().out.strip() == "No token usage data found. Start the server and send some messages."
    assert not (anima_dir / "token_usage").exists()
