"""Opt-in actual process integration, never reads or mounts the live runtime."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("ANIMAWORKS_RUN_DOCKER_SMOKE") != "1", reason="explicit Docker smoke opt-in required"
)
def test_slim_runtime_real_watcher_docker(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    process = subprocess.run(
        [sys.executable, str(repo / "scripts/slim_runtime_sandbox.py"), "--output", str(tmp_path / "sandbox")],
        cwd=repo,
        text=True,
        capture_output=True,
        timeout=270,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads((tmp_path / "sandbox/result.json").read_text())
    assert result["http_health"] == 200
    assert result["worker_task_model_calls"] > 0
    assert result["cron_contract_matches"] == result["command_runs"] == 12
    pipeline = result["watcher_pipeline"]
    assert pipeline["real_watcher"] and pipeline["http_submission"]
    assert pipeline["resume_attempts"] == 2
    assert pipeline["restart_no_replay"] and pipeline["dependency_order_verified"]
    assert {key: value["status"] for key, value in pipeline["tasks"].items()} == {
        "waiting": "done",
        "dependent": "done",
        "cancelled": "cancelled",
        "running-cancel": "cancelled",
    }
    assert len(pipeline["attempts"]) == 4
    assert pipeline["quiet_provider_cancel_seconds"] < 12
    assert pipeline["duplicate_ready_submission_one_attempt"] and pipeline["stale_finish_rejected"]
    assert pipeline["model_calls_by_task"] == {"waiting": 2, "dependent": 2, "running-cancel": 1}
    assert all(attempt["ended_at"] for attempt in pipeline["attempts"])
    assert all(not task["active"] for task in pipeline["tasks"].values())
    assert result["quality_evaluation"] == "not_run_deterministic_model"
