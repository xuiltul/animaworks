"""Execute the battle's real JavaScript reducer, without a second implementation."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_battle_model_behaviors():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for frontend model tests")
    # Node 24 rejects --experimental-default-type; since Node 22.7 the imported
    # server/static .js files are detected as ES modules from their syntax.
    result = subprocess.run(
        [
            node,
            "--test",
            "tests/unit/server/battle_model.test.mjs",
            "tests/unit/server/battle_combat.test.mjs",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
