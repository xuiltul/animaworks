"""Tests for swe/team_setup.py."""
from __future__ import annotations

import json

import pytest

from swe.team_setup import setup_team, teardown_team


@pytest.fixture
def runtime_dir(tmp_path):
    """Create a temporary isolated runtime directory."""
    rt = tmp_path / "swe-runtime"
    rt.mkdir()
    (rt / "animas").mkdir()
    config = {
        "credentials": {"vllm-local": {"type": "api_key", "api_key": "dummy"}},
        "animas": {},
    }
    (rt / "config.json").write_text(json.dumps(config))
    return rt


@pytest.fixture
def team_config(tmp_path):
    """Create a minimal team config."""
    cfg = {
        "port": 18502,
        "timeout_minutes": 30,
        "agents": {
            "test-arch": {
                "model": "claude-sonnet-4-6",
                "role": "engineer",
                "supervisor": None,
                "credential": None,
                "identity": "Test architect",
                "injection": "## Role\nTest",
            },
            "test-inv": {
                "model": "openai/qwen3.5-35b-a3b",
                "role": "researcher",
                "supervisor": "test-arch",
                "credential": "vllm-local",
                "identity": "Test investigator",
                "injection": "## Role\nTest",
            },
        },
    }
    path = tmp_path / "team.json"
    path.write_text(json.dumps(cfg))
    return path


class TestSetupTeam:
    def test_creates_agent_directories(self, runtime_dir, team_config):
        names, _ = setup_team(team_config, runtime_dir=runtime_dir)

        assert set(names) == {"test-arch", "test-inv"}
        assert (runtime_dir / "animas" / "test-arch").is_dir()
        assert (runtime_dir / "animas" / "test-inv").is_dir()

    def test_creates_status_json(self, runtime_dir, team_config):
        setup_team(team_config, runtime_dir=runtime_dir)

        status = json.loads(
            (runtime_dir / "animas" / "test-arch" / "status.json").read_text()
        )
        assert status["model"] == "claude-sonnet-4-6"
        assert status["enabled"] is True
        assert status["supervisor"] is None

    def test_creates_identity_md(self, runtime_dir, team_config):
        setup_team(team_config, runtime_dir=runtime_dir)

        identity = (runtime_dir / "animas" / "test-arch" / "identity.md").read_text()
        assert "Test architect" in identity

    def test_creates_injection_md(self, runtime_dir, team_config):
        setup_team(team_config, runtime_dir=runtime_dir)

        injection = (runtime_dir / "animas" / "test-inv" / "injection.md").read_text()
        assert "## Role" in injection

    def test_creates_subdirectories(self, runtime_dir, team_config):
        setup_team(team_config, runtime_dir=runtime_dir)

        agent_dir = runtime_dir / "animas" / "test-arch"
        assert (agent_dir / "state").is_dir()
        assert (agent_dir / "state" / "pending").is_dir()
        assert (agent_dir / "episodes").is_dir()
        assert (agent_dir / "knowledge").is_dir()

    def test_sets_credential(self, runtime_dir, team_config):
        setup_team(team_config, runtime_dir=runtime_dir)

        status = json.loads(
            (runtime_dir / "animas" / "test-inv" / "status.json").read_text()
        )
        assert status["credential"] == "vllm-local"

    def test_registers_in_config(self, runtime_dir, team_config):
        setup_team(team_config, runtime_dir=runtime_dir)

        config = json.loads((runtime_dir / "config.json").read_text())
        assert "test-arch" in config["animas"]
        assert "test-inv" in config["animas"]

    def test_idempotent(self, runtime_dir, team_config):
        setup_team(team_config, runtime_dir=runtime_dir)
        names, _ = setup_team(team_config, runtime_dir=runtime_dir)

        assert set(names) == {"test-arch", "test-inv"}

    def test_returns_runtime_dir(self, runtime_dir, team_config):
        _, returned_dir = setup_team(team_config, runtime_dir=runtime_dir)
        assert returned_dir == runtime_dir


class TestTeardownTeam:
    def test_removes_runtime(self, runtime_dir, team_config):
        setup_team(team_config, runtime_dir=runtime_dir)
        teardown_team(team_config, runtime_dir=runtime_dir)

        assert not runtime_dir.exists()

    def test_noop_if_not_exists(self, tmp_path, team_config):
        teardown_team(team_config, runtime_dir=tmp_path / "nonexistent")
