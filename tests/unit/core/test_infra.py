from __future__ import annotations

"""Tests for core.infra — infrastructure auto-start logic."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.infra import (
    _animas_need_neo4j,
    _is_port_open,
    _resolve_compose_command,
    _wait_for_neo4j,
    ensure_infra_services,
)

# ── _animas_need_neo4j ──────────────────────────────────────────────────


class TestAnimasNeedNeo4j:
    def test_detects_neo4j_backend(self, tmp_path: Path):
        (tmp_path / "alice").mkdir()
        (tmp_path / "alice" / "status.json").write_text(json.dumps({"enabled": True, "memory_backend": "neo4j"}))
        (tmp_path / "bob").mkdir()
        (tmp_path / "bob" / "status.json").write_text(json.dumps({"enabled": True, "memory_backend": "legacy"}))
        result = _animas_need_neo4j(tmp_path, ["alice", "bob"])
        assert result == ["alice"]

    def test_no_neo4j_returns_empty(self, tmp_path: Path):
        (tmp_path / "alice").mkdir()
        (tmp_path / "alice" / "status.json").write_text(json.dumps({"enabled": True}))
        result = _animas_need_neo4j(tmp_path, ["alice"])
        assert result == []

    def test_missing_status_json_skipped(self, tmp_path: Path):
        (tmp_path / "alice").mkdir()
        result = _animas_need_neo4j(tmp_path, ["alice"])
        assert result == []

    def test_corrupt_status_json_skipped(self, tmp_path: Path):
        (tmp_path / "alice").mkdir()
        (tmp_path / "alice" / "status.json").write_text("{invalid json")
        result = _animas_need_neo4j(tmp_path, ["alice"])
        assert result == []

    def test_multiple_neo4j_animas(self, tmp_path: Path):
        for name in ("a", "b", "c"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "status.json").write_text(json.dumps({"memory_backend": "neo4j"}))
        result = _animas_need_neo4j(tmp_path, ["a", "b", "c"])
        assert result == ["a", "b", "c"]


# ── _is_port_open ───────────────────────────────────────────────────────


class TestIsPortOpen:
    def test_closed_port(self):
        assert _is_port_open("127.0.0.1", 19999, timeout=0.5) is False

    @patch("core.infra.socket.create_connection")
    def test_open_port(self, mock_conn: MagicMock):
        mock_conn.return_value.__enter__ = MagicMock()
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)
        assert _is_port_open("127.0.0.1", 7687) is True


# ── _resolve_compose_command ─────────────────────────────────────────────


class TestResolveComposeCommand:
    @patch("shutil.which", side_effect=lambda x: "/usr/bin/docker" if x == "docker" else None)
    def test_docker_compose_v2(self, _mock: MagicMock):
        assert _resolve_compose_command() == ["docker", "compose"]

    @patch(
        "shutil.which",
        side_effect=lambda x: "/usr/local/bin/docker-compose" if x == "docker-compose" else None,
    )
    def test_docker_compose_v1(self, _mock: MagicMock):
        assert _resolve_compose_command() == ["docker-compose"]

    @patch("shutil.which", return_value=None)
    def test_no_docker(self, _mock: MagicMock):
        assert _resolve_compose_command() is None


# ── ensure_infra_services ────────────────────────────────────────────────


class TestEnsureInfraServices:
    @pytest.mark.asyncio
    async def test_no_neo4j_animas_is_noop(self, tmp_path: Path):
        (tmp_path / "animas" / "alice").mkdir(parents=True)
        (tmp_path / "animas" / "alice" / "status.json").write_text(json.dumps({"enabled": True}))
        with patch("core.infra._is_port_open") as mock_port:
            await ensure_infra_services(tmp_path / "animas", ["alice"], tmp_path)
            mock_port.assert_not_called()

    @pytest.mark.asyncio
    async def test_neo4j_already_running_skips_compose(self, tmp_path: Path):
        (tmp_path / "animas" / "sakura").mkdir(parents=True)
        (tmp_path / "animas" / "sakura" / "status.json").write_text(json.dumps({"memory_backend": "neo4j"}))
        with (
            patch("core.infra._is_port_open", return_value=True) as mock_port,
            patch("core.infra._run_docker_compose") as mock_compose,
        ):
            await ensure_infra_services(tmp_path / "animas", ["sakura"], tmp_path)
            assert mock_port.call_count >= 1
            mock_compose.assert_not_called()

    @pytest.mark.asyncio
    async def test_starts_neo4j_when_not_running(self, tmp_path: Path):
        (tmp_path / "animas" / "sakura").mkdir(parents=True)
        (tmp_path / "animas" / "sakura" / "status.json").write_text(json.dumps({"memory_backend": "neo4j"}))
        compose_file = tmp_path / "docker-compose.neo4j.yml"
        compose_file.write_text("services:\n  neo4j:\n    image: neo4j:5\n")

        with (
            patch("core.infra._is_port_open", return_value=False),
            patch("core.infra._run_docker_compose", new_callable=AsyncMock, return_value=True) as mock_compose,
            patch("core.infra._wait_for_neo4j", new_callable=AsyncMock, return_value=True),
        ):
            await ensure_infra_services(tmp_path / "animas", ["sakura"], tmp_path)
            mock_compose.assert_called_once_with(compose_file)

    @pytest.mark.asyncio
    async def test_missing_compose_file_warns(self, tmp_path: Path):
        (tmp_path / "animas" / "sakura").mkdir(parents=True)
        (tmp_path / "animas" / "sakura" / "status.json").write_text(json.dumps({"memory_backend": "neo4j"}))
        with patch("core.infra._is_port_open", return_value=False):
            await ensure_infra_services(tmp_path / "animas", ["sakura"], tmp_path)

    @pytest.mark.asyncio
    async def test_compose_failure_does_not_raise(self, tmp_path: Path):
        (tmp_path / "animas" / "sakura").mkdir(parents=True)
        (tmp_path / "animas" / "sakura" / "status.json").write_text(json.dumps({"memory_backend": "neo4j"}))
        compose_file = tmp_path / "docker-compose.neo4j.yml"
        compose_file.write_text("services:\n  neo4j:\n    image: neo4j:5\n")

        with (
            patch("core.infra._is_port_open", return_value=False),
            patch("core.infra._run_docker_compose", new_callable=AsyncMock, return_value=False),
        ):
            await ensure_infra_services(tmp_path / "animas", ["sakura"], tmp_path)


# ── _wait_for_neo4j ─────────────────────────────────────────────────────


class TestWaitForNeo4j:
    @pytest.mark.asyncio
    async def test_immediately_ready(self):
        with patch("core.infra._is_port_open", return_value=True):
            assert await _wait_for_neo4j(timeout=5) is True

    @pytest.mark.asyncio
    async def test_timeout(self):
        with (
            patch("core.infra._is_port_open", return_value=False),
            patch("core.infra.asyncio.sleep", new_callable=AsyncMock),
        ):
            assert await _wait_for_neo4j(timeout=1) is False
