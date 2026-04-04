# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Issue C (#155): Design fixes — layer violation removal, Governor notify, preset guard.

Tests:
1. core/_anima_heartbeat has no server.* imports
2. Governor _notify_supervisor sends message to supervisor
3. Governor _notify_supervisor calls call_human when no supervisor
4. apply_local_llm_presets_to_animas is guarded by auto_apply_presets
"""

import ast
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config.schemas import LocalLLMConfig

# ── Phase 1: Layer violation check ──────────────────────────────


class TestNoServerImportInHeartbeat:
    """core/_anima_heartbeat.py must not import from server.*."""

    def test_no_server_import(self):
        src_path = Path(__file__).resolve().parents[3] / "core" / "_anima_heartbeat.py"
        source = src_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("server"):
                violations.append(f"line {node.lineno}: from {node.module} import ...")

        assert violations == [], "server.* imports found in _anima_heartbeat.py:\n" + "\n".join(violations)


# ── Phase 2: Governor supervisor notification ───────────────────


class TestGovernorNotifySupervisor:
    """Governor._notify_supervisor sends to supervisor or call_human."""

    @pytest.fixture
    def governor(self, tmp_path):
        from server.usage_governor import UsageGovernor

        app = MagicMock()
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        gov = UsageGovernor(app, data_dir, animas_dir)
        return gov

    def _create_anima_status(self, animas_dir: Path, name: str, supervisor: str | None = None, enabled: bool = True):
        d = animas_dir / name
        d.mkdir(exist_ok=True)
        status = {"enabled": enabled, "role": "general"}
        if supervisor:
            status["supervisor"] = supervisor
        (d / "status.json").write_text(json.dumps(status), encoding="utf-8")

    @pytest.mark.asyncio
    async def test_notifies_supervisor(self, governor, tmp_path):
        animas_dir = tmp_path / "animas"
        self._create_anima_status(animas_dir, "worker", supervisor="boss")
        self._create_anima_status(animas_dir, "boss", enabled=True)

        with patch("core.messenger.Messenger") as MockMessenger:
            mock_instance = MagicMock()
            mock_instance.send = MagicMock()
            MockMessenger.return_value = mock_instance

            with patch("core.paths.get_shared_dir", return_value=tmp_path / "shared"):
                await governor._notify_supervisor("worker", "quota exceeded")

            mock_instance.send.assert_called_once()
            call_kwargs = mock_instance.send.call_args
            assert call_kwargs.kwargs["to"] == "boss"
            assert call_kwargs.kwargs["intent"] == "report"
            assert "worker" in call_kwargs.kwargs["content"]

    @pytest.mark.asyncio
    async def test_calls_human_when_no_supervisor(self, governor, tmp_path):
        animas_dir = tmp_path / "animas"
        self._create_anima_status(animas_dir, "top-anima")

        mock_notifier = MagicMock()
        mock_notifier.channel_count = 1
        mock_notifier.notify = AsyncMock()

        with patch("core.notification.notifier.HumanNotifier.from_config", return_value=mock_notifier):
            await governor._notify_supervisor("top-anima", "quota exceeded")
        mock_notifier.notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_calls_human_when_supervisor_disabled(self, governor, tmp_path):
        animas_dir = tmp_path / "animas"
        self._create_anima_status(animas_dir, "worker", supervisor="boss")
        self._create_anima_status(animas_dir, "boss", enabled=False)

        mock_notifier = MagicMock()
        mock_notifier.channel_count = 1
        mock_notifier.notify = AsyncMock()

        with patch("core.notification.notifier.HumanNotifier.from_config", return_value=mock_notifier):
            await governor._notify_supervisor("worker", "quota exceeded")
        mock_notifier.notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_crash_on_missing_status(self, governor, tmp_path):
        """No exception when anima has no status.json."""
        await governor._notify_supervisor("nonexistent", "test")


# ── Phase 3: Ollama preset guard ────────────────────────────────


class TestAutoApplyPresetsGuard:
    """apply_local_llm_presets_to_animas respects auto_apply_presets."""

    def test_default_false(self):
        cfg = LocalLLMConfig()
        assert cfg.auto_apply_presets is False

    def test_skips_when_disabled(self, tmp_path):
        from core.config.local_llm import apply_local_llm_presets_to_animas

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        (animas_dir / "test-anima").mkdir()
        (animas_dir / "test-anima" / "status.json").write_text(
            json.dumps({"role": "engineer", "model": "old-model", "credential": "ollama"}),
            encoding="utf-8",
        )

        config = MagicMock()
        config.local_llm = LocalLLMConfig(auto_apply_presets=False)

        result = apply_local_llm_presets_to_animas(animas_dir, config)
        assert result == []

        status = json.loads((animas_dir / "test-anima" / "status.json").read_text())
        assert status["model"] == "old-model"

    def test_applies_when_enabled(self, tmp_path):
        from core.config.local_llm import apply_local_llm_presets_to_animas

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        (animas_dir / "test-anima").mkdir()
        (animas_dir / "test-anima" / "status.json").write_text(
            json.dumps({"role": "engineer", "credential": "ollama"}),
            encoding="utf-8",
        )

        config = MagicMock()
        config.local_llm = LocalLLMConfig(auto_apply_presets=True)
        config.anima_defaults.credential = "ollama"

        result = apply_local_llm_presets_to_animas(animas_dir, config)
        assert len(result) >= 1

        status = json.loads((animas_dir / "test-anima" / "status.json").read_text())
        assert status["credential"] == "ollama"
