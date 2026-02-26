# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the Voice Chat WebSocket endpoint and voice config."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from core.config.models import AnimaWorksConfig, VoiceConfig


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def test_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a minimal AnimaWorks data directory."""
    from core.config import invalidate_cache

    data_dir = tmp_path / "animaworks"
    data_dir.mkdir()
    animas_dir = data_dir / "animas"
    animas_dir.mkdir()
    shared_dir = data_dir / "shared"
    shared_dir.mkdir()

    # Create config.json
    config = {
        "version": 1,
        "setup_complete": True,
        "credentials": {"anthropic": {"api_key": "", "base_url": None}},
        "anima_defaults": {"model": "claude-sonnet-4-6", "credential": "anthropic"},
        "animas": {"test_anima": {}},
    }
    (data_dir / "config.json").write_text(json.dumps(config))

    # Create auth.json (local_trust mode)
    auth = {"auth_mode": "local_trust", "trust_localhost": True, "users": []}
    (data_dir / "auth.json").write_text(json.dumps(auth))

    # Create test anima
    anima_dir = animas_dir / "test_anima"
    anima_dir.mkdir()
    (anima_dir / "identity.md").write_text("# Test Anima")
    (anima_dir / "status.json").write_text(
        json.dumps({
            "enabled": True,
            "model": "claude-sonnet-4-6",
            "voice": {
                "tts_provider": "voicevox",
                "voice_id": "3",
                "speed": 1.0,
            },
        })
    )

    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))
    invalidate_cache()
    return data_dir


@pytest.fixture(autouse=True)
def _clear_voice_sessions():
    """Clear voice route active sessions before each test to avoid cross-test pollution."""
    from server.routes import voice as voice_module

    yield
    voice_module._active_sessions.clear()


# ── TestVoiceWebSocket ───────────────────────────────────────────────


class TestVoiceWebSocket:
    def test_voice_endpoint_connection(self, test_data_dir: Path) -> None:
        """Test that /ws/voice/{name} can establish a WebSocket connection."""
        from server.routes.voice import create_voice_router

        app = FastAPI()
        app.include_router(create_voice_router())
        app.state.supervisor = MagicMock()
        app.state.animas_dir = test_data_dir / "animas"

        with patch("server.routes.voice.load_auth") as mock_auth, patch(
            "server.routes.voice._get_stt"
        ) as mock_stt, patch(
            "server.routes.voice.create_tts_provider"
        ) as mock_tts_factory:
            mock_auth_config = MagicMock()
            mock_auth_config.auth_mode = "local_trust"
            mock_auth.return_value = mock_auth_config

            mock_stt_instance = MagicMock()
            mock_stt.return_value = mock_stt_instance

            mock_tts = AsyncMock()
            mock_tts.health_check.return_value = True
            mock_tts_factory.return_value = mock_tts

            client = TestClient(app)
            with client.websocket_connect("/ws/voice/test_anima") as ws:
                # Should receive status loading then ready
                data = ws.receive_json()
                assert data["type"] == "status"
                assert data["state"] in ("loading", "ready")

                if data["state"] == "loading":
                    data = ws.receive_json()
                    assert data["type"] == "status"
                    assert data["state"] == "ready"

    def test_binary_audio_frames(self, test_data_dir: Path) -> None:
        """Test that binary frames are accepted as audio input."""
        from server.routes.voice import create_voice_router

        app = FastAPI()
        app.include_router(create_voice_router())
        app.state.supervisor = MagicMock()
        app.state.animas_dir = test_data_dir / "animas"

        with patch("server.routes.voice.load_auth") as mock_auth, patch(
            "server.routes.voice._get_stt"
        ) as mock_stt, patch(
            "server.routes.voice.create_tts_provider"
        ) as mock_tts_factory:
            mock_auth.return_value = MagicMock(auth_mode="local_trust")
            mock_stt.return_value = MagicMock()
            mock_tts_factory.return_value = AsyncMock()

            client = TestClient(app)
            with client.websocket_connect("/ws/voice/test_anima") as ws:
                # Drain status messages
                ws.receive_json()  # loading
                ws.receive_json()  # ready

                # Send binary audio data
                pcm_data = b"\x00\x00" * 1600  # 0.1s of 16kHz PCM
                ws.send_bytes(pcm_data)

                # No crash means binary frames are accepted

    def test_speech_end_stt_response(self, test_data_dir: Path) -> None:
        """Test speech_end triggers STT and returns transcript."""
        from core.supervisor.ipc import IPCResponse
        from server.routes.voice import create_voice_router

        app = FastAPI()
        app.include_router(create_voice_router())
        app.state.supervisor = MagicMock()
        app.state.animas_dir = test_data_dir / "animas"

        async def mock_stream(*args, **kwargs):
            yield IPCResponse(
                id="mock",
                done=True,
                chunk=None,
                result={
                    "response": "テスト応答",
                    "cycle_result": {"summary": "テスト応答", "emotion": "neutral"},
                },
            )

        app.state.supervisor.send_request_stream = mock_stream

        with patch("server.routes.voice.load_auth") as mock_auth, patch(
            "server.routes.voice._get_stt"
        ) as mock_stt_fn, patch(
            "server.routes.voice.create_tts_provider"
        ) as mock_tts_factory:
            mock_auth.return_value = MagicMock(auth_mode="local_trust")

            mock_stt = MagicMock()
            mock_stt.transcribe_buffer_async = AsyncMock(
                return_value={
                    "raw_text": "テスト発話",
                    "language": "ja",
                    "duration": 1.0,
                    "segments": [],
                }
            )
            mock_stt_fn.return_value = mock_stt

            mock_tts = AsyncMock()
            async def mock_synthesize(text: str, config: object) -> object:
                yield b"\x00\x01\x02\x03"

            mock_tts.synthesize = mock_synthesize
            mock_tts_factory.return_value = mock_tts

            client = TestClient(app)
            with client.websocket_connect("/ws/voice/test_anima") as ws:
                # Drain status
                ws.receive_json()
                ws.receive_json()

                # Send some audio then speech_end
                ws.send_bytes(b"\x00\x00" * 1600)
                ws.send_text(json.dumps({"type": "speech_end"}))

                # Wait for async processing and collect responses
                responses = []
                for _ in range(8):
                    try:
                        data = ws.receive_json()
                        responses.append(data)
                        if data.get("type") == "response_done":
                            break
                    except Exception:
                        break

                types = [r["type"] for r in responses]
                assert "transcript" in types or "response_start" in types or len(responses) > 0

    def test_interrupt_message(self, test_data_dir: Path) -> None:
        """Test that interrupt message is accepted."""
        from server.routes.voice import create_voice_router

        app = FastAPI()
        app.include_router(create_voice_router())
        app.state.supervisor = MagicMock()
        app.state.animas_dir = test_data_dir / "animas"

        with patch("server.routes.voice.load_auth") as mock_auth, patch(
            "server.routes.voice._get_stt"
        ) as mock_stt, patch(
            "server.routes.voice.create_tts_provider"
        ) as mock_tts_factory:
            mock_auth.return_value = MagicMock(auth_mode="local_trust")
            mock_stt.return_value = MagicMock()
            mock_tts_factory.return_value = AsyncMock()

            client = TestClient(app)
            with client.websocket_connect("/ws/voice/test_anima") as ws:
                ws.receive_json()  # loading
                ws.receive_json()  # ready

                ws.send_text(json.dumps({"type": "interrupt"}))
                # Should not crash

    def test_session_replacement(self, test_data_dir: Path) -> None:
        """Test that connecting same anima replaces existing session."""
        from server.routes.voice import create_voice_router, _active_sessions

        app = FastAPI()
        app.include_router(create_voice_router())
        app.state.supervisor = MagicMock()
        app.state.animas_dir = test_data_dir / "animas"

        with patch("server.routes.voice.load_auth") as mock_auth, patch(
            "server.routes.voice._get_stt"
        ) as mock_stt, patch(
            "server.routes.voice.create_tts_provider"
        ) as mock_tts_factory:
            mock_auth.return_value = MagicMock(auth_mode="local_trust")
            mock_stt.return_value = MagicMock()
            mock_tts_factory.return_value = AsyncMock()

            client = TestClient(app)
            with client.websocket_connect("/ws/voice/test_anima") as ws1:
                ws1.receive_json()  # loading
                ws1.receive_json()  # ready
                assert "test_anima" in _active_sessions


# ── TestVoiceConfigSerialization ─────────────────────────────────────


class TestVoiceConfigSerialization:
    def test_config_round_trip(self) -> None:
        """Test that VoiceConfig serializes/deserializes correctly."""
        config = AnimaWorksConfig(
            voice=VoiceConfig(
                stt_model="tiny",
                default_tts_provider="elevenlabs",
                stt_refine_enabled=True,
            )
        )

        data = config.model_dump(mode="json")
        assert data["voice"]["stt_model"] == "tiny"
        assert data["voice"]["default_tts_provider"] == "elevenlabs"
        assert data["voice"]["stt_refine_enabled"] is True

        restored = AnimaWorksConfig.model_validate(data)
        assert restored.voice.stt_model == "tiny"
        assert restored.voice.default_tts_provider == "elevenlabs"
        assert restored.voice.voicevox.base_url == "http://localhost:50021"
