"""Tests for core/tools/local_llm.py — Ollama LLM client."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from core.tools.local_llm import (
    DEFAULT_SERVERS,
    OllamaClient,
    OllamaServer,
    get_tool_schemas,
)


# ── OllamaServer ─────────────────────────────────────────────────


class TestOllamaServer:
    def test_dataclass(self):
        s = OllamaServer(name="test", url="http://localhost:11434")
        assert s.name == "test"
        assert s.url == "http://localhost:11434"


# ── OllamaClient init ────────────────────────────────────────────


class TestOllamaClientInit:
    def test_default_servers(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OLLAMA_SERVERS", raising=False)
        client = OllamaClient()
        assert "local" in client.servers
        assert len(client.servers) == 1

    def test_custom_servers_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OLLAMA_SERVERS", "local=http://localhost:11434,remote=http://10.0.0.1:11434")
        client = OllamaClient()
        assert "local" in client.servers
        assert "remote" in client.servers
        assert client.servers["local"].url == "http://localhost:11434"

    def test_invalid_env_falls_back(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OLLAMA_SERVERS", "invalid-format")
        client = OllamaClient()
        assert client.servers == dict(DEFAULT_SERVERS)

    def test_explicit_server(self):
        client = OllamaClient(server="local")
        assert client._server_name == "local"

    def test_explicit_model(self):
        client = OllamaClient(model="llama3:8b")
        assert client._model == "llama3:8b"


# ── _get_server ───────────────────────────────────────────────────


class TestGetServer:
    def test_named_server(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OLLAMA_SERVERS", raising=False)
        client = OllamaClient(server="local")
        server = client._get_server()
        assert server.name == "local"

    def test_unknown_server(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OLLAMA_SERVERS", raising=False)
        client = OllamaClient(server="nonexistent")
        with pytest.raises(ValueError, match="Unknown server"):
            client._get_server()

    def test_auto_calls_select(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OLLAMA_SERVERS", raising=False)
        client = OllamaClient(server="auto")
        mock_server = OllamaServer("local", "http://test")
        with patch.object(client, "select_server", return_value=mock_server):
            server = client._get_server()
        assert server == mock_server


# ── select_server ─────────────────────────────────────────────────


class TestSelectServer:
    def test_selects_least_loaded(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(
            "OLLAMA_SERVERS",
            "server-a=http://server-a:11434,server-b=http://server-b:11434",
        )
        client = OllamaClient()

        def mock_get(url, **kwargs):
            resp = MagicMock()
            if "server-a" in url:
                resp.json.return_value = {"models": []}  # 0 models
                resp.elapsed.total_seconds.return_value = 0.1
            else:
                resp.json.return_value = {"models": [{"name": "m1"}]}  # 1 model
                resp.elapsed.total_seconds.return_value = 0.1
            return resp

        with patch("core.tools.local_llm.httpx.get", side_effect=mock_get):
            server = client.select_server()
        assert server.name == "server-a"

    def test_handles_unreachable_server(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(
            "OLLAMA_SERVERS",
            "server-a=http://server-a:11434,server-b=http://server-b:11434",
        )
        client = OllamaClient()

        call_count = 0

        def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if "server-a" in url:
                raise Exception("unreachable")
            resp = MagicMock()
            resp.json.return_value = {"models": []}
            resp.elapsed.total_seconds.return_value = 0.1
            return resp

        with patch("core.tools.local_llm.httpx.get", side_effect=mock_get):
            server = client.select_server()
        assert server.name == "server-b"


# ── resolve_model ─────────────────────────────────────────────────


class TestResolveModel:
    def test_explicit_model(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OLLAMA_SERVERS", raising=False)
        client = OllamaClient(model="llama3:8b", server="local")
        assert client.resolve_model() == "llama3:8b"

    def test_resolve_from_server(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OLLAMA_SERVERS", raising=False)
        client = OllamaClient(server="local")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "qwen:14b"}, {"name": "llama3:70b"}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client._client, "get", return_value=mock_resp):
            model = client.resolve_model()
        assert model == "qwen:14b"

    def test_resolve_with_hint(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OLLAMA_SERVERS", raising=False)
        client = OllamaClient(server="local", hint="llama")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "qwen:14b"}, {"name": "llama3:70b"}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client._client, "get", return_value=mock_resp):
            model = client.resolve_model()
        assert model == "llama3:70b"

    def test_no_models_available(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OLLAMA_SERVERS", raising=False)
        client = OllamaClient(server="local")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": []}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client._client, "get", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="No models"):
                client.resolve_model()


# ── list_models ───────────────────────────────────────────────────


class TestListModels:
    def test_list_models(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OLLAMA_SERVERS", raising=False)
        client = OllamaClient(server="local")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "model-a"}, {"name": "model-b"}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client._client, "get", return_value=mock_resp):
            models = client.list_models()
        assert len(models) == 2


# ── server_status ─────────────────────────────────────────────────


class TestServerStatus:
    def test_server_status_single(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OLLAMA_SERVERS", raising=False)
        client = OllamaClient()

        def mock_get(url, **kwargs):
            resp = MagicMock()
            resp.json.return_value = {"models": []}
            return resp

        with patch("core.tools.local_llm.httpx.get", side_effect=mock_get):
            status = client.server_status()
        assert "local" in status

    def test_server_status_multi(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(
            "OLLAMA_SERVERS",
            "server-a=http://server-a:11434,server-b=http://server-b:11434",
        )
        client = OllamaClient()

        def mock_get(url, **kwargs):
            resp = MagicMock()
            resp.json.return_value = {"models": []}
            return resp

        with patch("core.tools.local_llm.httpx.get", side_effect=mock_get):
            status = client.server_status()
        assert "server-a" in status
        assert "server-b" in status

    def test_server_status_with_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OLLAMA_SERVERS", raising=False)
        client = OllamaClient()

        def mock_get(url, **kwargs):
            raise Exception("down")

        with patch("core.tools.local_llm.httpx.get", side_effect=mock_get):
            status = client.server_status()
        for name, info in status.items():
            assert "error" in info


# ── _apply_think ──────────────────────────────────────────────────


class TestApplyThink:
    def test_off(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OLLAMA_SERVERS", raising=False)
        client = OllamaClient()
        body = {"model": "llama3", "options": {}}
        client._apply_think(body, "off")
        assert body["options"]["think"] is False

    def test_on_regular_model(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OLLAMA_SERVERS", raising=False)
        client = OllamaClient()
        body = {"model": "llama3", "options": {}}
        client._apply_think(body, "high")
        assert body["options"]["think"] is True

    def test_on_gpt_oss_model(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OLLAMA_SERVERS", raising=False)
        client = OllamaClient()
        body = {"model": "gpt-oss:20b", "options": {}}
        client._apply_think(body, "medium")
        assert body["options"]["think"] == "medium"


# ── _get_alternate ────────────────────────────────────────────────


class TestGetAlternate:
    def test_alternate_with_two_servers(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(
            "OLLAMA_SERVERS",
            "server-a=http://server-a:11434,server-b=http://server-b:11434",
        )
        client = OllamaClient()
        alt = client._get_alternate(client.servers["server-a"])
        assert alt is not None
        assert alt.name == "server-b"

    def test_alternate_reverse(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(
            "OLLAMA_SERVERS",
            "server-a=http://server-a:11434,server-b=http://server-b:11434",
        )
        client = OllamaClient()
        alt = client._get_alternate(client.servers["server-b"])
        assert alt is not None
        assert alt.name == "server-a"

    def test_single_server_no_alternate(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OLLAMA_SERVERS", raising=False)
        client = OllamaClient()
        alt = client._get_alternate(client.servers["local"])
        assert alt is None


# ── get_tool_schemas ──────────────────────────────────────────────


class TestGetToolSchemas:
    def test_returns_schemas(self):
        schemas = get_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) == 4
        names = {s["name"] for s in schemas}
        assert names == {
            "local_llm_generate", "local_llm_chat",
            "local_llm_models", "local_llm_status",
        }

    def test_generate_requires_prompt(self):
        schemas = get_tool_schemas()
        gen = [s for s in schemas if s["name"] == "local_llm_generate"][0]
        assert "prompt" in gen["input_schema"]["required"]

    def test_chat_requires_messages(self):
        schemas = get_tool_schemas()
        chat = [s for s in schemas if s["name"] == "local_llm_chat"][0]
        assert "messages" in chat["input_schema"]["required"]
