from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from server.routes.setup import _resolve_codex_setup_model

pytest.importorskip("openai_codex")


@pytest.mark.parametrize("default", [True, False])
async def test_codex_setup_uses_available_model(default):
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.models.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(model="hidden-model", hidden=True, is_default=True),
            SimpleNamespace(model="account-model", hidden=False, is_default=default),
        ]
    )
    with patch("openai_codex.AsyncCodex", return_value=client):
        assert await _resolve_codex_setup_model() == "codex/account-model"
    client.__aexit__.assert_awaited_once()


async def test_codex_setup_does_not_guess_when_discovery_fails():
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.models.side_effect = RuntimeError("offline")
    with patch("openai_codex.AsyncCodex", return_value=client), pytest.raises(HTTPException) as error:
        await _resolve_codex_setup_model()
    assert error.value.status_code == 503
    client.__aexit__.assert_awaited_once()
