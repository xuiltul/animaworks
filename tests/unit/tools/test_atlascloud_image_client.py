# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Atlas Cloud request lifecycle and explicit backend routing tests."""

from __future__ import annotations

import io
from unittest.mock import patch

import httpx
import pytest
from PIL import Image

from core.config.models import ImageGenConfig
from core.tools.image import atlascloud
from core.tools.image.atlascloud import AtlasCloudImageClient
from core.tools.image_gen import _build_fullbody_client, _build_reference_client


@pytest.fixture
def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="JPEG")
    return buffer.getvalue()


def response(data: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=data, request=httpx.Request("GET", "https://api.atlascloud.ai"))


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> AtlasCloudImageClient:
    monkeypatch.setattr(atlascloud, "get_credential", lambda *args, **kwargs: "test-key")
    monkeypatch.setattr(atlascloud.time, "sleep", lambda seconds: None)
    return AtlasCloudImageClient()


def test_polling_submits_once_and_downloads_without_credential(client, image_bytes) -> None:
    completed = response(
        {"data": {"id": "prediction-1", "status": "completed", "outputs": ["https://cdn.example/a.jpg"]}}
    )
    with (
        patch.object(
            atlascloud.httpx, "post", return_value=response({"data": {"id": "prediction-1", "status": "processing"}})
        ) as post,
        patch.object(
            atlascloud.httpx,
            "get",
            side_effect=[
                response({}, 503),
                completed,
                httpx.Response(200, content=image_bytes, request=httpx.Request("GET", "https://cdn.example/a.jpg")),
            ],
        ) as get,
    ):
        result = client.generate_fullbody("A character", seed=42, negative_prompt="blurry")
    assert result.startswith(b"\x89PNG")
    assert post.call_count == 1
    assert post.call_args.kwargs["json"] == {
        "model": "bytedance/seedream-v4.5",
        "prompt": "A character",
        "size": "1728*2304",
    }
    assert get.call_count == 3
    assert get.call_args_list[0].args[0].endswith("/prediction/prediction-1")
    assert "headers" not in get.call_args_list[-1].kwargs


def test_reference_edits_normalize_image_and_use_edit_model(client, image_bytes) -> None:
    with patch.object(client, "_generate", return_value=b"result") as generate:
        assert client.generate_from_reference(image_bytes, "Smile", aspect_ratio="1:1") == b"result"
    payload = generate.call_args.args[0]
    assert payload["model"] == "bytedance/seedream-v4.5/edit"
    assert payload["size"] == "2048*2048"
    assert payload["images"][0].startswith("data:image/png;base64,iVBOR")


def test_face_reference_takes_precedence_over_style_reference(client, image_bytes) -> None:
    with patch.object(client, "generate_from_reference", return_value=b"result") as edit:
        assert client.generate_fullbody("Portrait", vibe_image=b"style", face_reference_image=image_bytes) == b"result"
    assert edit.call_args.args[0] == image_bytes


@pytest.mark.parametrize("status", [400, 429, 503])
def test_submit_errors_are_never_retried(client, status) -> None:
    with (
        patch.object(atlascloud.httpx, "post", return_value=response({}, status)) as post,
        pytest.raises(httpx.HTTPStatusError),
    ):
        client.generate_fullbody("Portrait")
    assert post.call_count == 1


def test_poll_transport_errors_are_bounded(client) -> None:
    with (
        patch.object(
            atlascloud.httpx, "post", return_value=response({"data": {"id": "p1", "status": "processing"}})
        ) as post,
        patch.object(atlascloud.httpx, "get", side_effect=httpx.ConnectError("offline")) as get,
        pytest.raises(httpx.ConnectError),
    ):
        client.generate_fullbody("Portrait")
    assert post.call_count == 1
    assert get.call_count == client.MAX_POLL_ERRORS


@pytest.mark.parametrize(
    "prediction",
    [
        {"id": "p1", "status": "failed"},
        {"id": "p1", "status": "completed", "outputs": []},
        {"id": "p1", "status": "completed", "outputs": ["http://example.com/a.png"]},
        {"id": "p1", "status": "unknown"},
        {"status": "processing"},
    ],
)
def test_invalid_or_failed_predictions_stop_without_resubmitting(client, prediction) -> None:
    with (
        patch.object(atlascloud.httpx, "post", return_value=response({"data": prediction})) as post,
        patch.object(atlascloud.httpx, "get") as get,
        pytest.raises((ValueError, RuntimeError)),
    ):
        client.generate_fullbody("Portrait")
    assert post.call_count == 1
    get.assert_not_called()


def test_poll_timeout_preserves_one_submission(client, monkeypatch) -> None:
    monkeypatch.setattr(client, "POLL_TIMEOUT", 0)
    with (
        patch.object(
            atlascloud.httpx, "post", return_value=response({"data": {"id": "p1", "status": "processing"}})
        ) as post,
        pytest.raises(TimeoutError, match="p1"),
    ):
        client.generate_fullbody("Portrait")
    assert post.call_count == 1


@pytest.mark.parametrize("factory", [_build_fullbody_client, _build_reference_client])
def test_explicit_backend_bypasses_codex_and_fal(client, factory) -> None:
    config = ImageGenConfig(backend="atlascloud")
    with (
        patch("core.tools.image_gen.AtlasCloudImageClient", return_value=client),
        patch("core.tools.image_gen.codex_available") as codex,
    ):
        assert factory(config) is client
    codex.assert_not_called()
    assert ImageGenConfig().backend == "api"
