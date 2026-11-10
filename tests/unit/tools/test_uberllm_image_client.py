# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""UberLLM image request shape, response handling, and explicit backend routing tests."""

from __future__ import annotations

import base64
import io
from unittest.mock import patch

import httpx
import pytest
from PIL import Image

from core.config.models import ImageGenConfig
from core.integrations.image import uberllm
from core.integrations.image.uberllm import UberLLMImageClient
from core.integrations.image_gen import _build_fullbody_client, _build_reference_client


@pytest.fixture
def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="JPEG")
    return buffer.getvalue()


def response(data: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=data, request=httpx.Request("POST", "https://api.uberllm.dev/v1/images"))


def image_response(image: bytes) -> httpx.Response:
    return response({"data": [{"b64_json": base64.b64encode(image).decode()}]})


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> UberLLMImageClient:
    monkeypatch.setattr(uberllm, "get_credential", lambda *args, **kwargs: "test-key")
    return UberLLMImageClient()


def test_generation_posts_once_to_fixed_host_and_returns_png(client, image_bytes) -> None:
    with patch.object(uberllm.httpx, "post", return_value=image_response(image_bytes)) as post:
        result = client.generate_fullbody("A character", seed=42, negative_prompt="blurry")
    assert result.startswith(b"\x89PNG")
    assert post.call_count == 1
    assert post.call_args.args[0] == "https://api.uberllm.dev/v1/images/generations"
    assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer test-key"}
    assert post.call_args.kwargs["json"] == {
        "model": "bytedance/seedream-4.5",
        "prompt": "A character",
        "size": "768x1024",
        "n": 1,
        "response_format": "b64_json",
    }


def test_jpeg_output(client, image_bytes) -> None:
    with patch.object(uberllm.httpx, "post", return_value=image_response(image_bytes)):
        assert client.generate_fullbody("A character", width=1024, height=1024, output_format="jpeg").startswith(
            b"\xff\xd8"
        )


def test_reference_edits_normalize_image_and_use_edit_model(client, image_bytes) -> None:
    with patch.object(uberllm.httpx, "post", return_value=image_response(image_bytes)) as post:
        client.generate_from_reference(image_bytes, "Smile", aspect_ratio="1:1")
    assert post.call_args.args[0] == "https://api.uberllm.dev/v1/images/edits"
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "bytedance/seedream-v4-edit"
    assert payload["size"] == "1024x1024"
    assert payload["image_url"].startswith("data:image/png;base64,iVBOR")


def test_face_reference_takes_precedence_over_style_reference(client, image_bytes) -> None:
    with patch.object(client, "generate_from_reference", return_value=b"result") as edit:
        assert client.generate_fullbody("Portrait", vibe_image=b"style", face_reference_image=image_bytes) == b"result"
    assert edit.call_args.args[0] == image_bytes


@pytest.mark.parametrize("status", [400, 402, 429, 503])
def test_request_errors_are_never_retried(client, status) -> None:
    with (
        patch.object(uberllm.httpx, "post", return_value=response({}, status)) as post,
        pytest.raises(httpx.HTTPStatusError),
    ):
        client.generate_fullbody("Portrait")
    assert post.call_count == 1


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"data": []},
        {"data": [{"url": "https://api.uberllm.dev/v1/media/artifacts/x/0"}]},
        {"data": [{"b64_json": "not base64!"}]},
    ],
)
def test_missing_or_invalid_images_raise(client, body) -> None:
    with patch.object(uberllm.httpx, "post", return_value=response(body)), pytest.raises(ValueError):
        client.generate_fullbody("Portrait")


def test_unsupported_ratio_and_format_are_rejected(client, image_bytes) -> None:
    with pytest.raises(ValueError):
        client.generate_from_reference(image_bytes, "Smile", aspect_ratio="21:9")
    with pytest.raises(ValueError):
        client.generate_fullbody("Portrait", output_format="webp")


@pytest.mark.parametrize("factory", [_build_fullbody_client, _build_reference_client])
def test_explicit_backend_bypasses_codex_and_fal(client, factory) -> None:
    config = ImageGenConfig(backend="uberllm")
    with (
        patch("core.integrations.image_gen.UberLLMImageClient", return_value=client),
        patch("core.integrations.image_gen.codex_available") as codex,
    ):
        assert factory(config) is client
    codex.assert_not_called()
    assert ImageGenConfig().backend == "api"
