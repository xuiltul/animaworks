"""Unit tests for NovelAI Vibe Transfer (V4+ encode-vibe)."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
import zipfile
from unittest.mock import MagicMock, call, patch

import httpx
import pytest

from core.tools._base import ToolConfigError
from core.tools.image_gen import (
    NOVELAI_API_URL,
    NOVELAI_ENCODE_URL,
    NOVELAI_MODEL,
    NovelAIClient,
    _retry,
)


# ── Helpers ──────────────────────────────────────────────


def _make_zip_png(png_data: bytes = b"PNG-BYTES") -> bytes:
    """Create a ZIP archive containing a single PNG file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("output.png", png_data)
    return buf.getvalue()


def _make_ok_response(
    content: bytes = b"",
    status_code: int = 200,
    text: str = "",
) -> MagicMock:
    """Create a mock httpx.Response with the given content."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.text = text
    resp.raise_for_status = MagicMock()
    return resp


def _make_error_response(
    status_code: int = 500,
    text: str = "Internal Server Error",
) -> MagicMock:
    """Create a mock httpx.Response that raises HTTPStatusError."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    req = MagicMock()
    req.url = "http://test"
    error = httpx.HTTPStatusError(
        f"{status_code}", request=req, response=resp,
    )
    resp.raise_for_status.side_effect = error
    resp.content = text.encode()
    return resp


# ── NovelAIClient.encode_vibe ────────────────────────────


class TestEncodeVibe:
    """Tests for NovelAIClient.encode_vibe()."""

    @pytest.fixture(autouse=True)
    def _set_token(self, monkeypatch: pytest.MonkeyPatch, tmp_path):
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("NOVELAI_TOKEN", "test-nai-token")

    def test_returns_binary_content(self):
        """encode_vibe should return raw binary vibe data."""
        vibe_data = b"\x00\x01\x02VIBE_ENCODED_DATA"
        mock_resp = _make_ok_response(content=vibe_data)

        with patch("core.tools.image_gen.httpx.post", return_value=mock_resp):
            client = NovelAIClient()
            result = client.encode_vibe(b"FAKE_IMAGE_BYTES")

        assert result == vibe_data

    def test_sends_correct_url(self):
        """encode_vibe should POST to NOVELAI_ENCODE_URL."""
        mock_resp = _make_ok_response(content=b"vibe")

        with patch("core.tools.image_gen.httpx.post", return_value=mock_resp) as mock_post:
            client = NovelAIClient()
            client.encode_vibe(b"IMG")

        actual_url = mock_post.call_args[0][0]
        assert actual_url == NOVELAI_ENCODE_URL

    def test_sends_base64_encoded_image(self):
        """encode_vibe should base64-encode the image in the request body."""
        image_bytes = b"\x89PNG_FAKE_DATA"
        expected_b64 = base64.b64encode(image_bytes).decode()
        mock_resp = _make_ok_response(content=b"vibe")

        with patch("core.tools.image_gen.httpx.post", return_value=mock_resp) as mock_post:
            client = NovelAIClient()
            client.encode_vibe(image_bytes)

        body = mock_post.call_args[1]["json"]
        assert body["image"] == expected_b64

    def test_sends_model_name(self):
        """encode_vibe should include the model name in the request body."""
        mock_resp = _make_ok_response(content=b"vibe")

        with patch("core.tools.image_gen.httpx.post", return_value=mock_resp) as mock_post:
            client = NovelAIClient()
            client.encode_vibe(b"IMG")

        body = mock_post.call_args[1]["json"]
        assert body["model"] == NOVELAI_MODEL

    def test_sends_information_extracted(self):
        """encode_vibe should include information_extracted in the body."""
        mock_resp = _make_ok_response(content=b"vibe")

        with patch("core.tools.image_gen.httpx.post", return_value=mock_resp) as mock_post:
            client = NovelAIClient()
            client.encode_vibe(b"IMG", information_extracted=0.5)

        body = mock_post.call_args[1]["json"]
        assert body["information_extracted"] == 0.5

    def test_default_information_extracted(self):
        """Default information_extracted should be 0.8."""
        mock_resp = _make_ok_response(content=b"vibe")

        with patch("core.tools.image_gen.httpx.post", return_value=mock_resp) as mock_post:
            client = NovelAIClient()
            client.encode_vibe(b"IMG")

        body = mock_post.call_args[1]["json"]
        assert body["information_extracted"] == 0.8

    def test_authorization_header(self):
        """encode_vibe should send Bearer token in Authorization header."""
        mock_resp = _make_ok_response(content=b"vibe")

        with patch("core.tools.image_gen.httpx.post", return_value=mock_resp) as mock_post:
            client = NovelAIClient()
            client.encode_vibe(b"IMG")

        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test-nai-token"
        assert headers["Content-Type"] == "application/json"

    def test_logs_error_on_non_200(self, caplog):
        """encode_vibe should log error detail before raising on non-200."""
        import logging

        resp_500 = _make_error_response(status_code=500, text="Server Error Detail")

        with (
            patch("core.tools.image_gen.httpx.post", return_value=resp_500),
            patch("core.tools.image_gen.time.sleep"),
            caplog.at_level(logging.ERROR, logger="animaworks.tools"),
        ):
            client = NovelAIClient()
            with pytest.raises(httpx.HTTPStatusError):
                client.encode_vibe(b"IMG")

        # The error should have been logged with status code and body text
        assert any(
            "encode-vibe error" in r.message and "500" in r.message
            for r in caplog.records
        )

    def test_retries_on_429(self):
        """encode_vibe should retry on 429 (rate limit) via _retry."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.text = "Rate limited"
        req = MagicMock()
        req.url = NOVELAI_ENCODE_URL
        error_429 = httpx.HTTPStatusError("429", request=req, response=resp_429)
        resp_429.raise_for_status.side_effect = error_429

        ok_resp = _make_ok_response(content=b"VIBE_OK")

        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return resp_429
            return ok_resp

        with (
            patch("core.tools.image_gen.httpx.post", side_effect=_side_effect),
            patch("core.tools.image_gen.time.sleep"),
        ):
            client = NovelAIClient()
            result = client.encode_vibe(b"IMG")

        assert result == b"VIBE_OK"
        assert call_count == 3

    def test_retries_on_500(self):
        """encode_vibe should retry on 500 (server error)."""
        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.text = "Internal error"
        req = MagicMock()
        req.url = NOVELAI_ENCODE_URL
        error_500 = httpx.HTTPStatusError("500", request=req, response=resp_500)
        resp_500.raise_for_status.side_effect = error_500

        ok_resp = _make_ok_response(content=b"VIBE_OK")

        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return resp_500
            return ok_resp

        with (
            patch("core.tools.image_gen.httpx.post", side_effect=_side_effect),
            patch("core.tools.image_gen.time.sleep"),
        ):
            client = NovelAIClient()
            result = client.encode_vibe(b"IMG")

        assert result == b"VIBE_OK"
        assert call_count == 2

    def test_no_retry_on_400(self):
        """encode_vibe should NOT retry on 400 (bad request)."""
        resp_400 = _make_error_response(status_code=400, text="Bad Request")

        with patch("core.tools.image_gen.httpx.post", return_value=resp_400):
            client = NovelAIClient()
            with pytest.raises(httpx.HTTPStatusError):
                client.encode_vibe(b"IMG")


# ── NovelAIClient.generate_fullbody with vibe_image ──────


class TestGenerateFullbodyVibeTransfer:
    """Tests for the vibe_image flow in generate_fullbody()."""

    @pytest.fixture(autouse=True)
    def _set_token(self, monkeypatch: pytest.MonkeyPatch, tmp_path):
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("NOVELAI_TOKEN", "test-nai-token")

    def _make_generate_response(self, png_data: bytes = b"GENERATED-PNG") -> MagicMock:
        """Create a mock response for the generate endpoint (ZIP format)."""
        zip_bytes = _make_zip_png(png_data)
        resp = MagicMock()
        resp.status_code = 200
        resp.content = zip_bytes
        resp.raise_for_status = MagicMock()
        return resp

    def test_vibe_image_calls_encode_vibe(self):
        """When vibe_image is provided, encode_vibe should be called first."""
        client = NovelAIClient()
        vibe_image = b"REFERENCE_IMAGE_BYTES"
        encoded_vibe = b"ENCODED_VIBE_DATA"

        generate_resp = self._make_generate_response()

        with (
            patch.object(client, "encode_vibe", return_value=encoded_vibe) as mock_encode,
            patch("core.tools.image_gen.httpx.post", return_value=generate_resp),
        ):
            result = client.generate_fullbody(
                prompt="1girl, test",
                vibe_image=vibe_image,
                vibe_strength=0.7,
                vibe_info_extracted=0.9,
            )

        mock_encode.assert_called_once_with(vibe_image, 0.9)
        assert result == b"GENERATED-PNG"

    def test_vibe_image_none_skips_encode_vibe(self):
        """When vibe_image is None, encode_vibe should NOT be called."""
        client = NovelAIClient()
        generate_resp = self._make_generate_response()

        with (
            patch.object(client, "encode_vibe") as mock_encode,
            patch("core.tools.image_gen.httpx.post", return_value=generate_resp),
        ):
            client.generate_fullbody(prompt="1girl, test", vibe_image=None)

        mock_encode.assert_not_called()

    def test_vibe_encoded_data_in_params(self):
        """Encoded vibe data should be base64'd in reference_image_multiple."""
        client = NovelAIClient()
        vibe_image = b"REFERENCE_IMAGE"
        encoded_vibe = b"\x00\x01ENCODED_VIBE"
        expected_b64 = base64.b64encode(encoded_vibe).decode()

        generate_resp = self._make_generate_response()

        with (
            patch.object(client, "encode_vibe", return_value=encoded_vibe),
            patch("core.tools.image_gen.httpx.post", return_value=generate_resp) as mock_post,
        ):
            client.generate_fullbody(
                prompt="1girl, test",
                vibe_image=vibe_image,
                vibe_strength=0.6,
                vibe_info_extracted=0.8,
            )

        # The generate call is the httpx.post call
        body = mock_post.call_args[1]["json"]
        params = body["parameters"]
        assert params["reference_image_multiple"] == [expected_b64]

    def test_vibe_strength_in_params(self):
        """vibe_strength should appear in reference_strength_multiple."""
        client = NovelAIClient()
        generate_resp = self._make_generate_response()

        with (
            patch.object(client, "encode_vibe", return_value=b"encoded"),
            patch("core.tools.image_gen.httpx.post", return_value=generate_resp) as mock_post,
        ):
            client.generate_fullbody(
                prompt="1girl",
                vibe_image=b"img",
                vibe_strength=0.75,
                vibe_info_extracted=0.85,
            )

        body = mock_post.call_args[1]["json"]
        params = body["parameters"]
        assert params["reference_strength_multiple"] == [0.75]
        assert params["reference_information_extracted_multiple"] == [0.85]

    def test_no_vibe_empty_reference_arrays(self):
        """Without vibe_image, reference arrays should be empty."""
        client = NovelAIClient()
        generate_resp = self._make_generate_response()

        with patch("core.tools.image_gen.httpx.post", return_value=generate_resp) as mock_post:
            client.generate_fullbody(prompt="1girl, test")

        body = mock_post.call_args[1]["json"]
        params = body["parameters"]
        assert params["reference_image_multiple"] == []
        assert params["reference_information_extracted_multiple"] == []
        assert params["reference_strength_multiple"] == []

    def test_generate_logs_error_on_non_200(self, caplog):
        """generate_fullbody should log error detail before raising on non-200."""
        import logging

        client = NovelAIClient()
        resp_500 = _make_error_response(status_code=500, text="Generation failed")

        with (
            patch("core.tools.image_gen.httpx.post", return_value=resp_500),
            patch("core.tools.image_gen.time.sleep"),
            caplog.at_level(logging.ERROR, logger="animaworks.tools"),
        ):
            with pytest.raises(httpx.HTTPStatusError):
                client.generate_fullbody(prompt="1girl, test")

        assert any(
            "generate error" in r.message and "500" in r.message
            for r in caplog.records
        )

    def test_generate_sends_to_api_url(self):
        """generate_fullbody should POST to NOVELAI_API_URL."""
        client = NovelAIClient()
        generate_resp = self._make_generate_response()

        with patch("core.tools.image_gen.httpx.post", return_value=generate_resp) as mock_post:
            client.generate_fullbody(prompt="test")

        actual_url = mock_post.call_args[0][0]
        assert actual_url == NOVELAI_API_URL

    def test_seed_included_when_set(self):
        """When seed is provided, it should appear in parameters."""
        client = NovelAIClient()
        generate_resp = self._make_generate_response()

        with patch("core.tools.image_gen.httpx.post", return_value=generate_resp) as mock_post:
            client.generate_fullbody(prompt="test", seed=42)

        body = mock_post.call_args[1]["json"]
        assert body["parameters"]["seed"] == 42

    def test_seed_omitted_when_none(self):
        """When seed is None, it should NOT appear in parameters."""
        client = NovelAIClient()
        generate_resp = self._make_generate_response()

        with patch("core.tools.image_gen.httpx.post", return_value=generate_resp) as mock_post:
            client.generate_fullbody(prompt="test", seed=None)

        body = mock_post.call_args[1]["json"]
        assert "seed" not in body["parameters"]

    def test_v4_prompt_structure(self):
        """generate_fullbody should include V4 structured prompt fields."""
        client = NovelAIClient()
        generate_resp = self._make_generate_response()

        with patch("core.tools.image_gen.httpx.post", return_value=generate_resp) as mock_post:
            client.generate_fullbody(prompt="1girl, black hair")

        body = mock_post.call_args[1]["json"]
        params = body["parameters"]
        assert "v4_prompt" in params
        assert params["v4_prompt"]["caption"]["base_caption"] == "1girl, black hair"
        assert "v4_negative_prompt" in params

    def test_full_vibe_flow_integration(self):
        """Integration-style test: vibe_image -> encode_vibe -> generate with encoded data.

        Verifies the full flow without mocking encode_vibe, using mocked HTTP
        calls for both encode-vibe and generate endpoints.
        """
        client = NovelAIClient()
        vibe_image = b"FULL_FLOW_REF_IMAGE"
        encoded_vibe = b"FULL_FLOW_ENCODED"

        # encode-vibe response
        encode_resp = _make_ok_response(content=encoded_vibe)
        # generate response
        generate_resp = MagicMock()
        generate_resp.status_code = 200
        generate_resp.content = _make_zip_png(b"FINAL_OUTPUT")
        generate_resp.raise_for_status = MagicMock()

        post_calls = []

        def _mock_post(url, **kwargs):
            post_calls.append(url)
            if url == NOVELAI_ENCODE_URL:
                return encode_resp
            return generate_resp

        with patch("core.tools.image_gen.httpx.post", side_effect=_mock_post):
            result = client.generate_fullbody(
                prompt="1girl, test",
                vibe_image=vibe_image,
                vibe_strength=0.6,
                vibe_info_extracted=0.8,
            )

        # Verify call order: encode first, then generate
        assert post_calls == [NOVELAI_ENCODE_URL, NOVELAI_API_URL]
        assert result == b"FINAL_OUTPUT"

    def test_full_vibe_flow_body_contents(self):
        """Verify the generated body contains correctly encoded vibe data."""
        client = NovelAIClient()
        vibe_image = b"REF_IMG"
        encoded_vibe = b"\xde\xad\xbe\xef_ENCODED"

        encode_resp = _make_ok_response(content=encoded_vibe)
        generate_resp = MagicMock()
        generate_resp.status_code = 200
        generate_resp.content = _make_zip_png(b"OUTPUT")
        generate_resp.raise_for_status = MagicMock()

        captured_bodies: list[dict] = []

        def _mock_post(url, **kwargs):
            captured_bodies.append({"url": url, "json": kwargs.get("json", {})})
            if url == NOVELAI_ENCODE_URL:
                return encode_resp
            return generate_resp

        with patch("core.tools.image_gen.httpx.post", side_effect=_mock_post):
            client.generate_fullbody(
                prompt="1girl",
                vibe_image=vibe_image,
                vibe_strength=0.55,
                vibe_info_extracted=0.75,
            )

        # Check encode-vibe request body
        encode_body = captured_bodies[0]["json"]
        assert encode_body["image"] == base64.b64encode(vibe_image).decode()
        assert encode_body["model"] == NOVELAI_MODEL
        assert encode_body["information_extracted"] == 0.75

        # Check generate request body
        gen_body = captured_bodies[1]["json"]
        params = gen_body["parameters"]
        expected_vibe_b64 = base64.b64encode(encoded_vibe).decode()
        assert params["reference_image_multiple"] == [expected_vibe_b64]
        assert params["reference_strength_multiple"] == [0.55]
        assert params["reference_information_extracted_multiple"] == [0.75]
