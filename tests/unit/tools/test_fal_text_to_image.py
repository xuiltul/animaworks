"""Unit tests for FalTextToImageClient in core/tools/image_gen.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from core.tools._base import ToolConfigError
from core.tools.image_gen import FalTextToImageClient, FAL_FLUX_PRO_SUBMIT_URL


# ── Constructor ──────────────────────────────────────────


class TestFalTextToImageInit:
    @pytest.fixture(autouse=True)
    def _isolate_credentials(self, monkeypatch: pytest.MonkeyPatch, tmp_path):
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))

    def test_requires_fal_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("FAL_KEY", raising=False)
        with pytest.raises(ToolConfigError):
            FalTextToImageClient()

    def test_reads_fal_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FAL_KEY", "test-key-123")
        client = FalTextToImageClient()
        assert client._key == "test-key-123"


# ── generate_fullbody ────────────────────────────────────


class TestFalGenerateFullbody:
    @pytest.fixture(autouse=True)
    def _set_key(self, monkeypatch: pytest.MonkeyPatch, tmp_path):
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("FAL_KEY", "test-fal-key")

    def _make_submit_response(self, request_id: str = "req-123"):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "request_id": request_id,
            "status_url": f"https://queue.fal.run/status/{request_id}",
            "response_url": f"https://queue.fal.run/result/{request_id}",
        }
        return resp

    def _make_status_response(self, status: str = "COMPLETED"):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"status": status}
        return resp

    def _make_result_response(self, image_url: str = "https://cdn.fal.ai/image.png"):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "images": [{"url": image_url}],
        }
        return resp

    def _make_image_response(self, content: bytes = b"PNG-IMAGE-DATA"):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.content = content
        return resp

    def test_successful_generation(self):
        client = FalTextToImageClient()

        submit_resp = self._make_submit_response()
        status_resp = self._make_status_response("COMPLETED")
        result_resp = self._make_result_response()
        image_resp = self._make_image_response(b"FINAL-PNG")

        with (
            patch("core.tools.image_gen.httpx.post", return_value=submit_resp),
            patch("core.tools.image_gen.httpx.get", side_effect=[status_resp, result_resp, image_resp]),
            patch("core.tools.image_gen.time.sleep"),
        ):
            result = client.generate_fullbody("1girl, black hair, full body")

        assert result == b"FINAL-PNG"

    def test_submit_sends_correct_payload(self):
        client = FalTextToImageClient()

        submit_resp = self._make_submit_response()
        status_resp = self._make_status_response("COMPLETED")
        result_resp = self._make_result_response()
        image_resp = self._make_image_response()

        with (
            patch("core.tools.image_gen.httpx.post", return_value=submit_resp) as mock_post,
            patch("core.tools.image_gen.httpx.get", side_effect=[status_resp, result_resp, image_resp]),
            patch("core.tools.image_gen.time.sleep"),
        ):
            client.generate_fullbody(
                "1girl, black hair",
                width=512,
                height=768,
                seed=42,
                guidance_scale=4.0,
            )

        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"]
        assert payload["prompt"] == "1girl, black hair"
        assert payload["image_size"]["width"] == 512
        assert payload["image_size"]["height"] == 768
        assert payload["seed"] == 42
        assert payload["guidance_scale"] == 4.0

    def test_submit_url_is_flux_pro(self):
        client = FalTextToImageClient()

        submit_resp = self._make_submit_response()
        status_resp = self._make_status_response("COMPLETED")
        result_resp = self._make_result_response()
        image_resp = self._make_image_response()

        with (
            patch("core.tools.image_gen.httpx.post", return_value=submit_resp) as mock_post,
            patch("core.tools.image_gen.httpx.get", side_effect=[status_resp, result_resp, image_resp]),
            patch("core.tools.image_gen.time.sleep"),
        ):
            client.generate_fullbody("test prompt")

        assert mock_post.call_args[0][0] == FAL_FLUX_PRO_SUBMIT_URL

    def test_authorization_header(self):
        client = FalTextToImageClient()

        submit_resp = self._make_submit_response()
        status_resp = self._make_status_response("COMPLETED")
        result_resp = self._make_result_response()
        image_resp = self._make_image_response()

        with (
            patch("core.tools.image_gen.httpx.post", return_value=submit_resp) as mock_post,
            patch("core.tools.image_gen.httpx.get", side_effect=[status_resp, result_resp, image_resp]),
            patch("core.tools.image_gen.time.sleep"),
        ):
            client.generate_fullbody("test")

        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Key test-fal-key"

    def test_task_failed(self):
        client = FalTextToImageClient()

        submit_resp = self._make_submit_response("req-fail")
        status_resp = self._make_status_response("FAILED")
        status_resp.json.return_value = {"status": "FAILED", "error": "bad input"}

        with (
            patch("core.tools.image_gen.httpx.post", return_value=submit_resp),
            patch("core.tools.image_gen.httpx.get", return_value=status_resp),
            patch("core.tools.image_gen.time.sleep"),
        ):
            with pytest.raises(RuntimeError, match="failed"):
                client.generate_fullbody("bad prompt")

    def test_task_timeout(self):
        client = FalTextToImageClient()
        # Override timeout for faster test
        client.POLL_TIMEOUT = 0.01

        submit_resp = self._make_submit_response("req-slow")
        pending_resp = self._make_status_response("IN_PROGRESS")

        # monotonic returns values that ensure deadline is passed
        times = [0.0, 100.0, 200.0]
        time_iter = iter(times)

        with (
            patch("core.tools.image_gen.httpx.post", return_value=submit_resp),
            patch("core.tools.image_gen.httpx.get", return_value=pending_resp),
            patch("core.tools.image_gen.time.sleep"),
            patch("core.tools.image_gen.time.monotonic", side_effect=lambda: next(time_iter)),
        ):
            with pytest.raises(TimeoutError, match="timed out"):
                client.generate_fullbody("slow prompt")

    def test_no_images_in_result(self):
        client = FalTextToImageClient()

        submit_resp = self._make_submit_response()
        status_resp = self._make_status_response("COMPLETED")
        result_resp = MagicMock()
        result_resp.status_code = 200
        result_resp.raise_for_status = MagicMock()
        result_resp.json.return_value = {"images": []}

        with (
            patch("core.tools.image_gen.httpx.post", return_value=submit_resp),
            patch("core.tools.image_gen.httpx.get", side_effect=[status_resp, result_resp]),
            patch("core.tools.image_gen.time.sleep"),
        ):
            with pytest.raises(ValueError, match="returned no images"):
                client.generate_fullbody("test")

    def test_ignores_vibe_parameters(self):
        """FalTextToImageClient should accept but ignore NovelAI-specific params."""
        client = FalTextToImageClient()

        submit_resp = self._make_submit_response()
        status_resp = self._make_status_response("COMPLETED")
        result_resp = self._make_result_response()
        image_resp = self._make_image_response()

        with (
            patch("core.tools.image_gen.httpx.post", return_value=submit_resp) as mock_post,
            patch("core.tools.image_gen.httpx.get", side_effect=[status_resp, result_resp, image_resp]),
            patch("core.tools.image_gen.time.sleep"),
        ):
            result = client.generate_fullbody(
                "1girl",
                vibe_image=b"some-image-data",
                vibe_strength=0.5,
                vibe_info_extracted=0.9,
                steps=50,
                scale=7.0,
                sampler="k_euler",
            )

        # Should succeed without passing vibe params to the API
        assert result is not None
        payload = mock_post.call_args[1]["json"]
        assert "vibe_image" not in payload
        assert "vibe_strength" not in payload
        assert "steps" not in payload
        assert "sampler" not in payload

    def test_seed_omitted_when_none(self):
        client = FalTextToImageClient()

        submit_resp = self._make_submit_response()
        status_resp = self._make_status_response("COMPLETED")
        result_resp = self._make_result_response()
        image_resp = self._make_image_response()

        with (
            patch("core.tools.image_gen.httpx.post", return_value=submit_resp) as mock_post,
            patch("core.tools.image_gen.httpx.get", side_effect=[status_resp, result_resp, image_resp]),
            patch("core.tools.image_gen.time.sleep"),
        ):
            client.generate_fullbody("test", seed=None)

        payload = mock_post.call_args[1]["json"]
        assert "seed" not in payload

    def test_poll_lifecycle(self):
        """Verify the submit → poll → fetch lifecycle."""
        client = FalTextToImageClient()

        submit_resp = self._make_submit_response("req-abc")
        pending_resp = self._make_status_response("IN_PROGRESS")
        completed_resp = self._make_status_response("COMPLETED")
        result_resp = self._make_result_response("https://cdn.fal.ai/img.png")
        image_resp = self._make_image_response(b"LIFECYCLE-PNG")

        get_responses = [pending_resp, pending_resp, completed_resp, result_resp, image_resp]

        with (
            patch("core.tools.image_gen.httpx.post", return_value=submit_resp),
            patch("core.tools.image_gen.httpx.get", side_effect=get_responses),
            patch("core.tools.image_gen.time.sleep") as mock_sleep,
        ):
            result = client.generate_fullbody("test lifecycle")

        assert result == b"LIFECYCLE-PNG"
        # Should have slept during polling (at least 2 times for pending)
        assert mock_sleep.call_count >= 2
