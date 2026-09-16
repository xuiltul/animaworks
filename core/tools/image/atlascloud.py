# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Optional Atlas Cloud backend for character images and reference edits."""

from __future__ import annotations

import io
import time
from typing import Any
from urllib.parse import quote, urlparse

import httpx
from PIL import Image

from core.tools._base import get_credential

from .utils import _image_to_data_uri

_API = "https://api.atlascloud.ai/api/v1/model"
_SIZES = {
    "1:1": "2048*2048",
    "4:3": "2304*1728",
    "3:4": "1728*2304",
    "16:9": "2848*1600",
    "9:16": "1600*2848",
    "3:2": "2496*1664",
    "2:3": "1664*2496",
    "21:9": "3136*1344",
}


class AtlasCloudImageClient:
    """Generate PNG/JPEG assets using Seedream 4.5 through Atlas Cloud.

    A generation POST is sent once. Only prediction GETs tolerate transient
    failures, and polling never creates another prediction.
    """

    POLL_INTERVAL = 2.0
    POLL_TIMEOUT = 240.0
    MAX_POLL_ERRORS = 3

    def __init__(self) -> None:
        self._key = get_credential("atlascloud", "image_gen", env_var="ATLASCLOUD_API_KEY")

    def _generate(self, payload: dict[str, Any], output_format: str) -> bytes:
        if output_format not in {"png", "jpeg", "jpg"}:
            raise ValueError("Atlas Cloud images support png, jpeg, or jpg output")
        headers = {"Authorization": f"Bearer {self._key}"}
        response = httpx.post(f"{_API}/generateImage", json=payload, headers=headers, timeout=60.0)
        response.raise_for_status()
        prediction = self._prediction(response)
        prediction_id = prediction.get("id")
        if not isinstance(prediction_id, str) or not prediction_id:
            raise ValueError("Atlas Cloud returned no prediction ID; generation was not retried")
        deadline = time.monotonic() + self.POLL_TIMEOUT
        errors = 0
        while prediction.get("status") not in {"completed", "succeeded"}:
            status = prediction.get("status")
            if status in {"failed", "canceled", "cancelled"}:
                raise RuntimeError(f"Atlas Cloud prediction {prediction_id} {status}")
            if status not in {"created", "starting", "queued", "pending", "processing"}:
                raise ValueError(f"Atlas Cloud prediction {prediction_id} has an unknown status")
            remaining = deadline - time.monotonic()
            if remaining <= self.POLL_INTERVAL:
                raise TimeoutError(f"Atlas Cloud prediction {prediction_id} timed out; generation was not retried")
            time.sleep(self.POLL_INTERVAL)
            try:
                response = httpx.get(
                    f"{_API}/prediction/{quote(prediction_id, safe='')}",
                    headers=headers,
                    timeout=min(30.0, deadline - time.monotonic()),
                )
                response.raise_for_status()
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in {429, 500, 502, 503, 504}:
                    raise
                errors += 1
                if errors >= self.MAX_POLL_ERRORS:
                    raise
                continue
            prediction = self._prediction(response)
        outputs = prediction.get("outputs")
        if not isinstance(outputs, list) or not outputs or not isinstance(outputs[0], str):
            raise ValueError(f"Atlas Cloud prediction {prediction_id} returned no image")
        if urlparse(outputs[0]).scheme != "https":
            raise ValueError("Atlas Cloud returned a non-HTTPS image URL")
        # Output downloads must not receive the API credential.
        image_response = httpx.get(outputs[0], timeout=60.0, follow_redirects=True)
        image_response.raise_for_status()
        with Image.open(io.BytesIO(image_response.content)) as image:
            output = io.BytesIO()
            image.convert("RGB").save(output, format="PNG" if output_format == "png" else "JPEG")
            return output.getvalue()

    @staticmethod
    def _prediction(response: httpx.Response) -> dict[str, Any]:
        data = response.json()
        if not isinstance(data, dict) or data.get("code", 200) != 200 or not isinstance(data.get("data"), dict):
            raise ValueError("Atlas Cloud returned an invalid prediction response")
        return data["data"]

    def generate_fullbody(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 768,
        height: int = 1024,
        seed: int | None = None,
        output_format: str = "png",
        vibe_image: bytes | None = None,
        face_reference_image: bytes | None = None,
        **kwargs: Any,
    ) -> bytes:
        """Generate at the closest supported 2K ratio, optionally using a reference.

        Seedream does not expose seed, negative-prompt, or NovelAI sampler/vibe
        strength controls. These compatibility arguments are not sent to the API.
        """
        if width <= 0 or height <= 0:
            raise ValueError("Image width and height must be positive")
        ratio = min(
            _SIZES, key=lambda value: abs(float(value.split(":")[0]) / float(value.split(":")[1]) - width / height)
        )
        reference = face_reference_image if face_reference_image is not None else vibe_image
        if reference is not None:
            return self.generate_from_reference(reference, prompt, aspect_ratio=ratio, output_format=output_format)
        return self._generate(
            {"model": "bytedance/seedream-v4.5", "prompt": prompt, "size": _SIZES[ratio]}, output_format
        )

    def generate_from_reference(
        self,
        reference_image: bytes,
        prompt: str,
        aspect_ratio: str = "3:4",
        output_format: str = "png",
        guidance_scale: float = 3.5,
        seed: int | None = None,
    ) -> bytes:
        """Edit a reference image; seed and guidance controls are unsupported."""
        if aspect_ratio not in _SIZES:
            raise ValueError(f"Unsupported Atlas Cloud image aspect ratio: {aspect_ratio}")
        with Image.open(io.BytesIO(reference_image)) as image:
            reference = io.BytesIO()
            image.convert("RGB").save(reference, format="PNG")
        return self._generate(
            {
                "model": "bytedance/seedream-v4.5/edit",
                "prompt": prompt,
                "size": _SIZES[aspect_ratio],
                "images": [_image_to_data_uri(reference.getvalue())],
            },
            output_format,
        )
