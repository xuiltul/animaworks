# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Optional UberLLM backend for character images and reference edits."""

from __future__ import annotations

import base64
import binascii
import io
from typing import Any

import httpx
from PIL import Image

from core.credentials import get_credential
from core.integrations.image.utils import _image_to_data_uri

_API = "https://api.uberllm.dev/v1"
_SIZES = {
    "1:1": "1024x1024",
    "4:3": "1024x768",
    "3:4": "768x1024",
    "3:2": "1536x1024",
    "2:3": "1024x1536",
}


class UberLLMImageClient:
    """Generate PNG/JPEG assets with Seedream through UberLLM's image routes.

    Each image is one synchronous request that returns the picture inline
    (``b64_json``). Requests are sent once and never retried by the client.
    """

    GENERATE_MODEL = "bytedance/seedream-4.5"
    EDIT_MODEL = "bytedance/seedream-v4-edit"
    TIMEOUT = 180.0

    def __init__(self) -> None:
        self._key = get_credential("uberllm", "image_gen", env_var="UBERLLM_API_KEY")

    def _generate(self, route: str, payload: dict[str, Any], output_format: str) -> bytes:
        if output_format not in {"png", "jpeg", "jpg"}:
            raise ValueError("UberLLM images support png, jpeg, or jpg output")
        response = httpx.post(
            f"{_API}/images/{route}",
            json={**payload, "n": 1, "response_format": "b64_json"},
            headers={"Authorization": f"Bearer {self._key}"},
            timeout=self.TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        items = data.get("data") if isinstance(data, dict) else None
        encoded = items[0].get("b64_json") if isinstance(items, list) and items and isinstance(items[0], dict) else None
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("UberLLM returned no image")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except binascii.Error as exc:
            raise ValueError("UberLLM returned an invalid image encoding") from exc
        with Image.open(io.BytesIO(raw)) as image:
            output = io.BytesIO()
            image.convert("RGB").save(output, format="PNG" if output_format == "png" else "JPEG")
            return output.getvalue()

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
        """Generate at the closest supported size, optionally using a reference.

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
            "generations", {"model": self.GENERATE_MODEL, "prompt": prompt, "size": _SIZES[ratio]}, output_format
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
            raise ValueError(f"Unsupported UberLLM image aspect ratio: {aspect_ratio}")
        with Image.open(io.BytesIO(reference_image)) as image:
            reference = io.BytesIO()
            image.convert("RGB").save(reference, format="PNG")
        return self._generate(
            "edits",
            {
                "model": self.EDIT_MODEL,
                "prompt": prompt,
                "size": _SIZES[aspect_ratio],
                "image_url": _image_to_data_uri(reference.getvalue()),
            },
            output_format,
        )
