# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Character image & 3-D model generation tool for AnimaWorks.

Pipeline:
  1. NovelAI V4.5 → anime full-body image (fallback: fal.ai Flux Pro)
  2. Flux Kontext [pro] (fal.ai) → bust-up from reference
  3. Flux Kontext [pro] (fal.ai) → chibi from reference
  4. Meshy Image-to-3D → GLB model from chibi image
  5. Meshy Rigging → rigged GLB + walking/running animations
  6. Meshy Animations → idle/sitting/waving/talking GLBs
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from core.tools._base import ToolConfigError, get_credential, logger
from core.tools._retry import retry_with_backoff

# ── Execution Profile ─────────────────────────────────────

EXECUTION_PROFILE: dict[str, dict[str, object]] = {
    "pipeline":   {"expected_seconds": 1800, "background_eligible": True},
    "3d":         {"expected_seconds": 600,  "background_eligible": True},
    "rigging":    {"expected_seconds": 600,  "background_eligible": True},
    "animations": {"expected_seconds": 600,  "background_eligible": True},
    "fullbody":   {"expected_seconds": 120,  "background_eligible": True},
    "bustup":     {"expected_seconds": 120,  "background_eligible": True},
    "chibi":      {"expected_seconds": 120,  "background_eligible": True},
}

# ── Constants ──────────────────────────────────────────────

NOVELAI_API_URL = "https://image.novelai.net/ai/generate-image"
NOVELAI_ENCODE_URL = "https://image.novelai.net/ai/encode-vibe"
NOVELAI_MODEL = "nai-diffusion-4-5-full"

FAL_KONTEXT_SUBMIT_URL = "https://queue.fal.run/fal-ai/flux-pro/kontext"
FAL_FLUX_PRO_SUBMIT_URL = "https://queue.fal.run/fal-ai/flux-pro/v1.1"
# Status/result URLs are extracted from the submit response
# (they omit the /kontext subpath per fal.ai queue convention)

MESHY_IMAGE_TO_3D_URL = "https://api.meshy.ai/openapi/v1/image-to-3d"
MESHY_TASK_URL_TPL = "https://api.meshy.ai/openapi/v1/image-to-3d/{task_id}"
MESHY_RIGGING_URL = "https://api.meshy.ai/openapi/v1/rigging"
MESHY_RIGGING_TASK_TPL = "https://api.meshy.ai/openapi/v1/rigging/{task_id}"
MESHY_ANIMATION_URL = "https://api.meshy.ai/openapi/v1/animations"
MESHY_ANIMATION_TASK_TPL = "https://api.meshy.ai/openapi/v1/animations/{task_id}"

# Default prompts for Kontext derivation
_BUSTUP_PROMPT = (
    "Generate a portrait of the same character from the chest up. "
    "Same outfit, same colors, same features. "
    "Anime illustration style, soft lighting, looking at viewer."
)
_CHIBI_PROMPT = (
    "Transform this character into a chibi / super-deformed version. "
    "2.5-head proportion, cute big eyes, simplified body. "
    "Same outfit colors and features. White background, full body, anime style."
)

# Expression-specific prompts for bustup image variants
_EXPRESSION_PROMPTS: dict[str, str] = {
    "neutral": (
        "The character with a calm relaxed expression, looking at viewer. "
        "Soft eyes, natural closed mouth, relaxed eyebrows. "
        "Arms at sides in a natural posture. "
        "Bust-up portrait, anime illustration, soft lighting. "
        "Same character identity, outfit, and hairstyle."
    ),
    "smile": (
        "Change the character's expression to a bright happy smile. "
        "Cheeks raised, eyes softened with warmth, gentle closed-mouth smile. "
        "Head tilted slightly to one side, one hand making a small wave or peace sign. "
        "Bust-up portrait, anime illustration, soft lighting. "
        "Same character identity, outfit, and hairstyle."
    ),
    "laugh": (
        "Change the character's expression to joyful laughing. "
        "Eyes squeezed shut happily, mouth wide open showing teeth, raised cheeks. "
        "Head tilted back with amusement, one hand near mouth. "
        "Bust-up portrait, anime illustration, soft lighting. "
        "Same character identity, outfit, and hairstyle."
    ),
    "troubled": (
        "Change the character's expression to worried and troubled. "
        "Eyebrows furrowed and pinched together, slight frown, nervous eyes. "
        "Both hands clasped together in front of chest, shoulders raised tensely. "
        "Bust-up portrait, anime illustration, soft lighting. "
        "Same character identity, outfit, and hairstyle."
    ),
    "surprised": (
        "Change the character's expression to shocked surprise. "
        "Eyes wide open as large as possible, eyebrows raised very high, "
        "mouth dropped open in a round shape. "
        "Both hands raised up near face, leaning back slightly in shock. "
        "Bust-up portrait, anime illustration, soft lighting. "
        "Same character identity, outfit, and hairstyle."
    ),
    "thinking": (
        "Change the character's expression to deep contemplation. "
        "Looking slightly upward, one eye slightly narrowed, thoughtful furrowed brow. "
        "One hand on chin in a classic thinking pose, slight pout. "
        "Bust-up portrait, anime illustration, soft lighting. "
        "Same character identity, outfit, and hairstyle."
    ),
    "embarrassed": (
        "Change the character's expression to deeply embarrassed. "
        "Bright red blush across entire face, eyes averted to the side. "
        "Both hands covering cheeks or pressing index fingers together nervously. "
        "Bust-up portrait, anime illustration, soft lighting. "
        "Same character identity, outfit, and hairstyle."
    ),
}

from core.schemas import VALID_EMOTIONS as _VALID_EXPRESSION_NAMES

# Verify prompts cover all valid emotions
assert set(_EXPRESSION_PROMPTS.keys()) == _VALID_EXPRESSION_NAMES, (
    f"Expression prompts mismatch: {set(_EXPRESSION_PROMPTS.keys())} != {_VALID_EXPRESSION_NAMES}"
)

# Per-expression guidance_scale for Flux Kontext.
# Higher values force stronger prompt adherence (more dramatic expression change).
_EXPRESSION_GUIDANCE: dict[str, float] = {
    "neutral": 4.0,
    "smile": 5.0,
    "laugh": 5.0,
    "troubled": 5.5,
    "surprised": 5.0,
    "thinking": 5.0,
    "embarrassed": 5.5,
}

assert set(_EXPRESSION_GUIDANCE.keys()) == _VALID_EXPRESSION_NAMES, (
    f"Expression guidance mismatch: {set(_EXPRESSION_GUIDANCE.keys())} != {_VALID_EXPRESSION_NAMES}"
)

# Default animation presets for office digital animas
# See https://docs.meshy.ai/api/animation-library for full catalog
_DEFAULT_ANIMATIONS: dict[str, int] = {
    "idle": 0,           # Standing idle
    "sitting": 32,       # Chair sit idle (female)
    "waving": 28,        # Big wave hello
    "talking": 307,      # Talking gesture
}

_HTTP_TIMEOUT = httpx.Timeout(30.0, read=120.0)
_DOWNLOAD_TIMEOUT = httpx.Timeout(60.0, read=300.0)

_RETRYABLE_CODES = {429, 500, 502, 503}


# ── Helpers ────────────────────────────────────────────────


def _retry(
    fn: Any,
    *,
    max_retries: int = 2,
    delay: float = 5.0,
    retryable_codes: set[int] | None = None,
) -> Any:
    """Execute *fn* with simple retry on transient HTTP errors.

    Delegates to the shared :func:`retry_with_backoff` utility while
    preserving the original filtering logic for non-retryable HTTP
    status codes.
    """
    codes = retryable_codes or _RETRYABLE_CODES

    class _NonRetryableHTTPError(Exception):
        """Wrapper for HTTP errors with non-retryable status codes."""

    def _guarded() -> Any:
        try:
            return fn()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in codes:
                raise _NonRetryableHTTPError from exc
            raise

    try:
        return retry_with_backoff(
            _guarded,
            max_retries=max_retries,
            base_delay=delay,
            max_delay=300.0,
            retry_on=(httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout),
        )
    except _NonRetryableHTTPError as exc:
        raise exc.__cause__ from None  # type: ignore[misc]


def _image_to_data_uri(image_bytes: bytes, mime: str = "image/png") -> str:
    """Encode raw image bytes as a ``data:`` URI."""
    b64 = base64.b64encode(image_bytes).decode()
    return f"data:{mime};base64,{b64}"


# ── NovelAIClient ──────────────────────────────────────────


class NovelAIClient:
    """NovelAI V4.5 API client for anime full-body image generation."""

    def __init__(self) -> None:
        self._token = get_credential("novelai", "image_gen", env_var="NOVELAI_TOKEN")

    def encode_vibe(
        self,
        image: bytes,
        information_extracted: float = 0.8,
    ) -> bytes:
        """Encode an image into V4+ vibe binary via /ai/encode-vibe.

        The NovelAI V4/V4.5 API requires images to be pre-encoded into
        vibe representations before they can be used as style references.
        This costs 2 Anlas per encoding.

        Returns:
            Binary vibe data (~48 KB).
        """
        b64 = base64.b64encode(image).decode()
        body = {
            "image": b64,
            "model": NOVELAI_MODEL,
            "information_extracted": information_extracted,
        }

        def _call() -> bytes:
            resp = httpx.post(
                NOVELAI_ENCODE_URL,
                json=body,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                timeout=_HTTP_TIMEOUT,
            )
            if resp.status_code != 200:
                logger.error(
                    "NovelAI encode-vibe error %d: %s",
                    resp.status_code,
                    resp.text[:500],
                )
            resp.raise_for_status()
            return resp.content

        return _retry(_call)

    def generate_fullbody(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1536,
        seed: int | None = None,
        steps: int = 28,
        scale: float = 5.0,
        sampler: str = "k_euler_ancestral",
        vibe_image: bytes | None = None,
        vibe_strength: float = 0.6,
        vibe_info_extracted: float = 0.8,
    ) -> bytes:
        """Generate a full-body anime character image.

        Returns:
            PNG image bytes.
        """
        neg = negative_prompt or (
            "lowres, bad anatomy, bad hands, missing fingers, extra digits, "
            "fewer digits, worst quality, low quality, blurry, jpeg artifacts, "
            "cropped, multiple views, logo, too many watermarks"
        )

        params: dict[str, Any] = {
            "width": width,
            "height": height,
            "scale": scale,
            "sampler": sampler,
            "steps": steps,
            "n_samples": 1,
            "ucPreset": 0,
            "qualityToggle": True,
            "sm": False,
            "sm_dyn": False,
            "dynamic_thresholding": False,
            "legacy": False,
            "cfg_rescale": 0,
            "noise_schedule": "native",
            "negative_prompt": neg,
            # V4/V4.5 structured prompt (required for nai-diffusion-4+)
            "v4_prompt": {
                "caption": {
                    "base_caption": prompt,
                    "char_captions": [],
                },
                "use_coords": False,
                "use_order": True,
            },
            "v4_negative_prompt": {
                "caption": {
                    "base_caption": neg,
                    "char_captions": [],
                },
                "legacy_uc": False,
            },
            "reference_image_multiple": [],
            "reference_information_extracted_multiple": [],
            "reference_strength_multiple": [],
        }
        if seed is not None:
            params["seed"] = seed

        # Vibe Transfer – V4+ requires pre-encoded vibe data
        if vibe_image is not None:
            encoded = self.encode_vibe(vibe_image, vibe_info_extracted)
            b64 = base64.b64encode(encoded).decode()
            params["reference_image_multiple"] = [b64]
            params["reference_information_extracted_multiple"] = [vibe_info_extracted]
            params["reference_strength_multiple"] = [vibe_strength]

        body = {
            "input": prompt,
            "model": NOVELAI_MODEL,
            "action": "generate",
            "parameters": params,
        }

        def _call() -> bytes:
            resp = httpx.post(
                NOVELAI_API_URL,
                json=body,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                timeout=_HTTP_TIMEOUT,
            )
            if resp.status_code != 200:
                logger.error(
                    "NovelAI generate error %d: %s",
                    resp.status_code,
                    resp.text[:500],
                )
            resp.raise_for_status()
            return self._extract_png(resp.content)

        return _retry(_call)

    @staticmethod
    def _extract_png(data: bytes) -> bytes:
        """Extract the first PNG from a ZIP-compressed response."""
        buf = io.BytesIO(data)
        with zipfile.ZipFile(buf) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".png"):
                    return zf.read(name)
        raise ValueError("NovelAI response ZIP contains no PNG file")


# ── FluxKontextClient ──────────────────────────────────────


class FluxKontextClient:
    """Flux Kontext [pro] client via fal.ai for reference-based generation."""

    POLL_INTERVAL = 2.0  # seconds
    POLL_TIMEOUT = 120.0  # seconds

    def __init__(self) -> None:
        self._key = get_credential("fal", "image_gen", env_var="FAL_KEY")

    def generate_from_reference(
        self,
        reference_image: bytes,
        prompt: str,
        aspect_ratio: str = "3:4",
        output_format: str = "png",
        guidance_scale: float = 3.5,
        seed: int | None = None,
    ) -> bytes:
        """Generate an image from a reference image with Flux Kontext.

        Returns:
            PNG (or JPEG) image bytes.
        """
        data_uri = _image_to_data_uri(reference_image)
        payload: dict[str, Any] = {
            "prompt": prompt,
            "image_url": data_uri,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
            "guidance_scale": guidance_scale,
            "num_images": 1,
            "safety_tolerance": "6",
        }
        if seed is not None:
            payload["seed"] = seed

        headers = {
            "Authorization": f"Key {self._key}",
            "Content-Type": "application/json",
        }

        # Submit task
        def _submit() -> dict[str, str]:
            resp = httpx.post(
                FAL_KONTEXT_SUBMIT_URL,
                json=payload,
                headers=headers,
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "request_id": data["request_id"],
                "status_url": data["status_url"],
                "response_url": data["response_url"],
            }

        submit_data = _retry(_submit)
        request_id = submit_data["request_id"]

        # Poll for completion (use URLs from submit response)
        result_url = submit_data["response_url"]
        status_url = submit_data["status_url"]
        deadline = time.monotonic() + self.POLL_TIMEOUT

        while time.monotonic() < deadline:
            time.sleep(self.POLL_INTERVAL)
            status_resp = httpx.get(
                status_url, headers=headers, timeout=_HTTP_TIMEOUT,
            )
            status_resp.raise_for_status()
            status_data = status_resp.json()
            if status_data.get("status") == "COMPLETED":
                break
            if status_data.get("status") == "FAILED":
                raise RuntimeError(
                    f"Flux Kontext task {request_id} failed: "
                    f"{status_data.get('error', 'unknown')}"
                )
        else:
            raise TimeoutError(
                f"Flux Kontext task {request_id} timed out after "
                f"{self.POLL_TIMEOUT}s"
            )

        # Fetch result
        result_resp = httpx.get(
            result_url, headers=headers, timeout=_HTTP_TIMEOUT,
        )
        result_resp.raise_for_status()
        result_data = result_resp.json()

        images = result_data.get("images", [])
        if not images:
            raise ValueError("Flux Kontext returned no images")

        image_url = images[0]["url"]
        img_resp = httpx.get(image_url, timeout=_DOWNLOAD_TIMEOUT)
        img_resp.raise_for_status()
        return img_resp.content


# ── FalTextToImageClient ──────────────────────────────────


class FalTextToImageClient:
    """Fal.ai Flux Pro text-to-image client (fallback for NovelAI)."""

    POLL_INTERVAL = 2.0  # seconds
    POLL_TIMEOUT = 120.0  # seconds

    def __init__(self) -> None:
        self._key = get_credential("fal", "image_gen", env_var="FAL_KEY")

    def generate_fullbody(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 768,
        height: int = 1024,
        seed: int | None = None,
        output_format: str = "png",
        guidance_scale: float = 3.5,
        # Accept and ignore NovelAI-specific params for interface compatibility
        steps: int = 28,
        scale: float = 5.0,
        sampler: str = "k_euler_ancestral",
        vibe_image: bytes | None = None,
        vibe_strength: float = 0.6,
        vibe_info_extracted: float = 0.8,
    ) -> bytes:
        """Generate a full-body character image from text prompt.

        Uses fal.ai Flux Pro v1.1 model.  Compatible with the same
        call signature as :meth:`NovelAIClient.generate_fullbody` but
        ignores NovelAI-specific parameters (vibe transfer, sampler, etc.).

        Returns:
            PNG image bytes.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "image_size": {"width": width, "height": height},
            "output_format": output_format,
            "guidance_scale": guidance_scale,
            "num_images": 1,
            "safety_tolerance": "6",
        }
        if seed is not None:
            payload["seed"] = seed

        headers = {
            "Authorization": f"Key {self._key}",
            "Content-Type": "application/json",
        }

        # Submit task
        def _submit() -> dict[str, str]:
            resp = httpx.post(
                FAL_FLUX_PRO_SUBMIT_URL,
                json=payload,
                headers=headers,
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "request_id": data["request_id"],
                "status_url": data["status_url"],
                "response_url": data["response_url"],
            }

        submit_data = _retry(_submit)
        request_id = submit_data["request_id"]

        # Poll for completion
        result_url = submit_data["response_url"]
        status_url = submit_data["status_url"]
        deadline = time.monotonic() + self.POLL_TIMEOUT

        while time.monotonic() < deadline:
            time.sleep(self.POLL_INTERVAL)
            status_resp = httpx.get(
                status_url, headers=headers, timeout=_HTTP_TIMEOUT,
            )
            status_resp.raise_for_status()
            status_data = status_resp.json()
            if status_data.get("status") == "COMPLETED":
                break
            if status_data.get("status") == "FAILED":
                raise RuntimeError(
                    f"Fal Flux Pro task {request_id} failed: "
                    f"{status_data.get('error', 'unknown')}"
                )
        else:
            raise TimeoutError(
                f"Fal Flux Pro task {request_id} timed out after "
                f"{self.POLL_TIMEOUT}s"
            )

        # Fetch result
        result_resp = httpx.get(
            result_url, headers=headers, timeout=_HTTP_TIMEOUT,
        )
        result_resp.raise_for_status()
        result_data = result_resp.json()

        images = result_data.get("images", [])
        if not images:
            raise ValueError("Fal Flux Pro returned no images")

        image_url = images[0]["url"]
        img_resp = httpx.get(image_url, timeout=_DOWNLOAD_TIMEOUT)
        img_resp.raise_for_status()
        return img_resp.content


# ── MeshyClient ────────────────────────────────────────────


class MeshyClient:
    """Meshy Image-to-3D API client."""

    POLL_INTERVAL = 10.0  # seconds
    POLL_TIMEOUT = 600.0  # seconds (10 min)

    def __init__(self) -> None:
        self._key = get_credential("meshy", "image_gen", env_var="MESHY_API_KEY")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key}"}

    def create_task(
        self,
        image_bytes: bytes,
        *,
        ai_model: str = "meshy-6",
        topology: str = "triangle",
        target_polycount: int = 30000,
        should_texture: bool = True,
        enable_pbr: bool = False,
    ) -> str:
        """Submit an image-to-3D task.

        Returns:
            Task ID string.
        """
        data_uri = _image_to_data_uri(image_bytes)
        body: dict[str, Any] = {
            "image_url": data_uri,
            "ai_model": ai_model,
            "topology": topology,
            "target_polycount": target_polycount,
            "should_texture": should_texture,
            "enable_pbr": enable_pbr,
        }

        def _call() -> str:
            resp = httpx.post(
                MESHY_IMAGE_TO_3D_URL,
                json=body,
                headers=self._headers(),
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["result"]

        return _retry(_call, max_retries=1, delay=10.0)

    def poll_task(self, task_id: str) -> dict[str, Any]:
        """Poll until task completes.

        Returns:
            Completed task dict with ``model_urls``.
        """
        url = MESHY_TASK_URL_TPL.format(task_id=task_id)
        deadline = time.monotonic() + self.POLL_TIMEOUT

        while time.monotonic() < deadline:
            resp = httpx.get(url, headers=self._headers(), timeout=_HTTP_TIMEOUT)
            resp.raise_for_status()
            task = resp.json()
            status = task.get("status", "")
            if status == "SUCCEEDED":
                return task
            if status in ("FAILED", "CANCELED"):
                err = task.get("task_error", {}).get("message", "unknown")
                raise RuntimeError(f"Meshy task {task_id} {status}: {err}")
            logger.debug(
                "Meshy task %s: %s (%d%%)",
                task_id, status, task.get("progress", 0),
            )
            time.sleep(self.POLL_INTERVAL)

        raise TimeoutError(
            f"Meshy task {task_id} timed out after {self.POLL_TIMEOUT}s"
        )

    def download_model(self, task: dict[str, Any], fmt: str = "glb") -> bytes:
        """Download the generated 3-D model.

        Args:
            task: Completed task dict (from :meth:`poll_task`).
            fmt: Model format key (``glb``, ``fbx``, ``obj``, ``usdz``).

        Returns:
            Raw model bytes.
        """
        model_urls = task.get("model_urls", {})
        url = model_urls.get(fmt)
        if not url:
            available = list(model_urls.keys())
            raise ValueError(
                f"Format '{fmt}' not available; got {available}"
            )
        resp = httpx.get(url, timeout=_DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
        return resp.content

    def create_rigging_task(self, input_task_id: str) -> str:
        """Submit a rigging task for a completed image-to-3D task.

        Returns:
            Rigging task ID.
        """
        body = {"input_task_id": input_task_id}

        def _call() -> str:
            resp = httpx.post(
                MESHY_RIGGING_URL,
                json=body,
                headers=self._headers(),
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["result"]

        return _retry(_call, max_retries=1, delay=10.0)

    def poll_rigging_task(self, task_id: str) -> dict[str, Any]:
        """Poll until a rigging task completes."""
        url = MESHY_RIGGING_TASK_TPL.format(task_id=task_id)
        deadline = time.monotonic() + self.POLL_TIMEOUT

        while time.monotonic() < deadline:
            resp = httpx.get(url, headers=self._headers(), timeout=_HTTP_TIMEOUT)
            resp.raise_for_status()
            task = resp.json()
            status = task.get("status", "")
            if status == "SUCCEEDED":
                return task
            if status in ("FAILED", "CANCELED"):
                err = task.get("task_error", {}).get("message", "unknown")
                raise RuntimeError(f"Meshy rigging {task_id} {status}: {err}")
            logger.debug(
                "Meshy rigging %s: %s (%d%%)",
                task_id, status, task.get("progress", 0),
            )
            time.sleep(self.POLL_INTERVAL)

        raise TimeoutError(
            f"Meshy rigging {task_id} timed out after {self.POLL_TIMEOUT}s"
        )

    def download_rigged_model(self, task: dict[str, Any], fmt: str = "glb") -> bytes:
        """Download the rigged character model.

        Args:
            task: Completed rigging task dict.
            fmt: Format (``glb`` or ``fbx``).

        Returns:
            Raw model bytes.
        """
        result = task.get("result", {})
        key = f"rigged_character_{fmt}_url"
        url = result.get(key)
        if not url:
            raise ValueError(f"Rigging result missing '{key}'")
        resp = httpx.get(url, timeout=_DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
        return resp.content

    def download_rigging_animations(self, task: dict[str, Any]) -> dict[str, bytes]:
        """Download built-in walking/running animations from rigging task.

        Prefers armature-only GLBs (skeleton + animation, no mesh) when
        available.  Falls back to full model GLBs.

        Returns:
            Dict mapping animation name to GLB bytes.
        """
        result = task.get("result", {})
        basic = result.get("basic_animations", {})
        animations: dict[str, bytes] = {}
        for name in ("walking", "running"):
            # Prefer armature-only (much smaller, ~50-500 KB vs ~27 MB)
            url = basic.get(f"{name}_armature_glb_url") or basic.get(f"{name}_glb_url")
            if url:
                logger.debug("Downloading %s animation …", name)
                resp = httpx.get(url, timeout=_DOWNLOAD_TIMEOUT)
                resp.raise_for_status()
                animations[name] = resp.content
        return animations

    # ── Animation API ──

    def create_animation_task(self, rig_task_id: str, action_id: int) -> str:
        """Submit an animation task for a rigged character.

        Args:
            rig_task_id: Completed rigging task ID.
            action_id: Animation preset ID (see Meshy animation library).

        Returns:
            Animation task ID.
        """
        body: dict[str, Any] = {
            "rig_task_id": rig_task_id,
            "action_id": action_id,
            "post_process": {"operation_type": "extract_armature"},
        }

        def _call() -> str:
            resp = httpx.post(
                MESHY_ANIMATION_URL,
                json=body,
                headers=self._headers(),
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["result"]

        return _retry(_call, max_retries=1, delay=10.0)

    def poll_animation_task(self, task_id: str) -> dict[str, Any]:
        """Poll until an animation task completes."""
        url = MESHY_ANIMATION_TASK_TPL.format(task_id=task_id)
        deadline = time.monotonic() + self.POLL_TIMEOUT

        while time.monotonic() < deadline:
            resp = httpx.get(url, headers=self._headers(), timeout=_HTTP_TIMEOUT)
            resp.raise_for_status()
            task = resp.json()
            status = task.get("status", "")
            if status == "SUCCEEDED":
                return task
            if status in ("FAILED", "CANCELED"):
                err = task.get("task_error", {}).get("message", "unknown")
                raise RuntimeError(f"Meshy animation {task_id} {status}: {err}")
            logger.debug(
                "Meshy animation %s: %s (%d%%)",
                task_id, status, task.get("progress", 0),
            )
            time.sleep(self.POLL_INTERVAL)

        raise TimeoutError(
            f"Meshy animation {task_id} timed out after {self.POLL_TIMEOUT}s"
        )

    def download_animation(self, task: dict[str, Any], fmt: str = "glb") -> bytes:
        """Download generated animation file.

        Args:
            task: Completed animation task dict.
            fmt: Format (``glb`` or ``fbx``).

        Returns:
            Raw animation bytes.
        """
        result = task.get("result", {})
        url = result.get(f"animation_{fmt}_url")
        if not url:
            raise ValueError(f"Animation result missing 'animation_{fmt}_url'")
        resp = httpx.get(url, timeout=_DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
        return resp.content


# ── gltf-transform helpers ────────────────────────────────────


def _run_gltf_transform(args: list[str], glb_path: Path) -> bool:
    """Run gltf-transform CLI command. Returns True on success."""
    import shutil
    import subprocess

    npx = shutil.which("npx")
    if npx is None:
        logger.warning("npx not found; skipping gltf-transform for %s", glb_path)
        return False

    cmd = [npx, "--yes", "@gltf-transform/cli", *args]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return True
    except FileNotFoundError:
        logger.warning("gltf-transform not available; skipping for %s", glb_path)
        return False
    except subprocess.CalledProcessError as exc:
        logger.warning("gltf-transform failed for %s: %s", glb_path, exc.stderr[:500].decode("utf-8", errors="replace") if exc.stderr else exc)
        return False
    except subprocess.TimeoutExpired:
        logger.warning("gltf-transform timed out for %s", glb_path)
        return False


_GLTF_MODULES_DIR: Path | None = None


def _ensure_gltf_transform_modules() -> Path:
    """Install @gltf-transform/core and /functions to a persistent cache.

    Returns the ``node_modules`` directory path to be used as ``NODE_PATH``.
    Packages are installed once and reused across calls.
    """
    global _GLTF_MODULES_DIR  # noqa: PLW0603
    if _GLTF_MODULES_DIR is not None and _GLTF_MODULES_DIR.exists():
        return _GLTF_MODULES_DIR

    import subprocess

    from core.paths import get_data_dir

    cache_dir = get_data_dir() / "cache" / "gltf_transform_modules"
    node_modules = cache_dir / "node_modules"

    if not (node_modules / "@gltf-transform" / "core").is_dir():
        cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["npm", "install", "--save",
             "@gltf-transform/core", "@gltf-transform/functions"],
            cwd=str(cache_dir),
            check=True, capture_output=True, timeout=120,
        )
        logger.info("Installed @gltf-transform modules to %s", cache_dir)

    _GLTF_MODULES_DIR = node_modules
    return node_modules


# ── fbx2gltf helpers ──────────────────────────────────────


_FBX2GLTF_PATH: Path | None = None


def _ensure_fbx2gltf() -> Path | None:
    """Install fbx2gltf to a persistent cache and return the binary path.

    Resolution order:
      1. Module-level cached path (fast path).
      2. System ``PATH`` via ``shutil.which("FBX2glTF")``.
      3. Persistent npm cache at ``~/.animaworks/cache/fbx2gltf/``.
      4. Fresh ``npm install --save fbx2gltf`` into the cache directory.

    Returns:
        Binary path on success, ``None`` if installation fails.
    """
    global _FBX2GLTF_PATH  # noqa: PLW0603
    if _FBX2GLTF_PATH is not None and _FBX2GLTF_PATH.exists():
        return _FBX2GLTF_PATH

    import shutil

    # Check if already on PATH
    system_path = shutil.which("FBX2glTF")
    if system_path:
        _FBX2GLTF_PATH = Path(system_path)
        return _FBX2GLTF_PATH

    import subprocess

    from core.paths import get_data_dir

    cache_dir = get_data_dir() / "cache" / "fbx2gltf"
    node_modules = cache_dir / "node_modules"

    # Check both .bin symlink and platform-specific binary location
    for found in _find_fbx2gltf_binary(node_modules):
        _FBX2GLTF_PATH = found
        return _FBX2GLTF_PATH

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["npm", "install", "--save", "fbx2gltf"],
            cwd=str(cache_dir),
            check=True,
            capture_output=True,
            timeout=120,
        )
        for found in _find_fbx2gltf_binary(node_modules):
            _FBX2GLTF_PATH = found
            logger.info("Installed fbx2gltf to %s", cache_dir)
            return _FBX2GLTF_PATH
    except Exception as exc:
        logger.warning("Failed to install fbx2gltf: %s", exc)
    return None


def _find_fbx2gltf_binary(node_modules: Path):
    """Yield fbx2gltf binary paths if they exist.

    The npm ``fbx2gltf`` package places the binary at a platform-specific
    path (e.g. ``node_modules/fbx2gltf/bin/Linux/FBX2glTF``) rather than
    the standard ``node_modules/.bin/`` symlink.
    """
    import platform

    # Standard .bin symlink
    candidate = node_modules / ".bin" / "fbx2gltf"
    if candidate.exists():
        yield candidate

    # Platform-specific binary (Linux, Darwin, Windows)
    os_name = platform.system()  # "Linux", "Darwin", "Windows"
    candidate = node_modules / "fbx2gltf" / "bin" / os_name / "FBX2glTF"
    if candidate.exists():
        yield candidate


def _convert_fbx_to_glb(fbx_path: Path, glb_path: Path) -> bool:
    """Convert an FBX file to GLB using fbx2gltf.

    Args:
        fbx_path: Source FBX file.
        glb_path: Destination GLB file.

    Returns:
        ``True`` on success, ``False`` on failure.
    """
    import subprocess

    fbx2gltf = _ensure_fbx2gltf()
    if fbx2gltf is None:
        return False
    try:
        subprocess.run(
            [str(fbx2gltf), "--binary", "--input", str(fbx_path),
             "--output", str(glb_path)],
            check=True,
            capture_output=True,
            timeout=120,
        )
        return glb_path.exists()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("fbx2gltf conversion failed for %s: %s", fbx_path, exc)
        return False


def _download_armature_animation(
    task: dict[str, Any],
    glb_path: Path,
) -> bool:
    """Download armature-only animation, converting from FBX if available.

    Tries the optimised path first: ``processed_armature_fbx_url`` (small,
    ~10-50 KB) downloaded as FBX then converted to GLB via *fbx2gltf*.

    Falls back to the legacy path on any failure: full-mesh GLB download
    followed by :func:`strip_mesh_from_glb`.

    Args:
        task: Completed animation task dict from Meshy API.
        glb_path: Destination path for the final ``.glb`` file.

    Returns:
        ``True`` when a usable GLB was written, ``False`` on total failure.
    """
    import tempfile

    result = task.get("result", {})
    armature_url = result.get("processed_armature_fbx_url")

    if armature_url:
        fbx_path: Path | None = None
        try:
            resp = httpx.get(armature_url, timeout=_DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(
                suffix=".fbx", delete=False,
            ) as f:
                f.write(resp.content)
                fbx_path = Path(f.name)
            if _convert_fbx_to_glb(fbx_path, glb_path):
                logger.info(
                    "Armature animation converted: %s (%d bytes)",
                    glb_path,
                    glb_path.stat().st_size,
                )
                return True
        except Exception as exc:
            logger.warning(
                "Armature download/conversion failed for %s, falling back: %s",
                glb_path.name,
                exc,
            )
        finally:
            if fbx_path is not None:
                fbx_path.unlink(missing_ok=True)

    # ── Fallback: full GLB + strip ──
    logger.warning(
        "Falling back to full GLB download + strip for %s", glb_path.name,
    )
    glb_url = result.get("animation_glb_url")
    if not glb_url:
        logger.error("No animation URL available for %s", glb_path.name)
        return False
    resp = httpx.get(glb_url, timeout=_DOWNLOAD_TIMEOUT)
    resp.raise_for_status()
    glb_path.write_bytes(resp.content)
    if not strip_mesh_from_glb(glb_path):
        logger.warning("Fallback strip also failed for %s", glb_path.name)
    return True


def strip_mesh_from_glb(glb_path: Path) -> bool:
    """Remove mesh/material/texture data from a GLB, keeping only skeleton + animation.

    Uses a Node.js script via gltf-transform programmatic API because
    the CLI does not have a dedicated 'strip meshes' command.

    Packages are installed to a persistent cache directory under
    ``~/.animaworks/cache/gltf_transform_modules/`` and referenced via
    ``NODE_PATH`` (npm 10.x compatible).

    Returns True on success, False if skipped or failed.
    """
    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    if node is None:
        logger.warning("node not found; skipping mesh strip for %s", glb_path)
        return False

    script = """\
const { NodeIO } = require("@gltf-transform/core");
const { prune } = require("@gltf-transform/functions");

(async () => {
    const io = new NodeIO();
    const doc = await io.read(process.argv[2]);
    for (const node of doc.getRoot().listNodes()) {
        node.setMesh(null);
    }
    for (const mat of doc.getRoot().listMaterials()) {
        mat.dispose();
    }
    for (const tex of doc.getRoot().listTextures()) {
        tex.dispose();
    }
    await doc.transform(prune());
    await io.write(process.argv[2], doc);
})();
"""

    script_path = None
    try:
        node_modules = _ensure_gltf_transform_modules()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cjs", delete=False,
        ) as fd:
            fd.write(script)
            script_path = fd.name

        import os
        env = {**os.environ, "NODE_PATH": str(node_modules)}
        subprocess.run(
            [node, script_path, str(glb_path)],
            check=True, capture_output=True, timeout=120, env=env,
        )
        logger.info("Stripped mesh from %s (now %d bytes)", glb_path, glb_path.stat().st_size)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("Mesh strip failed for %s: %s", glb_path, exc)
        return False
    finally:
        if script_path is not None:
            Path(script_path).unlink(missing_ok=True)


def optimize_glb(glb_path: Path) -> bool:
    """Apply Draco compression to a GLB file using gltf-transform.

    Returns True on success, False if skipped or failed.
    """
    tmp_path = glb_path.with_suffix(".opt.glb")
    try:
        if _run_gltf_transform(["optimize", str(glb_path), str(tmp_path)], glb_path):
            if _run_gltf_transform(["draco", str(tmp_path), str(glb_path)], glb_path):
                tmp_path.unlink(missing_ok=True)
                logger.info("Optimized %s (now %d bytes)", glb_path, glb_path.stat().st_size)
                return True
            else:
                # optimize succeeded but draco failed; keep optimized version
                tmp_path.rename(glb_path)
                return True
        return False
    finally:
        tmp_path.unlink(missing_ok=True)


def simplify_glb(glb_path: Path, target_ratio: float = 0.27, error_threshold: float = 0.01) -> bool:
    """Simplify mesh geometry using gltf-transform simplify (meshoptimizer).

    ``target_ratio=0.27`` reduces ~30K-polygon Meshy models to ~8K polygons.
    ``error_threshold`` controls maximum allowed deviation (lower = higher quality).

    Returns True on success, False if skipped or failed.
    """
    tmp_path = glb_path.with_suffix(".simp.glb")
    try:
        if _run_gltf_transform(
            ["simplify", str(glb_path), str(tmp_path),
             "--ratio", str(target_ratio),
             "--error", str(error_threshold)],
            glb_path,
        ):
            tmp_path.rename(glb_path)
            logger.info("Simplified %s (now %d bytes)", glb_path, glb_path.stat().st_size)
            return True
        return False
    finally:
        tmp_path.unlink(missing_ok=True)


def compress_textures(glb_path: Path, resolution: int = 1024) -> bool:
    """Resize textures and convert to WebP format using gltf-transform.

    Applies two-step optimization:
    1. Resize all textures to ``resolution x resolution``
    2. Convert textures to WebP

    Returns True on success, False if skipped or failed.
    """
    tmp_path = glb_path.with_suffix(".tex.glb")
    try:
        # Step 1: resize textures
        if not _run_gltf_transform(
            ["resize", str(glb_path), str(tmp_path),
             "--width", str(resolution), "--height", str(resolution)],
            glb_path,
        ):
            return False
        # Step 2: convert to WebP
        if _run_gltf_transform(
            ["webp", str(tmp_path), str(glb_path)],
            glb_path,
        ):
            tmp_path.unlink(missing_ok=True)
            logger.info(
                "Compressed textures in %s to webp@%d (now %d bytes)",
                glb_path, resolution, glb_path.stat().st_size,
            )
            return True
        # resize succeeded but webp failed — keep resized version
        tmp_path.rename(glb_path)
        return True
    finally:
        tmp_path.unlink(missing_ok=True)


# ── PipelineResult ─────────────────────────────────────────


@dataclass
class PipelineResult:
    """Result of the full character asset generation pipeline."""

    fullbody_path: Path | None = None
    bustup_path: Path | None = None
    bustup_paths: dict[str, Path] = field(default_factory=dict)
    chibi_path: Path | None = None
    model_path: Path | None = None
    rigged_model_path: Path | None = None
    animation_paths: dict[str, Path] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fullbody": str(self.fullbody_path) if self.fullbody_path else None,
            "bustup": str(self.bustup_path) if self.bustup_path else None,
            "bustup_expressions": {k: str(v) for k, v in self.bustup_paths.items()},
            "chibi": str(self.chibi_path) if self.chibi_path else None,
            "model": str(self.model_path) if self.model_path else None,
            "rigged_model": str(self.rigged_model_path) if self.rigged_model_path else None,
            "animations": {k: str(v) for k, v in self.animation_paths.items()},
            "errors": self.errors,
            "skipped": self.skipped,
        }


# ── ImageGenPipeline ───────────────────────────────────────


class ImageGenPipeline:
    """Orchestrates the full character asset generation pipeline.

    Steps:
      1. NovelAI V4.5 → full-body anime image
      2. Flux Kontext  → bust-up from reference  ─┐ independent
      3. Flux Kontext  → chibi from reference     ─┘
      4. Meshy Image-to-3D → GLB from chibi
      5. Meshy Rigging → rigged GLB + walking/running animations
      6. Meshy Animations → idle/sitting/waving/talking GLBs
    """

    ASSET_NAMES = {
        "fullbody": "avatar_fullbody.png",
        "bustup": "avatar_bustup.png",
        "chibi": "avatar_chibi.png",
        "model": "avatar_chibi.glb",
        "rigged_model": "avatar_chibi_rigged.glb",
    }
    # Animation files use pattern: anim_{name}.glb

    def __init__(
        self,
        anima_dir: Path,
        config: "ImageGenConfig | None" = None,
    ) -> None:
        from core.config.models import ImageGenConfig

        self._anima_dir = anima_dir
        self._assets_dir = anima_dir / "assets"
        self._config = config or ImageGenConfig()

    def generate_bustup_expression(
        self,
        reference_image: bytes,
        expression: str,
        skip_existing: bool = True,
    ) -> Path | None:
        """Generate a single expression variant of the bustup image.

        Args:
            reference_image: Full-body reference image bytes.
            expression: Expression name (e.g. "smile", "troubled").
            skip_existing: Skip if output file already exists.

        Returns:
            Path to generated image, or None on failure.
        """
        if expression not in _VALID_EXPRESSION_NAMES:
            logger.warning("Unknown expression: %s", expression)
            return None

        output_filename = (
            "avatar_bustup.png" if expression == "neutral"
            else f"avatar_bustup_{expression}.png"
        )
        output_path = self._assets_dir / output_filename

        if skip_existing and output_path.exists():
            logger.info("Skipping existing: %s", output_path)
            return output_path

        prompt = _EXPRESSION_PROMPTS[expression]

        # Apply style prefix/suffix
        if self._config.style_prefix:
            prompt = self._config.style_prefix + prompt
        if self._config.style_suffix:
            prompt = prompt + self._config.style_suffix

        self._assets_dir.mkdir(parents=True, exist_ok=True)

        kontext = FluxKontextClient()
        guidance = _EXPRESSION_GUIDANCE.get(expression, 5.0)
        result_bytes = kontext.generate_from_reference(
            reference_image=reference_image,
            prompt=prompt,
            aspect_ratio="3:4",
            guidance_scale=guidance,
        )

        output_path.write_bytes(result_bytes)
        logger.info("Generated expression '%s' (guidance=%.1f): %s", expression, guidance, output_path)
        return output_path

    def generate_all(
        self,
        prompt: str,
        negative_prompt: str = "",
        skip_existing: bool = True,
        steps: list[str] | None = None,
        animations: dict[str, int] | None = None,
        expressions: list[str] | None = None,
        vibe_image: bytes | None = None,
        vibe_strength: float | None = None,
        vibe_info_extracted: float | None = None,
        seed: int | None = None,
        progress_callback: "Callable[[str, str, int], None] | None" = None,
    ) -> PipelineResult:
        """Run the 6-step pipeline synchronously.

        Args:
            prompt: Character appearance tags for NovelAI.
            negative_prompt: Negative prompt for NovelAI.
            skip_existing: Skip steps whose output file already exists.
            steps: Subset of steps to run (default: all).
            animations: Dict of {name: action_id} to generate.
                Default: idle, sitting, waving, talking.
            expressions: Subset of expressions to generate.
            vibe_image: Override Vibe Transfer reference image bytes.
                When provided, takes precedence over config.style_reference.
            vibe_strength: Override Vibe Transfer strength (0.0-1.0).
                Falls back to config.vibe_strength.
            vibe_info_extracted: Override Vibe Transfer info extraction (0.0-1.0).
                Falls back to config.vibe_info_extracted.
            seed: Seed for reproducibility (fullbody generation only).
            progress_callback: Optional callback ``(step, status, pct)`` for
                real-time progress reporting.

        Returns:
            PipelineResult with paths and error info.
        """
        self._assets_dir.mkdir(parents=True, exist_ok=True)
        enabled = set(steps) if steps else {
            "fullbody", "bustup", "chibi", "3d", "rigging", "animations",
        }
        anim_map = animations if animations is not None else _DEFAULT_ANIMATIONS
        result = PipelineResult()

        def _notify(step: str, status: str, pct: int = 0) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(step, status, pct)
                except Exception:
                    logger.debug("progress_callback error (ignored)", exc_info=True)

        # ── Step 1: Full-body ──
        fullbody_bytes: bytes | None = None
        fullbody_path = self._assets_dir / self.ASSET_NAMES["fullbody"]

        if "fullbody" in enabled:
            if skip_existing and fullbody_path.exists():
                result.skipped.append("fullbody")
                fullbody_bytes = fullbody_path.read_bytes()
                result.fullbody_path = fullbody_path
            else:
                try:
                    _notify("fullbody", "generating", 0)
                    # Select text-to-image backend: NovelAI (primary) or Fal (fallback)
                    if os.environ.get("NOVELAI_TOKEN"):
                        logger.info("Step 1: Generating full-body with NovelAI …")
                        client: NovelAIClient | FalTextToImageClient = NovelAIClient()
                    elif os.environ.get("FAL_KEY"):
                        logger.info(
                            "Step 1: Generating full-body with Fal Flux Pro (fallback) …",
                        )
                        client = FalTextToImageClient()
                    else:
                        raise RuntimeError(
                            "No image generation API key configured. "
                            "Set NOVELAI_TOKEN or FAL_KEY."
                        )

                    # ── A: Load style reference for Vibe Transfer ──
                    # Direct vibe_image parameter takes precedence over config
                    effective_vibe: bytes | None = vibe_image
                    if effective_vibe is None and self._config.style_reference:
                        style_path = Path(self._config.style_reference).expanduser()
                        if style_path.exists():
                            effective_vibe = style_path.read_bytes()
                            logger.debug("Loaded style reference: %s", style_path)
                        else:
                            logger.warning(
                                "Style reference not found: %s", style_path
                            )

                    effective_vibe_strength = (
                        vibe_strength
                        if vibe_strength is not None
                        else self._config.vibe_strength
                    )
                    effective_vibe_info = (
                        vibe_info_extracted
                        if vibe_info_extracted is not None
                        else self._config.vibe_info_extracted
                    )

                    # ── B: Apply style prefix/suffix to prompt ──
                    styled_prompt = prompt
                    if self._config.style_prefix:
                        styled_prompt = self._config.style_prefix + styled_prompt
                    if self._config.style_suffix:
                        styled_prompt = styled_prompt + self._config.style_suffix

                    styled_negative = negative_prompt
                    if self._config.negative_prompt_extra:
                        if styled_negative:
                            styled_negative += ", " + self._config.negative_prompt_extra
                        else:
                            styled_negative = self._config.negative_prompt_extra

                    fullbody_bytes = client.generate_fullbody(
                        prompt=styled_prompt,
                        negative_prompt=styled_negative,
                        seed=seed,
                        vibe_image=effective_vibe,
                        vibe_strength=effective_vibe_strength,
                        vibe_info_extracted=effective_vibe_info,
                    )
                    fullbody_path.write_bytes(fullbody_bytes)
                    result.fullbody_path = fullbody_path
                    logger.info("Step 1 complete: %s", fullbody_path)
                    _notify("fullbody", "completed", 100)
                except Exception as exc:
                    result.errors.append(f"fullbody: {exc}")
                    logger.error("Step 1 failed: %s", exc)
                    _notify("fullbody", "error", 0)
        elif fullbody_path.exists():
            fullbody_bytes = fullbody_path.read_bytes()
            result.fullbody_path = fullbody_path

        if fullbody_bytes is None:
            # Cannot proceed without a reference image
            if not result.errors:
                result.errors.append(
                    "fullbody: No full-body image available as reference"
                )
            return result

        # ── Step 2 & 3: Bust-up and Chibi (sequential, same client) ──
        chibi_bytes: bytes | None = None

        if "bustup" in enabled:
            _notify("bustup", "generating", 0)
            expr_list = expressions or list(_EXPRESSION_PROMPTS.keys())
            logger.info("Step 2: Generating bustup expressions: %s", expr_list)

            # Step 2a: Generate neutral bustup first (from fullbody reference).
            # This produces a close-up where the face occupies ~40-50% of the
            # image, making it a much better reference for expression changes
            # than the original full-body image (~10-15% face area).
            bustup_ref_bytes: bytes | None = None
            neutral_path = self._assets_dir / "avatar_bustup.png"

            if "neutral" in expr_list:
                try:
                    path = self.generate_bustup_expression(
                        reference_image=fullbody_bytes,
                        expression="neutral",
                        skip_existing=skip_existing,
                    )
                    if path and path.exists():
                        bustup_ref_bytes = path.read_bytes()
                        result.bustup_paths["neutral"] = path
                        result.bustup_path = path
                except Exception as exc:
                    result.errors.append(f"bustup_neutral: {exc}")
                    logger.error("Bustup neutral failed: %s", exc)
            elif neutral_path.exists():
                # neutral not requested but exists on disk — use as reference
                bustup_ref_bytes = neutral_path.read_bytes()

            # Step 2b: Generate other expressions using bustup as reference.
            # Fall back to fullbody if bustup reference is unavailable.
            ref_for_expressions = bustup_ref_bytes or fullbody_bytes
            if bustup_ref_bytes:
                logger.info("Using neutral bustup as reference for expression variants")
            else:
                logger.warning("Neutral bustup unavailable, falling back to fullbody reference")

            for expr in expr_list:
                if expr == "neutral":
                    continue  # already handled above
                try:
                    path = self.generate_bustup_expression(
                        reference_image=ref_for_expressions,
                        expression=expr,
                        skip_existing=skip_existing,
                    )
                    if path:
                        result.bustup_paths[expr] = path
                except Exception as exc:
                    result.errors.append(f"bustup_{expr}: {exc}")
                    logger.error("Bustup expression '%s' failed: %s", expr, exc)

            if not result.bustup_path and result.bustup_paths:
                result.bustup_path = next(iter(result.bustup_paths.values()))
            logger.info("Step 2 complete: %d expressions generated", len(result.bustup_paths))
            _notify("bustup", "completed", 100)

        if "chibi" in enabled:
            chibi_path = self._assets_dir / self.ASSET_NAMES["chibi"]
            if skip_existing and chibi_path.exists():
                result.skipped.append("chibi")
                chibi_bytes = chibi_path.read_bytes()
                result.chibi_path = chibi_path
            else:
                try:
                    _notify("chibi", "generating", 0)
                    logger.info("Step 3: Generating chibi with Flux Kontext …")
                    kontext = FluxKontextClient()
                    chibi_bytes = kontext.generate_from_reference(
                        reference_image=fullbody_bytes,
                        prompt=_CHIBI_PROMPT,
                        aspect_ratio="1:1",
                    )
                    chibi_path.write_bytes(chibi_bytes)
                    result.chibi_path = chibi_path
                    logger.info("Step 3 complete: %s", chibi_path)
                    _notify("chibi", "completed", 100)
                except Exception as exc:
                    result.errors.append(f"chibi: {exc}")
                    logger.error("Step 3 failed: %s", exc)
                    _notify("chibi", "error", 0)

        # ── Step 4: 3-D model from chibi ──
        meshy_task_id: str | None = None  # tracked for rigging input
        meshy: MeshyClient | None = None

        if "3d" in enabled:
            if chibi_bytes is None:
                chibi_path = self._assets_dir / self.ASSET_NAMES["chibi"]
                if chibi_path.exists():
                    chibi_bytes = chibi_path.read_bytes()

            model_path = self._assets_dir / self.ASSET_NAMES["model"]
            if skip_existing and model_path.exists():
                result.skipped.append("3d")
                result.model_path = model_path
            elif chibi_bytes is None:
                result.errors.append("3d: No chibi image available for 3D conversion")
            else:
                try:
                    _notify("3d", "generating", 0)
                    logger.info("Step 4: Generating 3D model with Meshy …")
                    meshy = MeshyClient()
                    meshy_task_id = meshy.create_task(chibi_bytes)
                    logger.info("Meshy task created: %s", meshy_task_id)
                    task = meshy.poll_task(meshy_task_id)
                    glb_bytes = meshy.download_model(task, fmt="glb")
                    model_path.write_bytes(glb_bytes)
                    result.model_path = model_path
                    logger.info("Step 4 complete: %s", model_path)
                    _notify("3d", "completed", 100)
                except Exception as exc:
                    result.errors.append(f"3d: {exc}")
                    logger.error("Step 4 failed: %s", exc)
                    _notify("3d", "error", 0)

        # ── Step 5: Rigging ──
        rig_task_id: str | None = None

        if "rigging" in enabled:
            rigged_path = self._assets_dir / self.ASSET_NAMES["rigged_model"]
            if skip_existing and rigged_path.exists():
                result.skipped.append("rigging")
                result.rigged_model_path = rigged_path
            elif meshy_task_id is None:
                result.errors.append(
                    "rigging: No Meshy task_id available "
                    "(run 3d step first or provide model)"
                )
            else:
                try:
                    _notify("rigging", "generating", 0)
                    logger.info("Step 5: Rigging 3D model with Meshy …")
                    if meshy is None:
                        meshy = MeshyClient()
                    rig_task_id = meshy.create_rigging_task(meshy_task_id)
                    logger.info("Meshy rigging task created: %s", rig_task_id)
                    rig_task = meshy.poll_rigging_task(rig_task_id)

                    # Download rigged model
                    rigged_bytes = meshy.download_rigged_model(rig_task, fmt="glb")
                    rigged_path.write_bytes(rigged_bytes)
                    result.rigged_model_path = rigged_path
                    optimize_glb(rigged_path)
                    logger.info("Rigged model saved: %s", rigged_path)

                    # Download built-in walking/running animations
                    basic_anims = meshy.download_rigging_animations(rig_task)
                    for anim_name, anim_bytes in basic_anims.items():
                        anim_path = self._assets_dir / f"anim_{anim_name}.glb"
                        anim_path.write_bytes(anim_bytes)
                        if not strip_mesh_from_glb(anim_path):
                            result.errors.append(f"mesh_strip: failed for {anim_path.name}")
                            logger.warning("Mesh strip failed for %s, animation saved with embedded mesh", anim_path.name)
                        result.animation_paths[anim_name] = anim_path
                        logger.info(
                            "Animation '%s' saved: %s (%d bytes)",
                            anim_name, anim_path, len(anim_bytes),
                        )
                    logger.info("Step 5 complete: rigged + %d animations", len(basic_anims))
                    _notify("rigging", "completed", 100)
                except Exception as exc:
                    result.errors.append(f"rigging: {exc}")
                    logger.error("Step 5 failed: %s", exc)
                    _notify("rigging", "error", 0)

        # ── Step 6: Additional animations ──
        if "animations" in enabled and anim_map:
            if rig_task_id is None:
                result.errors.append(
                    "animations: No rigging task_id available "
                    "(run rigging step first)"
                )
            else:
                _notify("animations", "generating", 0)
                if meshy is None:
                    meshy = MeshyClient()
                for anim_name, action_id in anim_map.items():
                    anim_path = self._assets_dir / f"anim_{anim_name}.glb"
                    if skip_existing and anim_path.exists():
                        result.skipped.append(f"anim_{anim_name}")
                        result.animation_paths[anim_name] = anim_path
                        continue
                    try:
                        logger.info(
                            "Step 6: Generating '%s' animation (action_id=%d) …",
                            anim_name, action_id,
                        )
                        anim_task_id = meshy.create_animation_task(
                            rig_task_id, action_id,
                        )
                        logger.info(
                            "Animation task '%s' created: %s",
                            anim_name, anim_task_id,
                        )
                        anim_task = meshy.poll_animation_task(anim_task_id)
                        if not _download_armature_animation(anim_task, anim_path):
                            raise RuntimeError(
                                f"All download methods failed for {anim_path.name}"
                            )
                        result.animation_paths[anim_name] = anim_path
                        logger.info(
                            "Animation '%s' saved: %s (%d bytes)",
                            anim_name, anim_path, anim_path.stat().st_size,
                        )
                    except Exception as exc:
                        result.errors.append(f"anim_{anim_name}: {exc}")
                        logger.error(
                            "Animation '%s' failed: %s", anim_name, exc,
                        )
                _notify("animations", "completed", 100)

        return result


# ── Tool Schemas ───────────────────────────────────────────


def get_tool_schemas() -> list[dict]:
    """Return Anthropic tool_use schemas for image generation tools."""
    return [
        {
            "name": "generate_character_assets",
            "description": (
                "Generate a complete set of character avatar assets: "
                "full-body image, bust-up image, chibi image, 3D model, "
                "rigged model with skeleton, and animations "
                "(idle, sitting, waving, talking, walking, running). "
                "Requires NOVELAI_TOKEN, FAL_KEY, and MESHY_API_KEY."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Character appearance description using anime tags. "
                            "Example: '1girl, black hair, long hair, red eyes, "
                            "school uniform, full body, standing, white background'"
                        ),
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "Things to avoid in the generated image.",
                        "default": "",
                    },
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "fullbody", "bustup", "chibi",
                                "3d", "rigging", "animations",
                            ],
                        },
                        "description": (
                            "Which pipeline steps to run. "
                            "Default: all six steps."
                        ),
                    },
                    "supervisor_name": {
                        "type": "string",
                        "description": (
                            "Supervisor anima name. When specified, the "
                            "supervisor's avatar_fullbody.png is used as "
                            "the Vibe Transfer reference image for style "
                            "consistency."
                        ),
                    },
                    "skip_existing": {
                        "type": "boolean",
                        "description": (
                            "If true, skip steps whose output file already exists."
                        ),
                        "default": True,
                    },
                    "expressions": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "neutral", "smile", "laugh",
                                "troubled", "surprised", "thinking",
                                "embarrassed",
                            ],
                        },
                        "description": (
                            "Which bustup expressions to generate. "
                            "Default: neutral, smile, laugh, troubled, surprised."
                        ),
                    },
                },
                "required": ["prompt"],
            },
        },
        {
            "name": "generate_fullbody",
            "description": (
                "Generate an anime full-body character image using NovelAI V4.5. "
                "Saves to assets/avatar_fullbody.png. "
                "Requires NOVELAI_TOKEN."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Character appearance tags for NovelAI.",
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "Negative prompt.",
                        "default": "",
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width in pixels.",
                        "default": 1024,
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height in pixels.",
                        "default": 1536,
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Seed for reproducibility.",
                    },
                },
                "required": ["prompt"],
            },
        },
        {
            "name": "generate_bustup",
            "description": (
                "Generate a bust-up portrait from a reference image "
                "using Flux Kontext [pro]. Saves to assets/avatar_bustup.png. "
                "Requires FAL_KEY and an existing full-body image."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Bust-up generation prompt. "
                            "A default prompt is used if omitted."
                        ),
                    },
                },
                "required": [],
            },
        },
        {
            "name": "generate_chibi",
            "description": (
                "Generate a chibi / super-deformed version from a reference "
                "image using Flux Kontext [pro]. Saves to assets/avatar_chibi.png. "
                "Requires FAL_KEY and an existing full-body image."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Chibi generation prompt. "
                            "A default prompt is used if omitted."
                        ),
                    },
                },
                "required": [],
            },
        },
        {
            "name": "generate_3d_model",
            "description": (
                "Generate a 3D model (GLB) from a chibi image using "
                "Meshy Image-to-3D. Saves to assets/avatar_chibi.glb. "
                "Requires MESHY_API_KEY and an existing chibi image."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "ai_model": {
                        "type": "string",
                        "description": "Meshy model version.",
                        "default": "meshy-6",
                        "enum": ["meshy-5", "meshy-6"],
                    },
                    "target_polycount": {
                        "type": "integer",
                        "description": "Target polygon count.",
                        "default": 30000,
                    },
                },
                "required": [],
            },
        },
        {
            "name": "generate_rigged_model",
            "description": (
                "Rig a 3D model with a humanoid skeleton using Meshy. "
                "Also downloads built-in walking/running animations. "
                "Saves rigged model to assets/avatar_chibi_rigged.glb "
                "and animations to assets/anim_walking.glb, anim_running.glb. "
                "Requires MESHY_API_KEY and an existing 3D model."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "height_meters": {
                        "type": "number",
                        "description": (
                            "Approximate character height in meters. "
                            "Aids in scaling and rigging accuracy."
                        ),
                        "default": 1.0,
                    },
                },
                "required": [],
            },
        },
        {
            "name": "generate_animations",
            "description": (
                "Generate animation GLBs for a rigged character using "
                "Meshy's animation library. Default: idle, sitting, waving, "
                "talking. Requires MESHY_API_KEY and a completed rigging step."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "animations": {
                        "type": "object",
                        "description": (
                            "Dict of {name: action_id}. "
                            "Default: {idle: 0, sitting: 32, waving: 28, talking: 307}. "
                            "See Meshy animation library for all available action_ids."
                        ),
                        "additionalProperties": {"type": "integer"},
                    },
                },
                "required": [],
            },
        },
    ]


# ── CLI entry point ────────────────────────────────────────


def get_cli_guide() -> str:
    """Return CLI usage guide for image generation tools."""
    return """\
### 画像・3Dモデル生成 (image_gen)

A2モードのツール名: `generate_character_assets` / `generate_fullbody` / `generate_bustup` 等

```bash
# 全6ステップ一括生成（推奨）
animaworks-tool image_gen pipeline "1girl, black hair, ..." --negative "lowres, bad anatomy, ..." --anima-dir <anima_dir> -j

# 個別ステップ
animaworks-tool image_gen fullbody "prompt" --anima-dir <anima_dir> -j
animaworks-tool image_gen bustup --anima-dir <anima_dir> -j
animaworks-tool image_gen chibi --anima-dir <anima_dir> -j
animaworks-tool image_gen 3d --anima-dir <anima_dir> -j
animaworks-tool image_gen rigging <model.glb> -o <output_dir> -j
animaworks-tool image_gen animations <model.glb> -o <output_dir> -j
```"""


def cli_main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``animaworks-tool image_gen``."""
    parser = argparse.ArgumentParser(
        description="Character image & 3D model generation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- pipeline --
    p_pipe = sub.add_parser("pipeline", help="Run full 6-step pipeline")
    p_pipe.add_argument("prompt", help="Character appearance tags")
    p_pipe.add_argument("-n", "--negative", default="", help="Negative prompt")
    p_pipe.add_argument(
        "-d", "--anima-dir", required=True, help="Anima data directory",
    )
    p_pipe.add_argument(
        "--steps", nargs="*",
        choices=["fullbody", "bustup", "chibi", "3d", "rigging", "animations"],
        help="Steps to run (default: all)",
    )
    p_pipe.add_argument(
        "--no-skip", action="store_true", help="Regenerate even if files exist",
    )
    p_pipe.add_argument("-j", "--json", action="store_true", help="JSON output")

    # -- fullbody --
    p_full = sub.add_parser("fullbody", help="Generate full-body image only")
    p_full.add_argument("prompt", help="Character appearance tags")
    p_full.add_argument("-n", "--negative", default="", help="Negative prompt")
    p_full.add_argument("-o", "--output", default="avatar_fullbody.png")
    p_full.add_argument("-W", "--width", type=int, default=1024)
    p_full.add_argument("-H", "--height", type=int, default=1536)
    p_full.add_argument("-s", "--seed", type=int, default=None)
    p_full.add_argument("-j", "--json", action="store_true")

    # -- bustup --
    p_bust = sub.add_parser("bustup", help="Generate bust-up from reference")
    p_bust.add_argument("reference", help="Path to reference image")
    p_bust.add_argument("-p", "--prompt", default=_BUSTUP_PROMPT)
    p_bust.add_argument("-o", "--output", default="avatar_bustup.png")
    p_bust.add_argument("-j", "--json", action="store_true")

    # -- chibi --
    p_chibi = sub.add_parser("chibi", help="Generate chibi from reference")
    p_chibi.add_argument("reference", help="Path to reference image")
    p_chibi.add_argument("-p", "--prompt", default=_CHIBI_PROMPT)
    p_chibi.add_argument("-o", "--output", default="avatar_chibi.png")
    p_chibi.add_argument("-j", "--json", action="store_true")

    # -- 3d --
    p_3d = sub.add_parser("3d", help="Generate 3D model from image")
    p_3d.add_argument("image", help="Path to input image")
    p_3d.add_argument("-o", "--output", default="avatar_chibi.glb")
    p_3d.add_argument("--ai-model", default="meshy-6")
    p_3d.add_argument("--polycount", type=int, default=30000)
    p_3d.add_argument("-j", "--json", action="store_true")

    # -- rigging --
    p_rig = sub.add_parser("rigging", help="Rig a 3D model with skeleton")
    p_rig.add_argument("model", help="Path to GLB model file")
    p_rig.add_argument("-o", "--output-dir", default=".", help="Output directory")
    p_rig.add_argument(
        "--height", type=float, default=1.0,
        help="Character height in meters (default: 1.0 for chibi)",
    )
    p_rig.add_argument("-j", "--json", action="store_true")

    # -- animations --
    p_anim = sub.add_parser("animations", help="Generate animations for rigged model")
    p_anim.add_argument("model", help="Path to GLB model file")
    p_anim.add_argument("-o", "--output-dir", default=".", help="Output directory")
    p_anim.add_argument(
        "--height", type=float, default=1.0,
        help="Character height in meters (default: 1.0 for chibi)",
    )
    p_anim.add_argument(
        "--actions", nargs="*",
        help="Animation names to generate (default: idle sitting waving talking)",
    )
    p_anim.add_argument("-j", "--json", action="store_true")

    args = parser.parse_args(argv)

    # ── Execute ────────────────────────────────────────────

    if args.command == "pipeline":
        pipe = ImageGenPipeline(Path(args.anima_dir))
        result = pipe.generate_all(
            prompt=args.prompt,
            negative_prompt=args.negative,
            skip_existing=not args.no_skip,
            steps=args.steps,
        )
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            for k, v in result.to_dict().items():
                print(f"{k}: {v}")

    elif args.command == "fullbody":
        client = NovelAIClient()
        img = client.generate_fullbody(
            prompt=args.prompt,
            negative_prompt=args.negative,
            width=args.width,
            height=args.height,
            seed=args.seed,
        )
        Path(args.output).write_bytes(img)
        out = {"output": args.output, "size": len(img)}
        if args.json:
            print(json.dumps(out))
        else:
            print(f"Saved: {args.output} ({len(img)} bytes)")

    elif args.command == "bustup":
        ref_bytes = Path(args.reference).read_bytes()
        client = FluxKontextClient()
        img = client.generate_from_reference(
            reference_image=ref_bytes,
            prompt=args.prompt,
            aspect_ratio="3:4",
        )
        Path(args.output).write_bytes(img)
        out = {"output": args.output, "size": len(img)}
        if args.json:
            print(json.dumps(out))
        else:
            print(f"Saved: {args.output} ({len(img)} bytes)")

    elif args.command == "chibi":
        ref_bytes = Path(args.reference).read_bytes()
        client = FluxKontextClient()
        img = client.generate_from_reference(
            reference_image=ref_bytes,
            prompt=args.prompt,
            aspect_ratio="1:1",
        )
        Path(args.output).write_bytes(img)
        out = {"output": args.output, "size": len(img)}
        if args.json:
            print(json.dumps(out))
        else:
            print(f"Saved: {args.output} ({len(img)} bytes)")

    elif args.command == "3d":
        img_bytes = Path(args.image).read_bytes()
        client = MeshyClient()
        task_id = client.create_task(
            img_bytes, ai_model=args.ai_model, target_polycount=args.polycount,
        )
        print(f"Meshy task: {task_id}", file=sys.stderr)
        task = client.poll_task(task_id)
        glb = client.download_model(task, fmt="glb")
        Path(args.output).write_bytes(glb)
        out = {"output": args.output, "size": len(glb), "task_id": task_id}
        if args.json:
            print(json.dumps(out))
        else:
            print(f"Saved: {args.output} ({len(glb)} bytes)")

    elif args.command == "rigging":
        model_bytes = Path(args.model).read_bytes()
        data_uri = _image_to_data_uri(model_bytes, mime="model/gltf-binary")
        client = MeshyClient()

        # Submit rigging with model_url (data URI)
        body: dict[str, Any] = {
            "model_url": data_uri,
            "height_meters": args.height,
        }

        def _rig_call() -> str:
            resp = httpx.post(
                MESHY_RIGGING_URL,
                json=body,
                headers=client._headers(),
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["result"]

        rig_task_id = _retry(_rig_call, max_retries=1, delay=10.0)
        print(f"Rigging task: {rig_task_id}", file=sys.stderr)
        rig_task = client.poll_rigging_task(rig_task_id)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Any] = {"rig_task_id": rig_task_id}

        # Download rigged model
        rigged = client.download_rigged_model(rig_task, fmt="glb")
        rigged_path = out_dir / "avatar_chibi_rigged.glb"
        rigged_path.write_bytes(rigged)
        outputs["rigged_model"] = str(rigged_path)

        # Download built-in animations
        basic_anims = client.download_rigging_animations(rig_task)
        outputs["animations"] = {}
        for name, anim_bytes in basic_anims.items():
            anim_path = out_dir / f"anim_{name}.glb"
            anim_path.write_bytes(anim_bytes)
            outputs["animations"][name] = str(anim_path)

        if args.json:
            print(json.dumps(outputs, ensure_ascii=False, indent=2))
        else:
            print(f"Rigged model: {outputs['rigged_model']}")
            for name, path in outputs.get("animations", {}).items():
                print(f"Animation '{name}': {path}")

    elif args.command == "animations":
        model_bytes = Path(args.model).read_bytes()
        data_uri = _image_to_data_uri(model_bytes, mime="model/gltf-binary")
        client = MeshyClient()

        # First rig the model
        rig_body: dict[str, Any] = {
            "model_url": data_uri,
            "height_meters": args.height,
        }

        def _rig_call2() -> str:
            resp = httpx.post(
                MESHY_RIGGING_URL,
                json=rig_body,
                headers=client._headers(),
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["result"]

        rig_task_id = _retry(_rig_call2, max_retries=1, delay=10.0)
        print(f"Rigging task: {rig_task_id}", file=sys.stderr)
        client.poll_rigging_task(rig_task_id)

        # Determine animations to generate
        anim_map = _DEFAULT_ANIMATIONS.copy()
        if args.actions:
            anim_map = {
                name: _DEFAULT_ANIMATIONS.get(name, 0)
                for name in args.actions
            }

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs_anim: dict[str, Any] = {"rig_task_id": rig_task_id, "animations": {}}

        for name, action_id in anim_map.items():
            print(f"Generating '{name}' (action_id={action_id}) …", file=sys.stderr)
            anim_task_id = client.create_animation_task(rig_task_id, action_id)
            anim_task = client.poll_animation_task(anim_task_id)
            anim_bytes = client.download_animation(anim_task, fmt="glb")
            anim_path = out_dir / f"anim_{name}.glb"
            anim_path.write_bytes(anim_bytes)
            outputs_anim["animations"][name] = str(anim_path)

        if args.json:
            print(json.dumps(outputs_anim, ensure_ascii=False, indent=2))
        else:
            for name, path in outputs_anim["animations"].items():
                print(f"Animation '{name}': {path}")


# ── Dispatch ──────────────────────────────────────────


def dispatch(tool_name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool call to the appropriate handler."""
    if tool_name == "generate_character_assets":
        from core.config.models import load_config
        from core.paths import get_animas_dir

        anima_dir = Path(args.pop("anima_dir", ""))
        supervisor_name: str | None = args.pop("supervisor_name", None)
        config = load_config()
        image_config = config.image_gen

        # Use supervisor's fullbody image as Vibe Transfer reference
        if supervisor_name:
            supervisor_fullbody = (
                get_animas_dir() / supervisor_name / "assets" / "avatar_fullbody.png"
            )
            if supervisor_fullbody.exists():
                image_config = image_config.model_copy(
                    update={"style_reference": str(supervisor_fullbody)},
                )
                logger.info(
                    "Using supervisor image as vibe reference: %s",
                    supervisor_fullbody,
                )

        pipeline = ImageGenPipeline(anima_dir, config=image_config)
        result = pipeline.generate_all(
            prompt=args["prompt"],
            negative_prompt=args.get("negative_prompt", ""),
            skip_existing=args.get("skip_existing", True),
            steps=args.get("steps"),
            animations=args.get("animations"),
        )
        return result.to_dict()

    if tool_name == "generate_fullbody":
        anima_dir = Path(args.pop("anima_dir", ""))
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        client = NovelAIClient()
        img = client.generate_fullbody(
            prompt=args["prompt"],
            negative_prompt=args.get("negative_prompt", ""),
            width=args.get("width", 1024),
            height=args.get("height", 1536),
            seed=args.get("seed"),
        )
        out = assets_dir / "avatar_fullbody.png"
        out.write_bytes(img)
        return {"path": str(out), "size": len(img)}

    if tool_name == "generate_bustup":
        anima_dir = Path(args.pop("anima_dir", ""))
        assets_dir = anima_dir / "assets"
        ref_path = assets_dir / "avatar_fullbody.png"
        if not ref_path.exists():
            return {"error": "No full-body reference image found"}
        client = FluxKontextClient()
        img = client.generate_from_reference(
            reference_image=ref_path.read_bytes(),
            prompt=args.get("prompt", _BUSTUP_PROMPT),
            aspect_ratio="3:4",
        )
        out = assets_dir / "avatar_bustup.png"
        out.write_bytes(img)
        return {"path": str(out), "size": len(img)}

    if tool_name == "generate_chibi":
        anima_dir = Path(args.pop("anima_dir", ""))
        assets_dir = anima_dir / "assets"
        ref_path = assets_dir / "avatar_fullbody.png"
        if not ref_path.exists():
            return {"error": "No full-body reference image found"}
        client = FluxKontextClient()
        img = client.generate_from_reference(
            reference_image=ref_path.read_bytes(),
            prompt=args.get("prompt", _CHIBI_PROMPT),
            aspect_ratio="1:1",
        )
        out = assets_dir / "avatar_chibi.png"
        out.write_bytes(img)
        return {"path": str(out), "size": len(img)}

    if tool_name == "generate_3d_model":
        anima_dir = Path(args.pop("anima_dir", ""))
        assets_dir = anima_dir / "assets"
        chibi_path = assets_dir / "avatar_chibi.png"
        if not chibi_path.exists():
            return {"error": "No chibi image found for 3D conversion"}
        client = MeshyClient()
        task_id = client.create_task(
            chibi_path.read_bytes(),
            ai_model=args.get("ai_model", "meshy-6"),
            target_polycount=args.get("target_polycount", 30000),
        )
        task = client.poll_task(task_id)
        glb = client.download_model(task, fmt="glb")
        out = assets_dir / "avatar_chibi.glb"
        out.write_bytes(glb)
        return {"path": str(out), "size": len(glb), "task_id": task_id}

    if tool_name == "generate_rigged_model":
        import httpx as _httpx

        anima_dir = Path(args.pop("anima_dir", ""))
        assets_dir = anima_dir / "assets"
        glb_path = assets_dir / "avatar_chibi.glb"
        if not glb_path.exists():
            return {"error": "No 3D model found for rigging"}
        client = MeshyClient()
        data_uri = _image_to_data_uri(
            glb_path.read_bytes(), mime="model/gltf-binary",
        )
        body = {
            "model_url": data_uri,
            "height_meters": args.get("height_meters", 1.0),
        }
        resp = _httpx.post(
            MESHY_RIGGING_URL,
            json=body,
            headers=client._headers(),
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        rig_task_id = resp.json()["result"]
        rig_task = client.poll_rigging_task(rig_task_id)
        rigged = client.download_rigged_model(rig_task, fmt="glb")
        rigged_path = assets_dir / "avatar_chibi_rigged.glb"
        rigged_path.write_bytes(rigged)
        basic_anims = client.download_rigging_animations(rig_task)
        anim_results: dict[str, str] = {}
        for anim_name, anim_bytes in basic_anims.items():
            anim_path = assets_dir / f"anim_{anim_name}.glb"
            anim_path.write_bytes(anim_bytes)
            anim_results[anim_name] = str(anim_path)
        return {
            "rigged_model": str(rigged_path),
            "animations": anim_results,
            "rig_task_id": rig_task_id,
        }

    if tool_name == "generate_animations":
        import httpx as _httpx

        anima_dir = Path(args.pop("anima_dir", ""))
        assets_dir = anima_dir / "assets"
        glb_path = assets_dir / "avatar_chibi.glb"
        if not glb_path.exists():
            return {"error": "No 3D model found for animation"}
        client = MeshyClient()
        data_uri = _image_to_data_uri(
            glb_path.read_bytes(), mime="model/gltf-binary",
        )
        body = {"model_url": data_uri, "height_meters": 1.0}
        resp = _httpx.post(
            MESHY_RIGGING_URL,
            json=body,
            headers=client._headers(),
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        rig_task_id = resp.json()["result"]
        client.poll_rigging_task(rig_task_id)
        anim_map = args.get("animations") or _DEFAULT_ANIMATIONS
        anim_results_gen: dict[str, str] = {}
        for anim_name, action_id in anim_map.items():
            anim_task_id = client.create_animation_task(rig_task_id, action_id)
            anim_task = client.poll_animation_task(anim_task_id)
            anim_bytes = client.download_animation(anim_task, fmt="glb")
            anim_path = assets_dir / f"anim_{anim_name}.glb"
            anim_path.write_bytes(anim_bytes)
            anim_results_gen[anim_name] = str(anim_path)
        return {"animations": anim_results_gen, "rig_task_id": rig_task_id}

    raise ValueError(f"Unknown tool: {tool_name}")
