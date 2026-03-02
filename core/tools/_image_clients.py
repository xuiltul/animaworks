# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""API clients and shared constants for image/3D generation."""
from __future__ import annotations

import base64
import io
import time
import zipfile
from typing import Any

import httpx

from core.tools._base import ToolConfigError, get_credential, logger
from core.tools._retry import retry_with_backoff

__all__ = [
    "EXECUTION_PROFILE",
    "NOVELAI_API_URL",
    "NOVELAI_ENCODE_URL",
    "NOVELAI_MODEL",
    "FAL_KONTEXT_SUBMIT_URL",
    "FAL_FLUX_PRO_SUBMIT_URL",
    "MESHY_IMAGE_TO_3D_URL",
    "MESHY_TASK_URL_TPL",
    "MESHY_RIGGING_URL",
    "MESHY_RIGGING_TASK_TPL",
    "MESHY_ANIMATION_URL",
    "MESHY_ANIMATION_TASK_TPL",
    "_BUSTUP_PROMPT",
    "_CHIBI_PROMPT",
    "_EXPRESSION_PROMPTS",
    "_EXPRESSION_GUIDANCE",
    "_REALISTIC_BUSTUP_PROMPT",
    "_REALISTIC_EXPRESSION_PROMPTS",
    "_REALISTIC_EXPRESSION_GUIDANCE",
    "_convert_anime_to_realistic",
    "_DEFAULT_ANIMATIONS",
    "_HTTP_TIMEOUT",
    "_DOWNLOAD_TIMEOUT",
    "_RETRYABLE_CODES",
    "_retry",
    "_image_to_data_uri",
    "NovelAIClient",
    "FluxKontextClient",
    "FalTextToImageClient",
    "MeshyClient",
]

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
        "Change the character's expression to a bright smile. "
        "Eyes curving upward into happy crescents, "
        "rosy flushed cheeks, joyful and cheerful expression. "
        "Head tilted slightly to one side, hands relaxed naturally. "
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

# ── Realistic (photographic) prompts ──────────────────────

_REALISTIC_BUSTUP_PROMPT = (
    "Generate a portrait of the same person from the chest up. "
    "Same outfit, same colors, same features. "
    "Professional studio photograph, soft natural lighting, "
    "shallow depth of field, looking at viewer."
)

_REALISTIC_EXPRESSION_PROMPTS: dict[str, str] = {
    "neutral": (
        "The person with a calm relaxed expression, looking at viewer. "
        "Soft eyes, natural closed mouth, relaxed eyebrows. "
        "Arms at sides in a natural posture. "
        "Bust-up portrait, professional photograph, soft studio lighting. "
        "Same person identity, outfit, and hairstyle."
    ),
    "smile": (
        "Change the person's expression to a bright genuine smile. "
        "Eyes slightly narrowed with crow's feet, "
        "rosy cheeks, warm and cheerful expression. "
        "Head tilted slightly to one side, hands relaxed naturally. "
        "Bust-up portrait, professional photograph, soft studio lighting. "
        "Same person identity, outfit, and hairstyle."
    ),
    "laugh": (
        "Change the person's expression to joyful laughing. "
        "Eyes squeezed with laugh lines, mouth open showing teeth, raised cheeks. "
        "Head tilted back with amusement, one hand near mouth. "
        "Bust-up portrait, professional photograph, soft studio lighting. "
        "Same person identity, outfit, and hairstyle."
    ),
    "troubled": (
        "Change the person's expression to worried and concerned. "
        "Eyebrows furrowed and pinched together, slight frown, anxious eyes. "
        "Both hands clasped together in front of chest, shoulders raised tensely. "
        "Bust-up portrait, professional photograph, soft studio lighting. "
        "Same person identity, outfit, and hairstyle."
    ),
    "surprised": (
        "Change the person's expression to shocked surprise. "
        "Eyes wide open, eyebrows raised very high, "
        "mouth dropped open. "
        "Both hands raised up near face, leaning back slightly. "
        "Bust-up portrait, professional photograph, soft studio lighting. "
        "Same person identity, outfit, and hairstyle."
    ),
    "thinking": (
        "Change the person's expression to deep contemplation. "
        "Looking slightly upward, one eye slightly narrowed, thoughtful furrowed brow. "
        "One hand on chin in a classic thinking pose, slight pout. "
        "Bust-up portrait, professional photograph, soft studio lighting. "
        "Same person identity, outfit, and hairstyle."
    ),
    "embarrassed": (
        "Change the person's expression to visibly embarrassed. "
        "Red blush across cheeks, eyes averted to the side. "
        "Both hands covering cheeks or fidgeting nervously. "
        "Bust-up portrait, professional photograph, soft studio lighting. "
        "Same person identity, outfit, and hairstyle."
    ),
}

assert set(_REALISTIC_EXPRESSION_PROMPTS.keys()) == _VALID_EXPRESSION_NAMES, (
    f"Realistic expression prompts mismatch: "
    f"{set(_REALISTIC_EXPRESSION_PROMPTS.keys())} != {_VALID_EXPRESSION_NAMES}"
)

_REALISTIC_EXPRESSION_GUIDANCE: dict[str, float] = {
    "neutral": 4.0,
    "smile": 4.5,
    "laugh": 4.5,
    "troubled": 5.0,
    "surprised": 4.5,
    "thinking": 4.5,
    "embarrassed": 5.0,
}

assert set(_REALISTIC_EXPRESSION_GUIDANCE.keys()) == _VALID_EXPRESSION_NAMES, (
    f"Realistic expression guidance mismatch: "
    f"{set(_REALISTIC_EXPRESSION_GUIDANCE.keys())} != {_VALID_EXPRESSION_NAMES}"
)

# ── Anime → Realistic prompt conversion ──────────────────

_ANIME_QUALITY_TAGS = frozenset({
    "masterpiece", "best quality", "very aesthetic", "absurdres",
    "anime coloring", "clean lineart", "soft shading",
    "highres", "extremely detailed",
})

_LOCALE_ETHNICITY: dict[str, str] = {
    "ja": "Japanese",
    "en": "American",
    "ko": "Korean",
    "zh": "Chinese",
}

_DANBOORU_PERSON_TAGS = frozenset({"1girl", "1boy", "2girls", "2boys"})


def _convert_anime_to_realistic(anime_prompt: str, locale: str | None = None) -> str:
    """Convert a Danbooru-style anime prompt to a photographic prompt.

    Strips anime quality/style tags, replaces Danbooru shorthands with
    natural English (with locale-based ethnicity), and prepends
    realistic quality descriptors.
    """
    if locale is None:
        try:
            from core.config.models import load_config
            locale = load_config().locale or "ja"
        except Exception:
            locale = "ja"

    ethnicity = _LOCALE_ETHNICITY.get(locale, "")

    person_map: dict[str, str] = {
        "1girl": f"a young {ethnicity} woman" if ethnicity else "a young woman",
        "1boy": f"a young {ethnicity} man" if ethnicity else "a young man",
        "2girls": f"two young {ethnicity} women" if ethnicity else "two young women",
        "2boys": f"two young {ethnicity} men" if ethnicity else "two young men",
    }

    tags = [t.strip() for t in anime_prompt.split(",") if t.strip()]

    converted: list[str] = []
    for tag in tags:
        lower = tag.lower().strip()
        if lower in _ANIME_QUALITY_TAGS:
            continue
        natural = person_map.get(lower)
        if natural:
            converted.append(natural)
        else:
            converted.append(tag)

    realistic_prefix = [
        "professional photograph",
        "studio lighting",
        "high resolution",
        "realistic",
        "photorealistic",
    ]
    return ", ".join(realistic_prefix + converted)


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
