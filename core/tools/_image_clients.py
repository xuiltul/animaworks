# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""API clients and shared constants for image/3D generation.

Facade module — re-exports everything from core.tools.image for backward
compatibility. Use ``from core.tools._image_clients import NovelAIClient``
or ``from core.tools.image import NovelAIClient``.
"""

from __future__ import annotations

from core.tools.image import (
    _BUSTUP_PROMPT,
    _CHIBI_PROMPT,
    _DEFAULT_ANIMATIONS,
    _DOWNLOAD_TIMEOUT,
    _EXPRESSION_GUIDANCE,
    _EXPRESSION_PROMPTS,
    _HTTP_TIMEOUT,
    _REALISTIC_BUSTUP_PROMPT,
    _REALISTIC_EXPRESSION_GUIDANCE,
    _REALISTIC_EXPRESSION_PROMPTS,
    _RETRYABLE_CODES,
    EXECUTION_PROFILE,
    FAL_FLUX_PRO_SUBMIT_URL,
    FAL_KONTEXT_SUBMIT_URL,
    MESHY_ANIMATION_TASK_TPL,
    MESHY_ANIMATION_URL,
    MESHY_IMAGE_TO_3D_URL,
    MESHY_RIGGING_TASK_TPL,
    MESHY_RIGGING_URL,
    MESHY_TASK_URL_TPL,
    NOVELAI_API_URL,
    NOVELAI_ENCODE_URL,
    NOVELAI_MODEL,
    FalTextToImageClient,
    FluxKontextClient,
    LocalDiffusersClient,
    MeshyClient,
    NovelAIClient,
    _convert_anime_to_realistic,
    _image_to_data_uri,
    _retry,
)

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
    "LocalDiffusersClient",
    "NovelAIClient",
    "FluxKontextClient",
    "FalTextToImageClient",
    "MeshyClient",
]
