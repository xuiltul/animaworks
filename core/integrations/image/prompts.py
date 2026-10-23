# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Prompt constants for bustup, chibi, and expression variants."""

from __future__ import annotations

from core.schemas import VALID_EMOTIONS as _VALID_EXPRESSION_NAMES

# ── Anime prompts ───────────────────────────────────────────

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

_CHAT_ICON_PROMPT = (
    "Create a square chat avatar icon from this bust-up reference. "
    "Center the face in the frame; crop to a tight head-and-shoulders portrait. "
    "Same character identity, outfit colors, hairstyle, and features. "
    "Clean plain background, neutral soft lighting. "
    "Anime illustration style, high quality, looking at viewer."
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

assert set(_EXPRESSION_PROMPTS.keys()) == _VALID_EXPRESSION_NAMES, (
    f"Expression prompts mismatch: {set(_EXPRESSION_PROMPTS.keys())} != {_VALID_EXPRESSION_NAMES}"
)

# Per-expression guidance_scale for Flux Kontext.
_EXPRESSION_GUIDANCE: dict[str, float] = {
    "neutral": 3.5,
    "smile": 4.0,
    "laugh": 4.5,
    "troubled": 4.5,
    "surprised": 4.0,
    "thinking": 4.0,
    "embarrassed": 4.5,
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

_REALISTIC_FROM_ANIME_PROMPT = (
    "Turn the anime character in the reference image into a photograph of a real person. "
    "Keep the same gender, age impression, hairstyle, hair length, bangs, hair accessories, "
    "outfit design and colors, shoes, and accessories. "
    "Do not add hair ornaments, accessories, or items that are not in the reference image. "
    "Translate anime-only hair colors into a natural shade a real person could have "
    "(light blue or silver to light ash-gray, green or teal to dark brown-black with a subtle teal tint, "
    "pink to rose-brown, purple to dark plum-brown, red to dark burgundy-brown, orange to copper-brown); "
    "keep black, brown, and blonde hair as they are. Eyes become a natural dark brown. "
    "Realistic human proportions, full body, standing, plain white background, "
    "professional studio photograph, soft lighting, looking at viewer."
)

_REALISTIC_CHAT_ICON_PROMPT = (
    "Create a square chat avatar icon from this bust-up reference. "
    "Center the face in the frame; crop to a tight head-and-shoulders portrait. "
    "Same person identity, outfit, hairstyle, and features. "
    "Clean plain background, soft studio lighting. "
    "Professional photograph, natural skin texture, shallow depth of field, looking at viewer."
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
    f"Realistic expression prompts mismatch: {set(_REALISTIC_EXPRESSION_PROMPTS.keys())} != {_VALID_EXPRESSION_NAMES}"
)

_REALISTIC_EXPRESSION_GUIDANCE: dict[str, float] = {
    "neutral": 3.0,
    "smile": 3.5,
    "laugh": 4.0,
    "troubled": 4.0,
    "surprised": 3.5,
    "thinking": 3.5,
    "embarrassed": 4.0,
}

assert set(_REALISTIC_EXPRESSION_GUIDANCE.keys()) == _VALID_EXPRESSION_NAMES, (
    f"Realistic expression guidance mismatch: {set(_REALISTIC_EXPRESSION_GUIDANCE.keys())} != {_VALID_EXPRESSION_NAMES}"
)
