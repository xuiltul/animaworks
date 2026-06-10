from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""LoCoMo answer prompts, post-processing, and gate settings."""

import re
from typing import Any

LOCOMO_CONFIDENCE_DEFAULT = 0.35
LOCOMO_RRF_CONFIDENCE_DEFAULT = 0.02
LOCOMO_CONFIDENCE_MULTI_HOP_FACT = 0.05
LOCOMO_RRF_CONFIDENCE_MULTI_HOP_FACT = 0.005
LOCOMO_CONFIDENCE_ADVERSARIAL = LOCOMO_CONFIDENCE_DEFAULT
LOCOMO_RRF_CONFIDENCE_ADVERSARIAL = LOCOMO_RRF_CONFIDENCE_DEFAULT

LOCOMO_ANSWER_MAX_TOKENS = 512

_ADVERSARIAL_CATEGORY = 5

_COT_PREFIXES = (
    "we need to find",
    "we need to determine",
    "we need to",
    "let's ",
    "let us ",
    "the context ",
    "from the context",
    "from context",
    "based on the conversation",
    "based on the context",
    "based on context",
    "we are asked",
    "the question asks",
    "looking at ",
    "i need to ",
)

_WHICH_IS_RE = re.compile(r"which is (.+?)[\.\"]\s*$", re.IGNORECASE)
_ANSWER_CLAUSE_RE = re.compile(
    r"(?:the answer is|answer is|answer should be)\s+(.+?)[\.\"]?\s*$",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

ANSWER_SYSTEM = (
    "You are an expert assistant answering questions about past conversations based on the provided context."
)

ANSWER_TEMPLATE = """# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay special attention to timestamps (event_time) to determine when events occurred
3. If memories contain contradictory information, prioritize the most recent memory
4. Always convert relative time references (yesterday, last week, next month) to specific dates using the event_time timestamps
5. Timestamps represent the actual time the event occurred, not when it was mentioned
6. When multiple items are asked for, answer with a comma-separated list of short phrases
7. If the context supports a reasonable inference, provide your best answer — only say "No information available." when the context has absolutely no relevant information
8. Keep your answer as brief and direct as possible — a short phrase or sentence

Example:
Memory: (event_time: 2023-05-08T13:56:00) I went to the vet yesterday.
Question: When did I go to the vet?
Answer: 7 May 2023

Context:
{context}

Question: {question}
Answer:"""


def confidence_gate_for_category(category: int | None) -> dict[str, float]:
    """Return confidence gate thresholds for a LoCoMo question category."""
    if category == _ADVERSARIAL_CATEGORY:
        return {
            "confidence_threshold": LOCOMO_CONFIDENCE_ADVERSARIAL,
            "rrf_confidence_threshold": LOCOMO_RRF_CONFIDENCE_ADVERSARIAL,
        }
    return {
        "confidence_threshold": LOCOMO_CONFIDENCE_DEFAULT,
        "rrf_confidence_threshold": LOCOMO_RRF_CONFIDENCE_DEFAULT,
    }


def build_answer_user_content(
    question: str,
    context: str,
    *,
    category: int | None = None,
) -> str:
    """Format the user message for LoCoMo answer generation."""
    _ = category
    return ANSWER_TEMPLATE.format(
        context=context,
        question=question,
    )


def normalize_locomo_answer(
    raw: str,
    *,
    category: int | None = None,
    enable_category_normalization: bool = False,
) -> str:
    """Strip CoT leakage and compress verbose answers for token-F1 scoring."""
    text = (raw or "").strip()
    if not text:
        return "No information available."

    if re.search(r"(?i)^\s*answer:\s*", text):
        text = re.sub(r"(?i)^\s*answer:\s*", "", text, count=1).strip()

    low = text.lower()
    if "no information available" in low or "not mentioned" in low:
        if enable_category_normalization and category == _ADVERSARIAL_CATEGORY:
            return "No information available."
        if enable_category_normalization and category in (3, 4):
            pass
        else:
            return "No information available."

    which = _WHICH_IS_RE.search(text)
    if which:
        return which.group(1).strip().strip('"')

    answer_clause = _ANSWER_CLAUSE_RE.search(text)
    if answer_clause and any(low.startswith(p) for p in _COT_PREFIXES):
        clause = answer_clause.group(1).strip().strip('"')
        if 0 < len(clause.split()) <= 10:
            text = clause

    text = _strip_chain_of_thought(text)

    if enable_category_normalization and category == 2 and len(text.split()) > 8:
        years = _YEAR_RE.findall(text)
        if years:
            return years[-1]

    first_line = text.splitlines()[0].strip()
    if len(text.split()) > 20:
        text = first_line

    if enable_category_normalization and len(text.split()) > 15 and "," in text:
        parts = [p.strip() for p in text.split(",")]
        if parts and len(parts[-1].split()) <= 6:
            text = parts[-1]

    return text.strip() or "No information available."


def _strip_chain_of_thought(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text
    if any(lines[0].lower().startswith(p) for p in _COT_PREFIXES):
        for ln in reversed(lines):
            ln_low = ln.lower()
            if any(ln_low.startswith(p) for p in _COT_PREFIXES):
                continue
            if len(ln.split()) <= 15:
                return ln
        return lines[-1]
    return text


def merge_pipeline_gate_settings(
    settings: dict[str, Any],
    *,
    category: int | None,
    fact_evidence: bool = False,
) -> dict[str, float]:
    """Return configured pipeline gate thresholds with category-safe LoCoMo relaxations."""
    if category == 1 and fact_evidence:
        return {
            "confidence_threshold": min(
                float(settings.get("confidence_threshold", LOCOMO_CONFIDENCE_DEFAULT)),
                LOCOMO_CONFIDENCE_MULTI_HOP_FACT,
            ),
            "rrf_confidence_threshold": min(
                float(settings.get("rrf_confidence_threshold", LOCOMO_RRF_CONFIDENCE_DEFAULT)),
                LOCOMO_RRF_CONFIDENCE_MULTI_HOP_FACT,
            ),
        }
    return {
        "confidence_threshold": float(
            settings.get("confidence_threshold", LOCOMO_CONFIDENCE_DEFAULT),
        ),
        "rrf_confidence_threshold": float(
            settings.get("rrf_confidence_threshold", LOCOMO_RRF_CONFIDENCE_DEFAULT),
        ),
    }
