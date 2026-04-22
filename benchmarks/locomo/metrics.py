from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# LoCoMo benchmark evaluation metrics (LoCoMo-style; see snap-research/locomo task_eval/evaluation.py)
import asyncio
import logging
import re
import string
from collections import Counter
from typing import Any

try:
    from nltk.stem import PorterStemmer

    _PORTER: PorterStemmer | None = PorterStemmer()
except Exception:  # noqa: BLE001 - optional nltk
    _PORTER = None

logger = logging.getLogger(__name__)

# ── Constants ──────────

CATEGORY_NAMES: dict[int, str] = {
    1: "multi_hop",
    2: "temporal",
    3: "complex",
    4: "open_domain",
    5: "adversarial",
}

_ARTICLE_RE = re.compile(r"\b(a|an|the|and)\b", re.IGNORECASE)
_JUDGE_RETRIES = 3
_LLM_JUDGE_TEMPLATE = """You are evaluating a memory system's response to a question about past conversations.

Question: {question}
Reference answer: {reference}
System answer: {prediction}

Judge whether the system answer correctly addresses the question based on the reference answer.
Respond with EXACTLY one of:
- CORRECT: The answer captures the key information from the reference
- PARTIALLY_CORRECT: The answer has some relevant information but is incomplete or slightly inaccurate
- INCORRECT: The answer is wrong, irrelevant, or missing key information

Your judgment (one word):
"""

# ── Normalization and tokens ──────────


def _normalize_answer(text: str) -> str:
    """Lowercase, strip articles, strip punctuation, collapse whitespace.

    Ported from LoCoMo ``task_eval/evaluation.py`` ``normalize_answer``.

    Args:
        text: Raw answer or prediction string.

    Returns:
        Normalized string for token overlap metrics.
    """
    s = text.replace(",", "")

    def remove_articles(t: str) -> str:
        return _ARTICLE_RE.sub(" ", t)

    def white_space_fix(t: str) -> str:
        return " ".join(t.split())

    def remove_punc(t: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exclude)

    def lower(t: str) -> str:
        return t.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _stemmed_tokens(text: str) -> list[str]:
    """Return stemmed word tokens from normalized text.

    Uses :class:`nltk.stem.PorterStemmer` when NLTK is available; otherwise
    returns normalized tokens without stemming.

    Args:
        text: Input text.

    Returns:
        List of token strings.
    """
    norm = _normalize_answer(text)
    toks = norm.split() if norm else []
    if not toks or _PORTER is None:
        return toks
    return [_PORTER.stem(w) for w in toks]


# ── F1 and category eval ──────────


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 using stemmed tokens (LoCoMo ``f1_score``).

    Args:
        prediction: Model output.
        ground_truth: Reference answer.

    Returns:
        F1 in ``[0, 1]``, or 0.0 if either side is empty after normalization.
    """
    pred_toks = _stemmed_tokens(prediction)
    gt_toks = _stemmed_tokens(ground_truth)
    if not pred_toks or not gt_toks:
        return 0.0
    common: Counter[str] = Counter(pred_toks) & Counter(gt_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gt_toks)
    return (2.0 * precision * recall) / (precision + recall)


def _multi_hop_f1(prediction: str, answer: str) -> float:
    """Multi-hop F1: comma-split both sides, mean over refs of max F1 to each ref.

    Matches LoCoMo ``f1(prediction, ground_truth)`` in ``task_eval/evaluation.py``.

    Args:
        prediction: Model output.
        answer: Comma-separated reference spans.

    Returns:
        Aggregated F1 score.
    """
    predictions = [p.strip() for p in prediction.split(",") if p.strip()]
    ground_truths = [g.strip() for g in answer.split(",") if g.strip()]
    if not ground_truths or not predictions:
        return 0.0
    per_gt = [max(f1_score(p, gt) for p in predictions) for gt in ground_truths]
    return sum(per_gt) / len(per_gt)


def eval_by_category(prediction: str, answer: str, category: int) -> float:
    """LoCoMo-style per-category string metric (F1 or adversarial 0/1).

    Args:
        prediction: Model output.
        answer: Reference answer.
        category: LoCoMo category id (1–5).

    Returns:
        Score in ``[0, 1]``.
    """
    if category == 1:
        return _multi_hop_f1(prediction, answer)
    if category in (2, 4):
        return f1_score(prediction, answer)
    if category == 3:
        first = answer.split(";")[0].strip()
        return f1_score(prediction, first)
    if category == 5:
        low = prediction.lower()
        if "no information available" in low or "not mentioned" in low:
            return 1.0
        return 0.0
    logger.warning("Unknown LoCoMo category %s; using standard F1", category)
    return f1_score(prediction, answer)


# ── LLM judge ──────────


def _parse_judge_verdict(content: str) -> tuple[str, float]:
    """Map model text to verdict label and numeric score."""
    u = content.upper()
    if "PARTIALLY_CORRECT" in u or "PARTIALLY CORRECT" in u:
        return "partially_correct", 0.5
    if "INCORRECT" in u:
        return "incorrect", 0.0
    if "CORRECT" in u:
        return "correct", 1.0
    return "incorrect", 0.0


async def llm_judge(
    question: str,
    reference: str,
    prediction: str,
    *,
    model: str = "gpt-4o",
) -> dict[str, Any]:
    """LLM-based answer quality judgment via LiteLLM.

    Args:
        question: User question.
        reference: Gold reference answer.
        prediction: System answer to score.
        model: LiteLLM model id.

    Returns:
        ``{"verdict": str, "score": float}``. On total API failure, verdict
        ``"error"`` and score ``0.0``.
    """
    prompt = _LLM_JUDGE_TEMPLATE.format(
        question=question,
        reference=reference,
        prediction=prediction,
    )
    import litellm  # noqa: PLC0415

    last_err: Exception | None = None
    for attempt in range(1, _JUDGE_RETRIES + 1):
        try:
            resp = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = (resp.choices[0].message.content or "").strip()
            verdict, score = _parse_judge_verdict(content)
            return {"verdict": verdict, "score": score}
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning(
                "llm_judge attempt %s/%s failed: %s",
                attempt,
                _JUDGE_RETRIES,
                exc,
            )
            if attempt < _JUDGE_RETRIES:
                await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
    if last_err is not None:
        logger.error("llm_judge failed after %s attempts: %s", _JUDGE_RETRIES, last_err)
    return {"verdict": "error", "score": 0.0}


def llm_judge_sync(
    question: str,
    reference: str,
    prediction: str,
    *,
    model: str = "gpt-4o",
) -> dict[str, Any]:
    """Synchronous wrapper for :func:`llm_judge`; avoids per-call asyncio.run overhead.

    Uses ``litellm.completion`` directly instead of creating an event loop.
    """
    import litellm as _litellm  # noqa: PLC0415

    prompt = _LLM_JUDGE_TEMPLATE.format(
        question=question,
        reference=reference,
        prediction=prediction,
    )
    last_err: Exception | None = None
    for attempt in range(1, _JUDGE_RETRIES + 1):
        try:
            resp = _litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = (resp.choices[0].message.content or "").strip()
            verdict, score = _parse_judge_verdict(content)
            return {"verdict": verdict, "score": score}
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("llm_judge_sync attempt %s/%s failed: %s", attempt, _JUDGE_RETRIES, exc)
            if attempt < _JUDGE_RETRIES:
                import time  # noqa: PLC0415

                time.sleep(0.5 * (2 ** (attempt - 1)))
    if last_err is not None:
        logger.error("llm_judge_sync failed after %s attempts: %s", _JUDGE_RETRIES, last_err)
    return {"verdict": "error", "score": 0.0}


# ── Summary aggregation ──────────


def compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-item F1 and optional judge scores by category and overall.

    Args:
        results: Items with ``category`` (int), ``f1`` (float),
            ``judge_score`` (float or None).

    Returns:
        ``overall_f1``, ``overall_judge``, and ``by_category`` averages.
    """
    if not results:
        return {
            "overall_f1": 0.0,
            "overall_judge": None,
            "by_category": {},
        }

    overall_f1 = sum(r["f1"] for r in results) / len(results)
    js = [r["judge_score"] for r in results if r.get("judge_score") is not None]
    overall_judge: float | None
    if js:
        overall_judge = sum(js) / len(js)
    else:
        overall_judge = None

    by_acc: dict[str, tuple[float, int, float, int]] = {}
    for r in results:
        cat = int(r["category"])
        name = CATEGORY_NAMES.get(cat, f"category_{cat}")
        f1_sum, f1_cnt, j_sum, j_cnt = by_acc.get(name, (0.0, 0, 0.0, 0))
        f1_sum += float(r["f1"])
        f1_cnt += 1
        jv = r.get("judge_score")
        if jv is not None:
            j_sum += float(jv)
            j_cnt += 1
        by_acc[name] = (f1_sum, f1_cnt, j_sum, j_cnt)

    out_by: dict[str, dict[str, float | int | None]] = {}
    for name, (f1_sum, f1_cnt, j_sum, j_cnt) in by_acc.items():
        avg_f1 = f1_sum / f1_cnt if f1_cnt else 0.0
        j_avg: float | None = (j_sum / j_cnt) if j_cnt else None
        out_by[name] = {"f1": avg_f1, "judge": j_avg, "count": f1_cnt}

    return {
        "overall_f1": overall_f1,
        "overall_judge": overall_judge,
        "by_category": out_by,
    }
