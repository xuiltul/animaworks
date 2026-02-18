from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""NLI + LLM cascade validation for memory consolidation.

Validates that knowledge items extracted from episodes are faithful
to their source material, preventing hallucination during the
consolidation process.

Pipeline:
  1. NLI model check (fast, local) — entailment / contradiction / neutral
  2. LLM self-review (slow, API) — only for uncertain NLI results
"""

import json
import logging
import re

logger = logging.getLogger("animaworks.validation")


# ── KnowledgeValidator ────────────────────────────────────────


class KnowledgeValidator:
    """NLI + LLM cascade knowledge validation.

    Validates extracted knowledge items against source episodes using
    a two-stage pipeline:

    1. **NLI stage**: Local mDeBERTa model classifies premise-hypothesis
       pairs as entailment / contradiction / neutral.
    2. **LLM stage**: For uncertain NLI results, falls back to an LLM
       self-review that checks derivability from source episodes.

    Confidence scores:
    - NLI entailment (high confidence): 0.9
    - LLM review pass: 0.7
    - Contradiction or LLM rejection: item excluded
    """

    NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    CONTRADICTION_THRESHOLD = 0.7
    ENTAILMENT_THRESHOLD = 0.6

    def __init__(self) -> None:
        self._nli_pipeline = None
        self._nli_available = True

    # ── NLI model ─────────────────────────────────────────────

    def _load_nli_model(self) -> None:
        """Load the NLI model.  GPU -> CPU fallback.  On failure, NLI is skipped."""
        try:
            from transformers import pipeline as hf_pipeline

            try:
                self._nli_pipeline = hf_pipeline(
                    "text-classification",
                    model=self.NLI_MODEL,
                    device=0,
                )
                logger.info("NLI model loaded on GPU")
            except Exception:
                self._nli_pipeline = hf_pipeline(
                    "text-classification",
                    model=self.NLI_MODEL,
                    device=-1,
                )
                logger.info("NLI model loaded on CPU (GPU unavailable)")
        except Exception:
            logger.warning("NLI model load failed, falling back to LLM-only validation")
            self._nli_available = False

    def _nli_check(self, hypothesis: str, premise: str) -> tuple[str, float]:
        """Run NLI inference on a premise-hypothesis pair.

        Args:
            hypothesis: The knowledge claim to verify
            premise: The source episode text

        Returns:
            Tuple of (label, score) where label is one of
            'entailment', 'contradiction', 'neutral'
        """
        if not self._nli_available or self._nli_pipeline is None:
            return ("neutral", 0.0)
        try:
            result = self._nli_pipeline(
                f"{premise} [SEP] {hypothesis}",
                truncation=True,
            )
            label = result[0]["label"].lower()
            score = result[0]["score"]
            return (label, score)
        except Exception as e:
            logger.warning("NLI check failed: %s", e)
            return ("neutral", 0.0)

    # ── Main validation API ───────────────────────────────────

    async def validate(
        self,
        knowledge_items: list[dict],
        source_episodes: str,
        model: str = "anthropic/claude-sonnet-4-20250514",
    ) -> list[dict]:
        """Validate knowledge items against source episodes.

        Each item is checked via the NLI+LLM cascade:

        - NLI entailment (high confidence) -> confidence=0.9, accepted
        - NLI contradiction (high confidence) -> rejected
        - NLI neutral / low confidence -> LLM review
          - LLM pass -> confidence=0.7, accepted
          - LLM fail -> rejected

        Args:
            knowledge_items: List of dicts with at least a ``content`` key
            source_episodes: Raw episode text used as verification source
            model: LLM model for the fallback review stage

        Returns:
            Filtered list of accepted knowledge items with ``confidence``
            field populated
        """
        if self._nli_pipeline is None and self._nli_available:
            self._load_nli_model()

        results: list[dict] = []
        for item in knowledge_items:
            content = item.get("content", "")
            if not content:
                continue

            label, score = self._nli_check(content, source_episodes[:2000])

            if label == "entailment" and score >= self.ENTAILMENT_THRESHOLD:
                item["confidence"] = 0.9
                results.append(item)
                logger.debug(
                    "Knowledge accepted (NLI entailment, score=%.2f): %s",
                    score, content[:100],
                )
            elif label == "contradiction" and score >= self.CONTRADICTION_THRESHOLD:
                logger.warning(
                    "Knowledge rejected (NLI contradiction, score=%.2f): %s",
                    score, content[:100],
                )
            else:
                # Uncertain — fall back to LLM review
                llm_ok = await self._llm_review(content, source_episodes, model)
                if llm_ok:
                    item["confidence"] = 0.7
                    results.append(item)
                    logger.debug(
                        "Knowledge accepted (LLM review): %s", content[:100],
                    )
                else:
                    logger.warning(
                        "Knowledge rejected (LLM review): %s", content[:100],
                    )

        logger.info(
            "Validation complete: %d/%d items accepted",
            len(results), len(knowledge_items),
        )
        return results

    # ── LLM self-review ───────────────────────────────────────

    async def _llm_review(
        self,
        knowledge: str,
        episodes: str,
        model: str,
    ) -> bool:
        """LLM self-review: check if knowledge is derivable from episodes.

        Args:
            knowledge: The extracted knowledge text
            episodes: The source episode text
            model: LLM model identifier (LiteLLM format)

        Returns:
            True if the knowledge is judged valid, False otherwise.
            Returns True on LLM failure (conservative: let items through).
        """
        import litellm

        prompt = f"""以下の「知識」が「エピソード」から正しく導出されているか検証してください。

【エピソード（原文）】
{episodes[:3000]}

【抽出された知識】
{knowledge}

判定:
- この知識はエピソードの内容から正しく導出されていますか？
- エピソードに記載のない情報が追加されていませんか？
- 事実関係に誤りはありませんか？

回答は以下のJSON形式のみで出力してください:
{{"valid": true/false, "reason": "理由"}}"""

        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
            )
            text = response.choices[0].message.content or ""
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                valid = result.get("valid", False)
                reason = result.get("reason", "")
                logger.debug(
                    "LLM review result: valid=%s reason=%s",
                    valid, reason[:200],
                )
                return bool(valid)
        except Exception as e:
            logger.warning("LLM review failed: %s", e)

        # On failure, conservatively allow the item through
        return True
