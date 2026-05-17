from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shadow skill router for offline routing evaluation."""

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from core.memory.frontmatter import strip_frontmatter
from core.skills.models import SkillMetadata, SkillRiskMetadata

_ASCII_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_+.-]*", re.IGNORECASE)
_JP_SEGMENT_RE = re.compile(r"[ぁ-んァ-ヶー一-龯々]{2,}")
_ASCII_PART_RE = re.compile(r"[_+.-]+")
_JP_PARTICLE_CHARS = frozenset(
    {
        "を",
        "に",
        "へ",
        "は",
        "が",
        "で",
        "と",
        "や",
        "も",
        "か",
        "の",
    }
)
_ROUTING_STOPWORDS = {
    "する",
    "して",
    "してく",
    "したい",
    "お願い",
    "ください",
    "です",
    "ます",
    "これ",
    "それ",
    "今日",
    "明日",
    "もう少し",
    "話そう",
    "相談",
    "方針",
    "設計",
    "the",
    "and",
    "for",
    "with",
    "tool",
    "skill",
    "use",
    "md",
}


class SkillRouteCandidate(BaseModel):
    """Single router result for a matching skill."""

    name: str
    path: str
    score: float
    confidence: Literal["high", "medium", "low"]
    reasons: list[str] = Field(default_factory=list)
    is_common: bool = False
    is_procedure: bool = False
    risk: SkillRiskMetadata = Field(default_factory=SkillRiskMetadata)


class SkillRouter:
    """Hybrid router returning skill pointers and reasons only."""

    def __init__(
        self,
        *,
        min_score: float = 1.15,
        lexical_only_min_score: float = 3.0,
        medium_score: float = 2.5,
        high_score: float = 5.0,
        rrf_k: int = 60,
        rrf_weight: float = 8.0,
        include_body: bool = True,
    ) -> None:
        self.min_score = min_score
        self.lexical_only_min_score = lexical_only_min_score
        self.medium_score = medium_score
        self.high_score = high_score
        self.rrf_k = rrf_k
        self.rrf_weight = rrf_weight
        self.include_body = include_body

    def route(
        self,
        query: str,
        skills: Sequence[SkillMetadata],
        *,
        top_k: int = 3,
        dense_scores: Mapping[str, float] | None = None,
    ) -> list[SkillRouteCandidate]:
        """Return top skill candidates for *query*."""
        query_norm = _normalize(query)
        if not query_norm:
            return []

        query_tokens = _tokenize(query_norm)
        if not query_tokens:
            return []

        records = [
            _SkillRecord.from_metadata(meta, include_body=self.include_body) for meta in skills if _router_visible(meta)
        ]
        if not records:
            return []

        deterministic = self._deterministic_scores(query_norm, query_tokens, records)
        lexical = self._lexical_scores(query_tokens, records)
        dense = self._dense_scores(dense_scores or {}, records)

        deterministic_ranked = _ranked_keys(deterministic)
        lexical_ranked = _ranked_keys(lexical)
        dense_ranked = _ranked_keys(dense)

        rrf = _rrf_scores(
            (deterministic_ranked, 2.0),
            (lexical_ranked, 1.0),
            (dense_ranked, 1.0),
            k=self.rrf_k,
        )

        combined: dict[str, float] = {}
        for record in records:
            key = record.key
            source_boost = 0.05 if not record.meta.is_common and not record.meta.is_procedure else 0.0
            deterministic_score = deterministic.get(key, 0.0)
            lexical_score = lexical.get(key, 0.0)
            dense_score = dense.get(key, 0.0)
            if deterministic_score == 0 and dense_score == 0 and lexical_score < self.lexical_only_min_score:
                continue
            score = (
                deterministic_score + lexical_score + dense_score + (rrf.get(key, 0.0) * self.rrf_weight) + source_boost
            )
            if record.negative_penalty:
                score -= record.negative_penalty
            if score >= self.min_score:
                combined[key] = score

        ranked = sorted(
            (record for record in records if record.key in combined),
            key=lambda r: (
                combined[r.key],
                1 if not r.meta.is_common and not r.meta.is_procedure else 0,
                1 if r.meta.is_procedure else 0,
                r.meta.name,
            ),
            reverse=True,
        )

        candidates: list[SkillRouteCandidate] = []
        for record in ranked[:top_k]:
            score = round(combined[record.key], 4)
            candidates.append(
                SkillRouteCandidate(
                    name=record.meta.name,
                    path=record.pointer_path,
                    score=score,
                    confidence=self._confidence(score, record.reasons),
                    reasons=_dedupe(record.reasons),
                    is_common=record.meta.is_common,
                    is_procedure=record.meta.is_procedure,
                    risk=_merged_risk(record.meta),
                )
            )
        return candidates

    def metadata_gaps(self, skills: Sequence[SkillMetadata]) -> dict[str, list[str]]:
        """Return missing routing metadata hints by skill name."""
        gaps: dict[str, list[str]] = {}
        for meta in skills:
            missing: list[str] = []
            triggers = _merged_list(meta.trigger_phrases, meta.routing.trigger_phrases)
            use_when = _merged_list(meta.use_when, meta.routing.use_when)
            domains = _merged_list(meta.domains, meta.routing.domains, meta.tags)
            if not triggers:
                missing.append("trigger_phrases")
            if not use_when:
                missing.append("use_when")
            if not domains:
                missing.append("domains/tags")
            if missing:
                gaps[meta.name] = missing
        return gaps

    def _confidence(self, score: float, reasons: Sequence[str]) -> Literal["high", "medium", "low"]:
        if score >= self.high_score and _has_strong_signal(reasons):
            return "high"
        if score >= self.medium_score:
            return "medium"
        return "low"

    def _deterministic_scores(
        self,
        query_norm: str,
        query_tokens: set[str],
        records: list[_SkillRecord],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for record in records:
            meta = record.meta
            score = 0.0

            positive_fields = [
                ("name", [meta.name], 2.0),
                ("path", [record.pointer_path], 1.5),
                ("trigger", _merged_list(meta.trigger_phrases, meta.routing.trigger_phrases), 5.0),
                ("use_when", _merged_list(meta.use_when, meta.routing.use_when), 4.0),
                ("tag", _merged_list(meta.tags, meta.domains, meta.routing.domains), 2.5),
                ("tool", _merged_list(meta.allowed_tools, meta.requires_tools), 2.5),
                ("platform", meta.platforms, 2.0),
                ("example", _merged_list(meta.routing_examples, meta.routing.routing_examples), 2.0),
                ("description", [meta.description], 1.0),
            ]

            for label, values, weight in positive_fields:
                field_score, reasons = _score_phrases(query_norm, query_tokens, values, label, weight)
                score += field_score
                record.reasons.extend(reasons)

            negative_values = _merged_list(
                meta.negative_phrases,
                meta.do_not_use_when,
                meta.routing.negative_phrases,
                meta.routing.do_not_use_when,
            )
            penalty, reasons = _score_phrases(query_norm, query_tokens, negative_values, "negative", 6.0)
            if penalty:
                record.negative_penalty += penalty
                record.reasons.extend(f"negative:{reason}" for reason in reasons)

            if score > 0:
                scores[record.key] = score
        return scores

    def _lexical_scores(self, query_tokens: set[str], records: list[_SkillRecord]) -> dict[str, float]:
        doc_tokens = {record.key: _tokenize(record.search_text) for record in records}
        doc_freq: Counter[str] = Counter()
        for tokens in doc_tokens.values():
            for token in tokens:
                doc_freq[token] += 1

        total_docs = max(len(records), 1)
        avg_len = sum(len(tokens) for tokens in doc_tokens.values()) / total_docs
        avg_len = max(avg_len, 1.0)

        scores: dict[str, float] = {}
        for record in records:
            tokens = doc_tokens[record.key]
            if not tokens:
                continue
            counts = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                if token not in counts:
                    continue
                df = doc_freq[token]
                idf = math.log(1 + ((total_docs - df + 0.5) / (df + 0.5)))
                tf = counts[token]
                denom = tf + 1.2 * (1 - 0.75 + 0.75 * (len(tokens) / avg_len))
                score += idf * ((tf * 2.2) / denom)
            if score > 0:
                matched = sorted(query_tokens.intersection(tokens))
                if len(matched) < 2:
                    continue
                adjusted = min(score, 5.0)
                scores[record.key] = adjusted
                record.reasons.append(f"lexical:{','.join(matched[:6])}")
        return scores

    @staticmethod
    def _dense_scores(
        dense_scores: Mapping[str, float],
        records: list[_SkillRecord],
    ) -> dict[str, float]:
        if not dense_scores:
            return {}
        lookup: dict[str, _SkillRecord] = {}
        name_counts = Counter(record.meta.name for record in records)
        for record in records:
            if name_counts[record.meta.name] == 1:
                lookup[record.meta.name] = record
            lookup[record.pointer_path] = record
            if record.meta.path is not None:
                lookup[str(record.meta.path)] = record

        out: dict[str, float] = {}
        for raw_key, raw_score in dense_scores.items():
            record = lookup.get(raw_key)
            if record is None:
                continue
            score = max(float(raw_score), 0.0)
            if score > 0:
                out[record.key] = score
                record.reasons.append(f"dense:{score:.3f}")
        return out


class _SkillRecord(BaseModel):
    meta: SkillMetadata
    key: str
    pointer_path: str
    search_text: str
    reasons: list[str] = Field(default_factory=list)
    negative_penalty: float = 0.0

    @classmethod
    def from_metadata(cls, meta: SkillMetadata, *, include_body: bool) -> _SkillRecord:
        pointer_path = _pointer_path(meta)
        text_parts = [
            meta.name,
            pointer_path,
            meta.description,
            meta.category or "",
            " ".join(meta.tags),
            " ".join(meta.platforms),
            " ".join(meta.requires_tools),
            " ".join(meta.allowed_tools),
            " ".join(_merged_list(meta.use_when, meta.routing.use_when)),
            " ".join(_merged_list(meta.trigger_phrases, meta.routing.trigger_phrases)),
            " ".join(_merged_list(meta.domains, meta.routing.domains)),
            " ".join(_merged_list(meta.routing_examples, meta.routing.routing_examples)),
        ]
        if include_body:
            text_parts.append(_read_body(meta.path))
        return cls(
            meta=meta,
            key=str(meta.path or meta.name),
            pointer_path=pointer_path,
            search_text="\n".join(part for part in text_parts if part),
        )


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").casefold()


def _tokenize(text: str) -> set[str]:
    norm = _normalize(text)
    tokens: set[str] = set()
    for match in _ASCII_WORD_RE.findall(norm):
        if _is_routing_token(match):
            tokens.add(match)
        for part in _ASCII_PART_RE.split(match):
            if part != match and _is_routing_token(part):
                tokens.add(part)
    for segment in _JP_SEGMENT_RE.findall(norm):
        if _is_routing_token(segment):
            tokens.add(segment)
        for n in (2, 3, 4):
            if len(segment) < n:
                continue
            for i in range(0, len(segment) - n + 1):
                gram = segment[i : i + n]
                if _is_routing_token(gram):
                    tokens.add(gram)
    return tokens


def _is_routing_token(token: str) -> bool:
    if token in _ROUTING_STOPWORDS or len(token) < 2:
        return False
    if _ASCII_WORD_RE.fullmatch(token):
        return True
    if not _JP_SEGMENT_RE.fullmatch(token):
        return False
    if all("ぁ" <= char <= "ん" for char in token):
        return False
    if len(token) == 2:
        if "ー" in token:
            return False
        if any("ぁ" <= char <= "ん" for char in token):
            return False
    return not (len(token) <= 4 and any(char in _JP_PARTICLE_CHARS for char in token))


def _score_phrases(
    query_norm: str,
    query_tokens: set[str],
    values: Sequence[str],
    label: str,
    weight: float,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    for raw in values:
        value = _normalize(raw)
        if not value:
            continue
        if value in query_norm or query_norm in value:
            score += weight
            reasons.append(f"{label}:exact:{raw}")
            continue
        tokens = _tokenize(value)
        overlap = query_tokens.intersection(tokens)
        if not overlap:
            continue
        ratio = len(overlap) / max(len(tokens), 1)
        if ratio >= 0.35 or len(overlap) >= 2:
            part_score = weight * min(0.85, max(0.25, ratio))
            score += part_score
            reasons.append(f"{label}:overlap:{raw}")
        elif label in {"name", "path", "tag", "tool", "platform"} and any(
            _ASCII_WORD_RE.fullmatch(token) and len(token) >= 3 for token in overlap
        ):
            score += weight * 0.5
            reasons.append(f"{label}:identifier:{raw}")
    return score, reasons


def _ranked_keys(scores: Mapping[str, float]) -> list[str]:
    return [key for key, score in sorted(scores.items(), key=lambda item: item[1], reverse=True) if score > 0]


def _rrf_scores(*ranked_lists: tuple[list[str], float], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranked, weight in ranked_lists:
        for rank, key in enumerate(ranked, start=1):
            scores[key] += weight / (k + rank)
    return dict(scores)


def _merged_list(*values: Sequence[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for seq in values:
        for value in seq:
            item = str(value).strip()
            if not item:
                continue
            marker = _normalize(item)
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(item)
    return merged


def _merged_risk(meta: SkillMetadata) -> SkillRiskMetadata:
    base = meta.risk
    nested = meta.routing.risk
    return SkillRiskMetadata(
        read_only=base.read_only or nested.read_only,
        destructive=base.destructive or nested.destructive,
        external_send=base.external_send or nested.external_send,
        handles_untrusted_data=base.handles_untrusted_data or nested.handles_untrusted_data,
        requires_human_approval=base.requires_human_approval or nested.requires_human_approval,
        open_world=base.open_world or nested.open_world,
    )


def _dedupe(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _router_visible(meta: SkillMetadata) -> bool:
    try:
        from core.skills.curator import is_unloadable_lifecycle_state

        return not is_unloadable_lifecycle_state(meta.lifecycle_state)
    except Exception:
        return True


def _has_strong_signal(reasons: Sequence[str]) -> bool:
    strong_prefixes = (
        "trigger:",
        "use_when:",
        "tag:",
        "tool:",
        "platform:",
        "example:",
        "dense:",
        "name:exact:",
        "name:identifier:",
        "path:exact:",
        "path:identifier:",
    )
    return any(reason.startswith(strong_prefixes) for reason in reasons)


def _pointer_path(meta: SkillMetadata) -> str:
    path = meta.path
    if path is None:
        if meta.is_procedure:
            return f"procedures/{meta.name}.md"
        if meta.is_common:
            return f"common_skills/{meta.name}/SKILL.md"
        return f"skills/{meta.name}/SKILL.md"

    parts = list(path.parts)
    for marker in ("common_skills", "skills", "procedures"):
        if marker in parts:
            idx = parts.index(marker)
            return str(Path(*parts[idx:]))
    if meta.is_procedure:
        return f"procedures/{path.name}"
    if meta.is_common:
        return f"common_skills/{path.parent.name}/SKILL.md"
    if path.name == "SKILL.md":
        return f"skills/{path.parent.name}/SKILL.md"
    return str(path)


def _read_body(path: Path | None, *, max_chars: int = 8000) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        return strip_frontmatter(path.read_text(encoding="utf-8"))[:max_chars]
    except OSError:
        return ""
