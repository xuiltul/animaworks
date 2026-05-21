from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shadow skill router."""

from pathlib import Path

import scripts.evaluate_skill_router as evaluate_skill_router
from core.skills.loader import load_skill_metadata
from core.skills.models import SkillMetadata
from core.skills.router import SkillRouter


def _skill(path: Path, text: str) -> SkillMetadata:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return load_skill_metadata(path)


def test_loads_routing_metadata_from_frontmatter(tmp_path: Path) -> None:
    meta = _skill(
        tmp_path / "skills" / "gmail-tool" / "SKILL.md",
        "---\n"
        "name: gmail-tool\n"
        "description: Gmail操作\n"
        "tags: gmail\n"
        "use_when: メール下書き\n"
        "trigger_phrases:\n"
        "  - Gmail下書き\n"
        "routing:\n"
        "  domains: [email]\n"
        "  negative_phrases: 画像生成\n"
        "  risk:\n"
        "    external_send: true\n"
        "---\n"
        "# Gmail\n",
    )

    assert meta.tags == ["gmail"]
    assert meta.use_when == ["メール下書き"]
    assert meta.trigger_phrases == ["Gmail下書き"]
    assert meta.routing.domains == ["email"]
    assert meta.routing.negative_phrases == ["画像生成"]
    assert meta.routing.risk.external_send is True


def test_routes_by_trigger_phrase_and_returns_pointer(tmp_path: Path) -> None:
    meta = _skill(
        tmp_path / "skills" / "gmail-tool" / "SKILL.md",
        "---\n"
        "name: gmail-tool\n"
        "description: Gmail API操作\n"
        "trigger_phrases: [Gmail下書き, メール返信]\n"
        "use_when: [メールの本文確認や返信下書きが必要なとき]\n"
        "tags: [gmail, email]\n"
        "routing:\n"
        "  risk:\n"
        "    external_send: true\n"
        "---\n"
        "# Gmail\n",
    )

    candidates = SkillRouter(include_body=False).route("Gmailで返信メールの下書きを作って", [meta])

    assert candidates
    assert candidates[0].name == "gmail-tool"
    assert candidates[0].path == "skills/gmail-tool/SKILL.md"
    assert candidates[0].confidence in {"medium", "high"}
    assert candidates[0].risk.external_send is True
    assert candidates[0].activation_policy.injection == "body_allowed"


def test_trust_policy_does_not_change_relevance_score(tmp_path: Path) -> None:
    trusted = SkillMetadata(
        name="mail-helper",
        description="メール下書き",
        trigger_phrases=["メール下書き"],
        path=tmp_path / "skills" / "mail-helper" / "SKILL.md",
    )
    probation = SkillMetadata(
        name="mail-helper",
        description="メール下書き",
        trigger_phrases=["メール下書き"],
        path=tmp_path / "skills" / "mail-helper" / "SKILL.md",
        trust_level="community",
        promotion_status="probation",
        skill_policy={"use_mode": "candidate_hint", "injection": "pointer_preferred"},
    )

    router = SkillRouter(include_body=False)
    trusted_candidate = router.route("メール下書きを作って", [trusted])[0]
    probation_candidate = router.route("メール下書きを作って", [probation])[0]

    assert trusted_candidate.score == probation_candidate.score
    assert trusted_candidate.activation_policy.injection == "body_allowed"
    assert probation_candidate.activation_policy.injection == "pointer_preferred"


def test_negative_phrase_can_force_abstention(tmp_path: Path) -> None:
    meta = _skill(
        tmp_path / "common_skills" / "gmail-tool" / "SKILL.md",
        "---\n"
        "name: gmail-tool\n"
        "description: Gmail操作\n"
        "trigger_phrases: [Gmail, メール]\n"
        "negative_phrases: [画像生成]\n"
        "---\n"
        "# Gmail\n",
    ).model_copy(update={"is_common": True})

    candidates = SkillRouter(include_body=False).route("画像生成をして", [meta])

    assert candidates == []


def test_greeting_abstains_without_trigger() -> None:
    meta = SkillMetadata(
        name="cron-management",
        description="cron.mdの編集",
        trigger_phrases=["cron.md", "定時タスク"],
        tags=["cron"],
    )

    candidates = SkillRouter(include_body=False).route("おはよう、今日はどう？", [meta])

    assert candidates == []


def test_personal_skill_wins_common_tie(tmp_path: Path) -> None:
    personal = SkillMetadata(
        name="mail-helper",
        description="メール下書き",
        trigger_phrases=["メール下書き"],
        path=tmp_path / "animas" / "mei" / "skills" / "mail-helper" / "SKILL.md",
    )
    common = SkillMetadata(
        name="mail-helper-common",
        description="メール下書き",
        trigger_phrases=["メール下書き"],
        path=tmp_path / "common_skills" / "mail-helper-common" / "SKILL.md",
        is_common=True,
    )

    candidates = SkillRouter(include_body=False).route("メール下書きを作って", [common, personal], top_k=2)

    assert [candidate.name for candidate in candidates] == ["mail-helper", "mail-helper-common"]


def test_procedure_pointer_path(tmp_path: Path) -> None:
    meta = SkillMetadata(
        name="document-creation",
        description="PDF化",
        trigger_phrases=["PDF化"],
        path=tmp_path / "animas" / "mei" / "procedures" / "document-creation.md",
        is_procedure=True,
    )

    candidate = SkillRouter(include_body=False).route("PDF化してDownloadsに置いて", [meta])[0]

    assert candidate.path == "procedures/document-creation.md"


def test_metadata_gaps_report_missing_routing_hints() -> None:
    gaps = SkillRouter().metadata_gaps([SkillMetadata(name="bare-skill")])

    assert gaps == {"bare-skill": ["trigger_phrases", "use_when", "domains/tags"]}


def test_lexical_only_candidate_is_not_high_confidence(tmp_path: Path) -> None:
    meta = _skill(
        tmp_path / "skills" / "note-helper" / "SKILL.md",
        "---\n"
        "name: note-helper\n"
        "description: Notes\n"
        "---\n"
        "# Notes\n"
        "PDF Obsidian Markdown Downloads profile report contract invoice archive search memo.\n",
    )

    candidate = SkillRouter(
        include_body=True,
        lexical_only_min_score=0.1,
        medium_score=0.1,
        high_score=0.1,
    ).route(
        "PDF Obsidian Markdown Downloads report contract invoice search",
        [meta],
    )[0]

    assert candidate.confidence == "medium"


def test_dense_scores_by_name_ignore_duplicate_skill_names(tmp_path: Path) -> None:
    personal = SkillMetadata(
        name="shared-skill",
        path=tmp_path / "animas" / "mei" / "skills" / "shared-skill" / "SKILL.md",
    )
    common = SkillMetadata(
        name="shared-skill",
        path=tmp_path / "common_skills" / "shared-skill" / "SKILL.md",
        is_common=True,
    )

    candidates = SkillRouter(include_body=False).route(
        "unrelated query",
        [personal, common],
        dense_scores={"shared-skill": 10.0},
    )

    assert candidates == []


def test_evaluate_reports_no_skill_precision(monkeypatch, tmp_path: Path) -> None:
    fixture = tmp_path / "cases.yaml"
    fixture.write_text(
        "cases:\n"
        "  - id: actionable_miss\n"
        "    query: 給与計算して\n"
        "    expected_any: [gmail-tool]\n"
        "  - id: no_skill\n"
        "    query: おはよう\n"
        "    no_skill: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evaluate_skill_router,
        "_load_skills",
        lambda _anima: [SkillMetadata(name="gmail-tool", trigger_phrases=["Gmail"])],
    )

    result = evaluate_skill_router.evaluate(
        anima="mei",
        fixture=fixture,
        top_k=3,
        min_score=1.15,
        include_body=False,
    )

    assert result["summary"]["predicted_no_skill"] == 2
    assert result["summary"]["predicted_no_skill_correct"] == 1
    assert result["summary"]["no_skill_precision"] == 0.5
    assert result["cases"][0]["expected_metadata_gaps"]["gmail-tool"] == [
        "use_when",
        "domains/tags",
        "no_matching_signal",
    ]
