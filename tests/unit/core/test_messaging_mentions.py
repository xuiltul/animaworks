from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core.messaging.board import BoardPost, Principal
from core.messaging.board_fanout import fanout_board_mentions
from core.messaging.mentions import extract_mentions


def test_extract_mentions_matches_known_hyphenated_names_case_insensitively() -> None:
    assert extract_mentions("Hi @Team-Lead", {"team-lead"}) == {"team-lead"}


def test_extract_mentions_stops_at_unicode_punctuation() -> None:
    assert extract_mentions("@alice。 @alice、", {"alice"}) == {"alice"}


def test_extract_mentions_strips_trailing_hyphen_and_underscore() -> None:
    assert extract_mentions("@bob- @carol_", {"bob", "carol"}) == {"bob", "carol"}


def test_extract_mentions_ignores_unknown_names_without_partial_match() -> None:
    assert extract_mentions("@alice-team @unknown", {"alice", "bob"}) == set()


def test_extract_mentions_preserves_all_special_case() -> None:
    assert extract_mentions("@ALL", set()) == {"all"}


def test_fanout_uses_complete_known_names_and_punctuation(tmp_path: Path) -> None:
    shared_dir = tmp_path / "shared"
    (shared_dir / "channels").mkdir(parents=True)
    (shared_dir / "channels" / "general.jsonl").touch()
    animas_dir = tmp_path / "animas"
    for name in ("alice", "team-lead", "bob", "unknown"):
        (animas_dir / name).mkdir(parents=True)
        (animas_dir / name / "status.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
    post = BoardPost(
        channel="general",
        ts="2026-01-01T00:00:00+00:00",
        from_person="alice",
        text="@TEAM-LEAD。 @bob- @unknown_suffix",
        source="anima",
        id="p1",
        kind="anima",
        origin="internal",
        principal=Principal(kind="anima", name="alice"),
    )
    boundary = SimpleNamespace(cross_company=False, resolved_via="")

    with (
        patch("core.org.company.check_company_boundary", return_value=boundary),
        patch("core.messaging.messenger.Messenger.send") as send,
    ):
        notified, _ = fanout_board_mentions(shared_dir, post)

    assert set(notified) == {"team-lead", "bob"}
    assert {call.kwargs["to"] for call in send.call_args_list} == {"team-lead", "bob"}
