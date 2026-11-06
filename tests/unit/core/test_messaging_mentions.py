from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
    sockets_dir = tmp_path / "run" / "sockets"
    sockets_dir.mkdir(parents=True)
    for name in ("team-lead", "bob", "unknown"):
        (sockets_dir / f"{name}.sock").touch()
    messenger = MagicMock()
    messenger.shared_dir = tmp_path / "shared"
    boundary = SimpleNamespace(cross_company=False)

    with (
        patch("core.paths.get_data_dir", return_value=tmp_path),
        patch("core.messaging.messenger.is_channel_member", return_value=True),
        patch("core.org.company.check_company_boundary", return_value=boundary),
    ):
        fanout_board_mentions(
            messenger,
            "alice",
            "general",
            "@TEAM-LEAD。 @bob- @unknown_suffix",
            animas_dir=tmp_path,
        )

    assert {call.kwargs["to"] for call in messenger.send.call_args_list} == {"team-lead", "bob"}
