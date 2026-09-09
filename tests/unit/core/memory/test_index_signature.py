from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from core.memory.rag.index_signature import index_signature_error


@pytest.fixture(autouse=True)
def english_diagnostics(monkeypatch):
    monkeypatch.setattr("core.paths._get_locale", lambda: "en")


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        [],
        {"a.md": {"hash": "abc"}},
        {"embedding_model": "ruri"},
        {"embedding_model": "ruri", "embedding_e5_prefix": "false"},
    ],
)
def test_absent_signature_is_unknown_not_inferred_false(metadata):
    error = index_signature_error(metadata, "ruri", True)
    assert "unknown" in error
    assert "changed" not in error


@pytest.mark.parametrize("enabled", [True, False])
def test_matching_signature_is_compatible(enabled):
    assert index_signature_error({"embedding_model": "ruri", "embedding_e5_prefix": enabled}, "ruri", enabled) is None


def test_actual_mismatch_is_distinct_from_unknown():
    assert "prefix setting changed" in index_signature_error(
        {"embedding_model": "ruri", "embedding_e5_prefix": False},
        "ruri",
        True,
    )
    assert "model changed" in index_signature_error(
        {"embedding_model": "old", "embedding_e5_prefix": True},
        "ruri",
        True,
    )


@pytest.mark.parametrize("raw", ["{broken", "[]", json.dumps({"a.md": {"hash": "abc"}})])
def test_cli_does_not_write_or_claim_compatibility_for_unknown_metadata(tmp_path, raw, caplog):
    from cli.commands.index_cmd import _check_model_change

    path = tmp_path / "index_meta.json"
    path.write_text(raw)
    with (
        patch("core.memory.rag.singleton.get_embedding_model_name", return_value="ruri"),
        patch("core.memory.rag.singleton.get_embedding_e5_prefix_enabled", return_value=True),
    ):
        with pytest.raises(SystemExit):
            _check_model_change(tmp_path, full=False)
        # Explicit repair is still available, but no metadata is stamped by a
        # check alone (even when the operator intends a full rebuild).
        assert _check_model_change(tmp_path, full=True) == "ruri"
    assert "unknown" in caplog.text
    assert "False -> True" not in caplog.text
    assert path.read_text() == raw
