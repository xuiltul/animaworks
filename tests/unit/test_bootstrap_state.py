from __future__ import annotations

import json
from pathlib import Path

from core.anima.bootstrap_state import (
    STATE_COMPLETED,
    STATE_NEEDS_REPAIR,
    STATE_PENDING_USER_INPUT,
    finalize_bootstrap_run,
    get_bootstrap_status,
    repair_bootstrap_complete,
    repair_bootstrap_fresh,
    repair_bootstrap_retry,
)


def _make_anima_dir(tmp_path: Path, name: str = "midori") -> Path:
    anima_dir = tmp_path / "animas" / name
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "shortterm" / "chat").mkdir(parents=True)
    (anima_dir / "IDENTITY.md").write_text("# Identity\n\n未定義\n", encoding="utf-8")
    (anima_dir / "AGENTS.md").write_text("# Injection\n\n未定義\n", encoding="utf-8")
    (anima_dir / "BOOTSTRAP.md").write_text("# Bootstrap\n", encoding="utf-8")
    (anima_dir / "status.json").write_text('{"enabled": true}\n', encoding="utf-8")
    return anima_dir


def test_blank_anima_waits_for_user_input(tmp_path: Path) -> None:
    anima_dir = _make_anima_dir(tmp_path)

    status = get_bootstrap_status(anima_dir)

    assert status["state"] == STATE_PENDING_USER_INPUT
    assert status["needs_user_input"] is True
    assert status["needs_background_bootstrap"] is False
    assert status["needs_repair"] is False


def test_character_sheet_allows_background_bootstrap(tmp_path: Path) -> None:
    anima_dir = _make_anima_dir(tmp_path)
    (anima_dir / "character_sheet.md").write_text("# Character Sheet\n", encoding="utf-8")

    status = get_bootstrap_status(anima_dir)

    assert status["state"] == STATE_PENDING_USER_INPUT
    assert status["needs_user_input"] is False
    assert status["needs_background_bootstrap"] is True


def test_auto_resolved_is_needs_repair(tmp_path: Path) -> None:
    anima_dir = _make_anima_dir(tmp_path)
    (anima_dir / "BOOTSTRAP.md").rename(anima_dir / "BOOTSTRAP.md.auto_resolved")

    status = get_bootstrap_status(anima_dir)

    assert status["state"] == STATE_NEEDS_REPAIR
    assert status["needs_repair"] is True
    assert "unexpected_bootstrap_artifact:BOOTSTRAP.md.auto_resolved" in status["validation_errors"]


def test_finalize_archives_leftover_bootstrap_when_identity_defined(tmp_path: Path) -> None:
    anima_dir = _make_anima_dir(tmp_path)
    (anima_dir / "IDENTITY.md").write_text("# Midori\n\nDefined identity\n", encoding="utf-8")
    (anima_dir / "AGENTS.md").write_text("# Role\n\nDefined role\n", encoding="utf-8")

    status = finalize_bootstrap_run(anima_dir)

    assert status["state"] == STATE_COMPLETED
    assert not (anima_dir / "BOOTSTRAP.md").exists()
    archived = list((anima_dir / "state" / "bootstrap_archive").glob("bootstrap-*.md"))
    assert len(archived) == 1


def test_finalize_keeps_bootstrap_when_identity_incomplete(tmp_path: Path) -> None:
    anima_dir = _make_anima_dir(tmp_path)

    status = finalize_bootstrap_run(anima_dir)

    assert status["state"] == STATE_NEEDS_REPAIR
    assert (anima_dir / "BOOTSTRAP.md").exists()
    assert "identity_undefined" in status["validation_errors"]
    assert "injection_undefined" in status["validation_errors"]


def test_finalize_archives_leftover_character_sheet_when_identity_defined(tmp_path: Path) -> None:
    anima_dir = _make_anima_dir(tmp_path)
    (anima_dir / "IDENTITY.md").write_text("# Midori\n\nDefined identity\n", encoding="utf-8")
    (anima_dir / "AGENTS.md").write_text("# Role\n\nDefined role\n", encoding="utf-8")
    (anima_dir / "character_sheet.md").write_text("# Character Sheet\n", encoding="utf-8")

    status = finalize_bootstrap_run(anima_dir)

    assert status["state"] == STATE_COMPLETED
    assert not (anima_dir / "character_sheet.md").exists()
    assert list((anima_dir / "state" / "bootstrap_archive").glob("character_sheet-*.md"))


def test_finalize_keeps_character_sheet_when_identity_incomplete(tmp_path: Path) -> None:
    anima_dir = _make_anima_dir(tmp_path)
    (anima_dir / "character_sheet.md").write_text("# Character Sheet\n", encoding="utf-8")

    status = finalize_bootstrap_run(anima_dir)

    assert status["state"] == STATE_NEEDS_REPAIR
    assert (anima_dir / "character_sheet.md").exists()
    assert "character_sheet_unprocessed" in status["validation_errors"]


def test_interactive_bootstrap_completion_overrides_pending_state(tmp_path: Path) -> None:
    anima_dir = _make_anima_dir(tmp_path)
    from core.anima.bootstrap_state import initialize_bootstrap_state

    initialize_bootstrap_state(anima_dir)
    (anima_dir / "IDENTITY.md").write_text("# Midori\n\nDefined identity\n", encoding="utf-8")
    (anima_dir / "AGENTS.md").write_text("# Role\n\nDefined role\n", encoding="utf-8")
    (anima_dir / "BOOTSTRAP.md").unlink()

    status = get_bootstrap_status(anima_dir)

    assert status["state"] == STATE_COMPLETED
    assert status["needs_bootstrap"] is False


def test_missing_interactive_bootstrap_prompt_needs_repair(tmp_path: Path) -> None:
    anima_dir = _make_anima_dir(tmp_path)
    from core.anima.bootstrap_state import initialize_bootstrap_state

    initialize_bootstrap_state(anima_dir)
    (anima_dir / "BOOTSTRAP.md").unlink()

    status = get_bootstrap_status(anima_dir)

    assert status["state"] == STATE_NEEDS_REPAIR
    assert "bootstrap_missing" in status["validation_errors"]


def test_retry_restores_artifact_and_cleans_session_state(tmp_path: Path) -> None:
    anima_dir = _make_anima_dir(tmp_path)
    (anima_dir / "BOOTSTRAP.md").rename(anima_dir / "BOOTSTRAP.md.failed")
    (anima_dir / "shortterm" / "chat" / "codex_thread_id.txt").write_text("thread", encoding="utf-8")
    (anima_dir / "state" / "current_session_chat.json").write_text('{"session_id": "sid"}', encoding="utf-8")
    retries = tmp_path / "animas" / ".bootstrap_retries.json"
    retries.write_text(json.dumps({"midori": 3}), encoding="utf-8")

    status = repair_bootstrap_retry(anima_dir, retry_counts_file=retries)

    assert (anima_dir / "BOOTSTRAP.md").exists()
    assert not (anima_dir / "BOOTSTRAP.md.failed").exists()
    assert not (anima_dir / "shortterm" / "chat" / "codex_thread_id.txt").exists()
    assert not (anima_dir / "state" / "current_session_chat.json").exists()
    assert json.loads(retries.read_text(encoding="utf-8")) == {}
    assert status["state"] == STATE_PENDING_USER_INPUT


def test_complete_archives_artifacts_and_preserves_runtime_session(tmp_path: Path) -> None:
    anima_dir = _make_anima_dir(tmp_path)
    (anima_dir / "IDENTITY.md").write_text("# Midori\n\nDefined identity\n", encoding="utf-8")
    (anima_dir / "AGENTS.md").write_text("# Role\n\nDefined role\n", encoding="utf-8")
    (anima_dir / "BOOTSTRAP.md").rename(anima_dir / "BOOTSTRAP.md.failed")
    (anima_dir / "character_sheet.md").write_text("# Character Sheet\n", encoding="utf-8")
    (anima_dir / "shortterm" / "chat" / "codex_thread_id.txt").write_text("thread", encoding="utf-8")
    pending_dir = anima_dir / "state" / "pending"
    pending_dir.mkdir(parents=True)
    (pending_dir / "bootstrap-midori.json").write_text('{"task_id": "bootstrap-midori"}', encoding="utf-8")
    retries = tmp_path / "animas" / ".bootstrap_retries.json"
    retries.write_text(json.dumps({"midori": 3}), encoding="utf-8")

    status = repair_bootstrap_complete(anima_dir, retry_counts_file=retries)

    assert status["state"] == STATE_COMPLETED
    assert not (anima_dir / "BOOTSTRAP.md.failed").exists()
    assert not (anima_dir / "character_sheet.md").exists()
    assert (pending_dir / "bootstrap-midori.json").exists()
    assert (anima_dir / "shortterm" / "chat" / "codex_thread_id.txt").exists()
    assert json.loads(retries.read_text(encoding="utf-8")) == {}
    archived = list((anima_dir / "state" / "bootstrap_archive").glob("BOOTSTRAP.md.failed.*"))
    assert len(archived) == 1
    archived_character_sheets = list((anima_dir / "state" / "bootstrap_archive").glob("character_sheet-*.md"))
    assert len(archived_character_sheets) == 1


def test_fresh_recreates_blank_and_preserves_model_settings(tmp_path: Path) -> None:
    animas_dir = tmp_path / "animas"
    anima_dir = _make_anima_dir(tmp_path)
    (anima_dir / "status.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "model": "azure/gpt-4.1-mini",
                "credential": "codex-azure",
                "execution_mode": "C",
                "background_model": "azure/gpt-4.1-mini",
                "background_credential": "codex-azure",
            }
        ),
        encoding="utf-8",
    )
    (anima_dir / "episodes").mkdir()
    (anima_dir / "episodes" / "old.md").write_text("old", encoding="utf-8")

    new_dir, archive_path = repair_bootstrap_fresh(animas_dir, "midori", archive_root=tmp_path / "archive")

    assert archive_path.exists()
    assert new_dir.exists()
    assert not (new_dir / "episodes" / "old.md").exists()
    status = json.loads((new_dir / "status.json").read_text(encoding="utf-8"))
    assert status["model"] == "azure/gpt-4.1-mini"
    assert status["credential"] == "codex-azure"
    assert status["execution_mode"] == "C"
    assert status["background_model"] == "azure/gpt-4.1-mini"


def test_interactive_profile_remains_resumable_during_incremental_writes(tmp_path: Path) -> None:
    from core.anima.bootstrap_state import initialize_bootstrap_state

    anima_dir = _make_anima_dir(tmp_path)
    initialize_bootstrap_state(anima_dir)
    (anima_dir / "IDENTITY.md").write_text("# Identity\nA calm manager.", encoding="utf-8")
    status = get_bootstrap_status(anima_dir)
    assert status["state"] == STATE_PENDING_USER_INPUT
    assert not status["needs_repair"]
    (anima_dir / "AGENTS.md").write_text("# Role\nCoordinate work.", encoding="utf-8")
    # Graphics may still be running, or the Anima may ask a follow-up question.
    status = get_bootstrap_status(anima_dir)
    assert status["mode"] == "interactive"
    assert status["needs_user_input"]
    assert not status["needs_repair"]
    (anima_dir / "BOOTSTRAP.md").unlink()
    assert get_bootstrap_status(anima_dir)["state"] == STATE_COMPLETED
