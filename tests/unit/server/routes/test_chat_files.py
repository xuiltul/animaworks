"""Validation and storage tests for chat file attachments."""

from __future__ import annotations

import base64
from datetime import datetime

import pytest

from core.anima.messaging import _chat_received_meta
from server.routes import chat_files
from server.routes.chat_models import FileAttachment


def _file(name: str = "report.pdf", content: bytes = b"abc", media_type: str = "application/pdf") -> FileAttachment:
    return FileAttachment(name=name, media_type=media_type, data=base64.b64encode(content).decode("ascii"))


def test_file_attachment_limits_are_configured_as_planned() -> None:
    assert chat_files.MAX_CHAT_FILE_SIZE == 25 * 1024 * 1024
    assert chat_files.MAX_CHAT_FILE_COUNT == 10
    assert chat_files.MAX_CHAT_FILES_SIZE == 50 * 1024 * 1024


def test_validate_files_accepts_any_media_type_and_valid_base64() -> None:
    assert chat_files._validate_files([_file(media_type="image/svg+xml")]) is None
    assert chat_files._validate_files([_file(media_type="application/x-custom")]) is None


def test_validate_files_rejects_too_many_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_files, "MAX_CHAT_FILE_COUNT", 2)
    result = chat_files._validate_files([_file(), _file("two.pdf"), _file("three.pdf")])
    assert result is not None
    assert result[1] == 413
    assert "2" in result[0]


def test_validate_files_rejects_empty_and_malformed_base64() -> None:
    empty = chat_files._validate_files([_file(content=b"")])
    invalid = chat_files._validate_files([FileAttachment(name="bad.bin", media_type="", data="%%%not-base64%%")])
    assert empty is not None and empty[1] == 400
    assert invalid is not None and invalid[1] == 400


def test_validate_files_enforces_decoded_per_file_and_total_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_files, "MAX_CHAT_FILE_SIZE", 4)
    monkeypatch.setattr(chat_files, "MAX_CHAT_FILES_SIZE", 6)

    oversized_file = chat_files._validate_files([_file(content=b"12345")])
    oversized_total = chat_files._validate_files([_file(content=b"1234"), _file("second.txt", b"567")])

    assert oversized_file is not None and oversized_file[1] == 413
    assert oversized_total is not None and oversized_total[1] == 413


def test_sanitize_filename_removes_paths_controls_and_leading_dots() -> None:
    assert chat_files._sanitize_filename("../\\..hidden\x00 report.pdf") == "hidden report.pdf"
    assert chat_files._sanitize_filename("...\x00/\\") == "file"

    safe = chat_files._sanitize_filename("a" * 180 + ".pdf")
    assert len(safe) == 120
    assert safe.endswith(".pdf")


def test_save_files_writes_timestamped_paths_and_returns_metadata(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.paths.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr(chat_files, "now_local", lambda: datetime(2026, 10, 29, 10, 15, 0))

    saved = chat_files.save_files("mei", [_file(name="../estimate.pdf", content=b"PDF")])

    assert saved == [
        {
            "name": "estimate.pdf",
            "path": "attachments/20261029_101500_0_estimate.pdf",
            "size": 3,
            "media_type": "application/pdf",
        }
    ]
    assert (tmp_path / "animas" / "mei" / saved[0]["path"]).read_bytes() == b"PDF"


def test_build_file_message_adds_paths_for_anima_without_changing_empty_file_list() -> None:
    file = {
        "name": "estimate.pdf",
        "path": "attachments/20261029_101500_0_estimate.pdf",
        "size": 1258291,
        "media_type": "application/pdf",
    }
    assert chat_files.build_file_message("Please review", []) == "Please review"
    message = chat_files.build_file_message("Please review", [file])
    assert message.startswith("Please review\n\n")
    assert "estimate.pdf" in message
    assert "1.2 MB" in message or "1.2MB" in message
    assert file["path"] in message


def test_received_activity_metadata_contains_saved_files_when_present() -> None:
    files = [{"name": "brief.txt", "path": "attachments/brief.txt", "size": 12, "media_type": "text/plain"}]

    assert _chat_received_meta("thread-1", files) == {
        "from_type": "human",
        "thread_id": "thread-1",
        "files": files,
    }
    assert _chat_received_meta("default") == {"from_type": "human", "thread_id": "default"}
