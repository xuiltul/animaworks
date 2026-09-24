"""Unit tests for server/routes/chat_files.py — PDF/CSV attachment validation."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import os

import pytest
from fastapi import HTTPException

from server.routes.chat_files import _validate_files, save_files, SUPPORTED_FILE_TYPES
from server.routes.chat_models import FileAttachment, MAX_FILE_SIZE, MAX_FILE_PAYLOAD_SIZE


def _make_file(content: bytes, media_type: str, name: str) -> FileAttachment:
    return FileAttachment(
        data=base64.b64encode(content).decode(),
        media_type=media_type,
        name=name,
    )


class TestValidateFiles:
    """Tests for _validate_files security validation."""

    def test_valid_pdf(self):
        pdf = b"%PDF-1.4\n" + b"x" * 100
        assert _validate_files([_make_file(pdf, "application/pdf", "report.pdf")]) is None

    def test_valid_csv(self):
        csv = "col1,col2\nval1,val2\n".encode("utf-8")
        assert _validate_files([_make_file(csv, "text/csv", "data.csv")]) is None

    def test_valid_csv_alt_mime(self):
        csv = "col1,col2\nval1,val2\n".encode("utf-8")
        assert _validate_files([_make_file(csv, "application/csv", "data.csv")]) is None

    def test_valid_csv_excel_mime(self):
        csv = "col1,col2\nval1,val2\n".encode("utf-8")
        assert _validate_files([_make_file(csv, "application/vnd.ms-excel", "data.csv")]) is None

    def test_empty_list(self):
        assert _validate_files([]) is None

    def test_none(self):
        assert _validate_files(None) is None

    def test_rejects_unsupported_mime(self):
        content = b"not a pdf"
        result = _validate_files([_make_file(content, "application/json", "data.json")])
        assert result is not None  # Returns error message

    def test_rejects_extension_mismatch(self):
        """Extension must match expected for the MIME type."""
        pdf = b"%PDF-1.4\n" + b"x" * 100
        result = _validate_files([_make_file(pdf, "application/pdf", "report.csv")])
        assert result is not None

    def test_rejects_fake_pdf_magic(self):
        """Reject files that claim PDF but lack magic bytes."""
        content = b"This is not a PDF\n"
        result = _validate_files([_make_file(content, "application/pdf", "report.pdf")])
        assert result is not None

    def test_rejects_csv_with_null_bytes(self):
        csv = b"col1,col2\x00\nval1,val2\n"
        result = _validate_files([_make_file(csv, "text/csv", "data.csv")])
        assert result is not None

    def test_rejects_csv_non_utf8(self):
        csv = "日本語".encode("shift_jis")
        result = _validate_files([_make_file(csv, "text/csv", "data.csv")])
        assert result is not None

    def test_rejects_empty_file(self):
        """Empty decoded content should raise HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            _validate_files([_make_file(b"", "application/pdf", "empty.pdf")])
        assert exc_info.value.status_code == 400

    def test_rejects_oversized_file(self):
        """Files larger than MAX_FILE_SIZE should be rejected."""
        content = b"%PDF-1.4\n" + b"x" * (MAX_FILE_SIZE + 1)
        result = _validate_files([_make_file(content, "application/pdf", "huge.pdf")])
        assert result is not None

    def test_rejects_oversized_payload(self):
        """Total base64 payload exceeding MAX_FILE_PAYLOAD_SIZE should be rejected."""
        # Create files whose combined base64 exceeds MAX_FILE_PAYLOAD_SIZE
        content = b"%PDF-1.4\n" + b"x" * (14 * 1024 * 1024)
        file1 = _make_file(content, "application/pdf", "big1.pdf")
        file2 = _make_file(content, "application/pdf", "big2.pdf")
        file3 = _make_file(content, "application/pdf", "big3.pdf")
        result = _validate_files([file1, file2, file3])
        assert result is not None


class TestSaveFiles:
    """Tests for save_files path traversal prevention and correct saving."""

    def test_save_pdf(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "core.paths.get_data_dir",
            lambda: tmp_path,
        )
        pdf = b"%PDF-1.4\n" + b"x" * 100
        paths = save_files("alice", [_make_file(pdf, "application/pdf", "report.pdf")])
        assert len(paths) == 1
        assert paths[0].startswith("attachments/")
        assert paths[0].endswith(".pdf")
        # Verify file was saved
        saved = (tmp_path / "animas" / "alice" / paths[0]).read_bytes()
        assert saved == pdf

    def test_save_csv(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "core.paths.get_data_dir",
            lambda: tmp_path,
        )
        csv = "col1,col2\nval1,val2\n".encode("utf-8")
        paths = save_files("alice", [_make_file(csv, "text/csv", "data.csv")])
        assert len(paths) == 1
        assert paths[0].endswith(".csv")

    def test_sanitizes_filename(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "core.paths.get_data_dir",
            lambda: tmp_path,
        )
        pdf = b"%PDF-1.4\n" + b"x" * 100
        paths = save_files("alice", [_make_file(pdf, "application/pdf", "../../../etc/passwd.pdf")])
        assert len(paths) == 1
        filename = os.path.basename(paths[0])
        assert ".." not in filename
        assert "/" not in filename

    def test_empty_list(self, tmp_path):
        assert save_files("alice", []) == []


class TestSizeConstants:
    """Verify the raised size limits."""

    def test_max_file_size(self):
        assert MAX_FILE_SIZE == 15 * 1024 * 1024  # 15MB

    def test_max_file_payload_size(self):
        assert MAX_FILE_PAYLOAD_SIZE == 32 * 1024 * 1024  # 32MB

    def test_supported_file_types(self):
        assert "application/pdf" in SUPPORTED_FILE_TYPES
        assert "text/csv" in SUPPORTED_FILE_TYPES
        assert "application/csv" in SUPPORTED_FILE_TYPES
        assert "application/vnd.ms-excel" in SUPPORTED_FILE_TYPES
