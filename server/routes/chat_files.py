from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""PDF / CSV document attachment validation and persistence."""

import base64
import binascii
import re
import uuid
from pathlib import Path

from fastapi import HTTPException

from core.i18n import t
from core.time_utils import now_local
from server.routes.chat_models import MAX_FILE_PAYLOAD_SIZE, MAX_FILE_SIZE, FileAttachment

SUPPORTED_FILE_TYPES = {
    "application/pdf": ".pdf",
    "text/csv": ".csv",
    "application/csv": ".csv",
    "application/vnd.ms-excel": ".csv",
}


def _validate_files(files: list[FileAttachment]) -> str | None:
    """Validate PDF/CSV attachments without trusting client MIME or names.

    Returns an error message string on failure, ``None`` on success.
    Raises ``HTTPException(400)`` for zero-length decoded payloads so the
    caller never silently stores an empty file.
    """
    if not files:
        return None
    if sum(len(item.data) for item in files) > MAX_FILE_PAYLOAD_SIZE:
        return t("chat.file_payload_too_large")
    for item in files:
        extension = Path(item.name).suffix.lower()
        expected_extension = SUPPORTED_FILE_TYPES.get(item.media_type)
        if expected_extension is None or extension != expected_extension:
            return t("chat.unsupported_file_format")
        try:
            decoded = base64.b64decode(item.data, validate=True)
        except (binascii.Error, ValueError):
            return t("chat.invalid_file_data")
        if not decoded:
            raise HTTPException(status_code=400, detail=t("chat.invalid_file_data"))
        if len(decoded) > MAX_FILE_SIZE:
            return t("chat.file_too_large")
        # Magic-number validation: PDF must start with %PDF-
        if extension == ".pdf" and not decoded.startswith(b"%PDF-"):
            return t("chat.invalid_file_data")
        # CSV: reject NUL bytes and require valid UTF-8
        if extension == ".csv":
            if b"\x00" in decoded:
                return t("chat.invalid_file_data")
            try:
                decoded.decode("utf-8-sig")
            except UnicodeDecodeError:
                return t("chat.csv_encoding_invalid")
    return None


def save_files(anima_name: str, files: list[FileAttachment]) -> list[str]:
    """Save validated document attachments with controlled names and suffixes.

    Returns a list of relative paths like ``attachments/20260924_120000_<uuid>_0_report.pdf``.
    """
    if not files:
        return []
    from core.paths import get_data_dir

    attachments_dir = get_data_dir() / "animas" / anima_name / "attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now_local().strftime("%Y%m%d_%H%M%S")
    paths: list[str] = []
    for index, item in enumerate(files):
        suffix = SUPPORTED_FILE_TYPES[item.media_type]
        raw_stem = Path(item.name).stem
        # Sanitize filename: alphanumeric, dots, hyphens, underscores only
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_stem).strip("._")[:80] or "file"
        unique = uuid.uuid4().hex[:12]
        filename = f"{timestamp}_{unique}_{index}_{safe_stem}{suffix}"
        destination = attachments_dir / filename
        destination.write_bytes(base64.b64decode(item.data, validate=True))
        paths.append(f"attachments/{filename}")
    return paths
