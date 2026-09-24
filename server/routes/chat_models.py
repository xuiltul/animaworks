from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from pydantic import BaseModel, Field

from core.schemas import ImageData

MAX_CHAT_MESSAGE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_PAYLOAD_SIZE = 20 * 1024 * 1024  # 20MB total base64
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15MB decoded per file
MAX_FILE_PAYLOAD_SIZE = 32 * 1024 * 1024  # 32MB total base64

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MIME_TO_EXT = {
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}


class ImageAttachment(BaseModel):
    """A single base64-encoded image attachment."""

    data: str  # Base64 encoded string (no data: prefix)
    media_type: str  # "image/jpeg", "image/png", "image/gif", "image/webp"


class FileAttachment(BaseModel):
    """A single base64-encoded document attachment."""

    data: str
    media_type: str
    name: str


def _to_image_data(attachments: list[ImageAttachment]) -> list[ImageData]:
    """Convert API-layer ImageAttachment list to core-layer ImageData list."""
    return [{"data": img.data, "media_type": img.media_type} for img in attachments]


class ChatRequest(BaseModel):
    message: str
    from_person: str = "human"
    intent: str = ""
    images: list[ImageAttachment] = []
    files: list[FileAttachment] = Field(default_factory=list, max_length=10)
    resume: str | None = None
    last_event_id: str | None = None
    thread_id: str = "default"
    model: str | None = None


class ChatResponse(BaseModel):
    response: str
    anima: str
    images: list[dict[str, Any]] = []
