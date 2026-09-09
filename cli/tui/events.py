# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StreamStarted:
    response_id: str


@dataclass
class ChatDelta:
    text: str = ""


@dataclass
class ThinkingDelta:
    text: str = ""


@dataclass
class ThinkingStarted:
    pass


@dataclass
class ThinkingEnded:
    pass


@dataclass
class ToolStarted:
    tool_name: str
    tool_id: str
    detail: str = ""


@dataclass
class ToolDetail:
    tool_id: str
    tool_name: str
    detail: str = ""


@dataclass
class ToolEnded:
    tool_id: str
    tool_name: str
    result_summary: str | None = None
    input_summary: str | None = None
    is_error: bool = False


@dataclass
class ChainStarted:
    chain: str = ""


@dataclass
class StreamDone:
    summary: str = ""
    emotion: str = ""
    images: list = field(default_factory=list)


@dataclass
class StreamError:
    message: str
    code: str | None = None


@dataclass
class StatusChanged:
    status: str
    thread_id: str | None = None
