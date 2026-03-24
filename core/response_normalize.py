from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _extract_json_candidate(raw_text: str) -> str | None:
    text = (raw_text or "").strip()
    if not text:
        return None
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1].strip()
    return None


def _load_json_object(raw_text: str) -> dict[str, Any] | None:
    candidate = _extract_json_candidate(raw_text)
    if not candidate:
        return None
    try:
        parsed = json.loads(candidate)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None


def _extract_tool_call_text(payload: Mapping[str, Any]) -> str | None:
    if "function" in payload:
        nested = _coerce_mapping(payload.get("function"))
        if nested:
            return _extract_tool_call_text(nested)
    name = str(payload.get("name") or "").strip()
    args = _coerce_mapping(payload.get("arguments")) or {}
    if name == "post_channel":
        text = args.get("text")
        return str(text).strip() if text else None
    if name == "send_message":
        text = args.get("content")
        return str(text).strip() if text else None
    return None


def _format_status(status: str) -> str:
    return {
        "idle": "待機中",
        "busy": "処理中",
        "running": "実行中",
        "ok": "正常",
    }.get(status, status)


def _extract_status_summary(payload: Mapping[str, Any]) -> str | None:
    status_keys = {
        "status",
        "file_access_verified",
        "paths_verified",
        "access_method",
        "last_verified_file",
        "note",
    }
    if not any(key in payload for key in status_keys):
        return None
    lines: list[str] = []
    status = payload.get("status")
    if isinstance(status, str) and status.strip():
        lines.append(f"状態: {_format_status(status.strip())}")
    verified = payload.get("file_access_verified")
    if isinstance(verified, bool):
        lines.append(f"ファイルアクセス確認: {'可能' if verified else '未確認'}")
    access_method = payload.get("access_method")
    if isinstance(access_method, str) and access_method.strip():
        lines.append(f"確認方法: {access_method.strip()}")
    last_verified_file = payload.get("last_verified_file")
    if isinstance(last_verified_file, str) and last_verified_file.strip():
        lines.append(f"最終確認ファイル: {last_verified_file.strip()}")
    paths_verified = payload.get("paths_verified")
    if isinstance(paths_verified, list) and paths_verified:
        lines.append("確認済みパス:")
        lines.extend(f"- {path}" for path in paths_verified if isinstance(path, str) and path.strip())
    note = payload.get("note")
    if isinstance(note, str) and note.strip():
        lines.append(f"補足: {note.strip()}")
    if not lines:
        return None
    return "\n".join(lines)


def _extract_error_summary(payload: Mapping[str, Any]) -> str | None:
    status = payload.get("status")
    error_type = payload.get("error_type")
    message = payload.get("message")
    suggestion = payload.get("suggestion")
    if status != "error" and not error_type and not message:
        return None
    lines: list[str] = []
    if error_type:
        lines.append(f"エラー種別: {error_type}")
    if isinstance(message, str) and message.strip():
        lines.append(message.strip())
    elif status == "error":
        lines.append("エラーが発生しました。")
    if isinstance(suggestion, str) and suggestion.strip():
        lines.append(f"対処候補: {suggestion.strip()}")
    return "\n".join(lines) if lines else None


def normalize_user_facing_response_text(raw_text: str) -> str:
    payload = _load_json_object(raw_text)
    if not payload:
        return raw_text
    tool_text = _extract_tool_call_text(payload)
    if tool_text:
        return tool_text
    error_summary = _extract_error_summary(payload)
    if error_summary:
        return error_summary
    status_summary = _extract_status_summary(payload)
    if status_summary:
        return status_summary
    return raw_text
