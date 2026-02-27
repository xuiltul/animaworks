from __future__ import annotations

import json
import re
from typing import Any

_IMAGE_URL_RE = re.compile(r"https://[^\s\"'<>]+", re.IGNORECASE)
_IMAGE_PATH_RE = re.compile(
    r"(?:assets|attachments)/[A-Za-z0-9._/\-]+\.(?:png|jpe?g|gif|webp)",
    re.IGNORECASE,
)
_IMAGE_EXT_RE = re.compile(r"\.(?:png|jpe?g|gif|webp)(?:$|\?)", re.IGNORECASE)
_MAX_ARTIFACTS_PER_RESPONSE = 5


def extract_image_artifacts_from_tool_records(
    tool_call_records: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """Extract normalized image artifacts from tool call records."""
    if not tool_call_records:
        return []

    artifacts: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def _append(*, tool_name: str, path: str = "", url: str = "", source: str) -> None:
        if len(artifacts) >= _MAX_ARTIFACTS_PER_RESPONSE:
            return
        clean_path = path.strip()
        clean_url = url.strip()
        if not clean_path and not clean_url:
            return
        trust = "trusted" if source == "generated" else "untrusted"
        key = (source, clean_path, clean_url)
        if key in seen:
            return
        seen.add(key)
        item: dict[str, str] = {
            "type": "image",
            "source": source,
            "trust": trust,
            "provider": tool_name or "unknown",
        }
        if clean_path:
            item["path"] = clean_path
        if clean_url:
            item["url"] = clean_url
        artifacts.append(item)

    def _handle_string(text: str, tool_name: str) -> None:
        if not text:
            return
        for m in _IMAGE_PATH_RE.finditer(text):
            _append(tool_name=tool_name, path=m.group(0), source="generated")
        for m in _IMAGE_URL_RE.finditer(text):
            url = m.group(0).rstrip(").,")
            if _IMAGE_EXT_RE.search(url):
                _append(tool_name=tool_name, url=url, source="searched")

    def _walk(value: Any, tool_name: str) -> None:
        if len(artifacts) >= _MAX_ARTIFACTS_PER_RESPONSE:
            return
        if isinstance(value, dict):
            for key, val in value.items():
                key_l = str(key).lower()
                if isinstance(val, str):
                    if key_l in {"path", "file", "filepath", "asset_path"}:
                        if _IMAGE_PATH_RE.search(val):
                            _append(tool_name=tool_name, path=val, source="generated")
                    elif key_l in {"url", "image_url", "thumbnail", "src"}:
                        if val.startswith("https://") and _IMAGE_EXT_RE.search(val):
                            _append(tool_name=tool_name, url=val, source="searched")
                    _handle_string(val, tool_name)
                else:
                    _walk(val, tool_name)
            return
        if isinstance(value, list):
            for v in value:
                _walk(v, tool_name)
            return
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                    _walk(parsed, tool_name)
                except Exception:
                    pass
            _handle_string(value, tool_name)

    for record in tool_call_records:
        tool_name = str(record.get("tool_name", ""))
        result_summary = record.get("result_summary", "")
        _walk(result_summary, tool_name)
        if tool_name == "image_gen":
            for m in _IMAGE_PATH_RE.finditer(str(result_summary)):
                _append(tool_name=tool_name, path=m.group(0), source="generated")

    return artifacts

