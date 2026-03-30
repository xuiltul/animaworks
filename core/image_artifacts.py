from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)

_IMAGE_PATH_RE = re.compile(
    r"(?:assets|attachments)/[A-Za-z0-9._/\-]+\.(?:png|jpe?g|gif|webp)",
    re.IGNORECASE,
)
_IMAGE_EXT_RE = re.compile(r"\.(?:png|jpe?g|gif|webp)(?:$|\?)", re.IGNORECASE)

_LOCAL_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

_MD_IMAGE_RE = re.compile(
    r"!\[([^\]]*)\]\(((?:file://)?/[^)\s]+)\)",
)

_MAX_LOCAL_IMAGE_SIZE = 50 * 1024 * 1024  # 50 MB
_MAX_ARTIFACTS_PER_RESPONSE = 5
_ALLOWED_SEARCHED_IMAGE_HOSTS = {
    "cdn.search.brave.com",
    "images.unsplash.com",
    "images.pexels.com",
    "upload.wikimedia.org",
}
_PATH_KEYS = {"path", "file", "filepath", "asset_path"}
_URL_KEYS = {"url", "image_url", "thumbnail", "src"}


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

    def _is_allowed_searched_url(value: str) -> bool:
        if not value.startswith("https://"):
            return False
        if not _IMAGE_EXT_RE.search(value):
            return False
        parsed = urlparse(value)
        host = (parsed.hostname or "").lower()
        if not host:
            return False
        return any(host == d or host.endswith(f".{d}") for d in _ALLOWED_SEARCHED_IMAGE_HOSTS)

    def _walk(value: Any, tool_name: str) -> None:
        if len(artifacts) >= _MAX_ARTIFACTS_PER_RESPONSE:
            return
        if isinstance(value, dict):
            for key, val in value.items():
                key_l = str(key).lower()
                if isinstance(val, str):
                    if key_l in _PATH_KEYS:
                        if _IMAGE_PATH_RE.search(val):
                            _append(tool_name=tool_name, path=val, source="generated")
                    elif key_l in _URL_KEYS:
                        if _is_allowed_searched_url(val):
                            _append(tool_name=tool_name, url=val, source="searched")
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
                except json.JSONDecodeError:
                    return

    for record in tool_call_records:
        tool_name = str(record.get("tool_name", ""))
        result_summary = record.get("result_summary", "")
        _walk(result_summary, tool_name)
        if tool_name == "image_gen":
            for m in _IMAGE_PATH_RE.finditer(str(result_summary)):
                _append(tool_name=tool_name, path=m.group(0), source="generated")

    return artifacts


# ── Resolve local image paths ────────────────────────────


def _extract_local_path(raw: str) -> Path | None:
    """Convert a raw image src (``file:///...`` or ``/abs/path``) to a Path.

    Rejects symlinks to prevent unintended reads of arbitrary files.
    """
    s = raw.strip()
    if s.startswith("file://"):
        s = unquote(s[len("file://") :])
    p = Path(s)
    if not p.is_absolute():
        return None
    if p.suffix.lower() not in _LOCAL_IMAGE_EXTS:
        return None
    if p.is_symlink():
        return None
    return p


def _safe_attachment_name(dest_dir: Path, original_name: str) -> str:
    """Return a unique filename inside *dest_dir*."""
    candidate = original_name
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix
    counter = 1
    while (dest_dir / candidate).exists():
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def resolve_local_image_paths(
    text: str,
    anima_dir: Path,
) -> tuple[str, list[dict[str, str]]]:
    """Rewrite absolute/``file://`` image paths in markdown to ``attachments/``.

    For every ``![alt](file:///abs/path.jpg)`` or ``![alt](/abs/path.jpg)``
    found in *text*:
    1. Copy the file into ``{anima_dir}/attachments/``
    2. Replace the markdown src with ``attachments/{filename}``

    Returns ``(rewritten_text, new_artifact_list)``.
    """
    if "![" not in text:
        return text, []

    attachments_dir = anima_dir / "attachments"
    artifacts: list[dict[str, str]] = []
    copied: dict[str, str] = {}

    def _replace(m: re.Match) -> str:
        if len(artifacts) >= _MAX_ARTIFACTS_PER_RESPONSE:
            return m.group(0)
        alt = m.group(1)
        raw_src = m.group(2)
        local_path = _extract_local_path(raw_src)
        if local_path is None:
            return m.group(0)
        if not local_path.is_file():
            logger.debug("Local image not found, skipping: %s", local_path)
            return m.group(0)
        try:
            if local_path.stat().st_size > _MAX_LOCAL_IMAGE_SIZE:
                logger.warning("Local image too large, skipping: %s", local_path)
                return m.group(0)
        except OSError:
            return m.group(0)

        cache_key = str(local_path)
        if cache_key in copied:
            rel = copied[cache_key]
        else:
            attachments_dir.mkdir(parents=True, exist_ok=True)
            dest_name = _safe_attachment_name(attachments_dir, local_path.name)
            dest = attachments_dir / dest_name
            try:
                shutil.copy2(local_path, dest)
            except OSError:
                logger.warning("Failed to copy local image: %s", local_path, exc_info=True)
                return m.group(0)
            rel = f"attachments/{dest_name}"
            copied[cache_key] = rel
            artifacts.append(
                {
                    "type": "image",
                    "source": "local_file",
                    "trust": "trusted",
                    "provider": "filesystem",
                    "path": rel,
                }
            )
        return f"![{alt}]({rel})"

    rewritten = _MD_IMAGE_RE.sub(_replace, text)
    return rewritten, artifacts
