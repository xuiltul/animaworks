from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Common helpers for Hermes/OpenClaw migration importers."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from core.time_utils import now_iso

_SECRET_PATTERNS = [
    re.compile(
        r"(?i)['\"]?\b(api[_-]?key|token|secret|password)\b['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9._~+\-/=]{8,})['\"]?"
    ),
    re.compile(r"\b(sk-[A-Za-z0-9]{8,})\b"),
]


def safe_anima_name(value: str | None) -> str:
    name = str(value or "").strip()
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts or "/" in name or "\\" in name:
        raise ValueError("--target-anima is required and must be a simple anima name")
    return name


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            rel = str(child.relative_to(path)).replace("\\", "/")
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            h.update(child.read_bytes())
            h.update(b"\0")
        return h.hexdigest()
    h.update(path.read_bytes())
    return h.hexdigest()


def source_fingerprint(source_type: str, source_path: Path, target_path: str, *, extra: str = "") -> str:
    resolved = str(source_path.expanduser().resolve(strict=False))
    content_hash = sha256_path(source_path) if source_path.exists() else ""
    try:
        mtime = str(source_path.stat().st_mtime_ns)
    except OSError:
        mtime = ""
    raw = "\n".join([source_type, resolved, content_hash, mtime, target_path, extra])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_import_lock(path: Path) -> set[str]:
    if not path.exists():
        return set()
    fingerprints: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        fp = str(raw.get("source_fingerprint") or "")
        if fp:
            fingerprints.add(fp)
    return fingerprints


def append_import_lock(
    path: Path, *, fingerprint: str, source_system: str, action: str, target_path: str, batch_id: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": now_iso(),
        "source_system": source_system,
        "action": action,
        "target_path": target_path,
        "source_fingerprint": fingerprint,
        "import_batch_id": batch_id,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def write_backup_manifest(path: Path, *, data_dir: Path, targets: list[Path], batch_id: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for target in targets:
        item: dict[str, Any] = {
            "path": rel_to(target, data_dir),
            "exists": target.exists(),
        }
        if target.is_file():
            item["sha256"] = sha256_path(target)
            item["size_bytes"] = target.stat().st_size
        elif target.is_dir():
            item["sha256"] = sha256_path(target)
            item["type"] = "directory"
        items.append(item)
    manifest = {
        "generated_at": now_iso(),
        "batch_id": batch_id,
        "items": items,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def rel_to(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text_once(path: Path, content: str, *, replace: bool) -> bool:
    if path.exists() and not replace:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def detect_redacted_credentials(root: Path, *, max_files: int = 200) -> list[str]:
    warnings: list[str] = []
    scanned = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if scanned >= max_files:
            break
        if path.stat().st_size > 512_000:
            continue
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in _SECRET_PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(match.lastindex or 1)
                warnings.append(f"credential_detected:{path.name}:{_redact(value)}")
    return sorted(set(warnings))


def redact_credentials(text: str) -> str:
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(_redact_match, redacted)
    return redacted


def provenance_header(source_system: str, source_path: Path, batch_id: str) -> str:
    return (
        f"<!-- migrated_from: {source_system} -->\n"
        f"<!-- source_path: {source_path} -->\n"
        f"<!-- import_batch_id: {batch_id} -->\n\n"
    )


def _redact(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}***{value[-2:]}"


def _redact_match(match: re.Match[str]) -> str:
    if match.lastindex and match.lastindex >= 2:
        return f"{match.group(1)}: {_redact(match.group(2))}"
    return _redact(match.group(1))
