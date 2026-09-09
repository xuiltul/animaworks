# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Session persistence for the TUI.

A session is one JSON file under ``$ANIMAWORKS_TUI_DIR/sessions`` (or
``~/.animaworks/tui/sessions`` when the env var is not set). The TUI may
run on a different machine than the gateway, so the data dir is resolved
from ``ANIMAWORKS_TUI_DIR`` / the home directory rather than
``core.paths.get_data_dir()``.

All I/O functions accept an optional ``base_dir`` so they can be unit
tested against a temporary directory without touching the home dir.
"""

from __future__ import annotations

import json
import os
import random
import string
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

MAX_SESSIONS = 50

_TUI_ENV = "ANIMAWORKS_TUI_DIR"
_HOME_TUI = "~/.animaworks/tui"


def tui_base_dir() -> Path:
    """The TUI data directory (sessions, keybindings, ...)."""
    env = os.environ.get(_TUI_ENV)
    if env:
        return Path(env)
    return Path(os.path.expanduser(_HOME_TUI))


def session_dir_path() -> Path:
    """The directory that stores session files (created if missing, mode 0700)."""
    d = tui_base_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    return d


def keybindings_path() -> Path:
    """The keybindings configuration file."""
    return tui_base_dir() / "keybindings.json"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class SessionInfo:
    """Everything needed to restore/rejoin a TUI session."""

    session_id: str
    created_at: str
    updated_at: str
    gateway_url: str = "http://localhost:18500"
    from_person: str = "human"
    anima: str = ""
    thread_id: str = "default"
    sidebar_open: bool = True
    show_thinking: bool = False
    last_response_id: str | None = None
    last_event_id: str | None = None
    in_flight: bool = False
    model: str = ""
    recent_animas: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> SessionInfo:
        """Build a :class:`SessionInfo`, tolerating missing / extra fields."""
        known = {
            f
            for f in cls.__dataclass_fields__  # type: ignore[attr-defined]
        }
        cleaned = {k: v for k, v in data.items() if k in known}
        recent = cleaned.get("recent_animas")
        if not isinstance(recent, list):
            cleaned.pop("recent_animas", None)
        try:
            return cls(**{**cleaned, "recent_animas": list(recent or [])})
        except TypeError:
            # Reconstruct field-by-field so a broken entry can never crash us.
            return cls(
                session_id=str(data.get("session_id", "")),
                created_at=str(data.get("created_at", "")),
                updated_at=str(data.get("updated_at", "")),
                gateway_url=str(data.get("gateway_url", "http://localhost:18500")),
                from_person=str(data.get("from_person", "human")),
                anima=str(data.get("anima", "")),
                thread_id=str(data.get("thread_id", "default")),
                sidebar_open=bool(data.get("sidebar_open", True)),
                show_thinking=bool(data.get("show_thinking", False)),
                last_response_id=data.get("last_response_id"),
                last_event_id=data.get("last_event_id"),
                in_flight=bool(data.get("in_flight", False)),
                model=str(data.get("model", "")),
                recent_animas=list(data.get("recent_animas") or []),
            )


def new_session_id() -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    rnd = "".join(random.choices(string.digits, k=4))
    return f"{ts}-{rnd}"


def new_session(
    *,
    anima: str,
    thread_id: str = "default",
    gateway_url: str = "http://localhost:18500",
    from_person: str = "human",
) -> SessionInfo:
    """Create a brand-new session (does not touch disk)."""
    now = _now_iso()
    return SessionInfo(
        session_id=new_session_id(),
        created_at=now,
        updated_at=now,
        gateway_url=gateway_url,
        from_person=from_person,
        anima=anima,
        thread_id=thread_id,
        recent_animas=[anima] if anima else [],
    )


def save_session(session: SessionInfo, *, base_dir: Path | None = None) -> Path:
    """Persist a session and rotate old ones (returns the written path)."""
    d = base_dir if base_dir is not None else session_dir_path()
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    session.updated_at = _now_iso()
    path = d / f"{session.session_id}.json"
    path.write_text(json.dumps(asdict(session), ensure_ascii=False, indent=2), encoding="utf-8")
    _rotate(d)
    return path


def load_session(session_id: str, *, base_dir: Path | None = None) -> SessionInfo | None:
    """Load a single session, or ``None`` if missing / unreadable."""
    d = base_dir if base_dir is not None else session_dir_path()
    path = d / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    try:
        return SessionInfo.from_dict(data)
    except Exception:
        return None


def _iter_session_files(d: Path):
    return sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime)


def list_sessions(*, base_dir: Path | None = None) -> list[SessionInfo]:
    """Return all valid sessions, most recently updated first."""
    d = base_dir if base_dir is not None else session_dir_path()
    if not d.exists():
        return []
    infos: list[SessionInfo] = []
    for path in _iter_session_files(d):
        info = load_session(path.stem, base_dir=d)
        if info is not None:
            infos.append(info)
    infos.sort(key=lambda s: s.updated_at, reverse=True)
    return infos


def latest_session(*, base_dir: Path | None = None) -> SessionInfo | None:
    """The most recently updated session, or ``None``."""
    infos = list_sessions(base_dir=base_dir)
    return infos[0] if infos else None


def _rotate(d: Path) -> None:
    """Keep only the newest :data:`MAX_SESSIONS` files."""
    files = _iter_session_files(d)
    for path in files[:-MAX_SESSIONS]:
        try:
            path.unlink()
        except OSError:
            pass


def _local_timestamp(iso: str) -> str:
    """Render an ISO timestamp in the local timezone as ``YYYY-MM-DD HH:MM:SS``."""
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return (iso or "")[:19].replace("T", " ")


def sessions_table(infos: list[SessionInfo]) -> str:
    """Render a human-readable (tab separated) summary for ``--sessions``."""
    lines = ["session_id\tupdated\t\tanima\tthread"]
    for s in sorted(infos, key=lambda x: x.updated_at, reverse=True):
        ts = _local_timestamp(s.updated_at)
        lines.append(f"{s.session_id}\t{ts}\t{s.anima}\t{s.thread_id}")
    return "\n".join(lines)
