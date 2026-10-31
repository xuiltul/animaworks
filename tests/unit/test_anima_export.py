from __future__ import annotations

from pathlib import Path

from core.anima.export import export_anima
from core.anima.workspace_files import AGENTS_FILE, HEARTBEAT_FILE, IDENTITY_FILE, SOUL_FILE

SECRET = "sk-1234567890abcdef"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _create_anima(data_dir: Path) -> Path:
    anima_dir = data_dir / "animas" / "mei"
    _write(anima_dir / SOUL_FILE, f"# Soul\n\nKind and curious.\napi_key: {SECRET}\n")
    _write(anima_dir / IDENTITY_FILE, "# Identity: Mei\n\n## 基本プロフィール\n\n名前: Mei\n")
    _write(anima_dir / AGENTS_FILE, "# Agents\n\nVerify facts.\n")
    _write(anima_dir / HEARTBEAT_FILE, "# Heartbeat\n\nCheck the inbox.\n")
    _write(anima_dir / "permissions.json", f'{{"token": "{SECRET}"}}')
    _write(anima_dir / "status.json", '{"enabled": true}')
    _write(anima_dir / "cron.md", "# Cron\n")
    _write(
        anima_dir / "knowledge" / "00_project.md",
        "# Project Alpha\n\nAlpha is the current priority.\nKeep the roadmap small.\n",
    )
    _write(anima_dir / "episodes" / "2026-01-02.md", f"# Episode\n\nToday we used {SECRET}.\n")
    _write(anima_dir / "skills" / "review" / "SKILL.md", f"# Review\n\nUse {SECRET} carefully.\n")
    _write(data_dir / "shared" / "users" / "alice" / "index.md", f"# Alice\n\nOwner profile. token: {SECRET}\n")
    return anima_dir


def _all_text(root: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".md", ".json", ".txt"}
    )


def test_openclaw_export_layout_and_redaction(tmp_path: Path) -> None:
    data_dir = tmp_path / "runtime"
    _create_anima(data_dir)
    _write(
        data_dir / "animas" / "mei" / "knowledge" / "large.md",
        "# Large\n\n" + ("x" * 30_000),
    )
    out = tmp_path / "openclaw-export"

    result = export_anima(data_dir, "mei", "openclaw", out)

    assert result.format == "openclaw"
    for name in (AGENTS_FILE, SOUL_FILE, IDENTITY_FILE, HEARTBEAT_FILE, "USER.md", "MEMORY.md"):
        assert (out / name).is_file()
    assert (out / "memory" / "2026-01-02.md").is_file()
    assert (out / "skills" / "review" / "SKILL.md").is_file()
    assert (out / "ANIMAWORKS_EXPORT.md").is_file()
    assert not (out / "permissions.json").exists()
    assert not (out / "status.json").exists()
    assert not (out / "cron.md").exists()
    assert (out / "MEMORY.md").stat().st_size <= 20_000
    assert "## Project Alpha" in (out / "MEMORY.md").read_text(encoding="utf-8")
    text = _all_text(out)
    assert SECRET not in text
    assert "***" in text
    manifest = (out / "ANIMAWORKS_EXPORT.md").read_text(encoding="utf-8")
    assert "permissions.json" in manifest
    assert "Not exported" in manifest


def test_hermes_export_concatenates_soul_and_uses_memories(tmp_path: Path) -> None:
    data_dir = tmp_path / "runtime"
    anima_dir = _create_anima(data_dir)
    out = tmp_path / "hermes-export"

    result = export_anima(data_dir, "mei", "hermes", out)

    assert result.format == "hermes"
    soul = (out / SOUL_FILE).read_text(encoding="utf-8")
    identity = (anima_dir / IDENTITY_FILE).read_text(encoding="utf-8").rstrip()
    assert soul.startswith(identity)
    assert "Kind and curious." in soul
    assert "名前: Mei" in soul
    assert (out / AGENTS_FILE).is_file()
    assert (out / "memories" / "MEMORY.md").is_file()
    assert (out / "memories" / "USER.md").is_file()
    assert (out / "skills" / "review" / "SKILL.md").is_file()
    assert not (out / IDENTITY_FILE).exists()
    assert SECRET not in _all_text(out)
