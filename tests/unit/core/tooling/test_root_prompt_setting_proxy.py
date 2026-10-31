from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.platform.process_role import PROCESS_ROLE_ENV
from core.tooling.handler import ToolHandler


def test_task_runner_identity_write_is_forwarded_to_root(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path
    anima_dir = data_dir / "animas" / "alice"
    anima_dir.mkdir(parents=True)
    (anima_dir / "IDENTITY.md").write_text("Old identity\n", encoding="utf-8")
    (anima_dir / "AGENTS.md").write_text("Old injection\n", encoding="utf-8")
    (anima_dir / "status.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
    (anima_dir / "permissions.json").write_text(json.dumps({"version": 1, "file_roots": ["/"]}), encoding="utf-8")
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))
    monkeypatch.setenv(PROCESS_ROLE_ENV, "task_runner")

    response = MagicMock(status_code=200)
    handler = ToolHandler(
        anima_dir=anima_dir,
        memory=MagicMock(),
        messenger=MagicMock(),
        tool_registry=[],
    )
    with patch("core.host_api.host_api.post", return_value=response) as host_post:
        result = handler.handle("write_memory_file", {"path": "IDENTITY.md", "content": "New identity\n"})

    assert "root settings API" in result
    host_post.assert_called_once_with(
        "/api/internal/animas/alice/prompt-settings",
        json={"setting": "identity", "content": "New identity\n", "mode": "overwrite"},
        timeout=30.0,
    )
    assert (anima_dir / "IDENTITY.md").read_text(encoding="utf-8") == "Old identity\n"
