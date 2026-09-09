from __future__ import annotations

from pathlib import Path

from scripts.slim_runtime_measure import source_counts


def test_source_counts_include_untracked_modules_and_exclude_deleted_and_ui(tmp_path: Path, monkeypatch) -> None:
    names = ["core/existing.py", "core/taskboard/tasks.py", "core/deleted.py", "server/static/app.js"]
    for name in ("core/existing.py", "core/taskboard/tasks.py", "server/static/app.js"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("one\ntwo\n")
    monkeypatch.setattr("scripts.slim_runtime_measure.subprocess.check_output", lambda *a, **k: "\n".join(names))
    result = source_counts(tmp_path)
    assert result["files"] == 2
    assert result["physical_lines"] == 4
    assert "core/taskboard/tasks.py" in result["included_new_task_modules"]
