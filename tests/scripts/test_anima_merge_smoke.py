from __future__ import annotations

import json
from pathlib import Path

from scripts.anima_merge.__main__ import build_finalize_parser, build_parser
from scripts.anima_merge.service import AnimaMergeService


def _create_animas(data_dir: Path) -> tuple[Path, Path]:
    source = data_dir / "animas" / "source"
    target = data_dir / "animas" / "target"
    for anima, name in ((source, "source"), (target, "target")):
        anima.mkdir(parents=True)
        (anima / "IDENTITY.md").write_text(f"# {name}\n", encoding="utf-8")
        (anima / "status.json").write_text('{"enabled": true}\n', encoding="utf-8")
        (anima / "state").mkdir()
    return source, target


def test_dry_run_writes_manifest_and_leaves_both_animas_unchanged(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    source, target = _create_animas(data_dir)
    source_memory = source / "episodes" / "2026-09-01.md"
    target_memory = target / "episodes" / "2026-09-01.md"
    source_memory.parent.mkdir()
    target_memory.parent.mkdir()
    source_memory.write_text("source memory\n", encoding="utf-8")
    target_memory.write_text("target memory\n", encoding="utf-8")

    result = AnimaMergeService(data_dir, "source", "target").run()

    assert result.dry_run is True
    assert result.journal_path is None
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    assert manifest["source"] == "source"
    assert manifest["target"] == "target"
    assert manifest["collisions"]["episodes"] == [{"source": "episodes/2026-09-01.md", "target_date": "2026-09-01"}]
    assert source_memory.read_text(encoding="utf-8") == "source memory\n"
    assert target_memory.read_text(encoding="utf-8") == "target memory\n"
    assert not (data_dir / "state" / "merge_journal_source_target.json").exists()


def test_standalone_cli_keeps_merge_and_finalize_arguments() -> None:
    merge_args = build_parser().parse_args(["source", "target", "--dry-run", "--force"])
    assert (merge_args.source, merge_args.target) == ("source", "target")
    assert merge_args.dry_run is True
    assert merge_args.execute is False
    assert merge_args.force is True
    assert merge_args.resume is False

    finalize_args = build_finalize_parser().parse_args(["source", "target", "--execute", "--resume"])
    assert (finalize_args.source, finalize_args.target) == ("source", "target")
    assert finalize_args.execute is True
    assert finalize_args.resume is True
