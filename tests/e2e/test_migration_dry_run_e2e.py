from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from cli.commands.import_cmd import cmd_import_hermes


def test_hermes_cli_dry_run_fixture_does_not_modify_runtime(data_dir: Path, capsys) -> None:
    source = data_dir.parent / ".hermes"
    skill_dir = source / "skills" / "cli-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: cli-skill\ndescription: CLI Skill\n---\n\n# CLI Skill\n",
        encoding="utf-8",
    )

    cmd_import_hermes(
        Namespace(
            path=str(source),
            target_anima="mei",
            common_skills=False,
            apply=False,
            replace=False,
            json_output=True,
        )
    )

    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "dry_run"
    assert output["summary"]["planned"] == 1
    assert not (data_dir / "animas" / "mei" / "skills" / "cli-skill").exists()
    assert not (data_dir / "animas" / "mei" / "state" / "migrations").exists()
