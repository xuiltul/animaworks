# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cli.commands.company_cmd import register_company_command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="animaworks")
    subparsers = parser.add_subparsers(dest="command")
    register_company_command(subparsers)
    return parser


def _run(parser: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    args = parser.parse_args(argv)
    assert hasattr(args, "func")
    args.func(args)
    return args


@pytest.mark.parametrize(
    "argv",
    [
        ["company", "--help"],
        ["company", "create", "--help"],
        ["company", "list", "--help"],
        ["company", "assign", "--help"],
        ["company", "adopt", "--help"],
        ["company", "split", "--help"],
        ["company", "export", "--help"],
    ],
)
def test_company_help_for_group_and_every_leaf(
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _parser().parse_args(argv)

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "usage:" in output.lower()
    assert "company" in output


def test_company_parser_accepts_all_command_shapes() -> None:
    parser = _parser()

    create = parser.parse_args(["company", "create", "alpha", "--display-name", "Alpha Group"])
    assert create.company_command == "create"
    assert create.name == "alpha"
    assert create.display_name == "Alpha Group"

    listed = parser.parse_args(["company", "list"])
    assert listed.company_command == "list"

    assigned = parser.parse_args(["company", "assign", "alice", "bob", "--to", "alpha"])
    assert assigned.anima == ["alice", "bob"]
    assert assigned.to == "alpha"
    assert assigned.unassign is False

    unassigned = parser.parse_args(["company", "assign", "alice", "--unassign"])
    assert unassigned.anima == ["alice"]
    assert unassigned.to is None
    assert unassigned.unassign is True

    adopted = parser.parse_args(
        [
            "company",
            "adopt",
            "shared/one",
            "policy.md",
            "--to",
            "alpha",
            "--dest",
            "knowledge",
            "--no-symlink",
        ]
    )
    assert adopted.path == [Path("shared/one"), Path("policy.md")]
    assert adopted.to == "alpha"
    assert adopted.dest == "knowledge"
    assert adopted.no_symlink is True

    split = parser.parse_args(["company", "split", "--manifest", "split.json", "--execute"])
    assert split.manifest == Path("split.json")
    assert split.execute is True

    exported = parser.parse_args(["company", "export", "alpha", "--out", "migration"])
    assert exported.name == "alpha"
    assert exported.out == Path("migration")


def test_company_export_calls_core_and_prints_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = _parser()
    data_dir = tmp_path / "runtime"
    output_dir = tmp_path / "migration"
    result = SimpleNamespace(
        output_dir=output_dir,
        members=("alice", "bob"),
        skipped_symlinks=("animas/alice/external",),
        scan_hit_count=3,
    )

    with (
        patch("core.paths.get_data_dir", return_value=data_dir),
        patch("core.company.export_company", return_value=result) as export_company,
    ):
        _run(parser, ["company", "export", "alpha", "--out", str(output_dir)])

    export_company.assert_called_once_with("alpha", output_dir, data_dir=data_dir)
    assert capsys.readouterr().out.splitlines() == [
        f"Exported company 'alpha' to {output_dir}.",
        "Members: 2",
        "Skipped symlinks: 1",
        "Scan hits: 3",
    ]


def test_assign_requires_exactly_one_assignment_mode(capsys: pytest.CaptureFixture[str]) -> None:
    parser = _parser()

    with pytest.raises(SystemExit) as missing:
        parser.parse_args(["company", "assign", "alice"])
    assert missing.value.code == 2
    capsys.readouterr()

    with pytest.raises(SystemExit) as conflicting:
        parser.parse_args(["company", "assign", "alice", "--to", "alpha", "--unassign"])
    assert conflicting.value.code == 2


def test_company_cli_e2e_create_assign_adopt_and_split_idempotently(
    data_dir: Path,
    make_anima,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise the complete company workflow through registered argparse handlers."""
    parser = _parser()
    alice = make_anima("alice")
    bob = make_anima("bob")

    _run(parser, ["company", "create", "alpha", "--display-name", "Alpha Group"])
    assert "Created company 'alpha'." in capsys.readouterr().out
    assert (data_dir / "companies" / "alpha" / "company.json").is_file()

    _run(parser, ["company", "assign", "alice", "--to", "alpha"])
    assign_output = capsys.readouterr().out
    assert "ASSIGN alice -> alpha" in assign_output
    alice_status = json.loads((alice / "status.json").read_text(encoding="utf-8"))
    assert alice_status["company"] == "alpha"
    assert (data_dir / "companies" / "alpha" / "animas" / "alice").is_symlink()

    _run(parser, ["company", "list"])
    list_output = capsys.readouterr().out
    assert "NAME\tDISPLAY NAME\tANIMAS" in list_output
    assert "alpha\tAlpha Group\t1" in list_output
    assert "Unassigned animas:" in list_output
    assert "  - bob" in list_output

    manual = data_dir / "shared" / "alpha-manual.md"
    manual.write_text("# Alpha manual\n", encoding="utf-8")
    _run(
        parser,
        [
            "company",
            "adopt",
            "shared/alpha-manual.md",
            "--to",
            "alpha",
            "--dest",
            "knowledge",
        ],
    )
    adopt_output = capsys.readouterr().out
    adopted_manual = data_dir / "companies" / "alpha" / "knowledge" / "alpha-manual.md"
    assert "Moved:" in adopt_output
    assert "Backup:" in adopt_output
    assert "Symlink:" in adopt_output
    assert manual.is_symlink() and manual.resolve() == adopted_manual.resolve()

    split_source = data_dir / "shared" / "beta-assets"
    split_source.mkdir()
    (split_source / "asset.txt").write_text("asset", encoding="utf-8")
    manifest = data_dir / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "companies": [
                    {
                        "name": "beta",
                        "display_name": "Beta Group",
                        "members": ["bob"],
                        "adopt": [{"path": "shared/beta-assets", "dest": "shared"}],
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _run(parser, ["company", "split", "--manifest", str(manifest)])
    dry_run_lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert dry_run_lines == [
        "DRY-RUN CREATE beta",
        "DRY-RUN ASSIGN bob -> beta",
        "DRY-RUN ADOPT shared/beta-assets -> companies/beta/shared/beta-assets",
    ]
    assert not (data_dir / "companies" / "beta").exists()
    assert "company" not in json.loads((bob / "status.json").read_text(encoding="utf-8"))
    assert split_source.is_dir() and not split_source.is_symlink()

    _run(parser, ["company", "split", "--manifest", str(manifest), "--execute"])
    execute_lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert execute_lines == [
        "CREATE beta",
        "ASSIGN bob -> beta",
        "ADOPT shared/beta-assets -> companies/beta/shared/beta-assets",
    ]
    bob_status = json.loads((bob / "status.json").read_text(encoding="utf-8"))
    assert bob_status["company"] == "beta"
    split_destination = data_dir / "companies" / "beta" / "shared" / "beta-assets"
    assert split_source.is_symlink() and split_source.resolve() == split_destination.resolve()

    _run(parser, ["company", "split", "--manifest", str(manifest), "--execute"])
    repeated_lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert len(repeated_lines) == 3
    assert all(line.startswith("SKIP ") for line in repeated_lines)
    assert split_source.is_symlink() and split_source.resolve() == split_destination.resolve()
