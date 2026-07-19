# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.company import (
    CompanyError,
    SplitExecutionError,
    adopt_assets,
    assign_animas,
    create_company,
    list_companies,
    split_companies,
)


def _make_anima(data_dir: Path, name: str, *, company: str | None = None) -> Path:
    anima_dir = data_dir / "animas" / name
    anima_dir.mkdir(parents=True)
    status: dict[str, Any] = {"enabled": True, "model": "test-model"}
    if company is not None:
        status["company"] = company
    (anima_dir / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return anima_dir


def _read_status(anima_dir: Path) -> dict[str, Any]:
    return json.loads((anima_dir / "status.json").read_text(encoding="utf-8"))


def _summary_value(summary: object, key: str) -> object:
    if isinstance(summary, dict):
        return summary[key]
    return getattr(summary, key)


def _write_manifest(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class TestCreateAndListCompanies:
    def test_create_builds_skeleton_and_is_idempotent(self, tmp_path: Path) -> None:
        assert create_company("alpha-labs", "Alpha Labs", data_dir=tmp_path) is True

        company_dir = tmp_path / "companies" / "alpha-labs"
        config_path = company_dir / "company.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["name"] == "alpha-labs"
        assert config["display_name"] == "Alpha Labs"
        assert isinstance(config["created_at"], str) and config["created_at"]
        assert "Alpha Labs" in (company_dir / "vision.md").read_text(encoding="utf-8")
        for name in ("knowledge", "skills", "shared", "credentials"):
            assert (company_dir / name).is_dir()

        original_config = config_path.read_text(encoding="utf-8")
        (company_dir / "vision.md").write_text("Do not overwrite this vision.\n", encoding="utf-8")
        (company_dir / "skills").rmdir()

        create_company("alpha-labs", "Changed display name", data_dir=tmp_path)
        assert config_path.read_text(encoding="utf-8") == original_config
        assert (company_dir / "vision.md").read_text(encoding="utf-8") == "Do not overwrite this vision.\n"
        assert (company_dir / "skills").is_dir()
        assert create_company("alpha-labs", "Changed display name", data_dir=tmp_path) is False

    @pytest.mark.parametrize(
        "name",
        ["", "Alpha", "-alpha", "_alpha", "alpha space", "alpha/beta", "../alpha", "alpha.beta"],
    )
    def test_create_rejects_invalid_names(self, tmp_path: Path, name: str) -> None:
        with pytest.raises(CompanyError):
            create_company(name, data_dir=tmp_path)
        assert not (tmp_path / "companies").exists() or not any((tmp_path / "companies").iterdir())

    def test_list_reports_member_counts_and_unassigned_animas(self, tmp_path: Path) -> None:
        create_company("alpha", "Alpha Group", data_dir=tmp_path)
        create_company("beta", "Beta Group", data_dir=tmp_path)
        _make_anima(tmp_path, "alice", company="alpha")
        _make_anima(tmp_path, "bob", company="alpha")
        _make_anima(tmp_path, "carol")

        summaries, unassigned = list_companies(data_dir=tmp_path)

        by_name = {str(_summary_value(item, "name")): item for item in summaries}
        assert set(by_name) == {"alpha", "beta"}
        assert _summary_value(by_name["alpha"], "display_name") == "Alpha Group"
        assert _summary_value(by_name["alpha"], "member_count") == 2
        assert _summary_value(by_name["beta"], "member_count") == 0
        assert unassigned == ["carol"]


class TestAssignAnimas:
    def test_assign_transfer_and_unassign_update_status_and_relative_views(self, tmp_path: Path) -> None:
        create_company("alpha", data_dir=tmp_path)
        create_company("beta", data_dir=tmp_path)
        alice = _make_anima(tmp_path, "alice")
        bob = _make_anima(tmp_path, "bob")

        assert len(assign_animas(["alice", "bob"], company_name="alpha", data_dir=tmp_path)) == 2
        for name, anima_dir in (("alice", alice), ("bob", bob)):
            assert _read_status(anima_dir)["company"] == "alpha"
            link = tmp_path / "companies" / "alpha" / "animas" / name
            assert link.is_symlink()
            assert Path(link.readlink()) == Path("../../../animas") / name
            assert link.resolve() == anima_dir.resolve()

        assign_animas(["alice"], company_name="beta", data_dir=tmp_path)
        assert _read_status(alice)["company"] == "beta"
        assert not (tmp_path / "companies" / "alpha" / "animas" / "alice").exists()
        beta_link = tmp_path / "companies" / "beta" / "animas" / "alice"
        assert beta_link.is_symlink()
        assert Path(beta_link.readlink()) == Path("../../../animas/alice")

        assign_animas(["alice"], unassign=True, data_dir=tmp_path)
        assert "company" not in _read_status(alice)
        assert not beta_link.exists()
        assert _read_status(alice)["model"] == "test-model"

    def test_assign_rejects_missing_anima_or_company(self, tmp_path: Path) -> None:
        create_company("alpha", data_dir=tmp_path)
        _make_anima(tmp_path, "alice")

        with pytest.raises(CompanyError):
            assign_animas(["missing"], company_name="alpha", data_dir=tmp_path)
        with pytest.raises(CompanyError):
            assign_animas(["alice"], company_name="missing", data_dir=tmp_path)


class TestAdoptAssets:
    @pytest.mark.parametrize(
        ("relative_source", "expected_subdir"),
        [
            ("shared/alpha-data", "shared"),
            ("credentials/alpha-token.json", "credentials"),
            ("policy.md", "knowledge"),
            ("misc/data", "shared"),
        ],
    )
    def test_infers_destination_and_creates_backup_and_relative_symlink(
        self,
        tmp_path: Path,
        relative_source: str,
        expected_subdir: str,
    ) -> None:
        create_company("alpha", data_dir=tmp_path)
        source = tmp_path / relative_source
        if source.suffix:
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("original content", encoding="utf-8")
        else:
            source.mkdir(parents=True)
            (source / "payload.txt").write_text("original content", encoding="utf-8")

        result = adopt_assets([source], company_name="alpha", data_dir=tmp_path)[0]
        expected = tmp_path / "companies" / "alpha" / expected_subdir / source.name

        assert result.source == source
        assert result.destination == expected
        assert expected.exists()
        assert source.is_symlink()
        assert not source.readlink().is_absolute()
        assert source.resolve() == expected.resolve()
        assert result.symlink_created is True
        assert result.backup_path.exists()
        if expected.is_file():
            assert result.backup_path.read_text(encoding="utf-8") == "original content"
        else:
            assert (result.backup_path / "payload.txt").read_text(encoding="utf-8") == "original content"
        assert "company-adopt-" in result.backup_path.as_posix()
        backup_relative = result.backup_path.relative_to(tmp_path / "backup")
        assert backup_relative.parts[1:] == source.relative_to(tmp_path).parts

    def test_explicit_destination_and_no_symlink(self, tmp_path: Path) -> None:
        create_company("alpha", data_dir=tmp_path)
        source = tmp_path / "incoming" / "manual.txt"
        source.parent.mkdir()
        source.write_text("manual", encoding="utf-8")

        result = adopt_assets(
            [source],
            company_name="alpha",
            data_dir=tmp_path,
            dest="skills",
            leave_symlinks=False,
        )[0]

        assert result.destination == tmp_path / "companies" / "alpha" / "skills" / "manual.txt"
        assert result.destination.read_text(encoding="utf-8") == "manual"
        assert not source.exists()
        assert result.symlink_created is False
        assert result.backup_path.read_text(encoding="utf-8") == "manual"

    def test_dot_destination_places_asset_at_company_root(self, tmp_path: Path) -> None:
        create_company("alpha", data_dir=tmp_path)
        source = tmp_path / "overview.txt"
        source.write_text("overview", encoding="utf-8")

        result = adopt_assets([source], company_name="alpha", data_dir=tmp_path, dest=".")[0]

        assert result.destination == tmp_path / "companies" / "alpha" / "overview.txt"

    def test_refuses_destination_overwrite(self, tmp_path: Path) -> None:
        create_company("alpha", data_dir=tmp_path)
        source = tmp_path / "shared" / "item.txt"
        source.parent.mkdir()
        source.write_text("source", encoding="utf-8")
        destination = tmp_path / "companies" / "alpha" / "shared" / "item.txt"
        destination.write_text("keep", encoding="utf-8")

        with pytest.raises(CompanyError):
            adopt_assets([source], company_name="alpha", data_dir=tmp_path)

        assert source.read_text(encoding="utf-8") == "source"
        assert destination.read_text(encoding="utf-8") == "keep"

    def test_refuses_path_outside_data_dir(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        create_company("alpha", data_dir=data_dir)
        outside = tmp_path / "outside.txt"
        outside.write_text("outside", encoding="utf-8")

        with pytest.raises(CompanyError):
            adopt_assets([outside], company_name="alpha", data_dir=data_dir)
        assert outside.read_text(encoding="utf-8") == "outside"

    def test_refuses_symlink_input(self, tmp_path: Path) -> None:
        create_company("alpha", data_dir=tmp_path)
        target = tmp_path / "real.txt"
        target.write_text("real", encoding="utf-8")
        source = tmp_path / "link.txt"
        source.symlink_to(target.name)

        with pytest.raises(CompanyError):
            adopt_assets([source], company_name="alpha", data_dir=tmp_path)
        assert source.is_symlink()
        assert target.read_text(encoding="utf-8") == "real"


class TestSplitCompanies:
    def test_adopt_symlink_option_is_applied_per_entry_and_defaults_true(
        self,
        tmp_path: Path,
    ) -> None:
        without_link = tmp_path / "shared" / "without-link.txt"
        without_link.parent.mkdir(parents=True)
        without_link.write_text("move only", encoding="utf-8")
        default_link = tmp_path / "shared" / "default-link.txt"
        default_link.write_text("move and link", encoding="utf-8")
        manifest = _write_manifest(
            tmp_path / "split.json",
            {
                "companies": [
                    {
                        "name": "alpha",
                        "adopt": [
                            {"path": "shared/without-link.txt", "symlink": False},
                            {"path": "shared/default-link.txt"},
                        ],
                    }
                ]
            },
        )

        split_companies(manifest, execute=True, data_dir=tmp_path)

        destination = tmp_path / "companies" / "alpha" / "shared"
        assert not without_link.exists()
        assert (destination / without_link.name).read_text(encoding="utf-8") == "move only"
        assert default_link.is_symlink()
        assert default_link.resolve() == (destination / default_link.name).resolve()

    def test_adopt_symlink_option_must_be_boolean(self, tmp_path: Path) -> None:
        manifest = _write_manifest(
            tmp_path / "split.json",
            {
                "companies": [
                    {
                        "name": "alpha",
                        "adopt": [{"path": "shared/item.txt", "symlink": "false"}],
                    }
                ]
            },
        )

        with pytest.raises(CompanyError, match="symlink.*boolean"):
            split_companies(manifest, execute=False, data_dir=tmp_path)

    def test_dry_run_execute_and_second_execute_are_idempotent(self, tmp_path: Path) -> None:
        _make_anima(tmp_path, "alice")
        source = tmp_path / "shared" / "alpha-assets"
        source.mkdir(parents=True)
        (source / "guide.txt").write_text("guide", encoding="utf-8")
        manifest = _write_manifest(
            tmp_path / "split.json",
            {
                "companies": [
                    {
                        "name": "alpha",
                        "display_name": "Alpha Group",
                        "members": ["alice"],
                        "adopt": [{"path": "shared/alpha-assets", "dest": "shared"}],
                    }
                ]
            },
        )

        dry_run = split_companies(manifest, execute=False, data_dir=tmp_path)

        assert any("create" in action.lower() for action in dry_run)
        assert any("assign" in action.lower() for action in dry_run)
        assert any("adopt" in action.lower() or "move" in action.lower() for action in dry_run)
        assert not (tmp_path / "companies" / "alpha").exists()
        assert source.is_dir() and not source.is_symlink()
        assert "company" not in _read_status(tmp_path / "animas" / "alice")

        executed = split_companies(manifest, execute=True, data_dir=tmp_path)
        destination = tmp_path / "companies" / "alpha" / "shared" / "alpha-assets"
        assert executed
        assert _read_status(tmp_path / "animas" / "alice")["company"] == "alpha"
        assert source.is_symlink() and source.resolve() == destination.resolve()
        assert destination.is_dir()

        repeated = split_companies(manifest, execute=True, data_dir=tmp_path)
        assert repeated
        assert all("skip" in action.lower() for action in repeated)
        assert source.is_symlink() and source.resolve() == destination.resolve()

    def test_execute_stops_at_first_failure_and_reports_completed_operations(self, tmp_path: Path) -> None:
        _make_anima(tmp_path, "alice")
        _make_anima(tmp_path, "bob")
        manifest = _write_manifest(
            tmp_path / "partial.json",
            {
                "companies": [
                    {"name": "alpha", "members": ["alice"]},
                    {
                        "name": "beta",
                        "members": ["bob"],
                        "adopt": [{"path": "shared/missing"}],
                    },
                    {"name": "gamma"},
                ]
            },
        )

        with pytest.raises(SplitExecutionError) as exc_info:
            split_companies(manifest, execute=True, data_dir=tmp_path)

        assert "missing" in str(exc_info.value).lower()
        assert exc_info.value.completed_lines == [
            "CREATE alpha",
            "ASSIGN alice -> alpha",
            "CREATE beta",
            "ASSIGN bob -> beta",
        ]
        assert (tmp_path / "companies" / "alpha").is_dir()
        assert _read_status(tmp_path / "animas" / "alice")["company"] == "alpha"
        assert (tmp_path / "companies" / "beta").is_dir()
        assert _read_status(tmp_path / "animas" / "bob")["company"] == "beta"
        assert not (tmp_path / "companies" / "gamma").exists()
