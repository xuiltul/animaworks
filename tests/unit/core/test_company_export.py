# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from core.company import CompanyError, ExportResult, export_company, split_companies

PRIMARY_COMPANY = "primaryco"
SECONDARY_COMPANY = "secondaryco"
PRIMARY_DISPLAY_NAME = "Primary Division"
SECONDARY_DISPLAY_NAME = "Secondary Division"
PRIMARY_MEMBER = "primarymember"
PRIMARY_HELPER = "primaryhelper"
SECONDARY_MEMBER = "secondarymember"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_anima(data_dir: Path, name: str) -> Path:
    anima_dir = data_dir / "animas" / name
    (anima_dir / "knowledge").mkdir(parents=True)
    _write_json(anima_dir / "status.json", {"enabled": True, "model": "test-model"})
    (anima_dir / "identity.md").write_text(f"# {name}\n", encoding="utf-8")
    (anima_dir / "knowledge" / "owned.md").write_text(
        f"knowledge owned by {name}\n",
        encoding="utf-8",
    )
    return anima_dir


def _prepare_split_environment(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    for name in (PRIMARY_MEMBER, PRIMARY_HELPER, SECONDARY_MEMBER):
        _make_anima(data_dir, name)

    manifest = tmp_path / "split.json"
    _write_json(
        manifest,
        {
            "companies": [
                {
                    "name": PRIMARY_COMPANY,
                    "display_name": PRIMARY_DISPLAY_NAME,
                    "members": [PRIMARY_MEMBER, PRIMARY_HELPER],
                },
                {
                    "name": SECONDARY_COMPANY,
                    "display_name": SECONDARY_DISPLAY_NAME,
                    "members": [SECONDARY_MEMBER],
                },
            ]
        },
    )
    split_companies(manifest, execute=True, data_dir=data_dir)

    primary_dir = data_dir / "companies" / PRIMARY_COMPANY
    (primary_dir / "knowledge" / "company-policy.md").write_text(
        "company-owned policy\n",
        encoding="utf-8",
    )
    (primary_dir / "shared" / "company-data.json").write_text(
        '{"scope": "company"}\n',
        encoding="utf-8",
    )

    (data_dir / "common_knowledge").mkdir()
    (data_dir / "common_knowledge" / "handbook.md").write_text(
        "common handbook\n",
        encoding="utf-8",
    )
    (data_dir / "common_skills").mkdir()
    (data_dir / "common_skills" / "shared-skill.md").write_text(
        "common skill\n",
        encoding="utf-8",
    )

    _write_json(
        data_dir / "config.json",
        {
            "setup_complete": True,
            "credentials": {
                "provider": {
                    "api_key": {"$vault": "PROVIDER_API_KEY"},
                    "keys": {"client_secret": "credential-secret"},
                }
            },
            "animas": {
                PRIMARY_MEMBER: {
                    "model": "test-model",
                    "credential": "provider",
                    "company": PRIMARY_COMPANY,
                },
                PRIMARY_HELPER: {"model": "test-model", "company": PRIMARY_COMPANY},
                SECONDARY_MEMBER: {"model": "test-model", "company": SECONDARY_COMPANY},
            },
            "system": {"locale": "en"},
        },
    )
    return data_dir


def _assert_all_credential_values_redacted(value: Any) -> None:
    if isinstance(value, dict):
        assert value
        for child in value.values():
            _assert_all_credential_values_redacted(child)
    elif isinstance(value, list):
        for child in value:
            _assert_all_credential_values_redacted(child)
    else:
        assert value == "REDACTED"


def test_export_company_collects_transfer_bundle_and_handles_symlinks(tmp_path: Path) -> None:
    data_dir = _prepare_split_environment(tmp_path)
    member_dir = data_dir / "animas" / PRIMARY_MEMBER
    internal_link = member_dir / "knowledge" / "owned-link.md"
    internal_link.symlink_to("owned.md")

    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("must not be exported\n", encoding="utf-8")
    external_link = member_dir / "external-link.txt"
    external_link.symlink_to(outside_file)

    other_company_link = member_dir / "other-company-link.md"
    other_company_link.symlink_to(Path("../") / SECONDARY_MEMBER / "identity.md")

    output_dir = tmp_path / "export"
    result = export_company(PRIMARY_COMPANY, output_dir, data_dir=data_dir)

    assert isinstance(result, ExportResult)
    assert result.output_dir == output_dir
    assert result.members == (PRIMARY_HELPER, PRIMARY_MEMBER)
    assert any("external-link.txt" in item for item in result.skipped_symlinks)
    assert any("other-company-link.md" in item for item in result.skipped_symlinks)

    exported_member = output_dir / "animas" / PRIMARY_MEMBER
    assert (exported_member / "identity.md").read_text(encoding="utf-8") == (member_dir / "identity.md").read_text(
        encoding="utf-8"
    )
    assert (output_dir / "animas" / PRIMARY_HELPER / "knowledge" / "owned.md").is_file()
    assert not (output_dir / "animas" / SECONDARY_MEMBER).exists()

    exported_company = output_dir / "companies" / PRIMARY_COMPANY
    assert (exported_company / "knowledge" / "company-policy.md").read_text(
        encoding="utf-8"
    ) == "company-owned policy\n"
    assert (exported_company / "shared" / "company-data.json").is_file()
    assert not (exported_company / "animas").exists()
    assert not (output_dir / "companies" / SECONDARY_COMPANY).exists()

    assert (output_dir / "common_knowledge" / "handbook.md").read_text(encoding="utf-8") == "common handbook\n"
    assert (output_dir / "common_skills" / "shared-skill.md").read_text(encoding="utf-8") == "common skill\n"

    exported_config = json.loads((output_dir / "config.export.json").read_text(encoding="utf-8"))
    assert set(exported_config["animas"]) == {PRIMARY_MEMBER, PRIMARY_HELPER}
    assert SECONDARY_MEMBER not in json.dumps(exported_config["animas"])
    assert set(exported_config["credentials"]["provider"]) == {"api_key", "keys"}
    _assert_all_credential_values_redacted(exported_config["credentials"])
    assert "credential-secret" not in json.dumps(exported_config)

    exported_internal_link = exported_member / "knowledge" / "owned-link.md"
    assert exported_internal_link.is_symlink()
    assert Path(os.readlink(exported_internal_link)) == Path("owned.md")
    assert exported_internal_link.resolve() == (exported_member / "knowledge" / "owned.md").resolve()
    assert not (exported_member / "external-link.txt").is_symlink()
    assert not (exported_member / "external-link.txt").exists()
    assert not (exported_member / "other-company-link.md").is_symlink()
    assert not (exported_member / "other-company-link.md").exists()

    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "PROVIDER_API_KEY" in readme
    assert "vault" in readme.lower()
    assert "Neo4j" in readme
    assert "group_id" in readme
    assert f"animaworks index --anima {PRIMARY_MEMBER}" in readme
    assert f"animaworks index --anima {PRIMARY_HELPER}" in readme
    assert "systemd" in readme
    assert "external-link.txt" in readme
    assert "other-company-link.md" in readme

    assert (output_dir / "scan-report.md").is_file()


def test_export_company_scan_reports_other_company_context(tmp_path: Path) -> None:
    data_dir = _prepare_split_environment(tmp_path)
    context_file = data_dir / "animas" / PRIMARY_MEMBER / "knowledge" / "context.md"
    context_file.write_text(
        "Referenced member: secondarymember\n"
        "Referenced company: secondaryco\n"
        "Referenced display name: Secondary Division\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "export-with-hits"
    result = export_company(PRIMARY_COMPANY, output_dir, data_dir=data_dir)

    assert result.scan_hit_count == 3
    report = (output_dir / "scan-report.md").read_text(encoding="utf-8")
    assert "animas/primarymember/knowledge/context.md" in report
    assert "secondarymember" in report
    assert "secondaryco" in report
    assert "Secondary Division" in report


def test_export_company_scan_reports_no_hits_and_uses_anima_word_boundaries(
    tmp_path: Path,
) -> None:
    data_dir = _prepare_split_environment(tmp_path)
    (data_dir / "animas" / PRIMARY_MEMBER / "knowledge" / "context.md").write_text(
        "secondarymemberSuffix is not the other anima name as a complete word.\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "export-without-hits"
    result = export_company(PRIMARY_COMPANY, output_dir, data_dir=data_dir)

    assert result.scan_hit_count == 0
    report = (output_dir / "scan-report.md").read_text(encoding="utf-8")
    assert "0" in report
    assert "secondarymemberSuffix" not in report


def test_export_company_rejects_non_empty_output_without_modifying_it(tmp_path: Path) -> None:
    data_dir = _prepare_split_environment(tmp_path)
    output_dir = tmp_path / "existing-output"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("keep unchanged\n", encoding="utf-8")

    with pytest.raises(CompanyError, match="not empty|non-empty"):
        export_company(PRIMARY_COMPANY, output_dir, data_dir=data_dir)

    assert sentinel.read_text(encoding="utf-8") == "keep unchanged\n"
    assert list(output_dir.iterdir()) == [sentinel]
