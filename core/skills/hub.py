from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Skill Hub importer for external skill bundles."""

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.memory.frontmatter import parse_frontmatter
from core.paths import get_data_dir
from core.skills.guard import SkillScanner
from core.skills.hub_storage import activate_destination, pending_destination, removed_destination, rollback_activation
from core.skills.loader import load_skill_metadata
from core.skills.models import ScanResult, SkillBundle, SkillHubLockEntry, SkillScanVerdict, SkillTrustLevel
from core.skills.promotion_utils import safe_skill_name, scan_security_metadata
from core.time_utils import now_iso


@dataclass(slots=True)
class SkillHubResult:
    status: str
    skill_name: str
    target: str
    message: str = ""
    installed_path: str | None = None
    backup_path: str | None = None
    scan_verdict: str | None = None
    decision: bool | None = None
    reason: str = ""
    findings: list[dict[str, Any]] = field(default_factory=list)
    size_violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class _TargetPaths:
    target: str
    active_dir: Path
    quarantine_dir: Path
    lock_path: Path
    backup_dir: Path
    rel_base: Path

    def active_skill_dir(self, name: str) -> Path:
        return self.active_dir / name

    def quarantine_skill_dir(self, name: str) -> Path:
        return self.quarantine_dir / name

    def rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.rel_base))
        except ValueError:
            return str(path)


class SkillHub:
    """Stage, scan, install, inspect, and remove external skills."""

    def __init__(
        self,
        *,
        data_dir: Path | None = None,
        scanner: SkillScanner | None = None,
        actor: str = "cli",
    ) -> None:
        self.data_dir = (data_dir or get_data_dir()).resolve()
        self.scanner = scanner or SkillScanner()
        self.actor = actor

    def install(
        self,
        source: str,
        *,
        target: str,
        anima: str | None = None,
        dry_run: bool = False,
        replace: bool = False,
        force: bool = False,
        trust_level: SkillTrustLevel | str = SkillTrustLevel.community,
        quarantine: bool = False,
    ) -> SkillHubResult:
        trust = _coerce_import_trust(trust_level)
        # Hub imports use the fixed community gate; --force must not bypass external-source policy.
        del force
        paths = self._target_paths(target, anima=anima)
        tmp_parent = self.data_dir / "tmp" / "skill_hub"
        tmp_parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="stage-", dir=tmp_parent) as tmp:
            staging_root = Path(tmp)
            bundle = self._stage_source(source, staging_root)
            meta = load_skill_metadata(Path(bundle.skill_md_path))
            skill_name = safe_skill_name(meta.name or Path(bundle.staging_path).name)
            _validate_skill_name(skill_name, paths)
            scan = self.scanner.scan_skill(Path(bundle.staging_path), source=bundle.source_type)
            decision, reason = self.scanner.should_allow(scan, SkillTrustLevel.community, force=False)

            if scan.verdict == SkillScanVerdict.dangerous or scan.size_violations:
                return self._blocked_result(skill_name, paths.target, scan, decision, reason)
            if decision is None and not quarantine:
                return self._approval_required_result(skill_name, paths.target, scan, reason)
            if decision is False and not quarantine:
                return self._blocked_result(skill_name, paths.target, scan, decision, reason)

            destination = paths.quarantine_skill_dir(skill_name) if quarantine else paths.active_skill_dir(skill_name)
            install_trust = SkillTrustLevel.quarantine if quarantine else trust
            installed_path = paths.rel(destination / "SKILL.md")

            if dry_run:
                return SkillHubResult(
                    status="dry_run",
                    skill_name=skill_name,
                    target=paths.target,
                    installed_path=installed_path,
                    scan_verdict=scan.verdict.value,
                    decision=decision,
                    reason=reason,
                    message="Dry run completed; no files were installed.",
                    findings=[f.model_dump(mode="json") for f in scan.findings],
                    size_violations=scan.size_violations,
                )

            prepared = pending_destination(paths, skill_name)
            backup_path = None
            activated = False
            try:
                self._install_staged_bundle(
                    Path(bundle.staging_path),
                    prepared,
                    skill_name=skill_name,
                    trust_level=install_trust,
                    scan=scan,
                    source=bundle,
                )
                backup_path = activate_destination(prepared, destination, replace=replace, paths=paths)
                activated = True
                self._append_lock(
                    paths,
                    action="quarantine" if quarantine else "install",
                    skill_name=skill_name,
                    source=bundle,
                    scan=scan,
                    installed_path=installed_path,
                )
            except Exception:
                shutil.rmtree(prepared, ignore_errors=True)
                if activated:
                    rollback_activation(destination, backup_path, paths)
                raise
            return SkillHubResult(
                status="quarantine" if quarantine else "installed",
                skill_name=skill_name,
                target=paths.target,
                installed_path=installed_path,
                backup_path=backup_path,
                scan_verdict=scan.verdict.value,
                decision=decision,
                reason=reason,
                message="Skill installed to quarantine." if quarantine else "Skill installed.",
                findings=[f.model_dump(mode="json") for f in scan.findings],
                size_violations=scan.size_violations,
            )

    def list_skills(self, *, target: str, anima: str | None = None, quarantine: bool = False) -> list[dict[str, Any]]:
        paths = self._target_paths(target, anima=anima)
        root = paths.quarantine_dir if quarantine else paths.active_dir
        return [self._inspect_path(path, paths) for path in sorted(root.glob("*/SKILL.md"))]

    def inspect(self, skill_name: str, *, target: str, anima: str | None = None) -> dict[str, Any]:
        paths = self._target_paths(target, anima=anima)
        name = safe_skill_name(skill_name)
        _validate_skill_name(name, paths)
        for root, state in ((paths.active_dir, "active"), (paths.quarantine_dir, "quarantine")):
            skill_md = root / name / "SKILL.md"
            if skill_md.is_file():
                result = self._inspect_path(skill_md, paths)
                result["state"] = state
                return result
        raise FileNotFoundError(f"Skill not found: {name}")

    def remove(self, skill_name: str, *, target: str, anima: str | None = None) -> SkillHubResult:
        paths = self._target_paths(target, anima=anima)
        name = safe_skill_name(skill_name)
        _validate_skill_name(name, paths)
        for root in (paths.active_dir, paths.quarantine_dir):
            skill_dir = root / name
            if skill_dir.is_dir():
                installed_path = paths.rel(skill_dir / "SKILL.md")
                removed = removed_destination(paths, name)
                removed.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(skill_dir), str(removed))
                try:
                    self._append_lock(paths, action="remove", skill_name=name, installed_path=installed_path)
                except Exception:
                    if not skill_dir.exists():
                        shutil.move(str(removed), str(skill_dir))
                    raise
                shutil.rmtree(removed, ignore_errors=True)
                return SkillHubResult(
                    status="removed",
                    skill_name=name,
                    target=paths.target,
                    installed_path=installed_path,
                    message="Skill removed.",
                )
        raise FileNotFoundError(f"Skill not found: {name}")

    def promote_quarantine(
        self,
        skill_name: str,
        *,
        target: str,
        anima: str | None = None,
        approval_id: str,
        replace: bool = False,
        trust_level: SkillTrustLevel | str = SkillTrustLevel.community,
    ) -> SkillHubResult:
        if not approval_id:
            raise ValueError("approval_id is required")
        trust = _coerce_import_trust(trust_level)
        paths = self._target_paths(target, anima=anima)
        name = safe_skill_name(skill_name)
        _validate_skill_name(name, paths)
        source_dir = paths.quarantine_skill_dir(name)
        if not (source_dir / "SKILL.md").is_file():
            raise FileNotFoundError(f"Quarantine skill not found: {name}")
        scan = self.scanner.scan_skill(source_dir, source="community")
        decision, reason = self.scanner.should_allow(scan, SkillTrustLevel.community, force=False)
        if scan.verdict == SkillScanVerdict.dangerous or scan.size_violations:
            return self._blocked_result(name, paths.target, scan, decision, reason)
        if decision is False:
            return self._blocked_result(name, paths.target, scan, decision, reason)
        destination = paths.active_skill_dir(name)
        prepared = pending_destination(paths, name)
        backup_path = None
        activated = False
        try:
            _copy_tree(source_dir, prepared)
            self._rewrite_skill_metadata(prepared / "SKILL.md", name, trust, scan, approval_id=approval_id)
            backup_path = activate_destination(prepared, destination, replace=replace, paths=paths)
            activated = True
            installed_path = paths.rel(destination / "SKILL.md")
            self._append_lock(paths, action="promote", skill_name=name, scan=scan, installed_path=installed_path)
        except Exception:
            shutil.rmtree(prepared, ignore_errors=True)
            if activated:
                rollback_activation(destination, backup_path, paths)
            raise
        shutil.rmtree(source_dir, ignore_errors=True)
        return SkillHubResult(
            status="promoted",
            skill_name=name,
            target=paths.target,
            installed_path=installed_path,
            backup_path=backup_path,
            scan_verdict=scan.verdict.value,
            decision=decision,
            reason=reason,
            message="Quarantine skill promoted.",
        )

    def _stage_source(self, source: str, staging_root: Path) -> SkillBundle:
        if source.startswith("github:"):
            from core.skills.sources.github import stage_github_source

            staged, commit = stage_github_source(source, staging_root)
            source_type = "github"
        elif source.startswith(("http://", "https://")):
            from core.skills.sources.url import stage_url_source

            staged = stage_url_source(source, staging_root)
            commit = None
            source_type = "url"
        else:
            from core.skills.sources.local import stage_local_source

            staged = stage_local_source(source, staging_root)
            commit = None
            source_type = "local"
        skill_md = staged / "SKILL.md"
        if not skill_md.is_file():
            raise FileNotFoundError("Staged skill bundle does not contain SKILL.md")
        return SkillBundle(
            source_type=source_type,
            source_identifier=source,
            staging_path=str(staged),
            skill_md_path=str(skill_md),
            resolved_commit=commit,
        )

    def _target_paths(self, target: str, *, anima: str | None) -> _TargetPaths:
        if target == "common":
            common_dir = self.data_dir / "common_skills"
            return _TargetPaths(
                target="common",
                active_dir=common_dir / "community",
                quarantine_dir=common_dir / "quarantine",
                lock_path=self.data_dir / "shared" / "skill_hub_lock.jsonl",
                backup_dir=self.data_dir / "shared" / "skill_hub_backups",
                rel_base=self.data_dir,
            )
        if target == "personal":
            if not anima:
                raise ValueError("--anima is required for personal skill operations")
            anima_dir = self.data_dir / "animas" / _safe_anima_name(anima)
            return _TargetPaths(
                target="personal",
                active_dir=anima_dir / "skills",
                quarantine_dir=anima_dir / "skills" / "quarantine",
                lock_path=anima_dir / "state" / "skill_hub_lock.jsonl",
                backup_dir=anima_dir / "state" / "skill_hub_backups",
                rel_base=anima_dir,
            )
        raise ValueError("target must be 'personal' or 'common'")

    def _install_staged_bundle(
        self,
        staged: Path,
        destination: Path,
        *,
        skill_name: str,
        trust_level: SkillTrustLevel,
        scan: ScanResult,
        source: SkillBundle,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_tree(staged, destination)
        self._rewrite_skill_metadata(destination / "SKILL.md", skill_name, trust_level, scan, source=source)

    def _rewrite_skill_metadata(
        self,
        skill_md: Path,
        skill_name: str,
        trust_level: SkillTrustLevel,
        scan: ScanResult,
        *,
        source: SkillBundle | None = None,
        approval_id: str | None = None,
    ) -> None:
        meta, body = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        meta["name"] = skill_name
        meta["trust_level"] = trust_level.value
        meta["security"] = scan_security_metadata(scan)
        if source is not None:
            meta["source"] = {
                "type": source.source_type,
                "identifier": source.source_identifier,
                "origin": "skill_hub",
                "resolved_commit": source.resolved_commit,
            }
        if approval_id is not None:
            meta["approval_id"] = approval_id
            meta["approved_at"] = now_iso()
        frontmatter = yaml.dump(meta, allow_unicode=True, default_flow_style=False, sort_keys=False).strip()
        skill_md.write_text(f"---\n{frontmatter}\n---\n\n{body.lstrip()}", encoding="utf-8")

    def _append_lock(
        self,
        paths: _TargetPaths,
        *,
        action: str,
        skill_name: str,
        source: SkillBundle | None = None,
        scan: ScanResult | None = None,
        installed_path: str | None = None,
    ) -> None:
        paths.lock_path.parent.mkdir(parents=True, exist_ok=True)
        entry = SkillHubLockEntry(
            ts=now_iso(),
            action=action,
            skill_name=skill_name,
            target=paths.target,
            source_type=source.source_type if source else None,
            source_identifier=source.source_identifier if source else None,
            resolved_commit=source.resolved_commit if source else None,
            scan_verdict=scan.verdict.value if scan else None,
            installed_path=installed_path,
            actor=self.actor,
        )
        with paths.lock_path.open("a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

    def _inspect_path(self, skill_md: Path, paths: _TargetPaths) -> dict[str, Any]:
        meta = load_skill_metadata(skill_md)
        return {
            "name": meta.name,
            "description": meta.description,
            "path": paths.rel(skill_md),
            "trust_level": meta.trust_level.value,
            "scan_verdict": meta.security.verdict.value,
            "is_common": paths.target == "common",
        }

    @staticmethod
    def _blocked_result(
        skill_name: str,
        target: str,
        scan: ScanResult,
        decision: bool | None,
        reason: str,
    ) -> SkillHubResult:
        return SkillHubResult(
            status="blocked",
            skill_name=skill_name,
            target=target,
            scan_verdict=scan.verdict.value,
            decision=decision,
            reason=reason,
            message="Skill import blocked by scan policy.",
            findings=[f.model_dump(mode="json") for f in scan.findings],
            size_violations=scan.size_violations,
        )

    @staticmethod
    def _approval_required_result(skill_name: str, target: str, scan: ScanResult, reason: str) -> SkillHubResult:
        return SkillHubResult(
            status="approval_required",
            skill_name=skill_name,
            target=target,
            scan_verdict=scan.verdict.value,
            decision=None,
            reason=reason,
            message="Human approval is required before install.",
            findings=[f.model_dump(mode="json") for f in scan.findings],
            size_violations=scan.size_violations,
        )


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    shutil.copytree(source, destination, symlinks=False)


def _safe_anima_name(value: str) -> str:
    name = str(value).strip()
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts or "/" in name or "\\" in name:
        raise ValueError(f"Invalid anima name: {value}")
    return name


def _coerce_import_trust(value: SkillTrustLevel | str) -> SkillTrustLevel:
    trust = value if isinstance(value, SkillTrustLevel) else SkillTrustLevel(str(value))
    if trust not in {SkillTrustLevel.community, SkillTrustLevel.untrusted}:
        raise ValueError("Skill Hub imports may only be installed as community or untrusted")
    return trust


def _validate_skill_name(skill_name: str, paths: _TargetPaths) -> None:
    if paths.target == "personal" and skill_name == "quarantine":
        raise ValueError("'quarantine' is reserved for personal skill quarantine storage")


def result_json(value: SkillHubResult | list[dict[str, Any]] | dict[str, Any]) -> str:
    if isinstance(value, SkillHubResult):
        data = value.to_dict()
    else:
        data = value
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)
