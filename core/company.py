# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Company membership and cross-company boundary helpers.

Membership is deliberately resolved from each anima's ``status.json`` on
every call.  This keeps company assignment changes effective without a server
restart and avoids making the synchronized ``config.json`` entry authoritative.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.config.models import read_anima_company
from core.i18n import t

logger = logging.getLogger(__name__)

_COMPANY_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
_COMPANY_DIRECTORIES = ("knowledge", "skills", "shared", "credentials")
_ADOPT_DESTINATIONS = frozenset((*_COMPANY_DIRECTORIES, "."))


class CompanyError(Exception):
    """Raised when a company management operation cannot be completed."""


class SplitExecutionError(CompanyError):
    """A split execution stopped after completing one or more operations."""

    def __init__(self, message: str, completed_lines: Sequence[str]) -> None:
        super().__init__(message)
        self.completed_lines = list(completed_lines)


@dataclass(frozen=True, slots=True)
class CompanySummary:
    """One row in the company listing."""

    name: str
    display_name: str
    member_count: int


@dataclass(frozen=True, slots=True)
class AdoptResult:
    """The filesystem changes made by one asset adoption."""

    source: Path
    destination: Path
    backup_path: Path
    symlink_created: bool


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Summary of a completed company export."""

    output_dir: Path
    members: tuple[str, ...]
    skipped_symlinks: tuple[str, ...]
    scan_hit_count: int


def _resolve_data_dir(data_dir: Path | None) -> Path:
    if data_dir is not None:
        return data_dir
    from core.paths import get_data_dir

    return get_data_dir()


def _resolve_animas_dir(
    *,
    data_dir: Path | None,
    animas_dir: Path | None,
) -> Path:
    if animas_dir is not None:
        return animas_dir
    return _resolve_data_dir(data_dir) / "animas"


def _company_dir(company_name: str, data_dir: Path) -> Path | None:
    """Resolve a company directory without allowing membership path escape."""
    companies_dir = (data_dir / "companies").resolve()
    candidate = (companies_dir / company_name).resolve()
    if candidate.parent != companies_dir:
        logger.warning("Ignoring unsafe company name: %r", company_name)
        return None
    return candidate


def _validate_company_name(name: str) -> str:
    if not isinstance(name, str) or _COMPANY_NAME_RE.fullmatch(name) is None:
        raise CompanyError(f"Invalid company name: {name!r}")
    return name


def _managed_company_dir(name: str, data_dir: Path) -> Path:
    name = _validate_company_name(name)
    companies_root = data_dir / "companies"
    lexical_directory = companies_root / name
    if companies_root.is_symlink() or lexical_directory.is_symlink():
        raise CompanyError(f"Company management path cannot be a symlink: {lexical_directory}")
    directory = _company_dir(name, data_dir)
    if directory is None:
        raise CompanyError(f"Unsafe company path: {name}")
    return directory


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_json_object(path: Path, *, missing_ok: bool = False) -> dict[str, Any]:
    if missing_ok and not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CompanyError(f"File not found: {path}") from exc
    except (json.JSONDecodeError, OSError) as exc:
        raise CompanyError(f"Failed to read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CompanyError(f"Expected a JSON object: {path}")
    return value


def _company_is_complete(directory: Path) -> bool:
    config_path = directory / "company.json"
    if not config_path.is_file() or not (directory / "vision.md").is_file():
        return False
    if any(not (directory / child).is_dir() for child in _COMPANY_DIRECTORIES):
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(config, dict) and all(key in config for key in ("name", "display_name", "created_at"))


def create_company(
    name: str,
    display_name: str | None = None,
    data_dir: Path | None = None,
) -> bool:
    """Create or complete a company skeleton without overwriting existing values."""
    root = _resolve_data_dir(data_dir).resolve()
    directory = _managed_company_dir(name, root)
    if directory.exists() and not directory.is_dir():
        raise CompanyError(f"Company path is not a directory: {directory}")

    changed = False
    if not directory.exists():
        directory.mkdir(parents=True)
        changed = True

    config_path = directory / "company.json"
    if config_path.exists():
        config = _read_json_object(config_path)
    else:
        config = {}
    effective_display_name = display_name.strip() if isinstance(display_name, str) and display_name.strip() else name
    additions: dict[str, Any] = {
        "name": name,
        "display_name": effective_display_name,
        "created_at": datetime.now(UTC).isoformat(),
    }
    missing = {key: value for key, value in additions.items() if key not in config}
    if missing:
        config.update(missing)
        try:
            _atomic_write_json(config_path, config)
        except OSError as exc:
            raise CompanyError(f"Failed to write {config_path}: {exc}") from exc
        changed = True

    vision_path = directory / "vision.md"
    if not vision_path.exists():
        configured_display = config.get("display_name")
        vision_display = configured_display if isinstance(configured_display, str) and configured_display else name
        try:
            vision_path.write_text(
                t("company.vision_placeholder", display_name=vision_display) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise CompanyError(f"Failed to write {vision_path}: {exc}") from exc
        changed = True

    for child in _COMPANY_DIRECTORIES:
        child_path = directory / child
        if child_path.exists() and not child_path.is_dir():
            raise CompanyError(f"Company skeleton path is not a directory: {child_path}")
        if not child_path.exists():
            child_path.mkdir()
            changed = True
    return changed


def list_companies(data_dir: Path | None = None) -> tuple[list[CompanySummary], list[str]]:
    """List configured companies and Animas without company membership."""
    root = _resolve_data_dir(data_dir).resolve()
    companies_root = root / "companies"
    member_counts: dict[str, int] = {}
    unassigned: list[str] = []
    animas_root = root / "animas"
    if animas_root.is_dir():
        for anima_dir in sorted(animas_root.iterdir(), key=lambda item: item.name):
            if not anima_dir.is_dir() or anima_dir.is_symlink():
                continue
            company = read_anima_company(anima_dir)
            if company is None:
                unassigned.append(anima_dir.name)
            else:
                member_counts[company] = member_counts.get(company, 0) + 1

    summaries: list[CompanySummary] = []
    if companies_root.is_dir():
        for directory in sorted(companies_root.iterdir(), key=lambda item: item.name):
            if not directory.is_dir() or directory.is_symlink() or _COMPANY_NAME_RE.fullmatch(directory.name) is None:
                continue
            config = read_company_config(directory.name, data_dir=root)
            display_name = directory.name
            if config is not None:
                configured_display = config.get("display_name")
                if isinstance(configured_display, str) and configured_display.strip():
                    display_name = configured_display.strip()
            summaries.append(
                CompanySummary(
                    name=directory.name,
                    display_name=display_name,
                    member_count=member_counts.get(directory.name, 0),
                )
            )
    return summaries, unassigned


def _anima_dir(anima_name: str, data_dir: Path) -> Path:
    if (
        not isinstance(anima_name, str)
        or not anima_name
        or anima_name in {".", ".."}
        or Path(anima_name).name != anima_name
        or "\\" in anima_name
    ):
        raise CompanyError(f"Invalid Anima name: {anima_name!r}")
    animas_root = (data_dir / "animas").resolve()
    candidate = animas_root / anima_name
    if not candidate.is_dir() or candidate.resolve().parent != animas_root:
        raise CompanyError(f"Anima not found: {anima_name}")
    return candidate


def _company_must_exist(company_name: str, data_dir: Path) -> Path:
    directory = _managed_company_dir(company_name, data_dir)
    if not directory.is_dir() or read_company_config(company_name, data_dir=data_dir) is None:
        raise CompanyError(f"Company not found: {company_name}")
    return directory


def _membership_link(company_name: str, anima_name: str, data_dir: Path) -> Path | None:
    if _COMPANY_NAME_RE.fullmatch(company_name) is None:
        return None
    directory = _company_dir(company_name, data_dir)
    return None if directory is None else directory / "animas" / anima_name


def _expected_membership_target(anima_name: str) -> Path:
    return Path("../../../animas") / anima_name


def _membership_link_is_correct(link: Path, anima_name: str) -> bool:
    if not link.is_symlink():
        return False
    try:
        return Path(os.readlink(link)) == _expected_membership_target(anima_name)
    except OSError:
        return False


def _remove_membership_link(link: Path | None) -> None:
    if link is None or not (link.exists() or link.is_symlink()):
        return
    if not link.is_symlink():
        raise CompanyError(f"Refusing to remove non-symlink membership view: {link}")
    try:
        link.unlink()
    except OSError as exc:
        raise CompanyError(f"Failed to remove membership link {link}: {exc}") from exc


def _write_anima_company(anima_dir: Path, company_name: str | None) -> None:
    status_path = anima_dir / "status.json"
    status = _read_json_object(status_path, missing_ok=True)
    if company_name is None:
        status.pop("company", None)
    else:
        status["company"] = company_name
    try:
        _atomic_write_json(status_path, status)
    except OSError as exc:
        raise CompanyError(f"Failed to update {status_path}: {exc}") from exc


def assign_animas(
    anima_names: Sequence[str],
    *,
    company_name: str | None = None,
    unassign: bool = False,
    data_dir: Path | None = None,
) -> list[str]:
    """Assign Animas to a company, or remove their current assignments."""
    if unassign == (company_name is not None):
        raise CompanyError("Specify exactly one of company_name or unassign")
    root = _resolve_data_dir(data_dir).resolve()
    target_directory = None if unassign else _company_must_exist(company_name or "", root)

    names = list(dict.fromkeys(anima_names))
    if not names:
        raise CompanyError("At least one Anima name is required")
    anima_directories = {name: _anima_dir(name, root) for name in names}
    lines: list[str] = []
    for name in names:
        anima_directory = anima_directories[name]
        previous = read_anima_company(anima_directory)
        if unassign:
            if previous is None:
                lines.append(f"SKIP unassign {name}: already unassigned")
                continue
            _remove_membership_link(_membership_link(previous, name, root))
            _write_anima_company(anima_directory, None)
            lines.append(f"UNASSIGN {name} (from {previous})")
            continue

        assert company_name is not None and target_directory is not None
        target_link = target_directory / "animas" / name
        if previous == company_name and _membership_link_is_correct(target_link, name):
            lines.append(f"SKIP assign {name}: already assigned to {company_name}")
            continue
        _remove_membership_link(target_link)
        if previous != company_name:
            _remove_membership_link(_membership_link(previous, name, root) if previous else None)
        _write_anima_company(anima_directory, company_name)
        try:
            target_link.parent.mkdir(parents=True, exist_ok=True)
            target_link.symlink_to(_expected_membership_target(name), target_is_directory=True)
        except OSError as exc:
            raise CompanyError(f"Failed to create membership link {target_link}: {exc}") from exc
        lines.append(f"ASSIGN {name} -> {company_name}")
    return lines


@dataclass(frozen=True, slots=True)
class _AdoptPlan:
    source: Path
    destination: Path
    relative_source: Path


def _normalize_adopt_destination(dest: str | None) -> str | None:
    if dest is None:
        return None
    normalized = dest.strip().rstrip("/") or "."
    if normalized not in _ADOPT_DESTINATIONS:
        choices = ", ".join(sorted(_ADOPT_DESTINATIONS))
        raise CompanyError(f"Invalid adoption destination {dest!r}; choose one of: {choices}")
    return normalized


def _lexical_adopt_source(value: str | Path, data_dir: Path) -> Path:
    given = Path(value).expanduser()
    return given if given.is_absolute() else data_dir / given


def _infer_adopt_destination(source: Path, relative_source: Path) -> str:
    if relative_source.parts and relative_source.parts[0] == "shared":
        return "shared"
    if relative_source.parts and relative_source.parts[0] == "credentials":
        return "credentials"
    if source.suffix.lower() == ".md" and not source.is_dir():
        return "knowledge"
    return "shared"


def _expected_adopt_destination(
    source: Path,
    *,
    company_name: str,
    dest: str | None,
    data_dir: Path,
) -> tuple[Path, Path]:
    try:
        relative_source = source.relative_to(data_dir)
    except ValueError as exc:
        raise CompanyError(f"Adoption source is outside the data directory: {source}") from exc
    selected_dest = _normalize_adopt_destination(dest) or _infer_adopt_destination(source, relative_source)
    company_directory = _managed_company_dir(company_name, data_dir)
    destination_root = company_directory if selected_dest == "." else company_directory / selected_dest
    destination = destination_root / source.name
    return destination, relative_source


def _prepare_adopt(
    value: str | Path,
    *,
    company_name: str,
    dest: str | None,
    data_dir: Path,
) -> _AdoptPlan:
    source = _lexical_adopt_source(value, data_dir)
    if source.is_symlink():
        raise CompanyError(f"Refusing to adopt a symlink; specify its target instead: {source}")
    if not source.exists():
        raise CompanyError(f"Adoption source not found: {source}")
    resolved_source = source.resolve()
    if resolved_source == data_dir:
        raise CompanyError("The data directory itself cannot be adopted")
    try:
        relative_source = resolved_source.relative_to(data_dir)
    except ValueError as exc:
        raise CompanyError(f"Adoption source is outside the data directory: {source}") from exc
    destination, _ = _expected_adopt_destination(
        resolved_source,
        company_name=company_name,
        dest=dest,
        data_dir=data_dir,
    )
    if destination.exists() or destination.is_symlink():
        raise CompanyError(f"Adoption destination already exists: {destination}")
    try:
        destination.relative_to(_managed_company_dir(company_name, data_dir))
    except ValueError as exc:
        raise CompanyError(f"Unsafe adoption destination: {destination}") from exc
    return _AdoptPlan(source=resolved_source, destination=destination, relative_source=relative_source)


def adopt_assets(
    paths: Sequence[str | Path],
    *,
    company_name: str,
    dest: str | None = None,
    leave_symlinks: bool = True,
    data_dir: Path | None = None,
) -> list[AdoptResult]:
    """Move assets into a company, archiving them and leaving relative links."""
    root = _resolve_data_dir(data_dir).resolve()
    _company_must_exist(company_name, root)
    _normalize_adopt_destination(dest)
    if not paths:
        raise CompanyError("At least one adoption path is required")

    plans = [_prepare_adopt(value, company_name=company_name, dest=dest, data_dir=root) for value in paths]
    destinations: set[Path] = set()
    sources = [plan.source for plan in plans]
    for plan in plans:
        if plan.destination in destinations:
            raise CompanyError(f"Multiple adoption sources have the same destination: {plan.destination}")
        destinations.add(plan.destination)
    for index, source in enumerate(sources):
        for other in sources[index + 1 :]:
            if source in other.parents or other in source.parents:
                raise CompanyError(f"Cannot adopt nested sources together: {source}, {other}")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup_root = root / "backup" / f"company-adopt-{timestamp}"
    backup_paths: list[Path] = []
    try:
        for plan in plans:
            backup_path = backup_root / plan.relative_source
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if plan.source.is_dir():
                shutil.copytree(plan.source, backup_path, symlinks=True)
            else:
                shutil.copy2(plan.source, backup_path, follow_symlinks=False)
            backup_paths.append(backup_path)
    except OSError as exc:
        raise CompanyError(f"Failed to archive adoption source: {exc}") from exc

    results: list[AdoptResult] = []
    for plan, backup_path in zip(plans, backup_paths, strict=True):
        try:
            plan.destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(plan.source), str(plan.destination))
            symlink_created = False
            if leave_symlinks:
                relative_target = Path(os.path.relpath(plan.destination, start=plan.source.parent))
                plan.source.symlink_to(relative_target, target_is_directory=plan.destination.is_dir())
                symlink_created = True
        except OSError as exc:
            raise CompanyError(f"Failed to adopt {plan.source} to {plan.destination}: {exc}") from exc
        results.append(
            AdoptResult(
                source=plan.source,
                destination=plan.destination,
                backup_path=backup_path,
                symlink_created=symlink_created,
            )
        )
    return results


def _load_split_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    path = Path(manifest_path).expanduser().resolve()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CompanyError(f"Failed to read split manifest {path}: {exc}") from exc

    value: Any
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CompanyError(f"Invalid JSON manifest {path}: {exc}") from exc
    else:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml
            except ImportError as exc:
                raise CompanyError("YAML manifests require PyYAML; use a JSON manifest instead") from exc
            try:
                value = yaml.safe_load(text)
            except yaml.YAMLError as exc:
                raise CompanyError(f"Invalid YAML manifest {path}: {exc}") from exc

    if not isinstance(value, dict) or not isinstance(value.get("companies"), list):
        raise CompanyError("Split manifest must contain a 'companies' list")
    companies: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, raw_company in enumerate(value["companies"]):
        if not isinstance(raw_company, dict):
            raise CompanyError(f"Manifest company entry {index} must be an object")
        name = _validate_company_name(raw_company.get("name"))
        if name in seen_names:
            raise CompanyError(f"Duplicate company in manifest: {name}")
        seen_names.add(name)
        display_name = raw_company.get("display_name")
        if display_name is not None and not isinstance(display_name, str):
            raise CompanyError(f"display_name for {name} must be a string")
        members = raw_company.get("members", [])
        if not isinstance(members, list) or any(not isinstance(member, str) for member in members):
            raise CompanyError(f"members for {name} must be a list of Anima names")
        adopt = raw_company.get("adopt", [])
        if not isinstance(adopt, list):
            raise CompanyError(f"adopt for {name} must be a list")
        normalized_adopt: list[dict[str, Any]] = []
        for adopt_index, item in enumerate(adopt):
            if not isinstance(item, dict) or not isinstance(item.get("path"), (str, Path)):
                raise CompanyError(f"adopt entry {adopt_index} for {name} requires a path")
            item_dest = item.get("dest")
            if item_dest is not None and not isinstance(item_dest, str):
                raise CompanyError(f"dest in adopt entry {adopt_index} for {name} must be a string")
            _normalize_adopt_destination(item_dest)
            item_symlink = item.get("symlink", True)
            if not isinstance(item_symlink, bool):
                raise CompanyError(f"symlink in adopt entry {adopt_index} for {name} must be a boolean")
            normalized_adopt.append({"path": item["path"], "dest": item_dest, "symlink": item_symlink})
        companies.append(
            {
                "name": name,
                "display_name": display_name,
                "members": list(dict.fromkeys(members)),
                "adopt": normalized_adopt,
            }
        )
    return companies


def _display_company_path(path: Path, data_dir: Path) -> str:
    try:
        return path.relative_to(data_dir).as_posix()
    except ValueError:
        return str(path)


def _split_adopt_status(
    item: dict[str, Any],
    *,
    company_name: str,
    data_dir: Path,
) -> tuple[str | None, _AdoptPlan | None]:
    source = _lexical_adopt_source(item["path"], data_dir)
    item_dest = item.get("dest")
    if source.is_symlink():
        try:
            resolved = source.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise CompanyError(f"Invalid adoption symlink {source}: {exc}") from exc
        destination, _ = _expected_adopt_destination(
            source,
            company_name=company_name,
            dest=item_dest,
            data_dir=data_dir,
        )
        if resolved == destination.resolve() and destination.exists():
            source_display = _display_company_path(source, data_dir)
            destination_display = _display_company_path(destination, data_dir)
            return f"SKIP adopt {source_display}: already moved to {destination_display}", None
        raise CompanyError(f"Adoption source is an unexpected symlink: {source}")
    plan = _prepare_adopt(item["path"], company_name=company_name, dest=item_dest, data_dir=data_dir)
    return None, plan


def split_companies(
    manifest_path: str | Path,
    *,
    execute: bool = False,
    data_dir: Path | None = None,
) -> list[str]:
    """Plan or execute company creation, assignment, and asset adoption."""
    root = _resolve_data_dir(data_dir).resolve()
    companies = _load_split_manifest(manifest_path)
    lines: list[str] = []
    try:
        for company in companies:
            name = company["name"]
            company_directory = _managed_company_dir(name, root)
            if execute:
                changed = create_company(name, company["display_name"], data_dir=root)
                lines.append(f"CREATE {name}" if changed else f"SKIP create {name}: already exists")
            elif _company_is_complete(company_directory):
                lines.append(f"SKIP create {name}: already exists")
            else:
                lines.append(f"DRY-RUN CREATE {name}")

            for member in company["members"]:
                anima_directory = _anima_dir(member, root)
                expected_link = company_directory / "animas" / member
                if read_anima_company(anima_directory) == name and _membership_link_is_correct(expected_link, member):
                    lines.append(f"SKIP assign {member}: already assigned to {name}")
                elif execute:
                    lines.extend(assign_animas([member], company_name=name, data_dir=root))
                else:
                    lines.append(f"DRY-RUN ASSIGN {member} -> {name}")

            for item in company["adopt"]:
                skipped, plan = _split_adopt_status(item, company_name=name, data_dir=root)
                if skipped is not None:
                    lines.append(skipped)
                    continue
                assert plan is not None
                source_display = _display_company_path(plan.source, root)
                destination_display = _display_company_path(plan.destination, root)
                if execute:
                    result = adopt_assets(
                        [plan.source],
                        company_name=name,
                        dest=item.get("dest"),
                        leave_symlinks=item["symlink"],
                        data_dir=root,
                    )[0]
                    lines.append(
                        f"ADOPT {_display_company_path(result.source, root)} -> "
                        f"{_display_company_path(result.destination, root)}"
                    )
                else:
                    lines.append(f"DRY-RUN ADOPT {source_display} -> {destination_display}")
    except SplitExecutionError:
        raise
    except Exception as exc:
        message = str(exc) if isinstance(exc, CompanyError) else f"Unexpected split failure: {exc}"
        raise SplitExecutionError(message, lines) from exc
    return lines


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _copy_export_tree(
    source: Path,
    destination: Path,
    *,
    output_root: Path,
    skipped_symlinks: list[str],
    excluded_names: frozenset[str] = frozenset(),
) -> None:
    """Copy a tree without following links that cross its ownership boundary."""
    source_root = source.resolve()
    excluded_roots = tuple((source / name).resolve(strict=False) for name in excluded_names)
    destination.mkdir(parents=True, exist_ok=True)

    def copy_directory(current_source: Path, current_destination: Path) -> None:
        for item in sorted(current_source.iterdir(), key=lambda candidate: candidate.name):
            if current_source == source and item.name in excluded_names:
                continue
            target = current_destination / item.name
            if item.is_symlink():
                try:
                    link_value = Path(os.readlink(item))
                    resolved_target = (item.parent / link_value).resolve(strict=False)
                except (OSError, RuntimeError):
                    link_value = Path()
                    resolved_target = Path()
                if (
                    link_value
                    and not link_value.is_absolute()
                    and _path_is_within(resolved_target, source_root)
                    and not any(_path_is_within(resolved_target, root) for root in excluded_roots)
                ):
                    try:
                        target.symlink_to(link_value, target_is_directory=item.is_dir())
                    except OSError as exc:
                        raise CompanyError(f"Failed to preserve export symlink {item}: {exc}") from exc
                else:
                    skipped_symlinks.append(target.relative_to(output_root).as_posix())
                continue
            if item.is_dir():
                target.mkdir()
                copy_directory(item, target)
                continue
            if item.is_file():
                try:
                    shutil.copy2(item, target)
                except OSError as exc:
                    raise CompanyError(f"Failed to copy export file {item}: {exc}") from exc

    try:
        copy_directory(source, destination)
    except OSError as exc:
        raise CompanyError(f"Failed to copy export tree {source}: {exc}") from exc


def _redact_credential_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact_credential_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_redact_credential_value(child) for child in value]
    return "REDACTED"


def _collect_vault_references(value: Any) -> set[str]:
    references: set[str] = set()
    if isinstance(value, dict):
        reference = value.get("$vault")
        if isinstance(reference, str) and reference.strip():
            references.add(reference.strip())
        for child in value.values():
            references.update(_collect_vault_references(child))
    elif isinstance(value, list):
        for child in value:
            references.update(_collect_vault_references(child))
    return references


def _credential_leaf_paths(value: Any, prefix: str = "credentials") -> list[str]:
    if isinstance(value, dict):
        paths: list[str] = []
        for key, child in value.items():
            paths.extend(_credential_leaf_paths(child, f"{prefix}.{key}"))
        return paths
    if isinstance(value, list):
        paths = []
        for index, child in enumerate(value):
            paths.extend(_credential_leaf_paths(child, f"{prefix}[{index}]"))
        return paths
    return [prefix]


def _write_export_readme(
    output_dir: Path,
    *,
    company_name: str,
    members: tuple[str, ...],
    credentials: Any,
    vault_references: set[str],
    skipped_symlinks: list[str],
) -> None:
    credential_paths = sorted(set(_credential_leaf_paths(credentials)))
    lines = [
        f"# Company export: {company_name}",
        "",
        "This bundle contains redacted configuration only. Complete every item below in the new environment.",
        "",
        "## Startup checklist",
        "",
        "- [ ] Place `animas/`, `companies/`, `common_knowledge/`, and `common_skills/` under the new data_dir.",
        "- [ ] Review `config.export.json`, save it as `config.json`, and insert real credential values through a secure channel.",
        "- [ ] Migrate the required vault keys separately; no vault secret values are included here.",
        "- [ ] Export and import the Neo4j subgraph for every member, using the Anima name as `group_id`.",
        "- [ ] Rebuild every vectordb/index with the commands listed below.",
        "- [ ] Create and enable the systemd units for the new installation.",
        "- [ ] Verify file ownership and permissions, startup, messaging, company boundaries, memory search, and scheduled work.",
        "",
        "## Credential value locations",
        "",
    ]
    if credential_paths:
        lines.extend(f"- `{path}`" for path in credential_paths)
    else:
        lines.append("- (none in the source config)")
    lines.extend(["", "## Vault keys referenced by config", ""])
    if vault_references:
        lines.extend(f"- `{reference}`" for reference in sorted(vault_references))
    else:
        lines.append("- (none detected; inspect the source vault and deployment secrets separately)")
    lines.extend(["", "## Neo4j migration", ""])
    for member in members:
        lines.append(f"- [ ] Export and import records for `group_id={member}` (the group_id is the Anima name).")
    lines.extend(["", "## Vectordb rebuild", ""])
    lines.extend(f"- [ ] `animaworks index --anima {member}`" for member in members)
    lines.extend(["", "## systemd and verification", ""])
    lines.extend(
        [
            "- [ ] Create environment-specific systemd unit files and enable/start them.",
            "- [ ] Confirm every exported Anima starts and can read only its own Company layer.",
            "- [ ] Confirm credentials, Neo4j retrieval, indexes, messaging, and scheduled jobs work.",
            "",
            "## Skipped symlinks",
            "",
        ]
    )
    if skipped_symlinks:
        lines.extend(f"- `{path}`" for path in skipped_symlinks)
    else:
        lines.append("- (none)")
    try:
        (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        raise CompanyError(f"Failed to write export README: {exc}") from exc


def _other_company_scan_terms(
    company_name: str,
    *,
    data_dir: Path,
) -> list[tuple[str, str, re.Pattern[str]]]:
    terms: list[tuple[str, str, re.Pattern[str]]] = []
    seen: set[str] = set()
    summaries, _ = list_companies(data_dir=data_dir)
    for summary in summaries:
        if summary.name == company_name:
            continue
        company_values = (("company", summary.name), ("display_name", summary.display_name))
        for kind, value in company_values:
            if value and value not in seen:
                seen.add(value)
                terms.append((kind, value, re.compile(re.escape(value))))

        animas_root = data_dir / "animas"
        if not animas_root.is_dir():
            continue
        for anima_dir in sorted(animas_root.iterdir(), key=lambda path: path.name):
            if anima_dir.is_symlink() or not anima_dir.is_dir():
                continue
            if read_anima_company(anima_dir) != summary.name:
                continue
            if anima_dir.name in seen:
                continue
            seen.add(anima_dir.name)
            pattern = re.compile(rf"(?<!\w){re.escape(anima_dir.name)}(?!\w)")
            terms.append(("anima", anima_dir.name, pattern))
    return terms


def _write_scan_report(
    output_dir: Path,
    *,
    company_name: str,
    data_dir: Path,
) -> int:
    terms = _other_company_scan_terms(company_name, data_dir=data_dir)
    reported_by_file: dict[str, list[str]] = {}
    hit_count = 0
    for path in sorted(output_dir.rglob("*")):
        if path.is_symlink() or not path.is_file() or path.name == "scan-report.md":
            continue
        try:
            with path.open("rb") as binary_stream:
                if b"\0" in binary_stream.read(4096):
                    continue
        except OSError:
            continue
        relative = path.relative_to(output_dir).as_posix()
        file_hits: list[str] = []
        file_hit_count = 0
        try:
            with path.open("r", encoding="utf-8") as text_stream:
                for line_number, line in enumerate(text_stream, start=1):
                    for kind, term, pattern in terms:
                        matches = list(pattern.finditer(line))
                        if not matches:
                            continue
                        file_hit_count += len(matches)
                        if len(file_hits) < 5:
                            excerpt = line.strip()
                            if len(excerpt) > 300:
                                excerpt = excerpt[:297] + "..."
                            file_hits.append(f"- `{relative}:{line_number}` [{kind}: `{term}`] {excerpt}")
        except (OSError, UnicodeDecodeError):
            continue
        hit_count += file_hit_count
        if file_hits:
            reported_by_file[relative] = file_hits

    lines = [
        "# Other-company context scan",
        "",
        f"Total hits: {hit_count}",
        "",
        "This report is advisory and does not block export. Review every hit before transfer.",
        "",
    ]
    if reported_by_file:
        for relative, file_hits in reported_by_file.items():
            lines.extend([f"## `{relative}`", "", *file_hits, ""])
    else:
        lines.extend(["No other-company context was detected.", ""])
    try:
        (output_dir / "scan-report.md").write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:
        raise CompanyError(f"Failed to write export scan report: {exc}") from exc
    return hit_count


def export_company(
    company_name: str,
    output_dir: str | Path,
    *,
    data_dir: Path | None = None,
) -> ExportResult:
    """Create a portable, secret-redacted bundle for one company."""
    root = _resolve_data_dir(data_dir).resolve()
    company_directory = _company_must_exist(company_name, root)
    destination = Path(output_dir).expanduser().resolve()
    animas_root = root / "animas"
    members = (
        tuple(
            sorted(
                anima_dir.name
                for anima_dir in animas_root.iterdir()
                if anima_dir.is_dir() and not anima_dir.is_symlink() and read_anima_company(anima_dir) == company_name
            )
        )
        if animas_root.is_dir()
        else ()
    )
    copied_sources = [company_directory, *(animas_root / member for member in members)]
    copied_sources.extend(root / name for name in ("common_knowledge", "common_skills"))
    if any(_path_is_within(destination, source.resolve(strict=False)) for source in copied_sources):
        raise CompanyError(f"Export output directory cannot be inside copied source data: {destination}")
    if destination.is_symlink():
        raise CompanyError(f"Export output directory cannot be a symlink: {destination}")
    if destination.exists():
        if not destination.is_dir():
            raise CompanyError(f"Export output is not a directory: {destination}")
        try:
            if any(destination.iterdir()):
                raise CompanyError(f"Export output directory is not empty: {destination}")
        except OSError as exc:
            raise CompanyError(f"Failed to inspect export output directory {destination}: {exc}") from exc
    else:
        try:
            destination.mkdir(parents=True)
        except OSError as exc:
            raise CompanyError(f"Failed to create export output directory {destination}: {exc}") from exc

    skipped_symlinks: list[str] = []

    for member in members:
        _copy_export_tree(
            animas_root / member,
            destination / "animas" / member,
            output_root=destination,
            skipped_symlinks=skipped_symlinks,
        )
    _copy_export_tree(
        company_directory,
        destination / "companies" / company_name,
        output_root=destination,
        skipped_symlinks=skipped_symlinks,
        excluded_names=frozenset({"animas"}),
    )
    for common_name in ("common_knowledge", "common_skills"):
        common_source = root / common_name
        if common_source.is_dir() and not common_source.is_symlink():
            _copy_export_tree(
                common_source,
                destination / common_name,
                output_root=destination,
                skipped_symlinks=skipped_symlinks,
            )
        else:
            (destination / common_name).mkdir(parents=True, exist_ok=True)
            if common_source.is_symlink():
                skipped_symlinks.append(common_name)

    config_path = root / "config.json"
    config = _read_json_object(config_path)
    source_animas = config.get("animas")
    config["animas"] = (
        {member: source_animas[member] for member in members if member in source_animas}
        if isinstance(source_animas, dict)
        else {}
    )
    source_credentials = config.get("credentials", {})
    vault_references = _collect_vault_references(source_credentials)
    config["credentials"] = _redact_credential_value(source_credentials)
    try:
        _atomic_write_json(destination / "config.export.json", config)
    except OSError as exc:
        raise CompanyError(f"Failed to write exported config: {exc}") from exc

    _write_export_readme(
        destination,
        company_name=company_name,
        members=members,
        credentials=source_credentials,
        vault_references=vault_references,
        skipped_symlinks=skipped_symlinks,
    )
    scan_hit_count = _write_scan_report(
        destination,
        company_name=company_name,
        data_dir=root,
    )
    return ExportResult(
        output_dir=destination,
        members=members,
        skipped_symlinks=tuple(skipped_symlinks),
        scan_hit_count=scan_hit_count,
    )


def read_company_config(
    company_name: str,
    *,
    data_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Read ``companies/<name>/company.json`` when it is a JSON object."""
    if not isinstance(company_name, str) or not company_name.strip():
        return None
    root = _resolve_data_dir(data_dir)
    directory = _company_dir(company_name.strip(), root)
    if directory is None:
        return None
    config_path = directory / "company.json"
    if not config_path.is_file():
        return None
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", config_path, exc)
        return None
    return value if isinstance(value, dict) else None


def get_company(
    anima_name: str,
    *,
    data_dir: Path | None = None,
    animas_dir: Path | None = None,
) -> str | None:
    """Return an anima's current company membership from disk."""
    if not isinstance(anima_name, str) or not anima_name:
        return None
    root = _resolve_animas_dir(data_dir=data_dir, animas_dir=animas_dir)
    candidate = (root / anima_name).resolve()
    if candidate.parent != root.resolve():
        logger.warning("Ignoring unsafe anima name: %r", anima_name)
        return None
    return read_anima_company(candidate)


def get_company_display_name(
    company_name: str,
    *,
    data_dir: Path | None = None,
) -> str:
    """Return a company's display name, falling back to its directory name."""
    config = read_company_config(company_name, data_dir=data_dir)
    if config is not None:
        display_name = config.get("display_name")
        if isinstance(display_name, str) and display_name.strip():
            return display_name.strip()
    return company_name


def is_cross_company(
    anima_a: str,
    anima_b: str,
    *,
    data_dir: Path | None = None,
    animas_dir: Path | None = None,
) -> bool:
    """Return whether two assigned animas belong to different companies.

    An unassigned anima remains unrestricted for legacy compatibility.
    """
    root = _resolve_animas_dir(data_dir=data_dir, animas_dir=animas_dir)
    company_a = get_company(anima_a, animas_dir=root)
    company_b = get_company(anima_b, animas_dir=root)
    return company_a is not None and company_b is not None and company_a != company_b
