#!/usr/bin/env python3
from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""One-shot runtime bloat cleanup with archive-before-delete semantics."""

import argparse
import json
import logging
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from core.paths import get_data_dir
from core.time_utils import now_local, today_local
from core.tmp_cleanup import format_size

logger = logging.getLogger("animaworks.cleanup_runtime")
_CORRUPT_VECTORDB_RE = re.compile(r"^(?:vectordb-corrupt|corrupt-vectordb)[-_](?P<stamp>\d{8}[-_]?\d{6}|\d{14})")


@dataclass(frozen=True)
class CleanupTarget:
    """One path selected for runtime cleanup."""

    path: Path
    reason: str
    size_bytes: int


@dataclass(frozen=True)
class CleanupResult:
    """Result of a cleanup run."""

    dry_run: bool
    target_count: int
    estimated_reclaim_bytes: int
    archive_path: Path | None
    archiver: str
    deleted_count: int
    deleted_bytes: int
    skipped_open: tuple[Path, ...]
    errors: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "target_count": self.target_count,
            "estimated_reclaim_bytes": self.estimated_reclaim_bytes,
            "archive_path": str(self.archive_path) if self.archive_path else None,
            "archiver": self.archiver,
            "deleted_count": self.deleted_count,
            "deleted_bytes": self.deleted_bytes,
            "skipped_open": [str(path) for path in self.skipped_open],
            "errors": list(self.errors),
        }


def collect_cleanup_targets(
    data_dir: Path,
    *,
    corrupt_archive_age_days: int = 90,
) -> list[CleanupTarget]:
    """Collect explicit bloat residue targets without touching memory content."""
    root = data_dir.resolve()
    targets: list[CleanupTarget] = []

    animas_dir = root / "animas"
    if animas_dir.exists():
        for path in sorted(animas_dir.rglob("*.bloated.bak")):
            _add_target(targets, path, "bloated activity log backup")

        cutoff = now_local() - timedelta(days=corrupt_archive_age_days)
        for archive_dir in sorted(animas_dir.glob("*/archive")):
            if not archive_dir.is_dir():
                continue
            for path in sorted(archive_dir.iterdir()):
                if not path.is_dir():
                    continue
                if not (path.name.startswith("vectordb-corrupt-") or path.name.startswith("corrupt-vectordb-")):
                    continue
                archive_time = _corrupt_archive_datetime(path)
                if archive_time is None:
                    logger.warning("Skipping corrupt vectordb archive with unparsable timestamp: %s", path)
                    continue
                if archive_time >= cutoff:
                    continue
                _add_target(targets, path, f"corrupt vectordb archive older than {corrupt_archive_age_days}d")

    for path in sorted(root.glob("config.json.bak*")):
        _add_target(targets, path, "root config backup")
    for path in sorted(root.glob("*.jsonl")):
        _add_target(targets, path, "root debug jsonl dump")
    for name in ("mkdir", "Directories created"):
        path = root / name
        if path.exists():
            _add_target(targets, path, "accidental root artifact")

    root_vectordb = root / "vectordb"
    if root_vectordb.exists():
        _add_target(targets, root_vectordb, "abandoned root vectordb")

    deduped: dict[Path, CleanupTarget] = {}
    for target in targets:
        deduped[target.path.resolve()] = target
    return sorted(deduped.values(), key=lambda item: str(item.path))


def execute_cleanup(
    data_dir: Path,
    *,
    execute: bool = False,
    archive_name: str | None = None,
    corrupt_archive_age_days: int = 90,
) -> CleanupResult:
    """Run dry-run or archive+delete cleanup for selected targets."""
    root = data_dir.resolve()
    targets = collect_cleanup_targets(root, corrupt_archive_age_days=corrupt_archive_age_days)
    estimated = sum(target.size_bytes for target in targets)
    if not execute or not targets:
        return CleanupResult(
            dry_run=not execute,
            target_count=len(targets),
            estimated_reclaim_bytes=estimated,
            archive_path=None,
            archiver="none",
            deleted_count=0,
            deleted_bytes=0,
            skipped_open=(),
            errors=(),
        )

    eligible: list[CleanupTarget] = []
    skipped_open: list[Path] = []
    for target in targets:
        if _path_has_open_files(target.path):
            skipped_open.append(target.path)
            logger.warning("Cleanup target is open; skipping: %s", target.path)
        else:
            eligible.append(target)

    if not eligible:
        return CleanupResult(
            dry_run=False,
            target_count=len(targets),
            estimated_reclaim_bytes=estimated,
            archive_path=None,
            archiver="none",
            deleted_count=0,
            deleted_bytes=0,
            skipped_open=tuple(skipped_open),
            errors=(),
        )

    archive_dir = root / "archive" / (archive_name or f"cleanup-{today_local().strftime('%Y%m%d')}")
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"runtime-cleanup-{now_local().strftime('%Y%m%d%H%M%S')}.tar.zst"
    try:
        archiver = _archive_targets(root, archive_path, eligible)
    except RuntimeError as exc:
        return CleanupResult(
            dry_run=False,
            target_count=len(targets),
            estimated_reclaim_bytes=estimated,
            archive_path=None,
            archiver="none",
            deleted_count=0,
            deleted_bytes=0,
            skipped_open=tuple(skipped_open),
            errors=(f"archive failed: {exc}",),
        )

    deleted_count = 0
    deleted_bytes = 0
    errors: list[str] = []
    for target in eligible:
        try:
            _remove_path(target.path)
            deleted_count += 1
            deleted_bytes += target.size_bytes
        except OSError as exc:
            errors.append(f"{target.path}: {exc}")

    return CleanupResult(
        dry_run=False,
        target_count=len(targets),
        estimated_reclaim_bytes=estimated,
        archive_path=archive_path,
        archiver=archiver,
        deleted_count=deleted_count,
        deleted_bytes=deleted_bytes,
        skipped_open=tuple(skipped_open),
        errors=tuple(errors),
    )


def _add_target(targets: list[CleanupTarget], path: Path, reason: str) -> None:
    if path.is_symlink() or not path.exists():
        return
    targets.append(CleanupTarget(path=path, reason=reason, size_bytes=_path_size(path)))


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file() and not child.is_symlink())


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _corrupt_archive_datetime(path: Path) -> datetime | None:
    match = _CORRUPT_VECTORDB_RE.match(path.name)
    if not match:
        return None
    stamp = match.group("stamp").replace("_", "").replace("-", "")
    try:
        return datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=now_local().tzinfo)
    except ValueError:
        return None


def _path_has_open_files(path: Path) -> bool:
    """Return true when lsof can prove that *path* is open by a process."""
    lsof = shutil.which("lsof")
    if not lsof:
        return False
    cmd = [lsof, "+D", str(path)] if path.is_dir() else [lsof, "--", str(path)]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return result.returncode == 0


def _archive_targets(root: Path, archive_path: Path, targets: list[CleanupTarget]) -> str:
    rels = [str(target.path.resolve().relative_to(root)) for target in targets]
    tar = shutil.which("tar")
    if tar:
        result = subprocess.run(
            [tar, "--zstd", "-cf", str(archive_path), "-C", str(root), "--", *rels],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return "tar --zstd"

    zstd = shutil.which("zstd")
    if not zstd:
        raise RuntimeError("zstd archiver is unavailable")

    with tempfile.NamedTemporaryFile(
        prefix="runtime-cleanup-", suffix=".tar", dir=archive_path.parent, delete=False
    ) as f:
        temp_tar = Path(f.name)
    try:
        with tarfile.open(temp_tar, "w") as archive:
            for target in targets:
                archive.add(target.path, arcname=target.path.resolve().relative_to(root))
        result = subprocess.run(
            [zstd, "-q", "-f", "-o", str(archive_path), str(temp_tar)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            archive_path.unlink(missing_ok=True)
            detail = result.stderr.strip() or f"zstd exited with {result.returncode}"
            raise RuntimeError(detail)
    finally:
        temp_tar.unlink(missing_ok=True)
    return "python tarfile + zstd"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=get_data_dir(), help="Runtime data dir")
    parser.add_argument("--execute", action="store_true", help="Archive selected targets, then delete them")
    parser.add_argument("--archive-name", help="Archive subdirectory name under archive/")
    parser.add_argument("--corrupt-archive-age-days", type=int, default=90)
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output machine-readable JSON")
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    targets = collect_cleanup_targets(data_dir, corrupt_archive_age_days=args.corrupt_archive_age_days)
    result = execute_cleanup(
        data_dir,
        execute=args.execute,
        archive_name=args.archive_name,
        corrupt_archive_age_days=args.corrupt_archive_age_days,
    )

    if args.json_output:
        payload = result.as_dict()
        payload["targets"] = [
            {"path": str(target.path), "reason": target.reason, "size_bytes": target.size_bytes} for target in targets
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("Runtime cleanup" + (" (execute)" if args.execute else " (dry-run)"))
    print(f"Data dir: {data_dir}")
    print(f"Targets: {len(targets)}")
    print(f"Estimated reclaim: {format_size(result.estimated_reclaim_bytes)}")
    for target in targets:
        print(f"- {format_size(target.size_bytes):>9} {target.path} [{target.reason}]")
    if args.execute:
        print(f"Archive: {result.archive_path} ({result.archiver})")
        print(f"Deleted: {result.deleted_count} targets, {format_size(result.deleted_bytes)}")
        if result.skipped_open:
            print("Skipped open targets:")
            for path in result.skipped_open:
                print(f"- {path}")
        for error in result.errors:
            print(f"ERROR: {error}")


if __name__ == "__main__":
    main()
