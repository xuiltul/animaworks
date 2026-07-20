from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for internal cache path classification."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from core.execution._sdk_security import _check_a1_file_access
from core.file_access_policy import (
    find_denied_root,
    find_internal_cache_root,
    load_denied_roots,
    shell_internal_deny_paths,
)
from core.tooling.handler import ToolHandler


def test_internal_cache_matcher_covers_atomic_and_vector_variants(tmp_path: Path) -> None:
    anima_dir = tmp_path / "anima"
    anima_dir.mkdir()

    protected = [
        anima_dir / "state" / "bm25_longterm_index.json",
        anima_dir / "state" / ".bm25_longterm_index.json.random.tmp",
        anima_dir / "archive" / "vectordb-corrupt-1" / "chroma.sqlite3",
        anima_dir / "vectordb" / "chroma.sqlite3",
        anima_dir / "vectordb.staging-42" / "chroma.sqlite3",
        anima_dir / ".codex_home" / "auth.json",
    ]

    assert all(find_internal_cache_root(path, anima_dir) is not None for path in protected)
    assert find_internal_cache_root(anima_dir / "state" / "current_state.md", anima_dir) is None
    assert find_internal_cache_root(anima_dir / "knowledge" / "allowed.md", anima_dir) is None


def test_internal_cache_matcher_blocks_symlinks_in_both_directions(tmp_path: Path) -> None:
    anima_dir = tmp_path / "anima"
    archive = anima_dir / "archive"
    allowed = anima_dir / "knowledge"
    external = tmp_path / "external"
    archive.mkdir(parents=True)
    allowed.mkdir()
    external.mkdir()
    cache = archive / "cache.txt"
    cache.write_text("secret", encoding="utf-8")
    (allowed / "into-cache").symlink_to(cache)
    (archive / "out-of-cache").symlink_to(external / "public.txt")

    assert find_internal_cache_root(allowed / "into-cache", anima_dir) is not None
    assert find_internal_cache_root(archive / "out-of-cache", anima_dir) is not None


def test_shell_internal_denies_include_existing_vector_variants(tmp_path: Path) -> None:
    anima_dir = tmp_path / "anima"
    anima_dir.mkdir()
    staging = anima_dir / "vectordb.staging-123"
    staging.mkdir()

    denied = shell_internal_deny_paths(anima_dir)

    assert (anima_dir / "state").resolve() in denied
    assert (anima_dir / "archive").resolve() in denied
    assert (anima_dir / ".codex_home").resolve() in denied
    assert (anima_dir / "vectordb").resolve() in denied
    assert anima_dir.resolve() / "vectordb-*" in denied
    assert anima_dir.resolve() / "vectordb.*" in denied
    assert anima_dir.resolve() / "vectordb_*" in denied
    assert staging.resolve() in denied


def test_company_denies_merge_with_config_and_follow_symlinks(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = data_dir / "animas" / "agent"
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text(json.dumps({"company": "alpha"}), encoding="utf-8")
    explicit = tmp_path / "explicit-private"
    (anima_dir / "permissions.json").write_text(
        json.dumps({"version": 1, "file_roots_denied": [str(explicit)]}),
        encoding="utf-8",
    )

    own = data_dir / "companies" / "alpha"
    foreign = data_dir / "companies" / "beta"
    own.mkdir(parents=True)
    foreign.mkdir()
    own_file = own / "own.txt"
    foreign_file = foreign / "secret.txt"
    own_file.write_text("own", encoding="utf-8")
    foreign_file.write_text("secret", encoding="utf-8")
    alias = tmp_path / "foreign-alias"
    alias.symlink_to(foreign, target_is_directory=True)

    denied = load_denied_roots(anima_dir)

    assert explicit.resolve() in denied
    assert foreign.resolve() in denied
    assert own.resolve() not in denied
    assert find_denied_root(own_file, denied) is None
    assert find_denied_root(alias / "secret.txt", denied) == foreign.resolve()


def test_nested_deny_roots_collapse_into_outer_root(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = data_dir / "animas" / "agent"
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text(json.dumps({"company": "alpha"}), encoding="utf-8")

    own = data_dir / "companies" / "alpha"
    foreign = data_dir / "companies" / "beta"
    own.mkdir(parents=True)
    nested = foreign / "shared" / "beta"
    nested.mkdir(parents=True)
    # Legacy path kept as a compatibility symlink into the foreign company
    # tree — exactly the post-split layout that produced nested deny mounts.
    alias = data_dir / "shared-beta"
    alias.symlink_to(nested, target_is_directory=True)
    (anima_dir / "permissions.json").write_text(
        json.dumps({"version": 1, "file_roots_denied": [str(alias)]}),
        encoding="utf-8",
    )

    denied = load_denied_roots(anima_dir)

    assert foreign.resolve() in denied
    assert nested.resolve() not in denied
    assert find_denied_root(nested / "secret.txt", denied) == foreign.resolve()


def test_company_derived_deny_is_enforced_by_file_tools(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = data_dir / "animas" / "agent"
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text(json.dumps({"company": "alpha"}), encoding="utf-8")
    (anima_dir / "permissions.json").write_text(
        json.dumps({"version": 1, "file_roots": ["/"]}),
        encoding="utf-8",
    )
    own = data_dir / "companies" / "alpha"
    foreign = data_dir / "companies" / "beta"
    own.mkdir(parents=True)
    foreign.mkdir()
    own_file = own / "own.txt"
    foreign_file = foreign / "secret.txt"
    own_file.write_text("own", encoding="utf-8")
    foreign_file.write_text("secret", encoding="utf-8")

    handler = ToolHandler(anima_dir=anima_dir, memory=MagicMock(), tool_registry=[])

    assert handler._check_file_permission(str(own_file)) is None
    denied = handler._check_file_permission(str(foreign_file))
    assert denied is not None
    assert json.loads(denied)["error_type"] == "PermissionDenied"
    assert _check_a1_file_access(str(foreign_file), anima_dir, write=False) is not None

    (anima_dir / "status.json").write_text("{}", encoding="utf-8")
    assert handler._check_file_permission(str(foreign_file)) is None
    assert _check_a1_file_access(str(foreign_file), anima_dir, write=False) is None
