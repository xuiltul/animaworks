from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for internal cache path classification."""

from pathlib import Path

from core.file_access_policy import find_internal_cache_root, shell_internal_deny_paths


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
