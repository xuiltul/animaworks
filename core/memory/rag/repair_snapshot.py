"""Private inputs and metadata publication for the existing RAG rebuild path.

Only explicit full rebuilds use these helpers. They are not a writer lock:
the caller still has to quiesce writers for a strictly current promotion.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from core.i18n import t
from core.memory._io import atomic_write_text

ARTIFACT_DIR = ".rebuild"


@dataclass
class RebuildInputs:
    data_dir: Path
    anima_dir: Path
    manifest: dict
    file_stats: dict[str, os.stat_result]


def _source_files(root: Path) -> list[Path]:
    if root.is_symlink():
        raise ValueError(t("rag.rebuild_symlink_input", path=root))
    if not root.exists():
        return []
    if root.is_file():
        return [root]
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(t("rag.rebuild_symlink_input", path=path))
        if path.is_file():
            files.append(path)
    return files


def _fingerprint(root: Path) -> dict:
    files = {}
    for path in _source_files(root):
        before = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        after = path.stat()
        if (before.st_mtime_ns, before.st_size, before.st_ctime_ns) != (
            after.st_mtime_ns,
            after.st_size,
            after.st_ctime_ns,
        ):
            raise RuntimeError(t("rag.rebuild_input_changed"))
        key = "." if path == root else path.relative_to(root).as_posix()
        files[key] = {"sha256": digest, "mtime_ns": after.st_mtime_ns, "size": after.st_size}
    return {"exists": root.exists(), "files": files}


@contextmanager
def snapshot_inputs(anima_dir: Path, *, include_shared: bool) -> Iterator[RebuildInputs]:
    from core.company_resources import get_company_resources
    from core.paths import get_common_knowledge_dir, get_common_skills_dir, get_data_dir

    base = get_data_dir()
    name = anima_dir.name
    relative_anima = Path("animas") / name
    sources = [
        (anima_dir / scope, relative_anima / scope)
        for scope in ("knowledge", "episodes", "procedures", "skills", "facts")
    ]
    sources.extend(
        (anima_dir / path, relative_anima / path)
        for path in (
            "status.json",
            "state/conversation.json",
            "state/entity_registry.json",
            "state/skill_curator.jsonl",
        )
    )
    sources.extend((base / path, Path(path)) for path in (".ragignore", "config.json"))
    if include_shared:
        sources.extend(
            ((get_common_knowledge_dir(), Path("common_knowledge")), (get_common_skills_dir(), Path("common_skills")))
        )
        company = get_company_resources(anima_dir, data_dir=base)
        if company is not None:
            for scope, source in (("knowledge", company.knowledge_dir), ("skills", company.skills_dir)):
                sources.append((source, Path("companies") / company.name / scope))

    # Copy only source inputs, never live vectordb, index_meta or automatic
    # upsert quarantine state. Metadata/registry repair writes stay private.
    with tempfile.TemporaryDirectory(prefix=".rag-rebuild-inputs-", dir=anima_dir) as temporary:
        snapshot = Path(temporary)
        fingerprints = {str(source): _fingerprint(source) for source, _dest in sources}
        stats = {}
        for source, destination in sources:
            for path in _source_files(source):
                target = snapshot / destination
                if path != source:
                    target = target / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                original_stat = path.stat()
                shutil.copy2(path, target)
                stats[str(target)] = original_stat
            if source.is_dir():
                (snapshot / destination).mkdir(parents=True, exist_ok=True)
            if _fingerprint(snapshot / destination) != fingerprints[str(source)]:
                raise RuntimeError(t("rag.rebuild_input_changed"))
        if any(_fingerprint(source) != fingerprints[str(source)] for source, _dest in sources):
            raise RuntimeError(t("rag.rebuild_input_changed"))
        snapshot_anima = snapshot / relative_anima
        snapshot_anima.mkdir(parents=True, exist_ok=True)
        yield RebuildInputs(
            snapshot, snapshot_anima, {"owner": str(anima_dir.resolve()), "sources": fingerprints}, stats
        )


def save_rebuild_artifacts(staging: Path, inputs: RebuildInputs) -> None:
    target = staging / ARTIFACT_DIR
    target.mkdir()
    metadata = inputs.anima_dir / "index_meta.json"
    # Empty personal input can legitimately produce no index_meta file.
    contents = metadata.read_text(encoding="utf-8") if metadata.exists() else "{}"
    value = json.loads(contents)
    if not isinstance(value, dict):
        raise ValueError(t("rag.rebuild_invalid_metadata"))
    atomic_write_text(target / "index_meta.json", contents)
    atomic_write_text(target / "sources.json", json.dumps(inputs.manifest))


def validate_rebuild_sources(staging: Path, anima_dir: Path) -> None:
    manifest = json.loads((staging / ARTIFACT_DIR / "sources.json").read_text(encoding="utf-8"))
    if manifest.get("owner") != str(anima_dir.resolve()) or not isinstance(manifest.get("sources"), dict):
        raise ValueError(t("rag.rebuild_invalid_manifest"))
    for source, expected in manifest["sources"].items():
        if _fingerprint(Path(source)) != expected:
            raise RuntimeError(t("rag.rebuild_input_changed"))


def publish_rebuild_metadata(anima_dir: Path) -> None:
    path = anima_dir / "vectordb" / ARTIFACT_DIR / "index_meta.json"
    content = path.read_text(encoding="utf-8")
    if not isinstance(json.loads(content), dict):
        raise ValueError(t("rag.rebuild_invalid_metadata"))
    atomic_write_text(anima_dir / "index_meta.json", content)
