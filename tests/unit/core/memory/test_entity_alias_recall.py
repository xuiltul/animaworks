from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from core.memory.retrieval.entity import (
    EntityBoostConfig,
    apply_entity_boost,
    clear_entity_alias_index_cache,
    load_entity_alias_index,
)


def _write_registry(anima_dir: Path, entities: dict) -> Path:
    state = anima_dir / "state"
    state.mkdir(parents=True, exist_ok=True)
    path = state / "entity_registry.json"
    path.write_text(
        json.dumps({"version": 1, "entities": entities}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture(autouse=True)
def _clear_alias_cache() -> None:
    clear_entity_alias_index_cache()
    yield
    clear_entity_alias_index_cache()


@pytest.mark.unit
def test_alias_resolves_query_form_to_canonical_content(tmp_path: Path) -> None:
    """Query uses alias なつめ; candidate body only has canonical natsume → boosted."""
    anima_dir = tmp_path / "alice"
    _write_registry(
        anima_dir,
        {
            "natsume": {
                "canonical": "natsume",
                "aliases": ["なつめ", "natsume"],
                "source_fact_ids": ["fact-1"],
            }
        },
    )
    candidates = [
        {"content": "generic note without names", "score": 0.50},
        {"content": "natsume の週次タスクを整理した", "score": 0.40},
    ]

    boosted = apply_entity_boost(
        "なつめのタスク",
        candidates,
        EntityBoostConfig(enabled=True, category=None, anima_dir=anima_dir, boost=0.20, max_boost=0.80),
    )

    assert boosted[0]["content"] == "natsume の週次タスクを整理した"
    assert boosted[0]["entity_boost"] == pytest.approx(0.20)
    assert boosted[0]["score"] == pytest.approx(0.60)


@pytest.mark.unit
def test_one_hop_related_entity_gets_smaller_boost(tmp_path: Path) -> None:
    """Y co-mentioned with X via shared fact is boosted less than a direct X hit."""
    anima_dir = tmp_path / "alice"
    _write_registry(
        anima_dir,
        {
            "natsume": {
                "canonical": "natsume",
                "aliases": ["なつめ", "natsume"],
                "source_fact_ids": ["fact-shared"],
            },
            "projectx": {
                "canonical": "ProjectX",
                "aliases": ["projectx", "プロジェクトx"],
                "source_fact_ids": ["fact-shared"],
            },
            "unrelated": {
                "canonical": "unrelated",
                "aliases": ["unrelated"],
                "source_fact_ids": ["fact-other"],
            },
        },
    )
    candidates = [
        {"content": "unrelated system logs only", "score": 0.50, "entities": ["unrelated"]},
        {"content": "ProjectX milestone notes", "score": 0.45, "entities": ["ProjectX"]},
        {"content": "natsume owns this task", "score": 0.40, "entities": ["natsume"]},
    ]

    boosted = apply_entity_boost(
        "なつめの進捗",
        candidates,
        EntityBoostConfig(
            enabled=True,
            category=None,
            anima_dir=anima_dir,
            boost=0.20,
            max_boost=0.80,
            related_boost=0.10,
        ),
    )

    by_content = {row["content"]: row for row in boosted}
    direct = by_content["natsume owns this task"]
    related = by_content["ProjectX milestone notes"]
    none = by_content["unrelated system logs only"]

    assert direct["entity_boost"] == pytest.approx(0.20)
    assert related["entity_boost"] == pytest.approx(0.10)
    assert related["entity_related_overlap"] == ["projectx"]
    assert "entity_boost" not in none
    assert direct["score"] > related["score"] > none["score"]


@pytest.mark.unit
def test_missing_registry_falls_back_to_regex_only(tmp_path: Path) -> None:
    """No registry file → historical phrase intersection still works."""
    anima_dir = tmp_path / "alice"
    anima_dir.mkdir()
    candidates = [
        {"content": "Caroline recommended Becoming Nicole.", "score": 0.30},
        {"content": "generic filler", "score": 0.50},
    ]

    boosted = apply_entity_boost(
        "What did Caroline recommend?",
        candidates,
        EntityBoostConfig(enabled=True, category=None, anima_dir=anima_dir, boost=0.20, max_boost=0.20),
    )

    assert boosted[0]["content"] == "Caroline recommended Becoming Nicole."
    assert boosted[0]["entity_boost"] == pytest.approx(0.20)
    # Alias cross-form must NOT fire without registry.
    no_alias = apply_entity_boost(
        "なつめのタスク",
        [{"content": "natsume task list", "score": 0.4}],
        EntityBoostConfig(enabled=True, category=None, anima_dir=anima_dir, boost=0.20, max_boost=0.80),
    )
    assert "entity_boost" not in no_alias[0]


@pytest.mark.unit
def test_registry_mtime_change_reloads_cache(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    path = _write_registry(
        anima_dir,
        {
            "natsume": {
                "canonical": "natsume",
                "aliases": ["natsume"],
                "source_fact_ids": [],
            }
        },
    )

    first = load_entity_alias_index(anima_dir)
    assert first is not None
    assert "なつめ" not in first.alias_owner

    # Ensure mtime advances on coarse filesystems.
    time.sleep(0.02)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "entities": {
                    "natsume": {
                        "canonical": "natsume",
                        "aliases": ["なつめ", "natsume"],
                        "source_fact_ids": [],
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    # Bump mtime explicitly in case write is too fast.
    now = time.time() + 1.0
    import os

    os.utime(path, (now, now))

    second = load_entity_alias_index(anima_dir)
    assert second is not None
    assert second is not first
    assert first.alias_owner.get("なつめ") is None
    assert second.alias_owner.get("なつめ") == "natsume"


@pytest.mark.unit
def test_related_boost_defaults_to_half_primary(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    _write_registry(
        anima_dir,
        {
            "alpha": {
                "canonical": "alpha",
                "aliases": ["alpha"],
                "source_fact_ids": ["f1"],
            },
            "beta": {
                "canonical": "beta",
                "aliases": ["beta"],
                "source_fact_ids": ["f1"],
            },
        },
    )
    boosted = apply_entity_boost(
        "alpha status",
        [{"content": "beta release notes", "score": 0.3, "entities": ["beta"]}],
        EntityBoostConfig(enabled=True, category=None, anima_dir=anima_dir, boost=0.20, max_boost=0.80),
    )
    assert boosted[0]["entity_boost"] == pytest.approx(0.10)
