from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient


def _write_skill(root: Path, name: str, *, body: str = "Use this skill.", extra: str = "") -> Path:
    skill_dir = root / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    path.write_text(
        "---\n"
        f"name: {name}\n"
        "description: route test skill\n"
        f"{extra}"
        "---\n\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return path


def _make_app(animas_dir: Path):
    from fastapi import FastAPI

    from server.routes.skills import create_skills_router

    app = FastAPI()
    app.state.animas_dir = animas_dir
    app.include_router(create_skills_router(), prefix="/api")
    return app


async def test_list_set_get_and_clear_active_skills(tmp_path: Path) -> None:
    animas_dir = tmp_path / "animas"
    anima_dir = animas_dir / "alice"
    common_dir = tmp_path / "common_skills"
    anima_dir.mkdir(parents=True)
    common_dir.mkdir()
    _write_skill(anima_dir, "writer", body="Writer body.")

    app = _make_app(animas_dir)
    transport = ASGITransport(app=app)
    with patch("core.paths.get_common_skills_dir", return_value=common_dir):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            listed = await client.get("/api/animas/alice/skills")
            assert listed.status_code == 200
            assert listed.json()["skills"][0]["active"] is False

            updated = await client.put(
                "/api/animas/alice/skills/active",
                json={"thread_id": "default", "refs": ["writer"]},
            )
            assert updated.status_code == 200
            assert [item["name"] for item in updated.json()["accepted"]] == ["writer"]

            active = await client.get("/api/animas/alice/skills/active")
            assert [item["path"] for item in active.json()["accepted"]] == ["skills/writer/SKILL.md"]

            listed_after = await client.get("/api/animas/alice/skills")
            assert listed_after.json()["skills"][0]["active"] is True

            cleared = await client.put(
                "/api/animas/alice/skills/active",
                json={"thread_id": "default", "refs": []},
            )
            assert cleared.status_code == 200
            assert cleared.json()["accepted"] == []


async def test_active_skills_route_returns_rejections_and_requires_confirm(tmp_path: Path) -> None:
    animas_dir = tmp_path / "animas"
    anima_dir = animas_dir / "alice"
    common_dir = tmp_path / "common_skills"
    anima_dir.mkdir(parents=True)
    common_dir.mkdir()
    _write_skill(anima_dir, "warn-skill", extra="security:\n  verdict: warn\n")

    app = _make_app(animas_dir)
    transport = ASGITransport(app=app)
    with patch("core.paths.get_common_skills_dir", return_value=common_dir):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            rejected = await client.put(
                "/api/animas/alice/skills/active",
                json={"refs": ["warn-skill"]},
            )
            assert rejected.status_code == 200
            assert rejected.json()["accepted"] == []
            assert rejected.json()["rejections"] == [{"ref": "warn-skill", "reason": "security_warn"}]

            accepted = await client.put(
                "/api/animas/alice/skills/active",
                json={"refs": ["warn-skill"], "confirm_risk": True},
            )
            assert [item["name"] for item in accepted.json()["accepted"]] == ["warn-skill"]
            assert accepted.json()["warnings"] == [
                {
                    "ref": "warn-skill",
                    "reason": "security_warn_allowed",
                    "name": "warn-skill",
                    "path": "skills/warn-skill/SKILL.md",
                }
            ]


async def test_skills_route_validates_anima_and_thread(tmp_path: Path) -> None:
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir()
    app = _make_app(animas_dir)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.get("/api/animas/missing/skills")
        assert missing.status_code == 404

        invalid_name = await client.get("/api/animas/../skills")
        assert invalid_name.status_code in {404, 405}

        anima_dir = animas_dir / "alice"
        anima_dir.mkdir()
        invalid_thread = await client.get("/api/animas/alice/skills", params={"thread_id": "../bad"})
        assert invalid_thread.status_code == 400


async def test_trust_skill_route_promotes_probation_skill(tmp_path: Path) -> None:
    animas_dir = tmp_path / "animas"
    anima_dir = animas_dir / "alice"
    common_dir = tmp_path / "common_skills"
    anima_dir.mkdir(parents=True)
    common_dir.mkdir()
    _write_skill(
        anima_dir,
        "writer",
        body="Writer body.",
        extra=(
            "trust_level: community\n"
            "promotion_status: probation\n"
            "skill_policy:\n"
            "  use_mode: candidate_hint\n"
            "  injection: pointer_preferred\n"
        ),
    )

    app = _make_app(animas_dir)
    transport = ASGITransport(app=app)
    with patch("core.paths.get_common_skills_dir", return_value=common_dir):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            promoted = await client.post(
                "/api/animas/alice/skills/trust",
                json={"ref": "writer", "trusted_by": "user", "trust_reason": "human_instruction"},
            )

    assert promoted.status_code == 200
    data = promoted.json()
    assert data["status"] == "trusted"
    assert data["skill_name"] == "writer"
    text = (anima_dir / "skills" / "writer" / "SKILL.md").read_text(encoding="utf-8")
    assert "trust_level: trusted" in text
    assert "promotion_status: trusted" in text
    assert "trusted_by: user" in text
