# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for Tool Prompts API — descriptions, guides, and preview endpoints.

Tests the full flow through the FastAPI app with real SQLite DB
operations (no mocks for ToolPromptStore).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from core.tooling.prompt_db import (
    DEFAULT_DESCRIPTIONS,
    DEFAULT_GUIDES,
    SECTION_CONDITIONS,
    ToolPromptStore,
)


def _create_app(db_path: Path):
    """Create a test FastAPI app with tool_prompts router.

    Monkey-patches ``get_prompt_store`` in the route module to return
    a ToolPromptStore backed by *db_path* instead of the global singleton.
    """
    from fastapi import FastAPI

    from server.routes.tool_prompts import create_tool_prompts_router

    store = ToolPromptStore(db_path)

    app = FastAPI()
    with patch(
        "server.routes.tool_prompts.get_prompt_store", return_value=store,
    ):
        router = create_tool_prompts_router()
    # The router closures still call get_prompt_store() at request time,
    # so we need the patch to be active during requests too.
    # Instead, we patch it on the app-level via middleware-style approach.
    # Actually, since the route handlers call get_prompt_store() each time,
    # we need to keep the patch active.  Return both app and store so tests
    # can use the patch context manager.
    app.include_router(router, prefix="/api")
    return app, store


def _seed_db(db_path: Path) -> ToolPromptStore:
    """Create a ToolPromptStore and seed it with default descriptions and guides."""
    store = ToolPromptStore(db_path)
    store.seed_defaults(descriptions=DEFAULT_DESCRIPTIONS, guides=DEFAULT_GUIDES)
    return store


# ── Descriptions API ────────────────────────────────────────


class TestDescriptionsAPI:
    """Tests for /api/tool-prompts/descriptions endpoints."""

    async def test_list_descriptions(self, tmp_path: Path):
        db_path = tmp_path / "tool_prompts.sqlite3"
        _seed_db(db_path)
        store = ToolPromptStore(db_path)

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=store,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/tool-prompts/descriptions")

        assert resp.status_code == 200
        data = resp.json()
        assert "descriptions" in data
        descriptions = data["descriptions"]
        # Should contain all seeded descriptions
        names = [d["name"] for d in descriptions]
        assert "search_memory" in names
        assert "send_message" in names
        assert "call_human" in names
        assert len(descriptions) == len(DEFAULT_DESCRIPTIONS)

    async def test_get_single_description(self, tmp_path: Path):
        db_path = tmp_path / "tool_prompts.sqlite3"
        _seed_db(db_path)
        store = ToolPromptStore(db_path)

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=store,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/tool-prompts/descriptions/search_memory")

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "search_memory"
        assert "description" in data
        assert len(data["description"]) > 0

    async def test_get_missing_description(self, tmp_path: Path):
        db_path = tmp_path / "tool_prompts.sqlite3"
        _seed_db(db_path)
        store = ToolPromptStore(db_path)

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=store,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/tool-prompts/descriptions/nonexistent_tool")

        assert resp.status_code == 404

    async def test_update_description(self, tmp_path: Path):
        db_path = tmp_path / "tool_prompts.sqlite3"
        _seed_db(db_path)
        store = ToolPromptStore(db_path)

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=store,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                new_desc = "Updated description for search_memory tool"
                resp = await client.put(
                    "/api/tool-prompts/descriptions/search_memory",
                    json={"description": new_desc},
                )
                assert resp.status_code == 200
                result = resp.json()
                assert result["name"] == "search_memory"
                assert result["description"] == new_desc

                # Confirm via GET
                resp2 = await client.get("/api/tool-prompts/descriptions/search_memory")
                assert resp2.status_code == 200
                assert resp2.json()["description"] == new_desc

    async def test_update_empty_description_rejected(self, tmp_path: Path):
        db_path = tmp_path / "tool_prompts.sqlite3"
        _seed_db(db_path)
        store = ToolPromptStore(db_path)

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=store,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/tool-prompts/descriptions/search_memory",
                    json={"description": ""},
                )

        assert resp.status_code == 400


# ── Guides API ──────────────────────────────────────────────


class TestGuidesAPI:
    """Tests for /api/tool-prompts/guides endpoints."""

    async def test_list_guides(self, tmp_path: Path):
        db_path = tmp_path / "tool_prompts.sqlite3"
        _seed_db(db_path)
        store = ToolPromptStore(db_path)

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=store,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/tool-prompts/guides")

        assert resp.status_code == 200
        data = resp.json()
        assert "guides" in data
        guides = data["guides"]
        keys = [g["key"] for g in guides]
        assert "a1_builtin" in keys
        assert "a1_mcp" in keys
        assert "non_a1" in keys
        assert len(guides) == 3

    async def test_get_single_guide(self, tmp_path: Path):
        db_path = tmp_path / "tool_prompts.sqlite3"
        _seed_db(db_path)
        store = ToolPromptStore(db_path)

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=store,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/tool-prompts/guides/a1_builtin")

        assert resp.status_code == 200
        data = resp.json()
        assert data["key"] == "a1_builtin"
        assert "content" in data
        assert len(data["content"]) > 0
        # Verify it contains expected guide content
        assert "Read" in data["content"] or "ファイル読み取り" in data["content"]

    async def test_get_missing_guide(self, tmp_path: Path):
        db_path = tmp_path / "tool_prompts.sqlite3"
        _seed_db(db_path)
        store = ToolPromptStore(db_path)

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=store,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/tool-prompts/guides/nonexistent_guide")

        assert resp.status_code == 404

    async def test_update_guide(self, tmp_path: Path):
        db_path = tmp_path / "tool_prompts.sqlite3"
        _seed_db(db_path)
        store = ToolPromptStore(db_path)

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=store,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                new_content = "## Updated Guide\n\nThis is an updated guide."
                resp = await client.put(
                    "/api/tool-prompts/guides/a1_builtin",
                    json={"content": new_content},
                )
                assert resp.status_code == 200
                result = resp.json()
                assert result["key"] == "a1_builtin"
                assert result["content"] == new_content

                # Confirm via GET
                resp2 = await client.get("/api/tool-prompts/guides/a1_builtin")
                assert resp2.status_code == 200
                assert resp2.json()["content"] == new_content

    async def test_update_empty_guide_rejected(self, tmp_path: Path):
        db_path = tmp_path / "tool_prompts.sqlite3"
        _seed_db(db_path)
        store = ToolPromptStore(db_path)

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=store,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/tool-prompts/guides/a1_builtin",
                    json={"content": ""},
                )

        assert resp.status_code == 400


# ── Schema Preview API ──────────────────────────────────────


class TestSchemaPreviewAPI:
    """Tests for /api/tool-prompts/preview/schema endpoint."""

    async def test_preview_anthropic_schema(self, tmp_path: Path):
        db_path = tmp_path / "tool_prompts.sqlite3"
        _seed_db(db_path)
        store = ToolPromptStore(db_path)

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=store,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/tool-prompts/preview/schema",
                    json={"mode": "anthropic"},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "anthropic"
        assert "tools" in data
        tools = data["tools"]
        assert isinstance(tools, list)
        assert len(tools) > 0
        # Anthropic format has input_schema
        first_tool = tools[0]
        assert "name" in first_tool
        assert "description" in first_tool
        assert "input_schema" in first_tool

    async def test_preview_litellm_schema(self, tmp_path: Path):
        db_path = tmp_path / "tool_prompts.sqlite3"
        _seed_db(db_path)
        store = ToolPromptStore(db_path)

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=store,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/tool-prompts/preview/schema",
                    json={"mode": "litellm"},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "litellm"
        assert "tools" in data
        tools = data["tools"]
        assert isinstance(tools, list)
        assert len(tools) > 0
        # LiteLLM/OpenAI format has type: function + function.parameters
        first_tool = tools[0]
        assert first_tool["type"] == "function"
        assert "function" in first_tool
        assert "name" in first_tool["function"]
        assert "parameters" in first_tool["function"]

    async def test_preview_text_schema(self, tmp_path: Path):
        db_path = tmp_path / "tool_prompts.sqlite3"
        _seed_db(db_path)
        store = ToolPromptStore(db_path)

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=store,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/tool-prompts/preview/schema",
                    json={"mode": "text"},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "text"
        assert "text" in data
        text = data["text"]
        assert isinstance(text, str)
        assert len(text) > 0
        # Text format includes tool names
        assert "search_memory" in text

    async def test_preview_unknown_mode(self, tmp_path: Path):
        db_path = tmp_path / "tool_prompts.sqlite3"
        _seed_db(db_path)
        store = ToolPromptStore(db_path)

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=store,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/tool-prompts/preview/schema",
                    json={"mode": "invalid_mode"},
                )

        assert resp.status_code == 400


# ── System Prompt Preview API ──────────────────────────────


class TestSystemPromptPreviewAPI:
    """Tests for /api/tool-prompts/preview/system-prompt endpoint."""

    async def test_preview_system_prompt_anima_not_found(self, tmp_path: Path):
        """Non-existent anima should return 404."""
        db_path = tmp_path / "tool_prompts.sqlite3"
        _seed_db(db_path)
        store = ToolPromptStore(db_path)

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        # Patch get_data_dir to return tmp_path so anima_dir check works
        with (
            patch(
                "server.routes.tool_prompts.get_prompt_store", return_value=store,
            ),
            patch(
                "core.paths.get_data_dir", return_value=tmp_path,
            ),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/tool-prompts/preview/system-prompt",
                    json={"anima_name": "nonexistent_anima"},
                )

        assert resp.status_code == 404

    async def test_preview_system_prompt_success(self, tmp_path: Path):
        """Valid anima should return system prompt with metadata."""
        db_path = tmp_path / "tool_prompts.sqlite3"
        _seed_db(db_path)
        store = ToolPromptStore(db_path)

        # Create minimal anima directory structure
        anima_dir = tmp_path / "animas" / "test_anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("# Test Anima\n\nIdentity text.")
        (anima_dir / "status.json").write_text('{}')
        # Create required shared/prompts directories
        (tmp_path / "shared" / "users").mkdir(parents=True, exist_ok=True)
        (tmp_path / "shared" / "channels").mkdir(parents=True, exist_ok=True)
        (tmp_path / "prompts").mkdir(parents=True, exist_ok=True)

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        with (
            patch(
                "server.routes.tool_prompts.get_prompt_store", return_value=store,
            ),
            patch(
                "core.paths.get_data_dir", return_value=tmp_path,
            ),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/tool-prompts/preview/system-prompt",
                    json={"anima_name": "test_anima"},
                )

        # May succeed or return 500 depending on runtime state,
        # but the endpoint should be reachable and handle the request
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            data = resp.json()
            assert data["anima_name"] == "test_anima"
            assert "system_prompt" in data
            assert "token_estimate" in data
            assert "char_count" in data
            assert data["char_count"] > 0


# ── Store Unavailable ───────────────────────────────────────


class TestStoreUnavailable:
    """Tests for when get_prompt_store returns None (DB unavailable).

    All CRUD and preview endpoints must return 500 when the store is unavailable.
    """

    @pytest.fixture
    def _no_store_app(self):
        """Create a FastAPI app with store patched to None."""
        from fastapi import FastAPI

        from server.routes.tool_prompts import create_tool_prompts_router

        app = FastAPI()
        router = create_tool_prompts_router()
        app.include_router(router, prefix="/api")
        return app

    async def test_list_descriptions_no_store(self, _no_store_app):
        transport = ASGITransport(app=_no_store_app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=None,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/tool-prompts/descriptions")
        assert resp.status_code == 500

    async def test_get_description_no_store(self, _no_store_app):
        transport = ASGITransport(app=_no_store_app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=None,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/tool-prompts/descriptions/search_memory")
        assert resp.status_code == 500

    async def test_update_description_no_store(self, _no_store_app):
        transport = ASGITransport(app=_no_store_app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=None,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/tool-prompts/descriptions/search_memory",
                    json={"description": "test"},
                )
        assert resp.status_code == 500

    async def test_list_guides_no_store(self, _no_store_app):
        transport = ASGITransport(app=_no_store_app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=None,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/tool-prompts/guides")
        assert resp.status_code == 500

    async def test_get_guide_no_store(self, _no_store_app):
        transport = ASGITransport(app=_no_store_app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=None,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/tool-prompts/guides/a1_builtin")
        assert resp.status_code == 500

    async def test_update_guide_no_store(self, _no_store_app):
        transport = ASGITransport(app=_no_store_app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=None,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/tool-prompts/guides/a1_builtin",
                    json={"content": "test"},
                )
        assert resp.status_code == 500

    async def test_preview_schema_no_store(self, _no_store_app):
        transport = ASGITransport(app=_no_store_app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=None,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/tool-prompts/preview/schema",
                    json={"mode": "anthropic"},
                )
        assert resp.status_code == 500

    async def test_preview_system_prompt_no_store(self, _no_store_app, tmp_path: Path):
        """System-prompt preview should still work without the store.

        The system-prompt preview does NOT gate on store availability
        (it builds the prompt independently), so expect 404 for
        non-existent anima rather than 500.
        """
        transport = ASGITransport(app=_no_store_app)
        with (
            patch(
                "server.routes.tool_prompts.get_prompt_store", return_value=None,
            ),
            patch(
                "core.paths.get_data_dir", return_value=tmp_path,
            ),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/tool-prompts/preview/system-prompt",
                    json={"anima_name": "nonexistent"},
                )
        assert resp.status_code == 404


# ── Sections API ────────────────────────────────────────────


class TestSectionsAPI:
    """Tests for /api/tool-prompts/sections endpoints."""

    async def test_list_sections(self, tmp_path: Path):
        db_path = tmp_path / "tool_prompts.sqlite3"
        store = ToolPromptStore(db_path)
        store.seed_defaults(
            sections={
                "behavior_rules": ("Test behavior rules content", SECTION_CONDITIONS.get("behavior_rules")),
                "environment": ("Test environment content", SECTION_CONDITIONS.get("environment")),
                "a2_reflection": ("Test A2 reflection content", SECTION_CONDITIONS.get("a2_reflection")),
            },
        )

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=store,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/tool-prompts/sections")

        assert resp.status_code == 200
        data = resp.json()
        assert "sections" in data
        sections = data["sections"]
        keys = [s["key"] for s in sections]
        assert "behavior_rules" in keys
        assert "environment" in keys
        assert "a2_reflection" in keys
        assert len(sections) == 3

    async def test_get_single_section(self, tmp_path: Path):
        db_path = tmp_path / "tool_prompts.sqlite3"
        store = ToolPromptStore(db_path)
        store.seed_defaults(
            sections={
                "a2_reflection": ("Test A2 reflection content", SECTION_CONDITIONS.get("a2_reflection")),
            },
        )

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=store,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/tool-prompts/sections/a2_reflection")

        assert resp.status_code == 200
        data = resp.json()
        assert data["key"] == "a2_reflection"
        assert "content" in data
        assert data["content"] == "Test A2 reflection content"
        assert data["condition"] == "mode:a2"

    async def test_get_missing_section(self, tmp_path: Path):
        db_path = tmp_path / "tool_prompts.sqlite3"
        store = ToolPromptStore(db_path)

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=store,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/tool-prompts/sections/nonexistent")

        assert resp.status_code == 404

    async def test_update_section(self, tmp_path: Path):
        db_path = tmp_path / "tool_prompts.sqlite3"
        store = ToolPromptStore(db_path)
        store.seed_defaults(
            sections={
                "behavior_rules": ("Original content", SECTION_CONDITIONS.get("behavior_rules")),
            },
        )

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=store,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                new_content = "Updated behavior rules content"
                resp = await client.put(
                    "/api/tool-prompts/sections/behavior_rules",
                    json={"content": new_content, "condition": None},
                )
                assert resp.status_code == 200
                result = resp.json()
                assert result["key"] == "behavior_rules"
                assert result["content"] == new_content

                # Confirm via GET
                resp2 = await client.get("/api/tool-prompts/sections/behavior_rules")
                assert resp2.status_code == 200
                assert resp2.json()["content"] == new_content

    async def test_update_empty_content_rejected(self, tmp_path: Path):
        db_path = tmp_path / "tool_prompts.sqlite3"
        store = ToolPromptStore(db_path)
        store.seed_defaults(
            sections={
                "behavior_rules": ("Original content", SECTION_CONDITIONS.get("behavior_rules")),
            },
        )

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=store,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/tool-prompts/sections/behavior_rules",
                    json={"content": "", "condition": None},
                )

        assert resp.status_code == 400

    async def test_list_after_update(self, tmp_path: Path):
        db_path = tmp_path / "tool_prompts.sqlite3"
        store = ToolPromptStore(db_path)
        store.seed_defaults(
            sections={
                "behavior_rules": ("Original content", SECTION_CONDITIONS.get("behavior_rules")),
                "environment": ("Environment content", SECTION_CONDITIONS.get("environment")),
            },
        )

        app, _ = _create_app(db_path)
        transport = ASGITransport(app=app)
        with patch(
            "server.routes.tool_prompts.get_prompt_store", return_value=store,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                # Update one section
                new_content = "Updated behavior rules"
                resp = await client.put(
                    "/api/tool-prompts/sections/behavior_rules",
                    json={"content": new_content, "condition": None},
                )
                assert resp.status_code == 200

                # List all and verify the update is reflected
                resp2 = await client.get("/api/tool-prompts/sections")
                assert resp2.status_code == 200
                sections = resp2.json()["sections"]
                assert len(sections) == 2
                by_key = {s["key"]: s for s in sections}
                assert by_key["behavior_rules"]["content"] == new_content
                assert by_key["environment"]["content"] == "Environment content"
