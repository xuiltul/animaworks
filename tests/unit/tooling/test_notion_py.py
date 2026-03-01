"""Tests for core.tools.notion — Notion API client and tool interface."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.tools.notion import (
    FB_DB_SCHEMA,
    NotionClient,
    build_page_url,
    dispatch,
    find_task_page_id,
    get_tool_schemas,
)


# ── build_page_url ────────────────────────────────────────────


class TestBuildPageUrl:
    def test_strips_hyphens(self):
        assert build_page_url("309fbbca-1204-8030-bdf8-daff8c7b5a5b") == (
            "https://www.notion.so/309fbbca12048030bdf8daff8c7b5a5b"
        )

    def test_no_hyphens_passthrough(self):
        assert build_page_url("abc123") == "https://www.notion.so/abc123"

    def test_empty_string(self):
        assert build_page_url("") == "https://www.notion.so/"


# ── NotionClient.create_database ──────────────────────────────


class TestCreateDatabase:
    def test_sends_correct_payload(self):
        client = MagicMock(spec=NotionClient)
        client._request = MagicMock(return_value={"id": "new-db-id"})

        # Call the real method with mocked _request
        result = NotionClient.create_database(
            client,
            parent_page_id="page-123",
            title="FB一覧",
            properties=FB_DB_SCHEMA,
        )

        client._request.assert_called_once_with(
            "POST",
            "/databases",
            {
                "parent": {"type": "page_id", "page_id": "page-123"},
                "title": [{"type": "text", "text": {"content": "FB一覧"}}],
                "properties": FB_DB_SCHEMA,
            },
        )
        assert result == {"id": "new-db-id"}


# ── NotionClient.find_child_database ──────────────────────────


class TestFindChildDatabase:
    def test_returns_matching_block(self):
        client = MagicMock(spec=NotionClient)
        client._request = MagicMock(return_value={
            "results": [
                {
                    "type": "paragraph",
                    "paragraph": {},
                },
                {
                    "type": "child_database",
                    "id": "db-456",
                    "child_database": {"title": "FB一覧"},
                },
            ],
        })

        result = NotionClient.find_child_database(client, "page-123", "FB一覧")

        client._request.assert_called_once_with(
            "GET", "/blocks/page-123/children",
        )
        assert result is not None
        assert result["id"] == "db-456"

    def test_returns_none_when_not_found(self):
        client = MagicMock(spec=NotionClient)
        client._request = MagicMock(return_value={
            "results": [
                {"type": "paragraph", "paragraph": {}},
            ],
        })

        result = NotionClient.find_child_database(client, "page-123", "FB一覧")
        assert result is None

    def test_returns_none_on_empty_results(self):
        client = MagicMock(spec=NotionClient)
        client._request = MagicMock(return_value={"results": []})

        result = NotionClient.find_child_database(client, "page-123", "FB一覧")
        assert result is None

    def test_does_not_match_different_title(self):
        client = MagicMock(spec=NotionClient)
        client._request = MagicMock(return_value={
            "results": [
                {
                    "type": "child_database",
                    "id": "db-789",
                    "child_database": {"title": "別のDB"},
                },
            ],
        })

        result = NotionClient.find_child_database(client, "page-123", "FB一覧")
        assert result is None


# ── find_task_page_id ─────────────────────────────────────────


class TestFindTaskPageId:
    @patch("core.tools.notion._resolve_database_id", return_value="default-db")
    def test_returns_page_id_when_found(self, _mock_resolve):
        client = MagicMock(spec=NotionClient)
        client.query_database = MagicMock(return_value={
            "results": [{"id": "page-abc"}],
        })

        result = find_task_page_id(client, "TASK-3296")

        client.query_database.assert_called_once_with(
            database_id="default-db",
            filter={"property": "Name", "title": {"contains": "TASK-3296"}},
            page_size=1,
        )
        assert result == "page-abc"

    @patch("core.tools.notion._resolve_database_id", return_value="default-db")
    def test_returns_none_when_not_found(self, _mock_resolve):
        client = MagicMock(spec=NotionClient)
        client.query_database = MagicMock(return_value={"results": []})

        result = find_task_page_id(client, "TASK-9999")
        assert result is None


# ── dispatch: notion_create_fb_db ─────────────────────────────


class TestDispatchCreateFbDb:
    @patch("core.tools.notion.NotionClient")
    @patch("core.tools.notion._resolve_database_id", return_value="default-db")
    def test_creates_new_fb_db(self, _mock_resolve, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # find_task_page_id → page found
        mock_client.query_database.return_value = {
            "results": [{"id": "page-100"}],
        }
        # find_child_database → not found
        mock_client.find_child_database.return_value = None
        # create_database → success
        mock_client.create_database.return_value = {"id": "new-fb-db-id"}

        result = dispatch("notion_create_fb_db", {"task_number": "TASK-3296"})

        assert result["status"] == "created"
        assert result["database_id"] == "new-fb-db-id"
        assert "notion.so" in result["url"]
        mock_client.create_database.assert_called_once_with(
            "page-100", "FB一覧", FB_DB_SCHEMA,
        )

    @patch("core.tools.notion.NotionClient")
    @patch("core.tools.notion._resolve_database_id", return_value="default-db")
    def test_skips_if_already_exists(self, _mock_resolve, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_client.query_database.return_value = {
            "results": [{"id": "page-100"}],
        }
        mock_client.find_child_database.return_value = {
            "id": "existing-db-id",
            "type": "child_database",
        }

        result = dispatch("notion_create_fb_db", {"task_number": "TASK-3296"})

        assert result["status"] == "already_exists"
        assert result["database_id"] == "existing-db-id"
        mock_client.create_database.assert_not_called()

    @patch("core.tools.notion.NotionClient")
    @patch("core.tools.notion._resolve_database_id", return_value="default-db")
    def test_raises_if_task_not_found(self, _mock_resolve, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_client.query_database.return_value = {"results": []}

        with pytest.raises(ValueError, match="Task page not found"):
            dispatch("notion_create_fb_db", {"task_number": "TASK-9999"})


# ── dispatch: notion_add_fb ────────────────────────────────────


class TestDispatchAddFb:
    @patch("core.tools.notion.NotionClient")
    @patch("core.tools.notion._resolve_database_id", return_value="default-db")
    def test_adds_fb_record(self, _mock_resolve, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.create_page.return_value = {"id": "fb-page-id"}

        result = dispatch("notion_add_fb", {
            "db_id": "fb-db-123",
            "fb_id": "FB-001",
            "問題の概要": "ボタンが反応しない",
            "再現方法": "ログイン後にボタンをクリック",
            "期待する動作": "次の画面に遷移する",
            "タグ": ["UI", "バグ"],
            "ステータス": "対応中",
        })

        assert result["page_id"] == "fb-page-id"
        assert "notion.so" in result["url"]

        call_args = mock_client.create_page.call_args
        props = call_args[0][1]
        assert props["FBのID"]["title"][0]["text"]["content"] == "FB-001"
        assert props["問題の概要"]["rich_text"][0]["text"]["content"] == "ボタンが反応しない"
        assert props["ステータス"]["select"]["name"] == "対応中"
        assert len(props["タグ"]["multi_select"]) == 2

    @patch("core.tools.notion.NotionClient")
    @patch("core.tools.notion._resolve_database_id", return_value="default-db")
    def test_defaults_status_to_未対応(self, _mock_resolve, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.create_page.return_value = {"id": "fb-page-id"}

        dispatch("notion_add_fb", {
            "db_id": "fb-db-123",
            "fb_id": "FB-002",
        })

        call_args = mock_client.create_page.call_args
        props = call_args[0][1]
        assert props["ステータス"]["select"]["name"] == "未対応"

    @patch("core.tools.notion.NotionClient")
    @patch("core.tools.notion._resolve_database_id", return_value="default-db")
    def test_minimal_fields(self, _mock_resolve, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.create_page.return_value = {"id": "fb-page-id"}

        dispatch("notion_add_fb", {
            "db_id": "fb-db-123",
            "fb_id": "FB-003",
        })

        call_args = mock_client.create_page.call_args
        props = call_args[0][1]
        # Only title and status should be set
        assert "FBのID" in props
        assert "ステータス" in props
        assert "問題の概要" not in props
        assert "タグ" not in props


# ── get_tool_schemas ──────────────────────────────────────────


class TestGetToolSchemas:
    def test_includes_notion_create_fb_db(self):
        schemas = get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "notion_create_fb_db" in names

    def test_fb_db_schema_requires_task_number(self):
        schemas = get_tool_schemas()
        fb_schema = next(s for s in schemas if s["name"] == "notion_create_fb_db")
        assert "task_number" in fb_schema["input_schema"]["required"]

    def test_includes_notion_add_fb(self):
        schemas = get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "notion_add_fb" in names

    def test_add_fb_requires_db_id_and_fb_id(self):
        schemas = get_tool_schemas()
        add_fb = next(s for s in schemas if s["name"] == "notion_add_fb")
        assert "db_id" in add_fb["input_schema"]["required"]
        assert "fb_id" in add_fb["input_schema"]["required"]

    def test_has_five_schemas(self):
        schemas = get_tool_schemas()
        assert len(schemas) == 5


# ── FB_DB_SCHEMA ──────────────────────────────────────────────


class TestFbDbSchema:
    def test_has_required_fields(self):
        expected_fields = {
            "FBのID", "問題の概要", "再現方法", "期待する動作", "タグ", "ステータス",
        }
        assert set(FB_DB_SCHEMA.keys()) == expected_fields

    def test_title_field(self):
        assert "title" in FB_DB_SCHEMA["FBのID"]

    def test_status_select_options(self):
        options = FB_DB_SCHEMA["ステータス"]["select"]["options"]
        option_names = {o["name"] for o in options}
        assert option_names == {"未対応", "対応中", "解消済み"}
